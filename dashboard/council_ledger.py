"""
council_ledger.py — append-only forward test for the AI Council.

STATUS: column definitions and file layout only. Nothing writes yet, nothing calls this, and
no CSV is created until the first real lock. Built now rather than later on purpose - see
RULE 5 and the bug it caught in player_ledger.py.

WHY IT EXISTS BEFORE THE PREDICTOR DOES
The Council's whole claim is that it might forecast better than the model. That claim is only
testable against predictions written down before kickoff. When player_ledger.py was first
run it locked 485 GW4 rows a DAY AFTER GW4 was played; had there been no guard, those
retrodictions would have been scored as a forward test and the minutes model would have looked
clairvoyant. The `late` column is that guard, and it is here from the start so the Council can
never be graded on a game it already knows the result of.

DESIGN, FOLLOWING player_ledger.py
  * one row per fixture, keyed "{home}|{away}" - the same key record_ledger uses, stable when
    TV picks move kickoff times
  * a row is written ONCE at lock time and its prediction fields are never restated
  * grading fills the result columns later, in place
  * `late=1` marks a row locked at or after kickoff; excluded from every score

WHAT IT STORES THAT THE OTHER LEDGERS DO NOT
The model's and the market's numbers are copied in beside the Council's. That looks redundant
against ledger.csv, but it is the point: this ledger answers "was the Council closer than the
model, ON THE SAME FIXTURE" without a join, and a join across two append-only files written at
different moments is exactly how a denominator mismatch gets in. RULE 33 was that bug - the
props Brier averaged the model over 440 markets and Kalshi over 387, and ~88% of the apparent
edge was the mismatch rather than skill. Three RPS columns stored side by side cannot drift
apart that way.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "council_ledger.csv")

# Identity and provenance. `late` is the forward-test guard, not a diagnostic.
ID_COLS = [
    "fixture_key",      # "{home}|{away}" - matches record_ledger's convention
    "kickoff",          # ISO kickoff as known at lock time
    "locked_at",        # when this row was written
    "late",             # "1" if locked at/after kickoff -> a retrodiction, never scored
    "home_team",
    "away_team",
]

# What the production model said, copied in at lock time so scoring needs no join.
MODEL_COLS = ["model_home", "model_draw", "model_away"]

# Sportsbook implied probabilities at lock time. May be blank: books open ~1 week out.
MARKET_COLS = ["market_home", "market_draw", "market_away"]

# The Council's own forecast, plus how much its analysts agreed.
COUNCIL_COLS = ["council_home", "council_draw", "council_away", "consensus_score"]

# Filled in after the match. Every forecaster is scored on the SAME row, same denominator.
RESULT_COLS = [
    "actual_result",    # "H" / "D" / "A" - the letters the existing ledger already uses
    "council_brier",
    "council_rps",
    "model_rps",
    "market_rps",
]

COLS = ID_COLS + MODEL_COLS + MARKET_COLS + COUNCIL_COLS + RESULT_COLS

# Written once at lock time and never restated. Anything outside this set may be filled in
# later by grading; anything inside it is frozen the moment the row exists.
LOCKED_COLS = frozenset(ID_COLS + MODEL_COLS + MARKET_COLS + COUNCIL_COLS)

# Grading writes only these.
GRADED_COLS = frozenset(RESULT_COLS)


# ---------------------------------------------------------------------------------------
# SCORING — copied from build_dashboard.py, deliberately, and proved identical by test.
#
# Importing them would drag the whole desk in: build_dashboard imports prem_dixon_coles,
# numpy, pandas and every feed module, and it is where the Council will eventually be wired
# FROM - so importing it here would invert the dependency and set up a cycle the first time
# build_dashboard imports council_ledger. Copying two five-line pure functions is the lesser
# evil, and test_council_ledger.py imports the originals and asserts byte-identical output
# across a grid of inputs so the copies cannot drift.
#
# SCALE. The existing ledger STORES probabilities as 0-100 integers but hands the scorers
# 0-1 (it divides at the call site). CouncilPrediction is natively 0-1, so this ledger stores
# 0-1 and scores 0-1 - no conversion anywhere, because a silent factor-of-100 would make every
# Council score look brilliant and be meaningless.
# ---------------------------------------------------------------------------------------
_Y = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}


def brier(p, outcome):
    """Multiclass Brier for a 3-way (H,D,A) forecast. Lower = better. `p` is 0-1."""
    y = _Y[outcome]
    return round(sum((pi - yi) ** 2 for pi, yi in zip(p, y)), 4)


def rps(p, outcome):
    """Ranked probability score for ordered H>D>A. Lower = better. `p` is 0-1."""
    y = _Y[outcome]
    cp = [p[0], p[0] + p[1]]
    cy = [y[0], y[0] + y[1]]
    return round(sum((a - b) ** 2 for a, b in zip(cp, cy)) / 2.0, 4)


def outcome_from_result(result):
    """'2-1' -> 'H'. Returns None for anything unparseable rather than guessing."""
    try:
        h, a = (int(x) for x in str(result).strip().split("-"))
    except (ValueError, AttributeError):
        return None
    return "H" if h > a else ("A" if h < a else "D")


# ---------------------------------------------------------------------------------------
# STORAGE
# ---------------------------------------------------------------------------------------
def _f(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _blank(v):
    return v in (None, "")


# Every column that holds a number. Read back as float, or None when the cell is blank -
# NEVER as 0.0, because "no bookmaker price" and "the book says this cannot happen" are
# different facts and only one of them should ever be scored.
NUMERIC_COLS = frozenset(MODEL_COLS + MARKET_COLS + COUNCIL_COLS
                         + ["council_brier", "council_rps", "model_rps", "market_rps"])


def _typed(row):
    """One stored row with its numeric columns parsed.

    csv gives back strings for everything, so without this `record()` returned floats when it
    wrote a new row and strings when it handed back an existing one - the same function, two
    different types, decided by whether the fixture happened to be seen before. A caller doing
    arithmetic on the result would work all week and break on the second refresh.
    """
    out = dict(row)
    for c in NUMERIC_COLS:
        if c in out:
            out[c] = None if _blank(out[c]) else _f(out[c])
    return out


def load(path=None, typed=True):
    """{fixture_key: row} from disk. Empty dict when the ledger does not exist yet."""
    import csv
    p = path or PATH
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        rows = {r["fixture_key"]: r for r in csv.DictReader(f)}
    return {k: _typed(v) for k, v in rows.items()} if typed else rows


def save(rows, path=None):
    """Write every row back, in a stable order. Called only by record() and settle()."""
    import csv
    p = path or PATH
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows):
            w.writerow({c: ("" if rows[key].get(c) is None else rows[key].get(c, ""))
                        for c in COLS})
    return p


def locked(fixture_key, path=None):
    """The stored Council prediction for this fixture, or None.

    This is what stops the automation paying for the same forecast twice: the caller asks
    here first, and only calls predict() when this returns None. It is also what makes the
    forward test meaningful - a prediction that can be recomputed on every refresh is not a
    prediction, it is a running commentary.
    """
    row = load(path).get(fixture_key)
    return dict(row) if row else None


def is_locked(fixture_key, path=None):
    return fixture_key in load(path)


def record(context, prediction, *, now=None, path=None):
    """Lock one Council prediction for one fixture. Returns the stored row.

    THE FIRST valid prediction wins and is never rewritten. A later refresh that calls this
    again gets the ORIGINAL row back untouched - that is the whole point of a forward test,
    and it is why the automation must consult locked() first rather than relying on this.

    A prediction first recorded at or after kickoff is stamped late=1. It is kept, because it
    is still a record of what the desk showed, but summary() excludes it from every score.
    RULE 5 enforced in code: player_ledger.py once locked 485 rows a day after the gameweek
    was played, and without this flag they would have been graded as foresight.
    """
    from council import CouncilPrediction, MatchContext          # local: avoid import cycles
    if not isinstance(context, MatchContext):
        raise TypeError(f"context must be a MatchContext, got {type(context).__name__}")
    if not isinstance(prediction, CouncilPrediction):
        raise TypeError(
            f"prediction must be a CouncilPrediction, got {type(prediction).__name__}")

    now = now or _utcnow()
    rows = load(path)
    key = context.fixture_key()

    existing = rows.get(key)
    if existing:
        return dict(existing)            # already locked - hand back what is stored

    ch, cd, ca = prediction.probabilities()
    row = {c: (None if c in NUMERIC_COLS else "") for c in COLS}
    row.update({
        "fixture_key": key,
        "kickoff": context.kickoff or "",
        "locked_at": now,
        "late": "1" if _is_late(context.kickoff, now) else "",
        "home_team": context.home_team,
        "away_team": context.away_team,
        "model_home": _n(context.model_home), "model_draw": _n(context.model_draw),
        "model_away": _n(context.model_away),
        "market_home": _n(context.market_home), "market_draw": _n(context.market_draw),
        "market_away": _n(context.market_away),
        "council_home": round(ch, 6), "council_draw": round(cd, 6),
        "council_away": round(ca, 6),
        "consensus_score": round(prediction.consensus_score, 6),
    })
    rows[key] = row
    save(rows, path)
    return dict(row)


def _n(v):
    """A probability for storage, or '' when it was never supplied.

    Missing market data stays MISSING. Writing 0.0 for "no bookmaker price" would make an
    absent market look like a confident prediction of impossibility, and it would then be
    scored as one.
    """
    return None if v is None else round(float(v), 6)


def _utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="minutes")


def _is_late(kickoff, now):
    """True when `now` is at or after kickoff. Unparseable or absent kickoff -> not late.

    Deliberately permissive: the desk's kickoff strings are not always ISO ('Fri 18 Sep
    19:00'), and refusing to lock a prediction because a date failed to parse would be worse
    than failing to flag one. The flag is a guard against grading retrodictions, not a
    security boundary.
    """
    from datetime import datetime
    if not kickoff:
        return False
    try:
        ko = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
        nw = datetime.fromisoformat(str(now).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if (ko.tzinfo is None) != (nw.tzinfo is None):      # compare like with like
        ko = ko.replace(tzinfo=nw.tzinfo) if ko.tzinfo is None else ko
        nw = nw.replace(tzinfo=ko.tzinfo) if nw.tzinfo is None else nw
    return nw >= ko


def settle(results, *, now=None, path=None):
    """Grade finished fixtures. `results` maps fixture_key -> '2-1' (or 'H'/'D'/'A').

    Writes ONLY the grading columns. The locked prediction fields are read, never touched -
    asserted below rather than merely intended, because a grading pass that can rewrite a
    forecast is not a forward test.

    market_rps stays blank when no market price was stored. An ungraded cell is honest; a
    zero would quietly enter the averages.
    """
    now = now or _utcnow()
    rows = load(path)
    graded = 0
    for key, result in (results or {}).items():
        row = rows.get(key)
        if row is None or not _blank(row.get("actual_result")):
            continue                                  # unknown fixture, or already graded
        out = result if result in _Y else outcome_from_result(result)
        if out is None:
            continue

        before = {c: row.get(c) for c in LOCKED_COLS}   # for the no-overwrite assertion

        council = [_f(row.get(f"council_{s}")) for s in ("home", "draw", "away")]
        row["actual_result"] = out
        if all(p is not None for p in council):
            row["council_brier"] = brier(council, out)
            row["council_rps"] = rps(council, out)
        model = [_f(row.get(f"model_{s}")) for s in ("home", "draw", "away")]
        if all(p is not None for p in model):
            row["model_rps"] = rps(model, out)
        market = [_f(row.get(f"market_{s}")) for s in ("home", "draw", "away")]
        if all(p is not None for p in market):
            row["market_rps"] = rps(market, out)
        # else: market_rps stays "" - see the docstring

        after = {c: row.get(c) for c in LOCKED_COLS}
        if before != after:                            # cannot happen; proves it cannot
            changed = [c for c in LOCKED_COLS if before[c] != after[c]]
            raise RuntimeError(
                f"settle() modified locked column(s) {changed} on {key} - refusing to save")
        graded += 1

    if graded:
        save(rows, path)
    return graded


def summary(path=None):
    """Forward-test scorecard. Rows locked after kickoff are EXCLUDED from every number."""
    rows = list(load(path).values())
    late = [r for r in rows if r.get("late")]
    valid = [r for r in rows if not r.get("late")]
    done = [r for r in valid if not _blank(r.get("actual_result"))]

    def mean(col, src):
        v = [_f(r[col]) for r in src if not _blank(r.get(col))]
        return round(sum(v) / len(v), 4) if v else None

    with_market = [r for r in done if not _blank(r.get("market_rps"))]
    return {
        "tracked": len(rows), "late": len(late), "valid": len(valid), "graded": len(done),
        "council_brier": mean("council_brier", done),
        "council_rps": mean("council_rps", done),
        "model_rps": mean("model_rps", done),
        # market is averaged over ONLY the fixtures that had a price - comparing a mean over
        # 40 rows against a mean over 12 is the denominator bug that made the props edge look
        # 8x bigger than it was (RULE 33). Its n is reported so the comparison is checkable.
        "market_rps": mean("market_rps", with_market),
        "market_n": len(with_market),
        "mean_consensus": mean("consensus_score", valid),
    }

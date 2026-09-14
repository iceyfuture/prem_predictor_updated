"""
player_ledger.py — lock every player projection BEFORE kickoff, grade it after.

WHY THIS EXISTS
The desk has forward-tested match outcomes (ledger.csv, 380 locked) and prop markets
(props_ledger.csv, 3,194 locked) since day one. Players had nothing. `fpl_minutes.predict`
emitted a start probability and expected minutes every single build, `simulate_fpl.project_gw`
turned those into projected points for ~485 players a week, and the scorer model priced anytime
goals — and not one of those numbers was ever compared to what happened. The only player-level
grading that existed was `fpl_forward.csv`: 60 rows covering the ~15 players in the user's own
squad.

You cannot improve a model you do not score. This is the missing loop.

WHAT IT STORES
One row per (gameweek, player), written at lock time and never rewritten:

    locked   gw, id, name, team, pos, price, p_start, xmin, p_score, proj, locked_at
    graded   started, minutes, goals, assists, points, bonus, xgi
    scored   brier_start, ae_minutes, brier_score, err_points

RULE 5 APPLIES HERE TOO: a prediction only counts if it was written down before the match. The
lock happens on the first build of a gameweek and later builds leave those numbers alone — see
`_same()` in build_dashboard.py for why re-stamping unchanged rows is its own problem.

P(scores) is derived from the model's own xG/90 and expected minutes rather than from
prem_scorer's name-keyed output, so it grades the same pipeline the dashboard displays and
needs no surname matching:  P(>=1 goal) = 1 - exp(-xg90 * xmin/90).
"""
import csv, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATH = os.path.join(HERE, "player_ledger.csv")
ACTUALS = os.path.join(ROOT, "outputs", "fpl_player_gameweek_2026_27.csv")

COLS = ["key", "gw", "id", "name", "team", "pos", "price",
        "p_start", "xmin", "p_score", "proj", "locked_at", "late",
        "started", "minutes", "goals", "assists", "points", "bonus", "xgi",
        "graded", "brier_start", "ae_minutes", "brier_score", "err_points"]


def _f(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _i(v, d=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def _load():
    if not os.path.exists(PATH):
        return {}
    with open(PATH) as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def _actuals():
    """{(gw, id): row} of what each player actually did, plus {gw: set(player_ids seen)}.

    CRITICAL: this file holds only players who APPEARED - 405 of 658 players, and not one row
    with minutes==0. Grading against it alone conditions on getting on the pitch, which makes
    every rate look far higher than it is. Measured: doing that put the observed start rate at
    71% against a predicted 57% and made the model look badly miscalibrated. Counting the
    1,045 non-appearances turned that into 33.4% predicted vs 33.4% actual - perfectly
    calibrated. A player whose CLUB played and who has no row here did not play: 0 minutes.
    """
    if not os.path.exists(ACTUALS):
        return {}, {}
    with open(ACTUALS) as f:
        rows = list(csv.DictReader(f))
    act = {(_i(r.get("gw")), _i(r.get("id"))): r for r in rows}
    seen = {}
    for (g, pid) in act:
        seen.setdefault(g, set()).add(pid)
    return act, seen


def record(gw, rows, now, finished=False, first_ko=None):
    """Lock `rows` (from simulate_fpl.project_gw) for `gw`, then grade whatever has results.

    `finished` says the gameweek is complete, which is the only time a missing actuals row is
    treated as "did not play" (0 minutes) rather than "not in yet". Grading a live gameweek as
    zeros would score every unplayed fixture as a miss.

    `first_ko` is that gameweek's earliest kickoff. A row locked AFTER it is a retrodiction,
    not a prediction: the model has already seen the result. Those rows are kept (they are
    still a record of what was shown) but stamped `late=1` and excluded from every score. The
    first run of this ledger locked 485 GW4 rows a day after GW4 was played - without this
    flag those would have been silently scored as a forward test.
    """
    led = _load()
    n_new = 0
    for r in rows or []:
        pid = _i(r.get("id"), -1)
        if pid < 0:
            continue
        key = f"{gw}:{pid}"
        if key in led:
            continue                      # RULE 5: locked once, never restated
        late = "1" if (first_ko and now >= first_ko) else ""
        xmin = _f(r.get("xmin"), 0.0) or 0.0
        xg90 = _f(r.get("xg90"), 0.0) or 0.0
        led[key] = {
            "key": key, "gw": gw, "id": pid, "name": r.get("name", ""),
            "team": r.get("team", ""), "pos": r.get("pos", ""),
            "price": r.get("price", ""),
            "p_start": round(_f(r.get("p_start"), 0.0) or 0.0, 4),
            "xmin": round(xmin, 1),
            "p_score": round(1.0 - math.exp(-xg90 * xmin / 90.0), 4),
            "proj": round(_f(r.get("proj"), 0.0) or 0.0, 2),
            "locked_at": now, "late": late, "graded": "",
        }
        n_new += 1

    act, seen = _actuals()
    # which clubs had a fixture in each gameweek, from the players who did appear
    clubs_on = {}
    id_team = {_i(r["id"]): r.get("team", "") for r in led.values()}
    for g, ids in seen.items():
        clubs_on[g] = {id_team[i] for i in ids if i in id_team}

    n_graded = 0
    for rec in led.values():
        if rec.get("graded"):
            continue
        g, pid = _i(rec["gw"]), _i(rec["id"])
        a = act.get((g, pid))
        if a is None:
            # No row. Either his club has not played yet (not a data point), or it has and he
            # did not feature (a real 0). Decided by whether ANY of his club-mates appeared.
            if rec.get("team") not in clubs_on.get(g, set()):
                continue
            if not (finished or g < gw):
                continue                  # gameweek still in progress
            a = {}                        # club played, he did not
        mins = _i(a.get("minutes"))
        started = _i(a.get("starts"))
        goals = _i(a.get("goals_scored"))
        rec.update({
            "started": started, "minutes": mins, "goals": goals,
            "assists": _i(a.get("assists")), "points": _i(a.get("total_points")),
            "bonus": _i(a.get("bonus")),
            "xgi": a.get("expected_goal_involvements", ""),
            "graded": now,
            "brier_start": round((_f(rec["p_start"], 0.0) - (1 if started else 0)) ** 2, 4),
            "ae_minutes": round(abs(_f(rec["xmin"], 0.0) - mins), 1),
            "brier_score": round((_f(rec["p_score"], 0.0) - (1 if goals else 0)) ** 2, 4),
            "err_points": round(_f(rec["proj"], 0.0) - _i(a.get("total_points")), 2),
        })
        n_graded += 1

    with open(PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(led.values(), key=lambda x: (_i(x["gw"]), -_f(x.get("proj"), 0) or 0)):
            w.writerow(rec)

    return summary(led, n_new, n_graded)


def _cal(pairs, edges=(0, .1, .25, .5, .75, .9, 1.01)):
    """Calibration bands: predicted probability vs realised rate."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sub = [(p, y) for p, y in pairs if lo <= p < hi]
        if len(sub) >= 10:
            out.append({"band": f"{int(lo*100)}-{int(hi*100)}%", "n": len(sub),
                        "pred": round(sum(p for p, _ in sub) / len(sub) * 100, 1),
                        "actual": round(sum(y for _, y in sub) / len(sub) * 100, 1)})
    return out


def summary(led=None, n_new=0, n_graded=0):
    led = _load() if led is None else led
    # RULE 5: only rows locked BEFORE kickoff count as a forward test
    done = [r for r in led.values() if r.get("graded") and not r.get("late")]
    late = sum(1 for r in led.values() if r.get("late"))
    if not done:
        return {"tracked": len(led), "graded": 0, "locked_now": n_new, "late": late,
                "note": ("all graded rows were locked after kickoff and do not count as a "
                         "forward test" if late else "no graded rows yet")}

    def m(k):
        v = [_f(r[k]) for r in done if _f(r.get(k)) is not None]
        return round(sum(v) / len(v), 4) if v else None

    played = [r for r in done if _i(r.get("minutes")) > 0]
    starts = [(_f(r["p_start"], 0.0), 1 if _i(r.get("started")) else 0) for r in done]
    scores = [(_f(r["p_score"], 0.0), 1 if _i(r.get("goals")) else 0) for r in done]
    # a baseline that always guesses the base rate - the bar any model must clear
    base_s = sum(y for _, y in starts) / len(starts)
    base_g = sum(y for _, y in scores) / len(scores)
    return {
        "tracked": len(led), "graded": len(done), "locked_now": n_new, "graded_now": n_graded,
        "late": late,
        "played": len(played),
        "brier_start": m("brier_start"),
        "brier_start_base": round(sum((base_s - y) ** 2 for _, y in starts) / len(starts), 4),
        "mae_minutes": m("ae_minutes"),
        "mae_minutes_played": (round(sum(_f(r["ae_minutes"], 0.0) for r in played) / len(played), 1)
                               if played else None),
        "brier_score": m("brier_score"),
        "brier_score_base": round(sum((base_g - y) ** 2 for _, y in scores) / len(scores), 4),
        "bias_points": m("err_points"),
        "start_rate_pred": round(base_s * 100, 1),
        "start_rate_actual": round(sum(y for _, y in starts) / len(starts) * 100, 1),
        "cal_start": _cal(starts),
        "cal_score": _cal(scores),
    }

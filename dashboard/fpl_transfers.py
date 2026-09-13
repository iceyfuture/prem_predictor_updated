"""
Transfer planner — the squad you can actually GET TO, not the squad you'd draft from scratch.

build_team() drafts an optimal 15 every week. That is the right answer in GW1 and after a
wildcard, and the wrong answer every other week: you hold last week's squad and get ONE free
transfer, with every extra costing -4 points. A "best XI" you cannot reach is not advice.

This module starts from the squad actually held and asks what one (or two, or three) moves
are worth, netting off the hits. Objective is the points you would REALLY score: best legal
XI with the captain doubled - so a transfer that upgrades your captain is valued properly,
and a transfer that only improves a bench player is valued at ~0.

HORIZON — you do not own a player for one week
Valuing a transfer on the coming gameweek alone is what made this planner want to sell the
squad's best in-form player for +0.9: one bad fixture is enough to tip a fractional gain, and
the fixture after it never entered the sum. The objective is now the decay-weighted XI total
over HORIZON gameweeks. Nothing else about the search changed, and both numbers are reported -
`gross` is still what you would score THIS week, `gross_h` is what the decision is made on.

DECAY = 0.65 is a judgement, not a swept constant, and is deliberately steep: you get another
free transfer every week, so a squad three weeks out is only loosely the squad you will own.
It is stated here rather than buried because it moves recommendations.

FORM — a bar on selling a player who is outscoring his own baseline
`fpl_form.form_adj` is this season's underlying output per 90, z-scored within position and
shrunk by minutes (w = m/(m+900)). The projection already contains some of this - `project_gw`
blends this season's xG/xA share at the same shrinkage - so the guard is a THRESHOLD, not a
bonus added to the projection: a move must clear
    bar = FORM_GUARD * max(0, form_adj(out) - form_adj(in))
before it is recommended. Selling form is allowed; selling form for a rounding error is not.

Two things about that, both checked rather than assumed:
  * it uses `form_adj` (shrunk), NOT `heat`. `heat` at this point in the season is dominated by
    small samples - the current top 8 by heat have confidence 0.05-0.17, i.e. 45-170 minutes -
    and gating transfers on it would be exactly the trap fpl_form.py's docstring warns about.
  * FORM_GUARD is NOT derived from the observed relationship between form and points. That
    regression looks strong (pts90 = 3.6*form_raw, r = 0.88) and is almost entirely circular:
    pts90 is one of form_raw's own inputs. It measures what a player has already scored, not
    what he will score, and using it as a points conversion would be self-referential.
So FORM_GUARD is a stated judgement in the same class as the chip thresholds in fpl_chips.py.
Set it to 0.0 to switch the guard off; the recommendation at each setting is reported.

Assumptions, stated because they matter:
  * Selling price = current price. Real FPL sells at purchase price plus half the profit, so
    a risen player nets slightly less than this assumes. Affects affordability, not ranking.
  * free_transfers defaults to 1. FPL banks unused ones (capped), so pass the real number.
"""
import csv, json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HIT = 4.0                      # points cost of each transfer beyond the free allowance
QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
CLUB_CAP = 3
HORIZON = 3                    # gameweeks the objective looks over
DECAY = 0.65                   # weight on each further gameweek (judgement - see docstring)
FORM_GUARD = 0.75              # extra points a move must clear per unit of shrunk form sold


def entry_id():
    """Your FPL team id. FPL_ENTRY_ID env var wins, so a shared clone needs no edit -
    each person points it at their own team without touching a tracked file."""
    env = os.environ.get("FPL_ENTRY_ID")
    if env and env.strip().isdigit():
        return int(env.strip())
    p = os.path.join(HERE, "fpl_config.json")
    if os.path.exists(p):
        try:
            return json.load(open(p)).get("entry_id")
        except Exception:
            return None
    return None


def real_squad(gw, timeout=20):
    """The squad you ACTUALLY hold, from the FPL API - not the model's memory of what it once
    recommended. Picks for the live gameweek are private until its deadline, so we read the
    last gameweek whose picks are public and walk forward through any transfers already made.

    This exists because the model's own GW1 snapshot was TAINTED (locked after the round had
    finished) and using it as the starting squad produced advice for a team the user did not
    own - it recommended buying a player they already had.
    """
    eid = entry_id()
    if not eid:
        return None
    base = "https://fantasy.premierleague.com/api"
    hdr = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    def get(path):
        r = urllib.request.Request(f"{base}/{path}", headers=hdr)
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return json.load(f)

    picks = None
    for g in range(gw, 0, -1):                      # newest public gameweek wins
        try:
            picks = get(f"entry/{eid}/event/{g}/picks/")
            src_gw = g
            break
        except Exception:
            continue
    if not picks:
        return None
    ids = [p["element"] for p in picks["picks"]]
    try:
        for t in get(f"entry/{eid}/transfers/"):     # apply anything bought since
            if t.get("event", 0) > src_gw and t["element_out"] in ids:
                ids[ids.index(t["element_out"])] = t["element_in"]
    except Exception:
        pass
    hist = {}
    try:
        cur = get(f"entry/{eid}/history/").get("current", [])
        if cur:
            hist = cur[-1]
    except Exception:
        pass
    made = sum(1 for t in (picks.get("picks") or []) if False)   # placeholder, kept explicit
    return {"ids": ids, "src_gw": src_gw, "entry": eid,
            "bank": (hist.get("bank", 0) or 0) / 10.0,
            "value": (hist.get("value", 0) or 0) / 10.0,
            "chips_used": None}


def held_squad(gw):
    """Player ids the model locked LAST gameweek - what you are still holding."""
    p = os.path.join(HERE, "fpl_forward.csv")
    if not os.path.exists(p):
        return None
    prev = [r for r in csv.DictReader(open(p)) if int(r["gw"]) == gw - 1]
    return [int(r["element"]) for r in prev] or None


def _proj(s, k=0):
    """This player's projection in gameweek gw+k, falling back to the current week."""
    pr = s.get("projs")
    return pr[k] if pr and k < len(pr) else s["proj"]


def xi_points(squad, k=0):
    """Points the squad would actually score in gameweek gw+k: best legal XI, captain doubled."""
    by = {p: sorted([s for s in squad if s["pos"] == p], key=lambda x: -_proj(x, k)) for p in QUOTA}
    if len(by["GK"]) < 1:
        return 0.0, None, None, None
    best = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not (1 <= f <= 3):
                continue
            if len(by["DEF"]) < d or len(by["MID"]) < m or len(by["FWD"]) < f:
                continue
            xi = by["GK"][:1] + by["DEF"][:d] + by["MID"][:m] + by["FWD"][:f]
            cap = max(xi, key=lambda x: _proj(x, k))
            tot = sum(_proj(x, k) for x in xi) + _proj(cap, k)  # captain counts twice
            if best is None or tot > best[0]:
                best = (tot, xi, f"{d}-{m}-{f}", cap)
    return best if best else (0.0, None, None, None)


def _legal(squad, budget):
    if sum(s["price"] for s in squad) > budget + 1e-9:
        return False
    from collections import Counter
    c = Counter(s["ot"] for s in squad)
    if any(v > CLUB_CAP for v in c.values()):
        return False
    p = Counter(s["pos"] for s in squad)
    return all(p.get(k, 0) == v for k, v in QUOTA.items())


def horizon_points(squad, weights):
    """The objective: XI points over the next `len(weights)` gameweeks, decayed."""
    return sum(w * xi_points(squad, k)[0] for k, w in enumerate(weights))


def _moves(squad, pool, budget, beam, weights):
    """All single transfers from `squad`, best `beam` by resulting horizon points."""
    have = {s["id"] for s in squad}
    base = sum(s["price"] for s in squad)
    out = []
    for i, s in enumerate(squad):
        for c in pool:
            if c["id"] in have or c["pos"] != s["pos"]:
                continue
            if base - s["price"] + c["price"] > budget + 1e-9:
                continue
            nxt = squad[:i] + [c] + squad[i + 1:]
            if not _legal(nxt, budget):
                continue
            out.append((horizon_points(nxt, weights), s, c, nxt))
    out.sort(key=lambda x: -x[0])
    return out[:beam]


def plan(rows, held_ids, budget=100.0, free_transfers=1, max_transfers=3, beam=25,
         all_players=None, future_rows=None, form=None, form_guard=FORM_GUARD):
    """Best plan at each transfer count, net of -4 hits. Returns None if the squad is unknown.

    `future_rows` is [project_gw(gw+1), project_gw(gw+2), ...]; with it the objective becomes
    the decayed multi-week XI total instead of a single gameweek. Without it the behaviour is
    exactly the old one-week objective, so the module still works standalone.

    `form` is {player_id: form_adj} from fpl_form; a move that sells form has to clear a bar
    before it is recommended.

    A held player who is injured or suspended is dropped from `rows` (project_gw keeps players
    with expected minutes) but you STILL OWN HIM - he occupies a squad slot and his sale funds
    the transfer. Such players are carried at proj 0.0 from `all_players`, which is exactly
    right: worth nothing this week, and therefore first in line to be sold.
    """
    by_id = {r["id"]: r for r in rows if r.get("id") is not None}
    spare = {p["id"]: p for p in (all_players or [])}
    fut = [{r["id"]: r["proj"] for r in fr if r.get("id") is not None}
           for fr in (future_rows or [])][:max(0, HORIZON - 1)]
    weights = [DECAY ** k for k in range(len(fut) + 1)]
    for r in rows:
        if r.get("id") is not None:
            r["projs"] = [r["proj"]] + [f.get(r["id"], 0.0) for f in fut]
    squad = []
    for i in (held_ids or []):
        if i in by_id:
            squad.append(by_id[i])
        elif i in spare:
            q = spare[i]
            squad.append({"id": q["id"], "name": q["name"], "pos": q["pos"], "team": q["team"],
                          "ot": q["ot"], "price": q["price"], "proj": 0.0,
                          "projs": [0.0] * len(weights),
                          "own": q.get("own", 0.0), "unavailable": True})
    if len(squad) != 15:
        return None
    pool = [r for r in rows if r.get("id") is not None]
    fm = form or {}
    base_pts = xi_points(squad, 0)[0]
    base_by_gw = [xi_points(squad, k)[0] for k in range(len(weights))]
    base_h = horizon_points(squad, weights)
    bank = budget - sum(s["price"] for s in squad)

    def entry(n, hpts, sq, mv):
        hit = HIT * max(0, n - free_transfers)
        # the bar: selling a player who is outscoring his own baseline has to be worth more
        # than selling one who is not. Threshold, not a bonus - see the module docstring.
        bar = sum(max(0.0, fm.get(s["id"], 0.0) - fm.get(c["id"], 0.0)) for s, c in mv) * form_guard
        # the per-gameweek gains the weighted score is built from, so the number on screen can be
        # taken apart instead of taken on trust
        by_gw = [round(xi_points(sq, k)[0] - base_by_gw[k], 1) for k in range(len(weights))]
        return {"n": n, "gross": round(xi_points(sq, 0)[0] - base_pts, 1),
                "by_gw": by_gw,
                "gross_h": round(hpts - base_h, 1), "hit": hit, "bar": round(bar, 1),
                "net": round(hpts - base_h - hit - bar, 1),
                "xi": round(xi_points(sq, 0)[0], 1), "squad": sq,
                "moves": [{"out": s["name"], "out_team": s["team"], "out_proj": round(s["proj"], 1),
                           "out_price": s["price"], "in": c["name"], "in_team": c["team"],
                           "in_proj": round(c["proj"], 1), "in_price": c["price"],
                           "out_form": round(fm.get(s["id"], 0.0), 2),
                           "in_form": round(fm.get(c["id"], 0.0), 2),
                           "pos": s["pos"]} for s, c in mv]}

    options = [entry(0, base_h, squad, [])]
    states = [(base_h, squad, [])]
    for n in range(1, max_transfers + 1):
        nxt = []
        for _, sq, mv in states:
            for pts, s, c, ns in _moves(sq, pool, budget, beam, weights):
                nxt.append((pts, ns, mv + [(s, c)]))
        if not nxt:
            break
        seen, ded = set(), []
        for pts, sq, mv in sorted(nxt, key=lambda x: -x[0]):
            k = tuple(sorted(s["id"] for s in sq))
            if k in seen:
                continue
            seen.add(k); ded.append((pts, sq, mv))
        states = ded[:beam]
        # rank this transfer count on the net the user actually acts on, not on gross: the
        # beam is ordered by gross, and the form bar can reorder the top of it.
        best_n = max((entry(n, p, sq, mv) for p, sq, mv in states[:beam]), key=lambda o: o["net"])
        options.append(best_n)
    best = max(options, key=lambda o: o["net"])
    horizon = len(weights)
    move = " and ".join(f"{m['out']} \u2192 {m['in']}" for m in best["moves"])
    # The verdict says the MOVE and both numbers. "+2.8 pts over 3 GW" was the wrong sentence:
    # 2.8 is the DECAYED score, not the points you would gain across three weeks (that is ~4.0).
    return {"bank": round(bank, 1),
            "unavailable": [s["name"] for s in squad if s.get("unavailable")],
            "free_transfers": free_transfers, "base_xi": round(base_pts, 1),
            "horizon": horizon, "decay": DECAY, "form_guard": form_guard,
            "weights": [round(w, 2) for w in weights],
            "base_horizon": round(base_h, 1),
            "options": options, "best": best,
            "verdict": ("HOLD \u2014 no transfer is worth the free transfer" if best["n"] == 0 else
                        f"{move} \u2014 {best['by_gw'][0]:+.1f} this week, "
                        f"{best['net']:+.1f} on the {horizon}-week score")}


def decorate(squad, gw, weeks=None, team_meta=None):
    """Turn a planner squad into the same shape build_team() returns, so the forward test and
    the UI can consume the REACHABLE squad instead of the unreachable from-scratch draft."""
    tot, xi, formation, cap = xi_points(squad)
    if xi is None:
        return None
    fix = {}
    wk = next((w for w in (weeks or []) if w["gw"] == gw), None)
    if wk:
        for e in wk["fixtures"]:
            fix[e["home"]] = (e["away_abbr"], "H")
            fix[e["away"]] = (e["home_abbr"], "A")
    tm = team_meta or {}
    xi_ids = {id(x) for x in xi}
    vice = max([x for x in xi if x is not cap], key=lambda x: x["proj"]) if len(xi) > 1 else None
    out = []
    for r in squad:
        opp, ha = fix.get(r["ot"], ("", ""))
        meta = tm.get(r["ot"], {})
        out.append({"id": r.get("id"), "nm": r.get("name") or r.get("nm"), "pos": r["pos"],
                    "team": r["team"], "ot": r["ot"], "price": r["price"],
                    "proj": round(r["proj"], 1), "own": r.get("own", 0.0),
                    "color": meta.get("color", "#5B7A72"),
                    "abbr": meta.get("abbr", r["ot"][:3].upper()), "opp": opp, "ha": ha,
                    "start": id(r) in xi_ids, "cap": r is cap, "vice": r is vice,
                    "unavailable": bool(r.get("unavailable"))})
    bench = [b for b in out if not b["start"]]
    bench.sort(key=lambda b: (b["pos"] != "GK", -b["proj"]))
    return {"gw": gw, "cost": round(sum(r["price"] for r in squad), 1), "formation": formation,
            "xi_proj": round(sum(x["proj"] for x in xi), 1), "captain": (cap.get("name") or cap.get("nm")),
            "squad": out, "bench_order": [b["nm"] for b in bench]}


def entry_history(timeout=20):
    """Your OWN gameweek scores, straight from the FPL API.

    The forward test grades the squad the MODEL picked. That answers "is the model any good",
    which is not the same question as "how am I doing" - and the second one is the whole reason
    you open the page. Both are now carried so they can sit side by side.
    """
    eid = entry_id()
    if not eid:
        return None
    try:
        r = urllib.request.Request(
            f"https://fantasy.premierleague.com/api/entry/{eid}/history/",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(r, timeout=timeout) as f:
            d = json.load(f)
    except Exception:
        return None
    out, total = [], 0
    for g in d.get("current", []):
        total += g.get("points", 0)
        out.append({"gw": g.get("event"), "points": g.get("points"),
                    "bench": g.get("points_on_bench"), "rank": g.get("overall_rank"),
                    "gw_rank": g.get("rank"), "transfers": g.get("event_transfers"),
                    "hit": g.get("event_transfers_cost"), "running": total,
                    "value": (g.get("value") or 0) / 10.0, "bank": (g.get("bank") or 0) / 10.0})
    chips = [{"name": c.get("name"), "gw": c.get("event")} for c in d.get("chips", [])]
    return {"entry": eid, "weeks": out, "total": total, "chips_used": chips}

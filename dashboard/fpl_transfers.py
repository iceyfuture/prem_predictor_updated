"""
Transfer planner — the squad you can actually GET TO, not the squad you'd draft from scratch.

build_team() drafts an optimal 15 every week. That is the right answer in GW1 and after a
wildcard, and the wrong answer every other week: you hold last week's squad and get ONE free
transfer, with every extra costing -4 points. A "best XI" you cannot reach is not advice.

This module starts from the squad actually held and asks what one (or two, or three) moves
are worth, netting off the hits. Objective is the points you would REALLY score: best legal
XI with the captain doubled - so a transfer that upgrades your captain is valued properly,
and a transfer that only improves a bench player is valued at ~0.

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


def xi_points(squad):
    """Points the squad would actually score: best legal XI, captain doubled."""
    by = {p: sorted([s for s in squad if s["pos"] == p], key=lambda x: -x["proj"]) for p in QUOTA}
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
            cap = max(xi, key=lambda x: x["proj"])
            tot = sum(x["proj"] for x in xi) + cap["proj"]      # captain counts twice
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


def _moves(squad, pool, budget, beam):
    """All single transfers from `squad`, best `beam` by resulting XI points."""
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
            out.append((xi_points(nxt)[0], s, c, nxt))
    out.sort(key=lambda x: -x[0])
    return out[:beam]


def plan(rows, held_ids, budget=100.0, free_transfers=1, max_transfers=3, beam=25,
         all_players=None):
    """Best plan at each transfer count, net of -4 hits. Returns None if the squad is unknown.

    A held player who is injured or suspended is dropped from `rows` (project_gw keeps fit
    players only) but you STILL OWN HIM - he occupies a squad slot and his sale funds the
    transfer. Such players are carried at proj 0.0 from `all_players`, which is exactly right:
    worth nothing this week, and therefore first in line to be sold.
    """
    by_id = {r["id"]: r for r in rows if r.get("id") is not None}
    spare = {p["id"]: p for p in (all_players or [])}
    squad = []
    for i in (held_ids or []):
        if i in by_id:
            squad.append(by_id[i])
        elif i in spare:
            q = spare[i]
            squad.append({"id": q["id"], "name": q["name"], "pos": q["pos"], "team": q["team"],
                          "ot": q["ot"], "price": q["price"], "proj": 0.0,
                          "own": q.get("own", 0.0), "unavailable": True})
    if len(squad) != 15:
        return None
    pool = [r for r in rows if r.get("id") is not None]
    base_pts = xi_points(squad)[0]
    bank = budget - sum(s["price"] for s in squad)

    options, states = [{"n": 0, "gross": 0.0, "hit": 0.0, "net": 0.0, "moves": [],
                        "xi": round(base_pts, 1), "squad": squad}], [(base_pts, squad, [])]
    for n in range(1, max_transfers + 1):
        nxt = []
        for _, sq, mv in states:
            for pts, s, c, ns in _moves(sq, pool, budget, beam):
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
        pts, sq, mv = states[0]
        hit = HIT * max(0, n - free_transfers)
        options.append({"n": n, "gross": round(pts - base_pts, 1), "hit": hit,
                        "net": round(pts - base_pts - hit, 1), "xi": round(pts, 1), "squad": sq,
                        "moves": [{"out": s["name"], "out_team": s["team"], "out_proj": round(s["proj"], 1),
                                   "out_price": s["price"], "in": c["name"], "in_team": c["team"],
                                   "in_proj": round(c["proj"], 1), "in_price": c["price"],
                                   "pos": s["pos"]} for s, c in mv]})
    best = max(options, key=lambda o: o["net"])
    return {"bank": round(bank, 1),
            "unavailable": [s["name"] for s in squad if s.get("unavailable")], "free_transfers": free_transfers, "base_xi": round(base_pts, 1),
            "options": options, "best": best,
            "verdict": ("HOLD - no transfer pays for itself" if best["n"] == 0 else
                        f"make {best['n']} transfer{'s' if best['n'] > 1 else ''} (net {best['net']:+.1f} pts)")}


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

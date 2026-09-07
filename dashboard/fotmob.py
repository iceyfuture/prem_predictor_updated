"""
fotmob.py — the richer per-match team stat line, and an xG-based strength index.

WHY THIS EXISTS ALONGSIDE team_stats.py

`team_stats.py` already records a per-match team line from ESPN. ESPN carries 28 stats and they
are good ones, but it does not publish the measures that actually separate a good performance
from a lucky one:

    expected goals (xG)      ESPN has none. team_stats.py derives team xG by summing FPL's
                             per-player xG, which works but is second-hand and only exists
                             from 2023-24. FotMob publishes match xG directly.
    xGOT (post-shot xG)      shot quality AFTER it leaves the boot - separates finishing and
                             goalkeeping from chance creation. No other feed here has it.
    big chances              FotMob's own high-quality-chance count
    shots inside/outside box location, not just volume - 17 shots from distance is not 17 chances
    touches in opposition box territory in the area that matters

WHAT IT WRITES
    outputs/team_fotmob_2026_27.csv    one row per team per match, ~30 columns
    outputs/team_strength_xg.csv       the strength index built from it

THE INDEX
Ratings are fit by least squares on **xG**, not goals:

    xg_for(team i vs j, at home) = attack_i - defence_j + home_adv

That is the Dixon-Coles structure, solved on expected goals instead of scored ones. The point
is sample efficiency: over three matches, goals are close to noise (a 4-0 and a 1-0 are the same
three points and wildly different performances) while xG accumulates from every chance and
stabilises far sooner. Ridge shrinkage keeps a 3-match sample from producing 20 confident
ratings, and every rating is reported with the matches behind it.

This is a MEASUREMENT of performance so far. It is deliberately not merged into the Dixon-Coles
match model, which is validated walk-forward on goals - swapping its training target is a change
that would need its own out-of-sample test, not an assumption.
"""
import csv, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
CACHE = os.path.join(HERE, ".cache")
os.makedirs(CACHE, exist_ok=True)

LEAGUE = 47                      # Premier League
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Referer": "https://www.fotmob.com/", "Accept": "application/json"}
BASE = "https://www.fotmob.com/api/data"

# FotMob club names -> the names this project uses everywhere else
ALIAS = {"Manchester United": "Man United", "Manchester City": "Man City",
         "Newcastle United": "Newcastle", "Tottenham Hotspur": "Tottenham",
         "Nottingham Forest": "Nott'm Forest", "Brighton & Hove Albion": "Brighton",
         "AFC Bournemouth": "Bournemouth", "Coventry City": "Coventry",
         "Hull City": "Hull", "Ipswich Town": "Ipswich", "Leeds United": "Leeds",
         "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
         "Crystal Palace": "Crystal Palace", "Aston Villa": "Aston Villa"}

# FotMob stat title -> our column. Titles are stable; anything unmapped is ignored, so a
# FotMob rename degrades to a missing column rather than a crash.
MAP = {
    "Ball possession": "possession", "Expected goals (xG)": "xg", "xG open play": "xg_open",
    "xG set play": "xg_set", "xG non-penalty": "xg_np", "xG on target (xGOT)": "xgot",
    "Total shots": "shots", "Shots on target": "shots_on_target",
    "Shots off target": "shots_off_target", "Blocked shots": "blocked_shots",
    "Hit woodwork": "woodwork", "Shots inside box": "shots_inside_box",
    "Shots outside box": "shots_outside_box", "Big chances": "big_chances",
    "Big chances missed": "big_chances_missed", "Touches in opposition box": "touches_opp_box",
    "Accurate passes": "accurate_passes", "Passes": "passes",
    "Accurate long balls": "long_balls", "Accurate crosses": "crosses", "Throws": "throws",
    "Offsides": "offsides", "Tackles": "tackles", "Interceptions": "interceptions",
    "Blocks": "blocks", "Clearances": "clearances", "Keeper saves": "saves",
    "Duels won": "duels_won", "Ground duels won": "ground_duels_won",
    "Aerial duels won": "aerial_duels_won", "Successful dribbles": "dribbles",
    "Yellow cards": "yellows", "Red cards": "reds", "Fouls committed": "fouls",
    "Distance covered": "distance_m", "Number of sprints": "sprints",
}
COLS = ["gw", "date", "team", "opponent", "venue", "gf", "ga", "result"] + sorted(set(MAP.values()))


def _get(path, key, max_age_min=1440):
    p = os.path.join(CACHE, key)
    if os.path.exists(p):
        age = (time.time() - os.path.getmtime(p)) / 60
        if age < max_age_min:
            return json.load(open(p))
    d = json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, headers=UA), timeout=45))
    json.dump(d, open(p, "w"))
    return d


def _num(v):
    """FotMob mixes ints, '0.71', and '377 (82%)'. Take the leading number."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().split()[0].replace("%", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def fixtures(max_age_min=180):
    d = _get(f"/leagues?id={LEAGUE}&ccode3=USA", "fotmob_league.json", max_age_min)
    return (d.get("fixtures") or {}).get("allMatches") or []


def match_rows(mid, rnd, verbose=False):
    """Two rows (home, away) of stats for one match, or [] if it has no stat line yet."""
    try:
        d = _get(f"/matchDetails?matchId={mid}", f"fotmob_m{mid}.json", 60 * 24 * 30)
    except Exception as e:
        if verbose:
            print(f"    match {mid}: {e}")
        return []
    g = d.get("general") or {}
    hn = ALIAS.get((g.get("homeTeam") or {}).get("name"), (g.get("homeTeam") or {}).get("name"))
    an = ALIAS.get((g.get("awayTeam") or {}).get("name"), (g.get("awayTeam") or {}).get("name"))
    if not hn or not an:
        return []
    per = ((d.get("content") or {}).get("stats") or {}).get("Periods") or {}
    groups = (per.get("All") or {}).get("stats") or []
    vals = {}
    for grp in groups:
        for s in (grp.get("stats") or []):
            col = MAP.get(s.get("title"))
            st = s.get("stats")
            if not col or not isinstance(st, list) or len(st) != 2:
                continue
            h, a = _num(st[0]), _num(st[1])
            if h is None and a is None:
                continue
            vals.setdefault(col, (h, a))
    if not vals:
        return []
    # The score lives in header.teams[].score - general.status is null on finished matches.
    hg = ag = None
    ht = (d.get("header") or {}).get("teams") or []
    if len(ht) == 2:
        try:
            hg, ag = int(ht[0]["score"]), int(ht[1]["score"])
        except (KeyError, TypeError, ValueError):
            hg = ag = None
    date = ((g.get("matchTimeUTCDate") or "")[:10]) or ""

    def row(team, opp, venue, gf, ga, side):
        r = {"gw": rnd, "date": date, "team": team, "opponent": opp, "venue": venue,
             "gf": gf, "ga": ga,
             "result": ("W" if (gf or 0) > (ga or 0) else "D" if gf == ga else "L")
                       if gf is not None else ""}
        for col, pair in vals.items():
            r[col] = pair[side]
        return r
    return [row(hn, an, "H", hg, ag, 0), row(an, hn, "A", ag, hg, 1)]


def finished_rounds():
    """Every round with at least one played match - so the recurring build picks up a new
    matchweek on its own instead of needing the round numbers passed in."""
    out = set()
    for m in fixtures():
        st = m.get("status") or {}
        if st.get("finished") or (st.get("scoreStr") and "-" in str(st.get("scoreStr"))):
            try:
                out.add(int(m["round"]))
            except (KeyError, TypeError, ValueError):
                pass
    return tuple(sorted(out))


def collect(rounds=None, verbose=True):
    fx = fixtures()
    rounds = rounds or finished_rounds()
    want = [m for m in fx if str(m.get("round")) in {str(r) for r in rounds}]
    rows = []
    for m in want:
        got = match_rows(m["id"], int(m["round"]), verbose)
        if got:
            rows += got
        elif verbose:
            print(f"    no stat line yet: {m['id']} round {m.get('round')}")
        time.sleep(0.25)
    rows.sort(key=lambda r: (r["gw"], r["team"]))
    return rows


def write(rows):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "team_fotmob_2026_27.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return p


# --------------------------------------------------------------- strength index
RIDGE = 1.0          # shrinkage toward league average. With ~3 matches a club has one equation
                     # per game; without this, 20 confident ratings come out of 60 rows.


def strength_index(rows, ridge=RIDGE):
    """Least-squares attack/defence on xG:  xg_for(i vs j) = attack_i - defence_j + home_adv.

    Same structure as the Dixon-Coles match model, solved on expected goals rather than scored
    ones. Over three matches a 4-0 and a 1-0 are both a win and tell you very different things;
    xG accumulates from every chance, so it says more per match played.
    """
    import numpy as np
    teams = sorted({r["team"] for r in rows})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    use = [r for r in rows if r.get("xg") is not None]
    A = np.zeros((len(use), 2 * n + 1))
    y = np.zeros(len(use))
    for k, r in enumerate(use):
        A[k, idx[r["team"]]] = 1.0                    # attack of the team creating
        A[k, n + idx[r["opponent"]]] = -1.0           # defence of the team conceding
        A[k, 2 * n] = 1.0 if r["venue"] == "H" else 0.0
        y[k] = float(r["xg"])
    P = np.vstack([A, np.sqrt(ridge) * np.eye(2 * n + 1)])
    q = np.concatenate([y, np.zeros(2 * n + 1)])
    sol, *_ = np.linalg.lstsq(P, q, rcond=None)
    att, dfn, hadv = sol[:n], sol[n:2 * n], float(sol[2 * n])
    att = att - att.mean(); dfn = dfn - dfn.mean()

    # supporting per-match aggregates, straight from the rows
    agg = {}
    for r in rows:
        a = agg.setdefault(r["team"], {"n": 0, "xg": 0.0, "xga": 0.0, "xgot": 0.0,
                                       "bc": 0.0, "sib": 0.0, "tob": 0.0, "gf": 0, "ga": 0, "pts": 0})
        a["n"] += 1
        for src, dst in (("xg", "xg"), ("xgot", "xgot"), ("big_chances", "bc"),
                         ("shots_inside_box", "sib"), ("touches_opp_box", "tob")):
            if r.get(src) is not None:
                a[dst] += float(r[src])
        a["gf"] += r.get("gf") or 0
        a["ga"] += r.get("ga") or 0
        a["pts"] += 3 if r["result"] == "W" else 1 if r["result"] == "D" else 0
    xg_by = {}
    for r in rows:
        if r.get("xg") is not None:
            xg_by[(r["gw"], r["team"])] = float(r["xg"])
    for r in rows:
        o = xg_by.get((r["gw"], r["opponent"]))
        if o is not None:
            agg[r["team"]]["xga"] += o

    out = []
    for t in teams:
        a = agg[t]; i = idx[t]
        out.append({
            "team": t, "played": a["n"],
            # dfn is solved with a MINUS sign in the design matrix, so a high value already
            # means "concedes less xG". Negating it here inverted the whole table - Arsenal,
            # with the league's best xGA by a distance, came out 20th.
            "attack": round(float(att[i]), 3), "defense": round(float(dfn[i]), 3),
            "net": round(float(att[i] + dfn[i]), 3),
            "xg_pg": round(a["xg"] / a["n"], 2), "xga_pg": round(a["xga"] / a["n"], 2),
            "xgd_pg": round((a["xg"] - a["xga"]) / a["n"], 2),
            "xgot_pg": round(a["xgot"] / a["n"], 2),
            "big_ch_pg": round(a["bc"] / a["n"], 2),
            "shots_in_box_pg": round(a["sib"] / a["n"], 1),
            "touches_box_pg": round(a["tob"] / a["n"], 1),
            "gf": a["gf"], "ga": a["ga"], "pts": a["pts"],
            "gf_minus_xg": round(a["gf"] - a["xg"], 2),
        })
    out.sort(key=lambda r: -r["net"])
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out, hadv


def write_index(idx_rows):
    p = os.path.join(OUT, "team_strength_xg.csv")
    cols = ["rank", "team", "played", "net", "attack", "defense", "xg_pg", "xga_pg", "xgd_pg",
            "xgot_pg", "big_ch_pg", "shots_in_box_pg", "touches_box_pg",
            "gf", "ga", "pts", "gf_minus_xg"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(idx_rows)
    return p


if __name__ == "__main__":
    rnds = tuple(int(x) for x in sys.argv[1:]) or finished_rounds()
    rows = collect(rounds=rnds, verbose=False)
    p1 = write(rows)
    idx, hadv = strength_index(rows)
    p2 = write_index(idx)
    print(f"FotMob: {len(rows)} team-match rows over rounds {rnds} -> {os.path.basename(p1)}")
    print(f"xG strength index -> {os.path.basename(p2)}   (home advantage {hadv:+.2f} xG)\n")
    print(f"  {'#':>2}  {'club':<16}{'net':>7}{'att':>7}{'def':>7}{'xG':>6}{'xGA':>6}{'xGD':>7}"
          f"{'xGOT':>6}{'bigCh':>7}{'inBox':>7}{'pts':>5}{'G-xG':>7}")
    print("  " + "-" * 92)
    for r in idx:
        print(f"  {r['rank']:>2}  {r['team']:<16}{r['net']:>+7.2f}{r['attack']:>+7.2f}{r['defense']:>+7.2f}"
              f"{r['xg_pg']:>6.2f}{r['xga_pg']:>6.2f}{r['xgd_pg']:>+7.2f}{r['xgot_pg']:>6.2f}"
              f"{r['big_ch_pg']:>7.2f}{r['shots_in_box_pg']:>7.1f}{r['pts']:>5}{r['gf_minus_xg']:>+7.2f}")

"""
simulate_fpl.py — project FPL points per player for a matchweek by simulating the fixtures.

For each fixture we take the model's full score-matrix distribution (every scoreline weighted
by its probability = simulating all outcomes) and turn it into expected FPL points per player:

  goals      share of team goals (from prem_scorer historical shares, restricted to FIT players)
             x team expected goals x FPL goal points (GK/DEF 6, MID 5, FWD 4)
  assists    ~0.6 x team goals x share x 3
  clean sheet  P(opponent scores 0) x (GK/DEF 4, MID 1)
  conceded   -0.5 x opponent expected goals  (GK/DEF only)
  appearance +2 for likely starters (FPL status 'a' and ep_next above a floor)
  gk saves   small positive proportional to shots faced

Only FIT players (FPL status 'a') are ever projected, so injured/suspended players score 0 and
drop out of the picks automatically. Output: projected points per player, refreshable per GW.
"""
import csv, os, re, sys, unicodedata, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import prem_dixon_coles as dc      # noqa: E402
import supremacy_odds as so        # noqa: E402
import prem_scorer as ps           # noqa: E402
import feeds                        # noqa: E402
import build_dashboard as bd        # noqa: E402

GOAL = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
FPL2OUR = {"Man Utd": "Man United", "Spurs": "Tottenham", "Coventry City": "Coventry",
           "Hull City": "Hull", "Ipswich Town": "Ipswich"}
EP_FLOOR = 1.5   # ep_next below this = unlikely starter -> appearance downweighted


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[.\-' ]", "", s).lower()


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


# Evidence shrinkage on THIS SEASON's per-90 rates. After one gameweek a player with 1 minute
# shows 270 defensive actions per 90; rates only mean something once minutes accumulate. Same
# shrinkage philosophy as the promoted-club fix: w = m/(m+MIN_K) toward the historical/positional
# prior, so early-season noise cannot drive a pick and the signal fades in as the season runs.
MIN_K = 900.0          # ~10 full matches before this season's rate carries half the weight
# FPL defensive-contribution scoring: 2 pts at a threshold of defensive actions per match.
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12, "GK": 99}
DC_POINTS = 2.0


def _dc_points(dc90, pos, minutes=90.0):
    """Expected defensive-contribution points: P(actions >= threshold) * 2, Poisson on the rate.
    This term was missing entirely - the projection modelled goals, assists and clean sheets but
    not the defensive points FPL now awards, which understated defenders and holding midfielders."""
    thr = DC_THRESHOLD.get(pos, 99)
    if thr > 90 or dc90 <= 0:
        return 0.0
    lam = dc90 * (minutes / 90.0)
    if lam <= 0:
        return 0.0
    # P(X >= thr) for X ~ Poisson(lam)
    import math
    cdf, term = 0.0, math.exp(-lam)
    for k in range(thr):
        cdf += term
        term *= lam / (k + 1)
    return DC_POINTS * max(0.0, 1.0 - cdf)


def _avail(p):
    """Probability this player features, as a NUMBER rather than a fit/unfit boolean.

    Treating a doubt as "unavailable" zeroed players FPL itself still projects points for -
    Gibbs-White was a 75% doubt with ep_next 2.1 and our model valued him at 0.0, which both
    distorted the XI and made him look like free money in a transfer. `chance_of_playing_next_round`
    is the right signal; fall back on status when FPL has not published a percentage.
    """
    c = p.get("chance_of_playing_next_round")
    if c is not None:
        return max(0.0, min(1.0, float(c) / 100.0))
    return {"a": 1.0, "d": 0.75}.get(p.get("status"), 0.0)


def fpl_players():
    b = feeds._get(feeds.FPL, "fpl_bootstrap.json", max_age_min=120)
    teams = {t["id"]: t["name"] for t in b["teams"]}
    out = []
    for p in b["elements"]:
        tm = teams[p["team"]]; ot = FPL2OUR.get(tm, tm)
        out.append(dict(id=p["id"], name=p["web_name"], team=tm, ot=ot, pos=POS[p["element_type"]],
                        price=p["now_cost"] / 10.0, ep=float(p.get("ep_next") or 0),
                        own=float(p["selected_by_percent"]), status=p["status"],
                        fit=(p["status"] == "a"), avail=_avail(p),
                        mins=p.get("minutes", 0) or 0,
                        xg90=_num(p.get("expected_goals_per_90")),
                        xa90=_num(p.get("expected_assists_per_90")),
                        dc90=_num(p.get("defensive_contribution_per_90"))))
    return out


def project_gw(gw, players=None, model=None, shares=None, weeks=None):
    if players is None:
        players = fpl_players()
    cold = getattr(project_gw, "_cold", set())
    if model is None:
        model = dc.get_model()
        events = feeds.espn_events()
        weeks = feeds.to_matchweeks(events)
        clubs = sorted({e["home"] for e in events} | {e["away"] for e in events})
        model, cold = bd.apply_cold_start(model, clubs)
        project_gw._cold = cold
    if shares is None:
        shares = ps.load_shares()
    wk = next((w for w in weeks if w["gw"] == gw), None)
    if not wk:
        return []
    # index fit players by (team, surname) for share matching, and per team by position
    by_team = {}
    for p in players:
        by_team.setdefault(p["ot"], []).append(p)

    # Positional median defensive-contribution rate, from players with real minutes. Early-season
    # per-90 rates are wild (1 minute of football can read as 270 actions per 90), so each player's
    # rate is shrunk toward this rather than toward zero - zero would permanently understate
    # defenders and holding midfielders, which is the opposite of the bug being fixed.
    import statistics as _st
    # The minutes bar adapts: early in the season nobody has 180 minutes, so it steps down
    # rather than silently producing a median of zero for every position.
    dc_med = {}
    for _pos in ("GK", "DEF", "MID", "FWD"):
        _r = []
        for _bar in (180, 60, 1):
            _r = [q["dc90"] for q in players if q["pos"] == _pos and q["mins"] >= _bar]
            if len(_r) >= 20:
                break
        dc_med[_pos] = float(_st.median(_r)) if _r else 0.0

    proj = {id(p): 0.0 for p in players}
    playing = set()
    for fx in wk["fixtures"]:
        M, lam_h, lam_a = dc.score_matrix(model, fx["home"], fx["away"])
        cs_home = float(M[:, 0].sum())    # away scored 0
        cs_away = float(M[0, :].sum())    # home scored 0
        for team, lam, opp_lam, csp in [(fx["home"], lam_h, lam_a, cs_home),
                                        (fx["away"], lam_a, lam_h, cs_away)]:
            fit = [p for p in by_team.get(team, []) if p.get("avail", 0) > 0]
            if not fit:
                continue
            # likely XI: best GK + best 10 outfield by AVAILABILITY-WEIGHTED ep, so a doubt is
            # ranked below an equivalent fit player rather than excluded outright
            fit.sort(key=lambda p: -p["ep"] * p.get("avail", 1.0))
            gk = [p for p in fit if p["pos"] == "GK"][:1]
            outfield = [p for p in fit if p["pos"] != "GK"][:10]
            xi = gk + outfield
            # goal shares among the XI, exact-surname match to historical shares, capped
            sh = shares.get(team)
            surn = {}
            if sh is not None:
                for pl, val in sh.items():
                    surn[ps._surname(pl)] = max(surn.get(ps._surname(pl), 0.0), float(val))
            # historical shares already sum to ~1 across the squad — use directly (capped),
            # do NOT renormalise to the XI (that hands a promoted team's lone known scorer
            # ~100% of the goals and inflates him wildly).
            # THIS SEASON's xG/xA share of the projected XI, as a live alternative to the
            # historical shares (which cannot see transfers, new signings or a changed role -
            # the same blind spot that made the team ratings mis-price promoted clubs).
            sum_xg = sum(q["xg90"] for q in xi) or 0.0
            sum_xa = sum(q["xa90"] for q in xi) or 0.0
            for p in xi:
                hist = min(0.40, surn.get(ps._surname(p["name"]), 0.0))
                w = p["mins"] / (p["mins"] + MIN_K)          # trust in this season's rate
                cur_g = (p["xg90"] / sum_xg) if sum_xg > 0 else hist
                cur_a = (p["xa90"] / sum_xa) if sum_xa > 0 else hist
                share = min(0.40, (1 - w) * hist + w * cur_g)
                a_share = min(0.40, (1 - w) * hist + w * cur_a)
                eg = share * lam
                ea = 0.6 * lam * a_share
                wdc = p["mins"] / (p["mins"] + MIN_K)
                eff_dc = wdc * p["dc90"] + (1 - wdc) * dc_med.get(p["pos"], 0.0)
                pts = (2.0 + eg * GOAL[p["pos"]] + ea * 3 + csp * CS[p["pos"]]
                       + _dc_points(eff_dc, p["pos"]))
                if p["pos"] in ("GK", "DEF"):
                    pts -= 0.5 * opp_lam
                if p["pos"] == "GK":
                    pts += min(1.6, 0.45 * opp_lam)     # rough saves contribution
                if team in cold:
                    pts *= 0.70          # promoted-team projections are unreliable; discount
                proj[id(p)] += max(0.0, pts) * p.get("avail", 1.0)   # scale by chance of featuring
                playing.add(id(p))
    rows = []
    for p in players:
        if id(p) in playing and proj[id(p)] > 0:
            rows.append({**{k: p[k] for k in ("id", "name", "team", "ot", "pos", "price", "own", "ep", "fit")},
                         "avail": round(p.get("avail", 1.0), 2),
                         "xg90": round(p.get("xg90", 0.0), 3), "xa90": round(p.get("xa90", 0.0), 3),
                         "dc90": round(p.get("dc90", 0.0), 2), "mins": p.get("mins", 0),
                         "proj": round(proj[id(p)], 2)})
    rows.sort(key=lambda r: -r["proj"])
    return rows


def build_team(gw, players=None, model=None, shares=None, weeks=None, team_meta=None,
               budget=100.0, rows=None):
    """Generate a full FPL squad for ONE gameweek from that week's projections:
    2 GK / 5 DEF / 5 MID / 3 FWD, <= budget, <= 3 per club, fit players only, tilted to
    value + low ownership. Then choose the best legal starting XI and captain."""
    if rows is None:
        rows = project_gw(gw, players=players, model=model, shares=shares, weeks=weeks)
    QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    START = {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
    CAP, PCAP = 3, 10.0
    if not rows:
        return None
    from collections import Counter
    FLOOR = {p: min(r["price"] for r in rows if r["pos"] == p) for p in QUOTA}

    def dsc(r):
        return (r["proj"] / r["price"]) * (1 - min(r["own"], 45) / 95.0)
    squad, cnt, club = [], {p: 0 for p in QUOTA}, Counter()
    for r in sorted([x for x in rows if x["price"] <= PCAP], key=lambda x: -dsc(x)):
        pos = r["pos"]
        if cnt[pos] >= START[pos] or club[r["ot"]] >= CAP:
            continue
        reserve = sum(FLOOR[p] * (QUOTA[p] - cnt[p]) for p in QUOTA) - FLOOR[pos]
        if sum(x["price"] for x in squad) + r["price"] + reserve > budget:
            continue
        squad.append(r); cnt[pos] += 1; club[r["ot"]] += 1
    for pos in QUOTA:
        while cnt[pos] < QUOTA[pos]:
            c = sorted([r for r in rows if r["pos"] == pos and r not in squad and club[r["ot"]] < CAP],
                       key=lambda r: (r["price"], -r["proj"]))[0]
            squad.append(c); cnt[pos] += 1; club[c["ot"]] += 1

    def bank():
        return budget - sum(r["price"] for r in squad)
    for _ in range(60):
        best = None
        for i, r in enumerate(squad):
            for c in sorted([x for x in rows if x["pos"] == r["pos"] and x not in squad and x["price"] <= PCAP],
                            key=lambda x: -x["proj"]):
                if c["proj"] <= r["proj"]:
                    break
                if c["price"] - r["price"] <= bank() and (c["ot"] == r["ot"] or club[c["ot"]] < CAP) and c["own"] <= 50:
                    g = c["proj"] - r["proj"]
                    if not best or g > best[0]:
                        best = (g, i, c, r)
        if not best:
            break
        _, i, c, r = best; club[r["ot"]] -= 1; club[c["ot"]] += 1; squad[i] = c

    # fixture per club this GW
    wk = next((w for w in (weeks or []) if w["gw"] == gw), None)
    fix = {}
    if wk:
        for e in wk["fixtures"]:
            fix[e["home"]] = (e["away_abbr"], "H")
            fix[e["away"]] = (e["home_abbr"], "A")
    tm = team_meta or {}

    def deco(r):
        opp, ha = fix.get(r["ot"], ("", ""))
        meta = tm.get(r["ot"], {})
        return {"id": r.get("id"), "nm": r["name"], "pos": r["pos"], "team": r["team"], "ot": r["ot"],
                "price": r["price"], "proj": round(r["proj"], 1), "own": r["own"],
                "color": meta.get("color", "#5B7A72"), "abbr": meta.get("abbr", r["ot"][:3].upper()),
                "opp": opp, "ha": ha}
    squad = [deco(r) for r in squad]

    # best legal XI: 1 GK + 10 outfield, DEF 3-5 / MID 2-5 / FWD 1-3
    by = {p: sorted([s for s in squad if s["pos"] == p], key=lambda x: -x["proj"]) for p in QUOTA}
    best_xi = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not (1 <= f <= 3):
                continue
            xi = by["GK"][:1] + by["DEF"][:d] + by["MID"][:m] + by["FWD"][:f]
            tot = sum(x["proj"] for x in xi)
            if best_xi is None or tot > best_xi[0]:
                best_xi = (tot, xi, f"{d}-{m}-{f}")
    _, xi, formation = best_xi
    xi_ids = {id(x) for x in xi}
    bench = [s for s in squad if id(s) not in xi_ids]
    bench.sort(key=lambda s: (s["pos"] != "GK", -s["proj"]))
    cap = max(xi, key=lambda s: s["proj"])
    vice = max([s for s in xi if s is not cap], key=lambda s: s["proj"])
    for s in squad:
        s["start"] = id(s) in xi_ids
        s["cap"] = s is cap
        s["vice"] = s is vice
    return {"gw": gw, "cost": round(sum(s["price"] for s in squad), 1),
            "formation": formation, "xi_proj": round(sum(x["proj"] for x in xi), 1),
            "captain": cap["nm"], "squad": squad,
            "bench_order": [b["nm"] for b in bench]}


if __name__ == "__main__":
    gw = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = project_gw(gw)
    print(f"\nMATCHWEEK {gw} — projected FPL points (fit players only), top by position:\n")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"== {pos} ==")
        for r in [x for x in rows if x["pos"] == pos][:6]:
            print(f"  {r['name']:<15}{r['team']:<15}{r['price']:>5.1f}m  proj {r['proj']:>5.2f}  ep {r['ep']:>4}  own {r['own']:>4}%")
        print()

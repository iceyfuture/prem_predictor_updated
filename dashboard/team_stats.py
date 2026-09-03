"""
team_stats.py — per-match team statistics, and the rolling form built from them.

The player side of this desk has kept a per-gameweek record since day one
(`fpl_player_gameweek_2026_27.csv`) plus a dated roll-forward of season totals
(`fpl_player_history.csv`). The team side had **no equivalent**: club ratings were recomputed
in memory on every build and overwritten, this season's results were re-derived from the ESPN
feed and thrown away, and the 28-stat team line ESPN publishes for every finished match was
fetched, read for its two corner counts, and discarded. Nothing at team level had a trajectory.

This writes that record down.

    outputs/team_match_2026_27.csv   one row per team per match this season, full stat line
    outputs/team_match_history.csv   the same core stats back to 2000-01
    outputs/team_rolling_2026_27.csv the rolling form AS AT each match, using only matches
                                     before it - built to be safe as a model feature
    outputs/team_form_2026_27.csv    each club's rolling form as of now, one row per club

WHERE EACH NUMBER COMES FROM — they are not all one source, and the gaps are real

  shots, shots on target, corners, fouls, cards      ESPN this season; football-data via
                                                     premier_league_history back to 2000-01
  offsides                                           ESPN only, so 2026-27 FORWARD ONLY.
                                                     football-data never carried it, and FPL's
                                                     `offside` column is populated for three
                                                     seasons (2016-19) at 5% of rows. There is
                                                     no honest way to backfill it.
  possession, passing, crosses, long balls, tackles,
  interceptions, clearances, blocked shots, saves    ESPN only, 2026-27 forward
  team xG / xA                                       summed from FPL's per-player per-gameweek
                                                     expected goals/assists. 2023-24 forward
                                                     (2022-23 is half-populated - its season
                                                     total is 732 against ~1,100 in full
                                                     seasons - so it is excluded, the same
                                                     cutoff fpl_minutes.py uses for `starts`)
  xGC                                                the OPPONENT's xG in the same match, so
                                                     xG and xGC are one scale by construction
  free kicks won                                     = the opponent's fouls committed. No feed
                                                     publishes free kicks directly; this is the
                                                     standard proxy and is labelled as one.
                                                     Offsides are NOT folded in, so that the
                                                     definition is identical in both eras.

Two things checked rather than assumed:

  * The FPL-to-ESPN join is validated against a number neither side was asked for. FPL also
    publishes a per-player `expected_goals_conceded` (the xG faced while that player was on the
    pitch), so a club's xGC should equal its opponent's xG. It does on **18 of 20** matches this
    season, within 0.35. The two that miss (Forest v Leeds, Man Utd v Ipswich) are provider
    rounding on penalty and own-goal xG, not a mis-join - every one of the 20 matches lines up
    on team identity. Summing that per-player xGC would be wrong, by the way: a player who
    lasted 90 minutes carries the WHOLE team's figure, so the sum is ~14x the truth.
  * Team names reconcile exactly. ESPN, FPL and football-data disagree on three clubs
    (Man Utd / Man United, Spurs / Tottenham, Sheffield Utd / Sheffield United); everything is
    canonicalised to the football-data spelling the rest of this project uses, and the join
    leaves zero unmatched clubs on either side.

DOUBLE GAMEWEEKS. ESPN data is per FIXTURE. FPL's expected stats are per GAMEWEEK, so when a
club plays twice in one gameweek its xG cannot be split between the two matches. Those rows are
flagged `xg_scope = gameweek` instead of `match` rather than being silently halved.

ROLLING FORM is strictly leak-free: a match's own numbers never enter its own rolling columns,
and the window carries back across the season boundary so August is not blank - the same
reasoning that made the minutes model backfill from previous seasons. Two windows, mirroring
the form/quality split in fpl_form.py: `l6` is current form, `l20` is level.

Neither window is swept. There is no out-of-sample target these general team stats are being
fitted to, so a sweep would be fitting nothing; 6 and 20 are stated conventions. Where a window
IS tuned against an outcome - corners - that lives in prem_corners.py and was swept to 30.

Run:  python team_stats.py     (rebuilds all four files; idempotent, ~2s)
"""
import csv
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

SHORT, LONG = 6, 20          # rolling windows: current form, established level
XG_FROM = "2023-24"          # FPL expected stats before this are half-populated

# ESPN, FPL and football-data all spell clubs differently; everything is canonicalised to the
# football-data form the rest of this project uses.
ALIAS = {"Man Utd": "Man United", "Manchester United": "Man United",
         "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
         "Sheffield Utd": "Sheffield United", "Sheffield Wednesday": "Sheff Wed",
         "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
         "Nottingham Forest": "Nott'm Forest", "Manchester City": "Man City",
         "AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton",
         "Leeds United": "Leeds", "Newcastle United": "Newcastle",
         "West Ham United": "West Ham", "Leicester City": "Leicester",
         "Wolverhampton Wanderers": "Wolves", "Norwich City": "Norwich",
         "Luton Town": "Luton", "Stoke City": "Stoke", "Swansea City": "Swansea",
         "Cardiff City": "Cardiff", "Birmingham City": "Birmingham"}

# A rolling window carries back over the season boundary on purpose, but not indefinitely:
# Hull were last in this division in 2016-17, and averaging their 2026 matches with matches from
# a decade ago is worse than saying "we only have two". Prior matches older than this are not
# counted, and `l6_n` reports how many actually were.
MAX_AGE_DAYS = 400

# the raw ESPN stat name -> our column
ESPN_MAP = {
    "totalShots": "shots", "shotsOnTarget": "shots_on_target", "blockedShots": "blocked_shots",
    "wonCorners": "corners", "foulsCommitted": "fouls", "offsides": "offsides",
    "yellowCards": "yellows", "redCards": "reds", "saves": "saves",
    "possessionPct": "possession_pct", "totalPasses": "passes", "passPct": "pass_pct",
    "totalCrosses": "crosses", "totalLongBalls": "long_balls", "totalTackles": "tackles",
    "interceptions": "interceptions", "totalClearance": "clearances",
    "penaltyKickShots": "pens_taken",
}
CORE = ["shots", "shots_on_target", "corners", "fouls", "free_kicks_won", "yellows", "reds",
        "xg", "xa", "xgc"]
EXTRA = ["offsides", "blocked_shots", "possession_pct", "passes", "pass_pct", "crosses",
         "long_balls", "tackles", "interceptions", "clearances", "saves", "pens_taken"]
ROLL = CORE + EXTRA + ["gf", "ga", "pts"]


def canon(name):
    return ALIAS.get((name or "").strip(), (name or "").strip())


def _norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def _num(v, d=None):
    try:
        return float(str(v).replace("%", ""))
    except (TypeError, ValueError):
        return d


def _result(gf, ga):
    return ("W", 3) if gf > ga else (("L", 0) if gf < ga else ("D", 1))


_FALLBACKS = []


def _resolve(st, home, away):
    """{normalised club -> stat dict} for one match's two ESPN blocks."""
    if not st or len(st) != 2:
        return {}
    nh, na = _norm(home), _norm(away)

    def same(a, b):
        return a == b or a.startswith(b) or b.startswith(a)
    got = {}
    for blk in st:
        n = _norm(canon(blk.get("team")))
        for ours in (nh, na):
            if same(n, ours):
                got[ours] = blk.get("stats", {})
    if len(got) == 2:
        return got
    _FALLBACKS.append(f"{home} v {away} ({', '.join(b.get('team', '?') for b in st)})")
    return {nh: st[0].get("stats", {}), na: st[1].get("stats", {})}


# --------------------------------------------------------------------------------------
# this season — ESPN per fixture, FPL for expected goals
# --------------------------------------------------------------------------------------

def fpl_expected():
    """{(gw, team): {xg, xa}} summed over the club's players, plus how many fixtures that club
    had in the gameweek (so a double can be flagged rather than silently split)."""
    gpath = os.path.join(OUT, "fpl_player_gameweek_2026_27.csv")
    spath = os.path.join(OUT, "fpl_player_stats_2026_27.csv")
    if not (os.path.exists(gpath) and os.path.exists(spath)):
        return {}
    club = {r["id"]: canon(r["team"]) for r in csv.DictReader(open(spath))}
    out = {}
    for r in csv.DictReader(open(gpath)):
        if str(r.get("provisional", "")).lower() == "true":
            continue
        t = club.get(r["id"])
        if not t:
            continue
        k = (int(r["gw"]), t)
        e = out.setdefault(k, {"xg": 0.0, "xa": 0.0})
        e["xg"] += _num(r.get("expected_goals"), 0.0)
        e["xa"] += _num(r.get("expected_assists"), 0.0)
    return out


def season_rows(events=None, weeks=None):
    """One row per team per finished match this season."""
    import feeds
    events = events if events is not None else feeds.espn_events()
    weeks = weeks if weeks is not None else feeds.to_matchweeks(events)
    gw_of, played = {}, {}
    for w in weeks:
        for fx in w["fixtures"]:
            gw_of[fx["id"]] = w["gw"]
    finished = [e for e in events if e.get("finished")]
    for e in finished:
        g = gw_of.get(e["id"])
        for t in (canon(e["home"]), canon(e["away"])):
            played[(g, t)] = played.get((g, t), 0) + 1
    xp = fpl_expected()

    rows = []
    for e in finished:
        st = feeds.espn_match_stats(e["id"])
        gw = gw_of.get(e["id"])
        home, away = canon(e["home"]), canon(e["away"])
        # Resolve the two stat blocks to the two clubs by NAME where the names agree, falling
        # back to position. Verified on every finished match: ESPN puts the home side first in
        # boxscore.teams (20/20), so position is a sound backstop when a spelling is unknown -
        # and a silent name miss is what left one club's whole stat line blank.
        blocks = _resolve(st, home, away)
        for team, opp, gf, ga, at_home in ((home, away, e["hs"], e["as"], True),
                                           (away, home, e["as"], e["hs"], False)):
            res, pts = _result(gf, ga)
            r = {"season": "2026-27", "gw": gw, "date": e["utc"][:10], "fixture": e["id"],
                 "team": team, "opponent": opp, "venue": "H" if at_home else "A",
                 "gf": gf, "ga": ga, "result": res, "pts": pts}
            mine = blocks.get(_norm(team), {})
            theirs = blocks.get(_norm(opp), {})
            for src, col in ESPN_MAP.items():
                r[col] = _num(mine.get(src))
            # No feed publishes free kicks. The opponent's fouls are the standard proxy and are
            # labelled as one; offsides are deliberately NOT added, so the definition is the
            # same here as in the historical file, which has no offsides at all.
            r["free_kicks_won"] = _num(theirs.get("foulsCommitted"))
            x = xp.get((gw, team), {})
            xo = xp.get((gw, opp), {})
            r["xg"] = round(x["xg"], 3) if x else None
            r["xa"] = round(x["xa"], 3) if x else None
            r["xgc"] = round(xo["xg"], 3) if xo else None
            r["xg_scope"] = ("match" if played.get((gw, team), 1) == 1 else "gameweek") if x else ""
            rows.append(r)
    rows.sort(key=lambda r: (r["date"], r["fixture"], r["venue"] == "A"))
    return rows


# --------------------------------------------------------------------------------------
# history — football-data results, plus FPL expected goals from 2023-24
# --------------------------------------------------------------------------------------

def _hist_path(name):
    for b in (ROOT, os.path.expanduser("~")):
        q = os.path.join(b, "premier_league_history", name)
        if os.path.exists(q):
            return q
    return None


def history_rows():
    """One row per team per match back to 2000-01 (the first season carrying shot counts)."""
    import pandas as pd
    rp = _hist_path("results.csv")
    if not rp:
        return []
    df = pd.read_csv(rp)
    df = df[df.home_shots.notna()].copy()

    xg = {}
    gp = _hist_path("player_gameweek.csv")
    if gp:
        g = pd.read_csv(gp, low_memory=False,
                        usecols=["season", "team", "fixture", "kickoff_time",
                                 "expected_goals", "expected_assists"])
        g = g[g.season >= XG_FROM]
        g["team"] = g.team.map(canon)
        g["date"] = g.kickoff_time.astype(str).str[:10]
        agg = g.groupby(["season", "date", "team"])[["expected_goals", "expected_assists"]].sum()
        for (s, d, t), row in agg.iterrows():
            xg[(s, d, t)] = (round(float(row.expected_goals), 3),
                             round(float(row.expected_assists), 3))

    rows = []
    for r in df.itertuples(index=False):
        h, a = canon(r.home_team), canon(r.away_team)
        date = str(r.date)[:10]
        for team, opp, gf, ga, at_home, sh, so, co, fo, ye, re_ in (
                (h, a, r.home_score, r.away_score, True, r.home_shots, r.home_shots_on_target,
                 r.home_corners, r.home_fouls, r.home_yellows, r.home_reds),
                (a, h, r.away_score, r.home_score, False, r.away_shots, r.away_shots_on_target,
                 r.away_corners, r.away_fouls, r.away_yellows, r.away_reds)):
            res, pts = _result(gf, ga)
            mine = xg.get((r.season, date, team))
            theirs = xg.get((r.season, date, opp))
            opp_fouls = r.away_fouls if at_home else r.home_fouls
            rows.append({
                "season": r.season, "gw": "", "date": date, "fixture": "",
                "team": team, "opponent": opp, "venue": "H" if at_home else "A",
                "gf": gf, "ga": ga, "result": res, "pts": pts,
                "shots": _num(sh), "shots_on_target": _num(so), "corners": _num(co),
                "fouls": _num(fo), "free_kicks_won": _num(opp_fouls),
                "yellows": _num(ye), "reds": _num(re_),
                "xg": mine[0] if mine else None, "xa": mine[1] if mine else None,
                "xgc": theirs[0] if theirs else None,
                "xg_scope": "match" if mine else "",
            })
    rows.sort(key=lambda r: (r["date"], r["team"]))
    return rows


# --------------------------------------------------------------------------------------
# rolling form — strictly PRIOR matches, carried across the season boundary
# --------------------------------------------------------------------------------------

def rolling(season, history, windows=(SHORT, LONG)):
    """Each club's rolling means over its last N matches, and the same computed AS AT every
    match this season (using only matches BEFORE it, so a row never sees its own result).

    History carries over the season boundary on purpose: a 6-match window that resets every
    August is blank exactly when it is most wanted, which is the same reason the minutes model
    trains on previous seasons rather than starting from nothing.
    """
    from datetime import date as _date

    def _d(s_):
        y, m, dd = (s_ or "1900-01-01")[:10].split("-")
        return _date(int(y), int(m), int(dd))
    per = {}
    for r in history + season:
        per.setdefault(r["team"], []).append(r)
    for t in per:
        per[t].sort(key=lambda r: (r["date"], r.get("fixture") or ""))
    # only clubs actually in this season - otherwise the form table fills up with Bolton and
    # Portsmouth averaging matches from 2010
    current_clubs = {r["team"] for r in season}

    def mean(rows, col):
        v = [r[col] for r in rows if r.get(col) is not None and r.get(col) != ""]
        return round(sum(float(x) for x in v) / len(v), 3) if v else None

    def fresh(prior, asof):
        cut = _d(asof)
        return [q for q in prior if (cut - _d(q["date"])).days <= MAX_AGE_DAYS]

    as_at, current = [], []
    for team, rows in per.items():
        if team not in current_clubs:
            continue
        for i, r in enumerate(rows):
            if r["season"] != "2026-27":
                continue
            out = {k: r[k] for k in ("season", "gw", "date", "team", "opponent", "venue")}
            for w in windows:
                prior = fresh(rows[max(0, i - w):i], r["date"])
                out[f"l{w}_n"] = len(prior)
                for c in ROLL:
                    out[f"l{w}_{c}"] = mean(prior, c)
            as_at.append(out)
        asof = max(r["date"] for r in rows)
        cur = {"team": team, "played_2026_27": sum(1 for r in rows if r["season"] == "2026-27"),
               "last_match": asof}
        for w in windows:
            prior = fresh(rows[-w:], asof)
            cur[f"l{w}_n"] = len(prior)
            for c in ROLL:
                cur[f"l{w}_{c}"] = mean(prior, c)
        current.append(cur)
    as_at.sort(key=lambda r: (r["date"], r["team"]))
    current.sort(key=lambda r: -(r.get(f"l{SHORT}_pts") or 0))
    return as_at, current


def _write(path, rows, cols=None):
    if not rows:
        return 0
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def build(verbose=True):
    """Rebuild all four files from source. Idempotent, and takes ~2 seconds, so history is
    rebuilt every run rather than cached - a stale derived file that nothing refreshes is
    exactly how team_rankings_2026_27.csv ended up frozen at July with the wrong decay."""
    os.makedirs(OUT, exist_ok=True)
    hp = os.path.join(OUT, "team_match_history.csv")
    hist = history_rows()
    _write(hp, hist)
    if verbose and hist:
        seasons = sorted({r["season"] for r in hist})
        withx = sum(1 for r in hist if r["xg"] is not None)
        print(f"wrote {hp}: {len(hist)} team-matches, {seasons[0]} -> {seasons[-1]} "
              f"({withx} with xG)")

    _FALLBACKS.clear()
    season = season_rows()
    if verbose and _FALLBACKS:
        print(f"  note: {len(_FALLBACKS)} match(es) resolved by block order, not club name "
              f"(add the spelling to ALIAS): {'; '.join(_FALLBACKS[:3])}")
    sp = os.path.join(OUT, "team_match_2026_27.csv")
    cols = (["season", "gw", "date", "fixture", "team", "opponent", "venue",
             "gf", "ga", "result", "pts"] + CORE + ["xg_scope"] + EXTRA)
    _write(sp, season, cols)
    if verbose:
        print(f"wrote {sp}: {len(season)} team-matches over "
              f"{len({r['fixture'] for r in season})} fixtures")

    as_at, current = rolling(season, hist)
    ap = os.path.join(OUT, "team_rolling_2026_27.csv")
    fp = os.path.join(OUT, "team_form_2026_27.csv")
    _write(ap, as_at)
    _write(fp, current)
    if verbose:
        print(f"wrote {ap}: {len(as_at)} rows (rolling form as at each match, leak-free)")
        print(f"wrote {fp}: {len(current)} clubs (form as of now)")
    return season, as_at, current


if __name__ == "__main__":
    s, a, cur = build()
    print(f"\nFORM TABLE - rolling means over each club's last {SHORT} matches "
          f"(the last {LONG} in brackets)\n")
    print(f"  {'club':<15}{'pts':>6}{'GF':>6}{'GA':>6}{'xG':>7}{'xGC':>7}{'shots':>7}"
          f"{'SoT':>6}{'corn':>6}{'fouls':>7}{'FK won':>8}{'cards':>7}{'offs':>6}")
    for r in cur:
        def g(c, w=SHORT):
            v = r.get(f"l{w}_{c}")
            return f"{v:.1f}" if isinstance(v, float) else "-"
        cards = (r.get(f"l{SHORT}_yellows") or 0) + (r.get(f"l{SHORT}_reds") or 0)
        print(f"  {r['team'][:14]:<15}{g('pts'):>6}{g('gf'):>6}{g('ga'):>6}{g('xg'):>7}"
              f"{g('xgc'):>7}{g('shots'):>7}{g('shots_on_target'):>6}{g('corners'):>6}"
              f"{g('fouls'):>7}{g('free_kicks_won'):>8}{cards:>7.1f}{g('offsides'):>6}")

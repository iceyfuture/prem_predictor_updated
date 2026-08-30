"""
Cross-reference the confirmed 2026/27 squads (squads_2026_27.csv) against our
historical performance data, by matching player names to the CSVs we built.

Outputs into outputs/:
  squad_2026_27_linked.csv     every 26/27 squad player + their linked historical
                               rating/stats (blank if no Premier League history in our
                               data — e.g. new signings, youth, overseas arrivals).
  player_rankings_2026_27.csv  our player ranking filtered to ONLY players who are in a
                               2026/27 squad (i.e. players no longer in the Prem removed).
  team_rankings_2026_27.csv    our team ranking filtered to the 20 clubs in 2026/27.
  players_removed.csv          historical players dropped because they are no longer in
                               a Premier League squad for 2026/27.
"""
import csv, os, re, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SQUADS = os.path.join(HERE, "squads_2026_27.csv")
PLAYER_RANK = os.path.join(OUT, "player_rankings.csv")
TEAM_RANK = os.path.join(OUT, "team_rankings.csv")

# squad CSV team name -> our football-data.co.uk naming (used across all our files)
TEAM_MAP = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton & Hove Albion": "Brighton", "Chelsea": "Chelsea",
    "Coventry City": "Coventry", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds", "Liverpool": "Liverpool", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower().replace(".", " ").replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", s).strip()


PARTICLES = {"de", "del", "da", "dos", "das", "van", "von", "der", "den", "di",
             "du", "la", "le", "el", "al", "bin", "ibn", "junior", "jr"}

# team-name aliases: our player data uses FPL names (Man Utd, Spurs), team_rankings /
# the squads file use football-data names (Man United, Tottenham). Canonicalise both.
TEAM_ALIAS = {
    "man utd": "manutd", "man united": "manutd", "manchester united": "manutd",
    "man city": "mancity", "manchester city": "mancity",
    "spurs": "tottenham", "tottenham hotspur": "tottenham", "tottenham": "tottenham",
    "newcastle united": "newcastle", "newcastle": "newcastle",
    "nottingham forest": "nottmforest", "nottm forest": "nottmforest",
    "nott'm forest": "nottmforest",
    "wolverhampton": "wolves", "wolves": "wolves",
    "brighton and hove albion": "brighton", "brighton & hove albion": "brighton",
    "brighton": "brighton",
    "sheffield united": "sheffieldutd", "sheffield utd": "sheffieldutd",
}


def canon_team(t):
    n = norm(t)
    return TEAM_ALIAS.get(n, n.replace(" ", ""))

IDX = {}   # name -> {full, web, web_last, snlast, sntoken}


def add(d, k, r):
    if k:
        d.setdefault(k, []).append(r)


def load_history():
    rows = list(csv.DictReader(open(PLAYER_RANK)))
    full_i, web_i, weblast_i, snlast_i, sntok_i = {}, {}, {}, {}, {}
    for r in rows:
        full = norm(r["first_name"] + " " + r["second_name"])
        web = norm(r["player"])
        sn = norm(r["second_name"]).split()
        r["_fi"] = norm(r["first_name"])[:1]
        add(full_i, full, r)
        add(web_i, web, r)
        add(weblast_i, web.split()[-1] if web else "", r)
        if sn:
            add(snlast_i, sn[-1], r)
            for t in sn:
                if t not in PARTICLES and len(t) > 2:
                    add(sntok_i, t, r)
    IDX.update(full=full_i, web=web_i, weblast=weblast_i, snlast=snlast_i, sntok=sntok_i)
    IDX["_rows"] = rows
    return rows


def _resolve(cands, team, fi):
    """Narrow a candidate list: prefer the player whose prior PL club == the squad's
    club, then whose first initial matches; return a single row or None if ambiguous."""
    cands = list({id(r): r for r in cands}.values())
    if not cands:
        return None
    ct = canon_team(team)
    t = [r for r in cands if canon_team(r["latest_team"]) == ct]
    if len(t) == 1:
        return t[0]
    if t:
        cands = t
    f = [r for r in cands if r["_fi"] == fi]
    if len(f) == 1:
        return f[0]
    if f:
        cands = f
    # still ambiguous (e.g. duplicate entities): take the most recent, then best-rated
    cands.sort(key=lambda r: (r["latest_season"], float(r["rating"])), reverse=True)
    return cands[0] if cands else None


def match_player(name, team):
    n = norm(name)
    toks = n.split()
    fi = toks[0][:1]
    surnames = [toks[-1]]
    if len(toks) >= 2:
        surnames.append(toks[-2] + " " + toks[-1])
    # Pool exact-full + web-name + web-last + surname candidates together, then let
    # _resolve pick by team -> first-initial -> MOST RECENT. Pooling (rather than an
    # early return on the first tier) is what makes a player's current row beat stale
    # season-split duplicates (Bruno/Bruno Miguel; Gabriel "Magalhães" vs "dos Santos
    # Magalhães"), which otherwise an exact-full or web hit would grab first.
    pool = IDX["full"].get(n, []) + IDX["web"].get(n, [])
    for s in surnames:
        pool += IDX["web"].get(s, []) + IDX["weblast"].get(s, []) + IDX["snlast"].get(s, [])
    r = _resolve(pool, team, fi)
    if r:
        return r, "name"
    # weaker fallback: any (non-particle) second-name token
    pool = [r for s in surnames for r in IDX["sntok"].get(s, [])]
    r = _resolve(pool, team, fi)
    if r:
        return r, "token"
    # mononym fallback (single-name squad entries like "Rodri" = Rodrigo Hernandez):
    # match a first-name / web prefix, but ONLY within the same club, so it stays safe.
    if len(toks) == 1:
        ct = canon_team(team)
        pool = [r for r in IDX["_rows"] if canon_team(r["latest_team"]) == ct and
                (norm(r["first_name"]).startswith(n) or norm(r["player"]).startswith(n))]
        r = _resolve(pool, team, fi)
        if r:
            return r, "mononym"
    return None, "none"


def main():
    squads = list(csv.DictReader(open(SQUADS)))
    hist_rows = load_history()

    STAT_COLS = ["rating", "w_points", "w_goals", "w_assists", "w_goal_involvements",
                 "w_xgi", "gi_per90", "w_minutes", "seasons", "latest_team",
                 "latest_season", "position"]
    linked, matched_hist_ids = [], set()
    n_matched = 0
    for s in squads:
        team_pl = TEAM_MAP.get(s["Team"], s["Team"])
        m, how = match_player(s["Player"], team_pl)
        row = {"team_2026_27": team_pl, "player": s["Player"],
               "matched": bool(m), "match_type": how}
        if m:
            n_matched += 1
            matched_hist_ids.add(id(m))
            for c in STAT_COLS:
                row[c] = m.get(c, "")
            row["prior_pl_team"] = m.get("latest_team", "")
        else:
            for c in STAT_COLS:
                row[c] = ""
            row["prior_pl_team"] = ""
        linked.append(row)

    # write linked squad file
    cols = ["team_2026_27", "player", "matched", "match_type", "prior_pl_team",
            "position", "rating", "w_points", "w_goals", "w_assists",
            "w_goal_involvements", "w_xgi", "gi_per90", "w_minutes", "seasons",
            "latest_team", "latest_season"]
    with open(os.path.join(OUT, "squad_2026_27_linked.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in linked:
            w.writerow({c: r.get(c, "") for c in cols})

    # filtered player rankings: only historical players who ARE in a 26/27 squad
    kept = [r for r in hist_rows if id(r) in matched_hist_ids]
    removed = [r for r in hist_rows if id(r) not in matched_hist_ids]
    outcols = [c for c in csv.DictReader(open(PLAYER_RANK)).fieldnames]
    kept.sort(key=lambda r: float(r["rating"]), reverse=True)
    for i, r in enumerate(kept, 1):
        r["rank"] = i
    with open(os.path.join(OUT, "player_rankings_2026_27.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=outcols, extrasaction="ignore")
        w.writeheader(); w.writerows(kept)
    removed.sort(key=lambda r: float(r["rating"]), reverse=True)
    with open(os.path.join(OUT, "players_removed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=outcols, extrasaction="ignore")
        w.writeheader(); w.writerows(removed)

    # filtered team rankings: only the 20 clubs in 2026/27
    keep_teams = set(TEAM_MAP.values())
    trows = list(csv.DictReader(open(TEAM_RANK)))
    tkept = [r for r in trows if r["team"] in keep_teams]
    tmiss = keep_teams - {r["team"] for r in trows}
    for i, r in enumerate(tkept, 1):
        r["rank"] = i
    with open(os.path.join(OUT, "team_rankings_2026_27.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=trows[0].keys()); w.writeheader(); w.writerows(tkept)

    # ---- report ----
    print(f"Squad players: {len(squads)}   matched to history: {n_matched} "
          f"({100*n_matched/len(squads):.0f}%)   no PL history: {len(squads)-n_matched}")
    print(f"Historical players kept (in a 26/27 squad): {len(kept)}")
    print(f"Historical players REMOVED (no longer in Prem): {len(removed)}")
    print(f"Team rankings kept: {len(tkept)}/20   "
          f"(26/27 clubs with no rating in our 2018-26 window: {sorted(tmiss) or 'none'})")
    print("\nUnmatched squad players (no PL history in our data):")
    un = [r["player"] + f" ({r['team_2026_27']})" for r in linked if not r["matched"]]
    for i in range(0, len(un), 3):
        print("   " + " | ".join(un[i:i+3]))
    print(f"\nTop 5 removed-from-Prem by prior rating:")
    for r in removed[:5]:
        print(f"   {r['player']:<16} {r['latest_team']:<14} rating {r['rating']}")


if __name__ == "__main__":
    main()

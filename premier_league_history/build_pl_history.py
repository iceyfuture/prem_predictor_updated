#!/usr/bin/env python3
"""Build a martj42-style flat historical folder for Premier League club football.

Outputs into ~/premier_league_history/:
  results.csv              one row per PL match, 1993/94 -> 2025/26 (+ team perf stats)
  player_season_totals.csv one row per player per season, 2016-17 -> latest
  player_gameweek.csv      per-player per-gameweek rows, 2016-17 -> latest
  teams.csv                distinct teams seen in results, with season span
Sources: football-data.co.uk (results/stats), vaastav/Fantasy-Premier-League (players).
"""
import csv, io, os, sys, urllib.request
from datetime import datetime

OUT = os.path.expanduser("~/premier_league_history")
RAW = os.path.join(OUT, ".raw")
os.makedirs(RAW, exist_ok=True)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

def fetch(url, cache_name, binary=False):
    path = os.path.join(RAW, cache_name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        data = urllib.request.urlopen(req, timeout=45).read()
    except Exception as e:
        print(f"  ! {url} -> {e}")
        return None
    with open(path, "wb") as f:
        f.write(data)
    return data

# ---------------------------------------------------------------- results.csv
def season_label(start_year):
    return f"{start_year}-{str(start_year+1)[-2:]}"

def code(start_year):
    return f"{str(start_year)[-2:]}{str(start_year+1)[-2:]}"

def parse_date(s):
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s

RESULT_COLS = ["season","date","home_team","away_team","home_score","away_score","result",
    "ht_home_score","ht_away_score","ht_result","referee",
    "home_shots","away_shots","home_shots_on_target","away_shots_on_target",
    "home_fouls","away_fouls","home_corners","away_corners",
    "home_yellows","away_yellows","home_reds","away_reds"]

FD_MAP = {  # football-data col -> our col
    "FTHG":"home_score","FTAG":"away_score","FTR":"result",
    "HTHG":"ht_home_score","HTAG":"ht_away_score","HTR":"ht_result","Referee":"referee",
    "HS":"home_shots","AS":"away_shots","HST":"home_shots_on_target","AST":"away_shots_on_target",
    "HF":"home_fouls","AF":"away_fouls","HC":"home_corners","AC":"away_corners",
    "HY":"home_yellows","AY":"away_yellows","HR":"home_reds","AR":"away_reds"}

def build_results():
    print("== results.csv (football-data.co.uk) ==")
    rows = []
    teams = {}
    for yr in range(1993, 2026):
        url = f"https://www.football-data.co.uk/mmz4281/{code(yr)}/E0.csv"
        data = fetch(url, f"E0_{code(yr)}.csv")
        if not data:
            continue
        text = data.decode("utf-8-sig", errors="replace")
        rdr = csv.DictReader(io.StringIO(text))
        n = 0
        for r in rdr:
            r = { (k.strip() if k else k): (v.strip() if isinstance(v,str) else v) for k,v in r.items() }
            if not r.get("HomeTeam") or not r.get("Date"):
                continue
            out = {c: "" for c in RESULT_COLS}
            out["season"] = season_label(yr)
            out["date"] = parse_date(r["Date"])
            out["home_team"] = r["HomeTeam"]
            out["away_team"] = r["AwayTeam"]
            for fd, oc in FD_MAP.items():
                if r.get(fd) not in (None, ""):
                    out[oc] = r[fd]
            rows.append(out)
            for t in (r["HomeTeam"], r["AwayTeam"]):
                teams.setdefault(t, [yr, yr])
                teams[t][0] = min(teams[t][0], yr); teams[t][1] = max(teams[t][1], yr)
            n += 1
        print(f"  {season_label(yr)}: {n} matches")
    rows.sort(key=lambda x: (x["date"], x["home_team"]))
    with open(os.path.join(OUT,"results.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT,"teams.csv"),"w",newline="") as f:
        w = csv.writer(f); w.writerow(["team","first_season","last_season","seasons"])
        for t in sorted(teams):
            a,b = teams[t]; w.writerow([t, season_label(a), season_label(b), b-a+1])
    print(f"  -> {len(rows)} total matches, {len(teams)} teams\n")
    return len(rows)

# ---------------------------------------------- vaastav player data
POS = {"1":"GKP","2":"DEF","3":"MID","4":"FWD"}
VA = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

def load_team_map(season):
    data = fetch(f"{VA}/{season}/teams.csv", f"va_teams_{season}.csv")
    if not data: return {}
    rdr = csv.DictReader(io.StringIO(data.decode("utf-8", errors="replace")))
    return {r["id"]: r["name"] for r in rdr}

TOTAL_COLS = ["season","player_id","first_name","second_name","web_name","team","position",
    "total_points","minutes","goals_scored","assists","clean_sheets","goals_conceded",
    "bonus","bps","influence","creativity","threat","ict_index",
    "expected_goals","expected_assists","now_cost","selected_by_percent","starts"]

def build_season_totals(seasons):
    print("== player_season_totals.csv (vaastav) ==")
    out_rows = []
    for s in seasons:
        data = fetch(f"{VA}/{s}/players_raw.csv", f"va_raw_{s}.csv")
        if not data:
            print(f"  {s}: (missing)"); continue
        tmap = load_team_map(s)
        rdr = csv.DictReader(io.StringIO(data.decode("utf-8", errors="replace")))
        n = 0
        for r in rdr:
            out = {c:"" for c in TOTAL_COLS}
            out["season"] = s
            out["player_id"] = r.get("id","")
            out["first_name"] = r.get("first_name","")
            out["second_name"] = r.get("second_name","")
            out["web_name"] = r.get("web_name","")
            out["team"] = tmap.get(r.get("team",""), r.get("team",""))
            out["position"] = POS.get(r.get("element_type",""), r.get("element_type",""))
            for c in ["total_points","minutes","goals_scored","assists","clean_sheets",
                      "goals_conceded","bonus","bps","influence","creativity","threat",
                      "ict_index","expected_goals","expected_assists","now_cost",
                      "selected_by_percent","starts"]:
                out[c] = r.get(c,"")
            out_rows.append(out); n += 1
        print(f"  {s}: {n} players")
    with open(os.path.join(OUT,"player_season_totals.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=TOTAL_COLS); w.writeheader(); w.writerows(out_rows)
    print(f"  -> {len(out_rows)} player-seasons\n")

def build_gameweeks(seasons):
    print("== player_gameweek.csv (vaastav merged_gw) ==")
    all_cols, all_rows = ["season"], []
    seen = {"season"}
    for s in seasons:
        data = fetch(f"{VA}/{s}/gws/merged_gw.csv", f"va_gw_{s}.csv")
        if not data:
            print(f"  {s}: (missing)"); continue
        # merged_gw files are sometimes latin-1 in early seasons
        for enc in ("utf-8","latin-1"):
            try:
                text = data.decode(enc); break
            except UnicodeDecodeError:
                continue
        rdr = csv.DictReader(io.StringIO(text))
        cols = rdr.fieldnames or []
        for c in cols:
            if c not in seen: seen.add(c); all_cols.append(c)
        n = 0
        for r in rdr:
            r["season"] = s; all_rows.append(r); n += 1
        print(f"  {s}: {n} rows")
    with open(os.path.join(OUT,"player_gameweek.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows: w.writerow(r)
    print(f"  -> {len(all_rows)} player-gameweeks, {len(all_cols)} cols\n")

def discover_seasons():
    seasons = []
    for yr in range(2016, 2026):
        s = season_label(yr)
        if fetch(f"{VA}/{s}/players_raw.csv", f"va_raw_{s}.csv") is not None:
            seasons.append(s)
    return seasons

if __name__ == "__main__":
    build_results()
    seasons = discover_seasons()
    print(f"vaastav player seasons available: {seasons}\n")
    build_season_totals(seasons)
    build_gameweeks(seasons)
    print("DONE. Folder:", OUT)

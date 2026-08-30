"""
Live feeds for the Floodlit terminal — ESPN (fixtures + market odds) and FPL (team news).

Both are public, key-less endpoints. Every field read here was verified against a live
response before being used (see verify_feeds() at the bottom — run it any time the shapes
look suspect; it prints exactly what it found rather than guessing).

ESPN  https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates=YYYYMMDD-YYYYMMDD
      -> real 2026/27 fixtures, kickoff, venue, status, live/final scores,
         team abbreviation + club colours, and DraftKings moneyline / totals.
FPL   https://fantasy.premierleague.com/api/bootstrap-static/
      -> player status, injury news, chance-of-playing, price, form, expected points.

SEASON GUARD: the FPL API rolls over to a new season late. fpl_feed() reports which season
it is actually serving so the UI can label stale news instead of showing it as current.
"""
import json
import os
import urllib.request
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
os.makedirs(CACHE, exist_ok=True)

ESPN = ("https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/"
        "scoreboard?dates={}&limit=500")
FPL = "https://fantasy.premierleague.com/api/bootstrap-static/"
UA = {"User-Agent": "Mozilla/5.0"}

FINISHED = {"STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FINAL_PEN", "STATUS_FINAL_AET"}
LIVE = {"STATUS_IN_PROGRESS", "STATUS_HALFTIME", "STATUS_FIRST_HALF", "STATUS_SECOND_HALF"}

# ESPN display name -> the team names used across this repo (football-data style)
ESPN_TEAMS = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton", "Chelsea": "Chelsea",
    "Coventry City": "Coventry", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds", "Liverpool": "Liverpool", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
}
# FPL short/long names -> same canonical set
FPL_TEAMS = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Burnley": "Burnley",
    "Chelsea": "Chelsea", "Coventry": "Coventry", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Fulham": "Fulham", "Hull": "Hull", "Ipswich": "Ipswich",
    "Leeds": "Leeds", "Liverpool": "Liverpool", "Man City": "Man City",
    # FPL uses the full club name for the promoted sides - without these the alias lookup
    # falls through and "Coventry City" / "Hull City" / "Ipswich Town" leak in as separate teams
    "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Man Utd": "Man United", "Newcastle": "Newcastle", "Nott'm Forest": "Nott'm Forest",
    "Spurs": "Tottenham", "Sunderland": "Sunderland", "West Ham": "West Ham",
    "Wolves": "Wolves",
}


def _get(url, cache_key, max_age_min=180):
    """Fetch with a short-lived disk cache so repeated builds don't hammer the feed."""
    p = os.path.join(CACHE, cache_key)
    if os.path.exists(p):
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(p))).total_seconds() / 60
        if age < max_age_min:
            return json.load(open(p))
    data = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45))
    json.dump(data, open(p, "w"))
    return data


# --------------------------------------------------------------------- odds
def american_to_decimal(v):
    try:
        v = float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    return round(1 + (100 / abs(v) if v < 0 else v / 100), 3)


def parse_odds(comp):
    """DraftKings moneyline -> decimal odds + vig-free implied probabilities."""
    odds = [o for o in (comp.get("odds") or []) if isinstance(o, dict)]
    if not odds:
        return None
    o = odds[0]
    ml = o.get("moneyline") or {}
    def side(k):
        d = (ml.get(k) or {})
        node = d.get("close") or d.get("open") or {}
        return american_to_decimal(node.get("odds"))
    h, dr, a = side("home"), side("draw"), side("away")
    if not (h and dr and a):
        return None
    inv = [1 / h, 1 / dr, 1 / a]
    s = sum(inv)
    tot = o.get("total") or {}
    return {
        "provider": (o.get("provider") or {}).get("displayName", "book"),
        "dec": {"h": h, "d": dr, "a": a},
        "imp": {"h": round(inv[0] / s, 4), "d": round(inv[1] / s, 4), "a": round(inv[2] / s, 4)},
        "overround": round((s - 1) * 100, 1),
        "ou_line": o.get("overUnder"),
    }


# --------------------------------------------------------------------- ESPN
def espn_events(start="20260801", end="20270601", chunk_days=45):
    """All PL events in the window, fetched in chunks (ESPN caps a single range)."""
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    seen, out = set(), []
    cur = s
    while cur < e:
        nxt = min(cur + timedelta(days=chunk_days), e)
        rng = f"{cur.strftime('%Y%m%d')}-{nxt.strftime('%Y%m%d')}"
        try:
            data = _get(ESPN.format(rng), f"espn_{rng}.json")
        except Exception as ex:
            print(f"  ! ESPN {rng}: {ex}")
            cur = nxt + timedelta(days=1)
            continue
        for ev in data.get("events", []):
            if ev["id"] in seen:
                continue
            seen.add(ev["id"])
            c = ev["competitions"][0]
            try:
                h = next(x for x in c["competitors"] if x["homeAway"] == "home")
                a = next(x for x in c["competitors"] if x["homeAway"] == "away")
            except StopIteration:
                continue
            hn = ESPN_TEAMS.get(h["team"]["displayName"])
            an = ESPN_TEAMS.get(a["team"]["displayName"])
            if not hn or not an:            # not a PL club we know -> report, don't guess
                print(f"  ! unmapped ESPN team: {h['team']['displayName']} / {a['team']['displayName']}")
                continue
            st = ev["status"]["type"]["name"]
            out.append({
                "id": ev["id"], "utc": ev["date"], "home": hn, "away": an,
                "home_abbr": h["team"].get("abbreviation", hn[:3].upper()),
                "away_abbr": a["team"].get("abbreviation", an[:3].upper()),
                "home_color": "#" + (h["team"].get("color") or "5B7A72"),
                "away_color": "#" + (a["team"].get("color") or "5B7A72"),
                "venue": (c.get("venue") or {}).get("fullName", ""),
                "status": st,
                "finished": st in FINISHED, "live": st in LIVE,
                "hs": int(h["score"]) if str(h.get("score", "")).isdigit() else None,
                "as": int(a["score"]) if str(a.get("score", "")).isdigit() else None,
                "odds": parse_odds(c),
            })
        cur = nxt + timedelta(days=1)
    out.sort(key=lambda x: x["utc"])
    return out


def to_matchweeks(events):
    """Reconstruct matchweeks: ESPN has no round number, so group in date order,
    starting a new week when a club would appear twice or the week reaches 10 games."""
    weeks, cur, used = [], [], set()
    for ev in events:
        if len(cur) >= 10 or ev["home"] in used or ev["away"] in used:
            weeks.append(cur); cur, used = [], set()
        cur.append(ev); used |= {ev["home"], ev["away"]}
    if cur:
        weeks.append(cur)
    return [{"gw": i, "fixtures": w} for i, w in enumerate(weeks, 1)]


# ---------------------------------------------------------------------- FPL
def fpl_feed():
    """Team news / injuries / prices. Includes a guard reporting which season FPL serves."""
    b = _get(FPL, "fpl_bootstrap.json", max_age_min=120)
    teams = {t["id"]: FPL_TEAMS.get(t["name"], t["name"]) for t in b.get("teams", [])}
    events = b.get("events", [])
    first = (events[0].get("deadline_time") or "")[:4] if events else ""
    cur = next((e for e in events if e.get("is_current")), None)
    nxt = next((e for e in events if e.get("is_next")), None)
    season = f"{first}/{str(int(first)+1)[-2:]}" if first.isdigit() else "unknown"

    news = []
    for p in b.get("elements", []):
        status, chance = p.get("status", "a"), p.get("chance_of_playing_next_round")
        if status != "a" or (p.get("news") or ""):
            flag = {"i": "out", "s": "out", "u": "out", "d": "doubt"}.get(status, "back")
            link = (p.get("scout_news_link") or "").strip()
            news.append({
                "id": p.get("id"), "who": p.get("web_name", ""), "team": teams.get(p.get("team"), ""),
                "what": (p.get("news") or "").strip() or _status_text(status, chance),
                "flag": flag, "chance": chance,
                "added": (p.get("news_added") or "")[:16].replace("T", " "),
                "url": link if link.startswith("http") else "",
                "_s": (p.get("news_added") or ""),
            })
    # newest first — a live wire, not grouped by severity
    news.sort(key=lambda x: x["_s"], reverse=True)
    for n in news:
        n.pop("_s")
    picks = []

    def ep(p):
        try:
            return float(p.get("ep_next") or 0)
        except (TypeError, ValueError):
            return 0.0
    POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    for p in sorted([e for e in b.get("elements", []) if e.get("status") == "a"],
                    key=ep, reverse=True)[:8]:
        picks.append({"pos": POS.get(p.get("element_type"), "?"), "nm": p.get("web_name", ""),
                      "team": teams.get(p.get("team"), ""),
                      "price": f"{p.get('now_cost',0)/10:.1f}", "xpts": f"{ep(p):.1f}",
                      "form": p.get("form", "0")})
    return {
        "season": season,
        "current_gw": (cur or nxt or {}).get("name", ""),
        "deadline": (nxt or {}).get("deadline_time", ""),
        "teams": sorted(set(teams.values())),
        "news": news, "picks": picks,
    }


def fpl_event_live(gw):
    """Actual FPL points scored by every player in a finished gameweek (for the forward-test).
    Returns {element_id: total_points}. Empty until the gameweek is played."""
    try:
        data = _get(f"https://fantasy.premierleague.com/api/event/{gw}/live/",
                    f"fpl_live_gw{gw}.json", max_age_min=180)
    except Exception:
        return {}
    out = {}
    for e in data.get("elements", []):
        out[e.get("id")] = (e.get("stats") or {}).get("total_points", 0)
    return out


def _status_text(status, chance):
    label = {"i": "injured", "d": "doubtful", "s": "suspended", "u": "unavailable"}.get(status, "")
    if chance not in (None, 100):
        return f"{label} - {chance}% chance".strip(" -")
    return label or "flagged"


# ------------------------------------------------------------------ verify
def verify_feeds():
    print("== ESPN ==")
    evs = espn_events()
    wks = to_matchweeks(evs)
    withodds = sum(1 for e in evs if e["odds"])
    print(f"  {len(evs)} fixtures, {len(wks)} matchweeks, {withodds} with moneyline odds")
    print(f"  first: {evs[0]['utc'][:16]} {evs[0]['home']} v {evs[0]['away']} @ {evs[0]['venue']}")
    print(f"  last : {evs[-1]['utc'][:16]} {evs[-1]['home']} v {evs[-1]['away']}")
    sizes = {}
    for w in wks:
        sizes[len(w["fixtures"])] = sizes.get(len(w["fixtures"]), 0) + 1
    print(f"  matchweek sizes: {sizes}")
    print("== FPL ==")
    f = fpl_feed()
    print(f"  serving season {f['season']} | {f['current_gw']} | {len(f['news'])} news rows")
    print(f"  teams: {len(f['teams'])}")
    for n in f["news"][:3]:
        print(f"    [{n['flag']}] {n['who']} ({n['team']}): {n['what'][:60]}")


if __name__ == "__main__":
    verify_feeds()


def espn_match_stats(event_id, max_age_min=100000):
    """Full team stat line for ONE finished match (corners, shots, possession, passing,
    tackles, cards...). Cached forever once the game is over, so the season costs one call
    per match, not one per build."""
    try:
        d = _get_url(f"https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/summary?event={event_id}",
                     f"espn_sum_{event_id}.json", max_age_min)
    except Exception:
        return None
    out = []
    for t in (d.get("boxscore") or {}).get("teams", []):
        st = {s.get("name"): s.get("displayValue") for s in t.get("statistics", [])}
        out.append({"team": t.get("team", {}).get("displayName", ""), "stats": st})
    return out if len(out) == 2 else None


def _get_url(url, cache_key, max_age_min):
    p = os.path.join(CACHE, cache_key)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(p))).total_seconds() / 60
        if age < max_age_min:
            return json.load(open(p))
    data = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45))
    json.dump(data, open(p, "w"))
    return data

"""
apifootball.py — per-player, per-fixture statistics under a licence that survives sharing.

WHY THIS AND NOT FOTMOB
FotMob has no public player-stats endpoint; getting it means scraping player pages, which is
against their terms and breaks without warning. This project is published to a public URL and
shared, so that is the wrong foundation. API-Football carries the same Opta-style stats, the
key is already provisioned, and the volume needed is trivial: 10 Premier League fixtures a
gameweek is 10 requests against a 100/day free allowance.

WHAT IT ADDS OVER THE FPL FEED
build_player_stats.py already pulls xG/xA/xGI/xGC per player from FPL's bootstrap-static. Those
are the OUTCOME expected-stats. What is missing is the VOLUME underneath them:

    shots, shots on target, key passes, passes, pass accuracy, dribbles,
    duels won, tackles, fouls, touches, position, rating, captain, substitute

That distinction is the whole point. Measured on 520 team-seasons, shot DIFFERENCE beats goal
difference as a predictor after four games (r=0.618 vs 0.553, bootstrap 95% CI on the gap
[+0.003, +0.127]). The same logic runs at player level: after four games a striker's shots and
box touches are a bigger sample than his goals, and should predict his next goals better.

AUTH
Key comes from the environment variable API_FOOTBALL_KEY and is never written to disk or
committed. Without it every function returns empty and the caller carries on unchanged.

BUDGET
A finished fixture's player stats never change, so they are cached permanently under
dashboard/.cache/apifootball/. Only unplayed or newly-finished fixtures cost a request.
"""
import json, os, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache", "apifootball")
BASE = "https://v3.football.api-sports.io"
LEAGUE_EPL = 39
TIMEOUT = 20


def _key():
    return os.environ.get("API_FOOTBALL_KEY", "").strip()


def available():
    return bool(_key())


def _get(path, params, cache_key=None, ttl=None):
    """GET with an on-disk cache. `ttl=None` means cache forever (finished fixtures)."""
    if cache_key:
        os.makedirs(CACHE, exist_ok=True)
        cp = os.path.join(CACHE, cache_key + ".json")
        if os.path.exists(cp) and (ttl is None or time.time() - os.path.getmtime(cp) < ttl):
            try:
                with open(cp) as f:
                    return json.load(f)
            except (OSError, ValueError):
                pass                                   # corrupt cache, refetch
    k = _key()
    if not k:
        return None
    qs = "&".join(f"{a}={b}" for a, b in params.items())
    req = urllib.request.Request(f"{BASE}/{path}?{qs}", headers={"x-apisports-key": k})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = json.load(r)
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        print(f"  ! api-football {path} failed ({e})")
        return None
    if body.get("errors"):
        print(f"  ! api-football {path} returned errors: {body['errors']}")
        return None
    if cache_key:
        try:
            with open(os.path.join(CACHE, cache_key + ".json"), "w") as f:
                json.dump(body, f)
        except OSError:
            pass
    return body


def quota():
    """{used, limit} for today, or None. Costs nothing against the allowance."""
    b = _get("status", {})
    try:
        req = b["response"]["requests"]
        return {"used": req["current"], "limit": req["limit_day"]}
    except (TypeError, KeyError):
        return None


def fixtures(season, league=LEAGUE_EPL, ttl=6 * 3600):
    """[{id, utc, round, home, away, hg, ag, status}] for the season."""
    b = _get("fixtures", {"league": league, "season": season},
             cache_key=f"fixtures_{league}_{season}", ttl=ttl)
    out = []
    for f in (b or {}).get("response", []):
        fx, tm, gl = f.get("fixture", {}), f.get("teams", {}), f.get("goals", {})
        out.append({
            "id": fx.get("id"), "utc": (fx.get("date") or "")[:16].replace("T", " "),
            "round": (f.get("league", {}).get("round") or "").replace("Regular Season - ", ""),
            "home": tm.get("home", {}).get("name"), "away": tm.get("away", {}).get("name"),
            "hg": gl.get("home"), "ag": gl.get("away"),
            "status": (fx.get("status", {}) or {}).get("short", ""),
        })
    return out


# the volume stats that FPL does not carry, flattened out of the nested response
def _flat(s):
    g = lambda *p: _dig(s, *p)
    return {
        "minutes": g("games", "minutes"), "rating": g("games", "rating"),
        "position": g("games", "position"), "captain": g("games", "captain"),
        "substitute": g("games", "substitute"),
        "shots": g("shots", "total"), "shots_on": g("shots", "on"),
        "goals": g("goals", "total"), "assists": g("goals", "assists"),
        "conceded": g("goals", "conceded"), "saves": g("goals", "saves"),
        "passes": g("passes", "total"), "key_passes": g("passes", "key"),
        "pass_pct": g("passes", "accuracy"),
        "tackles": g("tackles", "total"), "blocks": g("tackles", "blocks"),
        "interceptions": g("tackles", "interceptions"),
        "duels": g("duels", "total"), "duels_won": g("duels", "won"),
        "dribbles": g("dribbles", "attempts"), "dribbles_won": g("dribbles", "success"),
        "fouls_drawn": g("fouls", "drawn"), "fouls_committed": g("fouls", "committed"),
        "yellow": g("cards", "yellow"), "red": g("cards", "red"),
        "pens_scored": g("penalty", "scored"), "pens_missed": g("penalty", "missed"),
    }


def _dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def player_stats(fixture_id, finished=True):
    """[{fixture, team, player_id, name, ...volume stats}] for one fixture.

    A finished fixture is cached forever - its stats cannot change - so a full season backfill
    costs one request per fixture once and nothing thereafter.
    """
    b = _get("fixtures/players", {"fixture": fixture_id},
             cache_key=f"players_{fixture_id}" if finished else None,
             ttl=None if finished else 0)
    rows = []
    for side in (b or {}).get("response", []):
        team = _dig(side, "team", "name")
        for p in side.get("players", []):
            info = p.get("player", {}) or {}
            stats = (p.get("statistics") or [{}])[0]
            rows.append(dict(fixture=fixture_id, team=team,
                             player_id=info.get("id"), name=info.get("name"),
                             **_flat(stats)))
    return rows

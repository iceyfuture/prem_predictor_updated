"""
Kalshi EPL feed — public, read-only (no API key), same pattern as the World Cup bot.

Series KXEPLGAME carries one event per fixture with a 3-way market (home / away / Tie),
which maps 1:1 onto our model's 1X2 probabilities.

Kalshi quotes in dollars 0.00-1.00, which IS the implied probability of that outcome.
To take YES you pay the ASK; to fade it you take the BID. So:
    edge (buy YES) = model_prob - yes_ask      (in probability points)
    EV%             = model_prob / yes_ask - 1

Every field below was read off a live response (see verify() at the bottom).
Public market URL (verified against kalshi.com):
    https://kalshi.com/markets/kxeplgame/english-premier-league-game/<event_ticker lower>
"""
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
os.makedirs(CACHE, exist_ok=True)

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXEPLGAME"
UA = {"User-Agent": "Mozilla/5.0"}
WEB = "https://kalshi.com/markets/kxeplgame/english-premier-league-game/{}"

# Kalshi display name -> the team names used across this repo
KALSHI_TEAMS = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Chelsea": "Chelsea",
    "Coventry": "Coventry", "Coventry City": "Coventry", "Crystal Palace": "Crystal Palace",
    "Everton": "Everton", "Fulham": "Fulham", "Hull City": "Hull", "Hull": "Hull",
    "Ipswich Town": "Ipswich", "Ipswich": "Ipswich", "Leeds United": "Leeds", "Leeds": "Leeds",
    "Liverpool": "Liverpool", "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle": "Newcastle", "Newcastle United": "Newcastle",
    "Nottingham": "Nott'm Forest", "Nottingham Forest": "Nott'm Forest",
    "Sunderland": "Sunderland", "Tottenham": "Tottenham", "Wolverhampton": "Wolves",
}


def _get(path, cache_key, max_age_min=30):
    p = os.path.join(CACHE, cache_key)
    if os.path.exists(p):
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(p))).total_seconds() / 60
        if age < max_age_min:
            return json.load(open(p))
    data = json.load(urllib.request.urlopen(
        urllib.request.Request(API + path, headers=UA), timeout=30))
    json.dump(data, open(p, "w"))
    return data


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_markets(max_age_min=30):
    """One row per fixture: {home, away, url, legs{h,d,a}, vig, thin}. Prices are
    probabilities (0-1). Unmapped team names are PRINTED, never guessed."""
    try:
        d = _get(f"/events?series_ticker={SERIES}&status=open&with_nested_markets=true&limit=200",
                 "kalshi_epl.json", max_age_min)
    except Exception as e:
        print(f"  ! Kalshi fetch failed: {e}")
        return {}
    out = {}
    for e in d.get("events", []):
        title = e.get("title", "")
        if " vs " not in title:
            continue
        hk, ak = [t.strip() for t in title.split(" vs ", 1)]
        h, a = KALSHI_TEAMS.get(hk), KALSHI_TEAMS.get(ak)
        if not h or not a:
            print(f"  ! unmapped Kalshi team: {hk!r} / {ak!r}")
            continue
        legs = {}
        for m in e.get("markets", []):
            sub = (m.get("yes_sub_title") or "").strip()
            key = "d" if sub.lower() in ("tie", "draw") else (
                "h" if KALSHI_TEAMS.get(sub) == h else ("a" if KALSHI_TEAMS.get(sub) == a else None))
            if not key:
                continue
            bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
            last = _f(m.get("last_price_dollars"))
            mid = (bid + ask) / 2 if (bid is not None and ask is not None) else last
            legs[key] = {"bid": bid, "ask": ask, "last": last, "mid": mid,
                         "vol": _f(m.get("volume_fp")) or 0, "oi": _f(m.get("open_interest_fp")) or 0,
                         "ticker": m.get("ticker")}
        if len(legs) != 3:
            continue
        mids = [legs[k]["mid"] for k in ("h", "d", "a") if legs[k]["mid"] is not None]
        vig = (sum(mids) - 1) * 100 if len(mids) == 3 else None
        spread = max((legs[k]["ask"] - legs[k]["bid"]) for k in legs
                     if legs[k]["ask"] is not None and legs[k]["bid"] is not None)
        vol = sum(legs[k]["vol"] for k in legs)
        out[(h, a)] = {
            "home": h, "away": a, "event": e.get("event_ticker"),
            "url": WEB.format((e.get("event_ticker") or "").lower()),
            "sub": e.get("sub_title", ""), "legs": legs, "vig": round(vig, 1) if vig else None,
            "spread": round(spread, 3), "vol": round(vol), "thin": (vol < 500 or spread > 0.10),
        }
    return out


PROP_SERIES = {"btts": "KXEPLBTTS", "total": "KXEPLTOTAL", "spread": "KXEPLSPREAD",
               # Added 2026-08-30. We had been pricing 4 of the 30+ EPL series Kalshi lists,
               # and fighting for scraps on the one market (1X2) three separate tests say we
               # cannot beat - while these sat unpriced, all downstream of models we already have.
               "tcorners": "KXEPLTCORNERS",   # TEAM corners: our only market with validated
                                              # out-of-sample signal (+7.7% MAE vs naive)
               "score": "KXEPLSCORE",         # correct score - literally the Dixon-Coles matrix
               "teamtotal": "KXEPLTEAMTOTAL"} # a team's goals - a marginal of that same matrix
_OVER = re.compile(r"over\s+([\d.]+)", re.I)
_SPREAD = re.compile(r"^(.*?)\s+wins by more than\s+([\d.]+)", re.I)
# "Aston Villa: 5+"  /  "Arsenal: 6+ corners"
_TCORN = re.compile(r"^(.*?):\s*(\d+)\+", re.I)
# "2-1", "Aston Villa 2-1" etc - a correct-score line
_SCORE = re.compile(r"(\d+)\s*[-\u2013]\s*(\d+)")
# "Arsenal 2+ goals" / "Arsenal: 2+"
_TTOT = re.compile(r"^(.*?)[:\s]\s*(\d+)\+", re.I)


def fetch_props(max_age_min=30):
    """Prop markets keyed by (home, away):
        {"btts": leg, "total": [{line, leg}], "spread": [{side, line, leg}]}
    A `leg` is {bid, ask, mid, vol, ticker, url}. These markets are thin/placeholder well
    before kickoff (zero volume, very wide quotes) - `thin` on each leg says so."""
    out = {}
    for kind, series in PROP_SERIES.items():
        try:
            d = _get(f"/events?series_ticker={series}&status=open&with_nested_markets=true&limit=200",
                     f"kalshi_{kind}.json", max_age_min)
        except Exception as e:
            print(f"  ! Kalshi {series}: {e}")
            continue
        for e in d.get("events", []):
            title = (e.get("title") or "").split(":")[0]
            if " vs " not in title:
                continue
            hk, ak = [t.strip() for t in title.split(" vs ", 1)]
            h, a = KALSHI_TEAMS.get(hk), KALSHI_TEAMS.get(ak)
            if not h or not a:
                continue
            url = WEB.format((e.get("event_ticker") or "").lower())
            slot = out.setdefault((h, a), {"btts": None, "total": [], "spread": [],
                                           "tcorners": [], "score": [], "teamtotal": []})
            for m in e.get("markets", []):
                sub = (m.get("yes_sub_title") or "").strip()
                bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
                vol = _f(m.get("volume_fp")) or 0
                leg = {"bid": bid, "ask": ask,
                       "mid": ((bid + ask) / 2 if bid is not None and ask is not None else None),
                       "vol": vol, "ticker": m.get("ticker"), "url": url,
                       # a quote is unusable if nothing has traded or the spread is huge
                       "thin": (vol < 50 or (bid is not None and ask is not None and ask - bid > 0.08))}
                if kind == "btts":
                    slot["btts"] = leg
                elif kind == "total":
                    mo = _OVER.search(sub)
                    if mo:
                        slot["total"].append({"line": float(mo.group(1)), "leg": leg})
                elif kind == "spread":
                    ms = _SPREAD.match(sub)
                    if ms:
                        team = KALSHI_TEAMS.get(ms.group(1).strip())
                        if team in (h, a):
                            slot["spread"].append({"side": "h" if team == h else "a",
                                                   "line": float(ms.group(2)), "leg": leg})
                elif kind == "tcorners":
                    mc = _TCORN.match(sub)
                    if mc:
                        team = KALSHI_TEAMS.get(mc.group(1).strip())
                        if team in (h, a):
                            slot["tcorners"].append({"side": "h" if team == h else "a",
                                                     "line": int(mc.group(2)), "leg": leg})
                elif kind == "score":
                    msc = _SCORE.search(sub)
                    if msc:
                        slot["score"].append({"hs": int(msc.group(1)), "as": int(msc.group(2)),
                                              "leg": leg})
                elif kind == "teamtotal":
                    mt = _TTOT.match(sub)
                    if mt:
                        team = KALSHI_TEAMS.get(mt.group(1).strip())
                        if team in (h, a):
                            slot["teamtotal"].append({"side": "h" if team == h else "a",
                                                      "line": int(mt.group(2)), "leg": leg})
    for v in out.values():
        v["total"].sort(key=lambda x: x["line"])
        v["spread"].sort(key=lambda x: (x["side"], x["line"]))
        v["tcorners"].sort(key=lambda x: (x["side"], x["line"]))
        v["teamtotal"].sort(key=lambda x: (x["side"], x["line"]))
        v["score"].sort(key=lambda x: (x["hs"], x["as"]))
    return out


# ---------------------------------------------------------------- model prop pricing
def price_props(M):
    """Every prop below comes from the SAME Dixon-Coles score matrix that prices 1X2 -
    no extra model. M[i][j] = P(home i goals, away j goals)."""
    n = len(M)
    rng = range(n)
    btts = float(sum(M[i][j] for i in range(1, n) for j in range(1, n)))
    over = {k: float(sum(M[i][j] for i in rng for j in rng if i + j > k)) for k in range(6)}
    hby = {k: float(sum(M[i][j] for i in rng for j in rng if i - j > k)) for k in range(6)}
    aby = {k: float(sum(M[i][j] for i in rng for j in rng if j - i > k)) for k in range(6)}
    # Correct score is the single most natural output of a Dixon-Coles model - it IS the
    # matrix cell - and we were computing it and throwing it away.
    score = {(i, j): float(M[i][j]) for i in rng for j in rng if i < 7 and j < 7}
    # A team's own goal count is just the marginal of that same matrix.
    h_at_least = {k: float(sum(M[i][j] for i in rng for j in rng if i >= k)) for k in range(6)}
    a_at_least = {k: float(sum(M[i][j] for i in rng for j in rng if j >= k)) for k in range(6)}
    return {"btts": btts, "over": over, "home_by": hby, "away_by": aby,
            "score": score, "home_goals_at_least": h_at_least, "away_goals_at_least": a_at_least}


def settle_prop(kind, side, line, hs, as_, hc=None, ac=None):
    """Did this prop settle YES? kind in btts|total|spread|score|teamtotal|tcorners.
    `hc`/`ac` are the actual corner counts, needed only for tcorners."""
    if kind == "btts":
        return 1 if (hs >= 1 and as_ >= 1) else 0
    if kind == "total":
        return 1 if (hs + as_) > line else 0
    if kind == "score":
        want_h, want_a = line          # line carries the (home, away) pair for a correct score
        return 1 if (hs == want_h and as_ == want_a) else 0
    if kind == "teamtotal":
        return 1 if ((hs if side == "h" else as_) >= line) else 0
    if kind == "tcorners":
        got = hc if side == "h" else ac
        if got is None:
            return None                # corners unknown -> cannot settle, leave ungraded
        return 1 if got >= line else 0
    d = (hs - as_) if side == "h" else (as_ - hs)
    return 1 if d > line else 0


def compare(model_probs, k):
    """model_probs = (ph, pd, pa) as fractions. Returns per-leg comparison + best entry.
    Buying YES costs the ASK, so a real entry needs model_prob > ask (not > mid)."""
    rows = []
    for i, key in enumerate(("h", "d", "a")):
        leg = k["legs"][key]
        mp = model_probs[i]
        mid, ask = leg["mid"], leg["ask"]
        rows.append({
            "leg": key, "model": round(mp * 100, 1),
            "kalshi_mid": round(mid * 100, 1) if mid is not None else None,
            "ask": round(ask * 100, 1) if ask is not None else None,
            "diff": round((mp - mid) * 100, 1) if mid is not None else None,
            "ev": round((mp / ask - 1) * 100, 1) if ask else None,
        })
    priced = [r for r in rows if r["ev"] is not None]
    best = max(priced, key=lambda r: r["ev"]) if priced else None
    return {"rows": rows, "best": best}


def verify():
    ks = fetch_markets(max_age_min=0)
    print(f"Kalshi {SERIES}: {len(ks)} fixtures with a full 3-way market")
    for (h, a), k in list(ks.items())[:12]:
        L = k["legs"]
        print(f"  {h} v {a:<16} mid {L['h']['mid']:.2f}/{L['d']['mid']:.2f}/{L['a']['mid']:.2f}"
              f"  vig {k['vig']}%  vol {k['vol']}  {'THIN' if k['thin'] else ''}")
        print(f"     {k['url']}")


if __name__ == "__main__":
    verify()

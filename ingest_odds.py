"""
Ingest bookmaker odds from football-data.co.uk (same source as results.csv) so the model
can be benchmarked against the market.

PROVIDER CHOICE
---------------
Football-data publishes several price columns per fixture. We prefer the **average
closing** line (AvgC*) over Pinnacle closing (PSC*), which inverts the historical
convention of treating Pinnacle as the sharpest available line. Reason, measured in
.state/odds_raw (see dashboard/test_odds_provenance.py):

  * Pinnacle closing covers 380/380 played fixtures in every season from 2018/19 to
    2024/25, then 210/380 in 2025/26. The feed stops dead after 2026-01-08 and is
    absent for every fixture from 2026-01-17 onward. Both the opening and closing
    Pinnacle columns die together, so this is a broken upstream feed, not a market
    that stopped pricing games.
  * Average closing covers 380/380 for every season from 2019/20 onward.
  * On the 210 fixtures of 2025/26 where both exist, Pinnacle scores *worse* than the
    average line (paired dBrier +0.0027). That reverses its sign in all six prior
    seasons, but the difference-in-differences is t=+1.46 once clustered by matchday,
    so degraded accuracy is SUSPECTED, NOT ESTABLISHED. The coverage collapse is the
    hard evidence; the accuracy effect is not claimed.

Choosing one provider that is complete across the whole window matters more than
choosing the nominally sharpest one: a benchmark that silently switches provider
mid-season is not a benchmark. Every row therefore records which provider it came
from, and consumers compare model against market within a single provider.

PROVENANCE
----------
Each row stores provider, price type (opening/closing), the raw overround, the file we
read it from, that file's SHA-256 prefix and the time we collected it. football-data
does not timestamp individual quotes, so `collected_at` is the mtime of our cached raw
download: an upper bound on when we observed the price, not when the book posted it.

Odds are converted to implied probabilities and the overround is removed by normalising
to sum 1. The pre-normalisation overround is kept so staleness can be monitored.
Writes ~/prem_predictor/odds.csv keyed by date|home|away.
"""
import csv, hashlib, io, os, urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, ".state", "odds_raw")
OUT = os.path.join(HERE, "odds.csv")
os.makedirs(RAW, exist_ok=True)
UA = "Mozilla/5.0"

# (provider, price_type, (home, draw, away) columns) in preference order.
PROVIDERS = [
    ("avg_closing",      "closing", ("AvgCH", "AvgCD", "AvgCA")),
    ("pinnacle_closing", "closing", ("PSCH", "PSCD", "PSCA")),
    ("b365_closing",     "closing", ("B365CH", "B365CD", "B365CA")),
    ("avg_opening",      "opening", ("AvgH", "AvgD", "AvgA")),
    ("pinnacle_opening", "opening", ("PSH", "PSD", "PSA")),
    ("b365_opening",     "opening", ("B365H", "B365D", "B365A")),
]
PRIMARY = PROVIDERS[0][0]

# Football-data's Pinnacle feed for E0 stops here; see module docstring.
PINNACLE_LAST_SEEN = "2026-01-08"


def code(y):
    return f"{str(y)[-2:]}{str(y+1)[-2:]}"


def current_season(today=None):
    """Season start year: a PL season starting in August is labelled by its first year."""
    d = today or datetime.now(timezone.utc).date()
    return d.year if d.month >= 7 else d.year - 1


def parse_date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s.strip()


def fetch(y):
    """Return (bytes, source_file, collected_at_iso, sha8) or None. Cached downloads are
    reused; collection time is the cache file's mtime."""
    p = os.path.join(RAW, f"E0_{code(y)}.csv")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        url = f"https://www.football-data.co.uk/mmz4281/{code(y)}/E0.csv"
        try:
            data = urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": UA}), timeout=45).read()
        except Exception as e:
            print("  !", url, e)
            return None
        open(p, "wb").write(data)
    data = open(p, "rb").read()
    collected = datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).isoformat(timespec="seconds")
    return data, os.path.basename(p), collected, hashlib.sha256(data).hexdigest()[:8]


def quotes(row):
    """Every provider that priced this fixture, in preference order.

    Returns a list of dicts with provider, price_type, odds, normalised probs and the
    raw overround. No fallback logic here -- callers choose, so the choice is visible.
    """
    out = []
    for provider, price_type, cols in PROVIDERS:
        try:
            oh, od, oa = (float(row[c]) for c in cols)
        except (KeyError, ValueError, TypeError):
            continue
        if not (oh > 1 and od > 1 and oa > 1):
            continue
        inv = [1 / oh, 1 / od, 1 / oa]
        s = sum(inv)
        out.append({"provider": provider, "price_type": price_type,
                    "odds": (oh, od, oa), "probs": tuple(v / s for v in inv),
                    "overround": s})
    return out


def pick(row):
    """(chosen, alternate) quotes for a fixture, or (None, None) if nothing priced it.

    The alternate is the best quote from a *different* provider, kept so model-vs-market
    results can be re-run against a second source without re-downloading.
    """
    qs = quotes(row)
    if not qs:
        return None, None
    chosen = qs[0]
    alt = next((q for q in qs[1:] if q["provider"] != chosen["provider"]), None)
    return chosen, alt


FIELDS = ["date", "home_team", "away_team", "odds_h", "odds_d", "odds_a",
          "p_h", "p_d", "p_a", "src", "provider", "price_type", "overround",
          "collected_at", "source_file", "source_sha8",
          "alt_provider", "alt_p_h", "alt_p_d", "alt_p_a"]


def main(y0=2018, y1=None):
    if y1 is None:
        y1 = current_season()
    rows, coverage = [], []
    for y in range(y0, y1 + 1):
        got = fetch(y)
        if not got:
            continue
        data, src_file, collected, sha8 = got
        rdr = csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="replace")))
        seen, by_provider = 0, {}
        for r in rdr:
            r = {(k.strip() if k else k): v for k, v in r.items()}
            if not r.get("HomeTeam") or not r.get("Date"):
                continue
            seen += 1
            chosen, alt = pick(r)
            if not chosen:
                continue
            by_provider[chosen["provider"]] = by_provider.get(chosen["provider"], 0) + 1
            ph, pd_, pa = chosen["probs"]
            rows.append({
                "date": parse_date(r["Date"]), "home_team": r["HomeTeam"].strip(),
                "away_team": r["AwayTeam"].strip(),
                "odds_h": chosen["odds"][0], "odds_d": chosen["odds"][1], "odds_a": chosen["odds"][2],
                "p_h": round(ph, 4), "p_d": round(pd_, 4), "p_a": round(pa, 4),
                "src": chosen["provider"], "provider": chosen["provider"],
                "price_type": chosen["price_type"], "overround": round(chosen["overround"], 4),
                "collected_at": collected, "source_file": src_file, "source_sha8": sha8,
                "alt_provider": alt["provider"] if alt else "",
                "alt_p_h": round(alt["probs"][0], 4) if alt else "",
                "alt_p_d": round(alt["probs"][1], 4) if alt else "",
                "alt_p_a": round(alt["probs"][2], 4) if alt else "",
            })
        coverage.append((code(y), seen, by_provider))
        mix = ", ".join(f"{k}={v}" for k, v in sorted(by_provider.items()))
        warn = "  <-- MIXED PROVIDERS" if len(by_provider) > 1 else ""
        print(f"  {code(y)}: {sum(by_provider.values())}/{seen} priced   {mix}{warn}")
    if not rows:
        print("! no odds ingested; leaving", OUT, "untouched")
        return
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    used = sorted({r["provider"] for r in rows})
    print(f"-> {len(rows)} matches with odds -> {OUT}")
    print(f"   providers used: {', '.join(used)}   primary: {PRIMARY}")
    off = sum(1 for r in rows if r["provider"] != PRIMARY)
    if off:
        print(f"   ! {off} rows are NOT from {PRIMARY}; compare model vs market within one "
              f"provider (see backtest.py --provider)")


if __name__ == "__main__":
    main()

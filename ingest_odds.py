"""
Ingest bookmaker CLOSING odds from football-data.co.uk (same source as results.csv) so
the model can be benchmarked against the market.

For each match we take the sharpest closing line available, preferring:
  Pinnacle closing (PSCH/PSCD/PSCA) -> average closing (AvgCH/..) -> Bet365 closing
  (B365CH/..) -> Pinnacle (PSH/..) -> Bet365 (B365H/..).
Odds are converted to implied probabilities and the bookmaker overround is removed by
normalising to sum 1. Writes ~/prem_predictor/odds.csv keyed by date|home|away.
"""
import csv, io, os, urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, ".state", "odds_raw")
OUT = os.path.join(HERE, "odds.csv")
os.makedirs(RAW, exist_ok=True)
UA = "Mozilla/5.0"

# preference order of (home,draw,away) odds column triples
TRIPLES = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"),
           ("B365CH", "B365CD", "B365CA"), ("PSH", "PSD", "PSA"),
           ("B365H", "B365D", "B365A")]


def code(y):
    return f"{str(y)[-2:]}{str(y+1)[-2:]}"


def parse_date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s.strip()


def fetch(y):
    p = os.path.join(RAW, f"E0_{code(y)}.csv")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return open(p, "rb").read()
    url = f"https://www.football-data.co.uk/mmz4281/{code(y)}/E0.csv"
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=45).read()
    except Exception as e:
        print("  !", url, e); return None
    open(p, "wb").write(data)
    return data


def pick_odds(row):
    for h, d, a in TRIPLES:
        try:
            oh, od, oa = float(row[h]), float(row[d]), float(row[a])
            if oh > 1 and od > 1 and oa > 1:
                inv = [1/oh, 1/od, 1/oa]
                s = sum(inv)
                return oh, od, oa, inv[0]/s, inv[1]/s, inv[2]/s, h[:-1] if h.endswith(("H",)) else h
        except (KeyError, ValueError, TypeError):
            continue
    return None


def main(y0=2018, y1=2025):
    rows = []
    for y in range(y0, y1 + 1):
        data = fetch(y)
        if not data:
            continue
        rdr = csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="replace")))
        n = 0
        for r in rdr:
            r = {(k.strip() if k else k): v for k, v in r.items()}
            if not r.get("HomeTeam") or not r.get("Date"):
                continue
            got = pick_odds(r)
            if not got:
                continue
            oh, od, oa, ph, pd_, pa, src = got
            rows.append({"date": parse_date(r["Date"]), "home_team": r["HomeTeam"].strip(),
                         "away_team": r["AwayTeam"].strip(),
                         "odds_h": oh, "odds_d": od, "odds_a": oa,
                         "p_h": round(ph, 4), "p_d": round(pd_, 4), "p_a": round(pa, 4),
                         "src": src})
            n += 1
        print(f"  {code(y)}: {n} matches with odds")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"-> {len(rows)} matches with odds -> {OUT}")


if __name__ == "__main__":
    main()

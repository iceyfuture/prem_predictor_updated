"""
Corners model — a DIFFERENT method from the goals model, because corners behave differently.

I first built this with the same weighted-MLE Dixon-Coles architecture as the goals model.
It did not work: walk-forward it scored team-corner MAE 2.314 vs 2.207 for a plain rolling
average, and TOTAL corners came out worse than naive. Corner rates track current tactics and
personnel, which an 8-year window with a 3-year half-life tracks far too slowly. So this model
uses the method that actually validated.

METHOD (leak-free, validated walk-forward on 9,170 matches):
    expected home corners = mean(home team's corners-for in its last N home-or-away matches,
                                 away team's corners-against in its last N)
    ...and symmetrically for the away side.  N = 30 (swept: 8/12/20/30/50/80).

VALIDATION (out-of-sample, never using a match to predict itself):
    TEAM corners   MAE 2.207 vs naive 2.391  (+7.7%),  correlation 0.375   <- real signal
    TOTAL corners  MAE 2.838 vs naive 2.849  (+0.4%),  correlation 0.123   <- almost none

That gap is the headline: WHO wins corners is predictable; HOW MANY the game produces is
dominated by match-specific noise (prediction sd 0.75 against an actual sd of 3.57). Team
corner markets are worth pricing; total-corner markets are close to a coin flip and are
flagged `weak` so nothing downstream treats them as an edge.

Kalshi quotes corners as ">= N" (e.g. "8+ corners"), not "> N.5".
"""
import os
from collections import defaultdict, deque
from math import lgamma

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 30          # matches per team, swept
MIN_PRIOR = 8        # need this many before we trust a team's rate
MAXC = 25


def _find(*rel, default=None):
    bases = [HERE, os.path.dirname(HERE), os.path.expanduser("~")]
    for b in bases:
        for r in rel:
            q = os.path.join(b, r)
            if os.path.exists(q):
                return q
    return os.path.join(HERE, default or rel[0])


DATA = _find("premier_league_history/results.csv", "data/results.csv", default="data/results.csv")


def load(extra=None):
    """History plus any freshly-finished matches (ESPN gives corners per game)."""
    df = pd.read_csv(DATA, usecols=["date", "home_team", "away_team", "home_corners", "away_corners"])
    df = df[df.home_corners.notna() & df.away_corners.notna()].copy()
    df["date"] = pd.to_datetime(df.date)
    if extra is not None and len(extra):
        extra = extra.copy()
        extra["date"] = pd.to_datetime(extra["date"])
        have = {(d, frozenset((h, a))) for d, h, a in zip(df.date, df.home_team, df.away_team)}
        keep = [(r.date, frozenset((r.home_team, r.away_team))) not in have
                for r in extra.itertuples(index=False)]
        df = pd.concat([df, extra[pd.Series(keep, index=extra.index)]], ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def build(extra=None, window=WINDOW):
    """Current corner rates per team from its last `window` matches."""
    df = load(extra)
    cf, ca = defaultdict(lambda: deque(maxlen=window)), defaultdict(lambda: deque(maxlen=window))
    for r in df.itertuples(index=False):
        cf[r.home_team].append(r.home_corners); ca[r.home_team].append(r.away_corners)
        cf[r.away_team].append(r.away_corners); ca[r.away_team].append(r.home_corners)
    league_for = float(pd.concat([df.home_corners, df.away_corners]).tail(760).mean())
    return {"for": {t: float(np.mean(v)) for t, v in cf.items() if len(v) >= MIN_PRIOR},
            "against": {t: float(np.mean(v)) for t, v in ca.items() if len(v) >= MIN_PRIOR},
            "n": {t: len(v) for t, v in cf.items()},
            "league": league_for, "window": window,
            "asof": str(df.date.max().date()), "matches": int(len(df))}


def expected(model, home, away):
    """Expected corners for each side. Falls back to the league rate for unknown teams."""
    L = model["league"]
    hf = model["for"].get(home, L); ha_ = model["against"].get(home, L)
    af = model["for"].get(away, L); aa = model["against"].get(away, L)
    return (hf + aa) / 2.0, (af + ha_) / 2.0


def _pois(lam, k=MAXC):
    ks = np.arange(k + 1)
    return np.exp(-lam + ks * np.log(max(lam, 1e-6)) - np.array([lgamma(i + 1) for i in ks]))


def price(model, home, away):
    lam, mu = expected(model, home, away)
    ph, pa = _pois(lam), _pois(mu)
    tot = np.convolve(ph, pa)
    ge = lambda dist, n: float(dist[n:].sum())
    known = home in model["for"] and away in model["for"]
    return {
        "xc_home": round(lam, 2), "xc_away": round(mu, 2), "xc_total": round(lam + mu, 2),
        "known": known,
        # team markets carry real signal
        "home_at_least": {n: round(ge(ph, n) * 100, 1) for n in range(2, 13)},
        "away_at_least": {n: round(ge(pa, n) * 100, 1) for n in range(2, 13)},
        # total-corner markets validated at only +0.4% over naive -> never treat as an edge
        "total_at_least": {n: round(ge(tot, n) * 100, 1) for n in range(6, 17)},
        "total_weak": True,
    }


if __name__ == "__main__":
    m = build()
    print(f"Corners model: {m['matches']} matches through {m['asof']}, window {m['window']}, "
          f"league rate {m['league']:.2f}/team\n")
    cur = sorted(((t, m["for"][t], m["against"].get(t, 0)) for t in m["for"] if m["n"][t] >= 30),
                 key=lambda x: -x[1])
    print(f"  {'team':<16}{'corners for':>12}{'against':>10}")
    for t, f, a in cur[:8]:
        print(f"  {t:<16}{f:>12.2f}{a:>10.2f}")
    print("  ...")
    for t, f, a in cur[-4:]:
        print(f"  {t:<16}{f:>12.2f}{a:>10.2f}")

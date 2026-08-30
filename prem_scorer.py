"""
Anytime-goalscorer model for the Premier League — the same approach as the World Cup
scorer predictor, adapted to club football and the confirmed 2026/27 squads.

  * A player's scoring weight = their recency-weighted goals (w_goals, decay 8**(-age/4),
    from our historical data via squad_2026_27_linked.csv).
  * A player's SHARE of his team = w_goals / sum(w_goals over that club's 2026/27 squad).
    Because the pool is the actual current squad, transfers are handled correctly (a
    striker's past goals move with him to his new club); players with no Premier League
    goal history (new overseas signings, youth) get share 0 and aren't listed.
  * For a fixture with team expected goals lambda (from Dixon-Coles), a player's goals in
    the match ~ Poisson(share * lambda), so P(scores anytime) = 1 - exp(-THETA*share*lam).

THETA is a playing-time discount (1.0 = shares already encode minutes via goals scored;
lower it if a scorer backtest later shows over-prediction).
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def _find(*rel, default=None):
    bases = [HERE, os.path.dirname(HERE)]
    for b in bases:
        for r in rel:
            q = os.path.join(b, r)
            if os.path.exists(q):
                return q
    return os.path.join(HERE, default or rel[0])


LINKED = _find("outputs/squad_2026_27_linked.csv", "data/squad_2026_27_linked.csv",
               "../data/squad_2026_27_linked.csv", default="outputs/squad_2026_27_linked.csv")
THETA = 1.0
TOP_N = 5


def _surname(name):
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    n = re.sub(r"[.\-']", " ", n).lower()
    toks = [t for t in n.split() if t not in
            ("de", "van", "dos", "da", "der", "den", "el", "al", "di")]
    return toks[-1] if toks else ""


def load_shares():
    df = pd.read_csv(LINKED)
    df["w_goals"] = pd.to_numeric(df.get("w_goals"), errors="coerce").fillna(0.0)
    shares = {}
    for team, g in df.groupby("team_2026_27"):
        tot = g.w_goals.sum()
        if tot <= 0:
            continue
        s = (g[g.w_goals > 0].set_index("player").w_goals / tot).sort_values(ascending=False)
        shares[team] = s
    return shares


def match_scorers(team, lam, shares, avail=None, n=TOP_N):
    """P(scores anytime) = 1 - exp(-THETA * adjusted_share * lam).

    `avail` (optional) maps (team, surname) -> availability weight in [0,1] built from FPL
    status/chance-of-playing: 0 = out/suspended, chance/100 = doubt, 1 = fit. Unavailable
    players are removed and their goal share is redistributed to whoever is expected to
    play, so the expected team goals still sum to lam. Falls back to historical shares when
    no availability is supplied (e.g. before FPL rolls the new season on)."""
    if team not in shares:
        return []
    items = list(shares[team].items())          # (player, historical share), sums ~1
    if avail:
        adj = [(pl, sh * avail.get((team, _surname(pl)), 1.0)) for pl, sh in items]
        tot = sum(w for _, w in adj)
        items = [(pl, w / tot) for pl, w in adj] if tot > 0 else items
    items.sort(key=lambda x: -x[1])
    return [(pl, float(1.0 - np.exp(-THETA * sh * lam))) for pl, sh in items[:n] if sh > 0]


if __name__ == "__main__":
    import sys
    shares = load_shares()
    if len(sys.argv) == 2:
        team = sys.argv[1]
        print(f"{team} recent scorer shares:")
        for pl, sh in shares.get(team, pd.Series(dtype=float)).head(10).items():
            print(f"  {sh*100:5.1f}%  {pl}")
    else:
        print("teams with scorer data:", ", ".join(sorted(shares)))

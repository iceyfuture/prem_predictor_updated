"""
Goal-supremacy recent-form rating + odds mapping  (Buchdahl / Football-Data 2003).

Methodology only (its dates & 2003 coefficients are ignored — we refit on our data):

  1. Each team carries a RECENT-FORM goal-supremacy rating = sum of (goals for - goals
     against) over its last N matches (default 6).
  2. A fixture's MATCH RATING = home_form_supremacy - away_form_supremacy. Because the
     mapping below is fit on real home/away/draw frequencies, home advantage is baked in
     (a match rating of 0 still yields ~46% home win in the paper's data / whatever our
     data says).
  3. The match rating maps to result probabilities with the paper's functional forms,
     refit here by weighted least squares on OUR Premier League history:
        P(home win) : linear     h(x) = a1*x + a0
        P(draw)     : quadratic  d(x) = b2*x^2 + b1*x + b0
        P(away win) : 1 - P(home) - P(draw)
     Probabilities are clipped and renormalised. Fair odds = 1 / probability.

This is a SEPARATE, recent-form signal from Dixon-Coles' long-window attack/defense
ratings; blending the two (see build_blend.py) is what strengthens the model.
"""
import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
def _find(*rel, default=None):
    bases = [HERE, os.path.dirname(HERE), os.path.expanduser("~")]
    for b in bases:
        for r in rel:
            q = os.path.join(b, r)
            if os.path.exists(q):
                return q
    return os.path.join(HERE, default or rel[0])


DATA = _find("premier_league_history/results.csv", "data/results.csv", "../data/results.csv",
             default="data/results.csv")
STATE_DIR = os.path.dirname(_find(".state/supremacy_map.json", "state/supremacy_map.json",
                                  "../state/supremacy_map.json", default=".state/supremacy_map.json"))
MAP_CACHE = os.path.join(STATE_DIR, "supremacy_map.json")

FORM_N = 6          # last N matches define recent-form supremacy (paper uses 6)
MIN_PRIOR = 3       # need at least this many prior matches to use/fit a fixture


def load_results():
    df = pd.read_csv(DATA, usecols=["date", "home_team", "away_team",
                                    "home_score", "away_score"])
    df = df[df.home_score.notna() & df.away_score.notna()].copy()
    df["date"] = pd.to_datetime(df.date)
    df["home_score"] = df.home_score.astype(int)
    df["away_score"] = df.away_score.astype(int)
    return df.sort_values("date").reset_index(drop=True)


def rolling_form(df, n=FORM_N):
    """For each match, attach each team's pre-match goal-supremacy over its last n games
    (chronological, any venue). Returns arrays aligned to df rows plus prior-game counts."""
    from collections import deque, defaultdict
    hist = defaultdict(lambda: deque(maxlen=n))
    hf, af, hc, ac = [], [], [], []
    for r in df.itertuples(index=False):
        hf.append(sum(hist[r.home_team])); hc.append(len(hist[r.home_team]))
        af.append(sum(hist[r.away_team])); ac.append(len(hist[r.away_team]))
        gd = r.home_score - r.away_score
        hist[r.home_team].append(gd)
        hist[r.away_team].append(-gd)
    df = df.copy()
    df["home_form"], df["away_form"] = hf, af
    df["home_prior"], df["away_prior"] = hc, ac
    df["match_rating"] = df["home_form"] - df["away_form"]
    return df


def fit_mapping(train_df=None):
    """Fit the rating->P(home)/P(draw) curves on our history (paper's functional forms)."""
    df = rolling_form(load_results()) if train_df is None else train_df
    d = df[(df.home_prior >= MIN_PRIOR) & (df.away_prior >= MIN_PRIOR)].copy()
    d["res"] = np.sign(d.home_score - d.away_score)     # 1 home, 0 draw, -1 away
    # empirical rates per integer match rating, weighted by sample count
    g = d.groupby("match_rating")
    stats = pd.DataFrame({
        "n": g.size(),
        "home": g.apply(lambda x: (x.res == 1).mean(), include_groups=False),
        "draw": g.apply(lambda x: (x.res == 0).mean(), include_groups=False),
    }).reset_index()
    x, w = stats.match_rating.values.astype(float), stats.n.values.astype(float)
    a = np.polyfit(x, stats.home.values, 1, w=w)     # linear: home win %
    b = np.polyfit(x, stats.draw.values, 2, w=w)     # quadratic: draw %
    model = {"home_lin": a.tolist(), "draw_quad": b.tolist(),
             "form_n": FORM_N, "n_train": int(len(d)),
             "rating_min": float(d.match_rating.min()),
             "rating_max": float(d.match_rating.max())}
    return model


def probs_from_rating(model, rating):
    a = model["home_lin"]; b = model["draw_quad"]
    ph = a[0] * rating + a[1]
    pd_ = b[0] * rating ** 2 + b[1] * rating + b[2]
    ph = float(np.clip(ph, 0.02, 0.96))
    pd_ = float(np.clip(pd_, 0.02, 0.60))
    pa = 1.0 - ph - pd_
    p = np.clip(np.array([ph, pd_, max(pa, 0.02)]), 1e-4, None)
    p = p / p.sum()
    return float(p[0]), float(p[1]), float(p[2])


def get_mapping(refit=False):
    if not refit and os.path.exists(MAP_CACHE):
        return json.load(open(MAP_CACHE))
    m = fit_mapping()
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump(m, open(MAP_CACHE, "w"))
    return m


def current_form(df=None, n=FORM_N):
    """Latest recent-form supremacy per team (for predicting upcoming fixtures)."""
    from collections import deque, defaultdict
    if df is None:
        df = load_results()
    hist = defaultdict(lambda: deque(maxlen=n))
    for r in df.itertuples(index=False):
        gd = r.home_score - r.away_score
        hist[r.home_team].append(gd)
        hist[r.away_team].append(-gd)
    return {t: sum(dq) for t, dq in hist.items()}


if __name__ == "__main__":
    m = fit_mapping()
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump(m, open(MAP_CACHE, "w"))
    print("Fitted supremacy mapping on", m["n_train"], "matches")
    print(f"  home win %:  {m['home_lin'][0]*100:+.2f}*rating + {m['home_lin'][1]*100:.2f}")
    print(f"  draw %:      {m['draw_quad'][0]*100:+.3f}*r^2 {m['draw_quad'][1]*100:+.2f}*r "
          f"+ {m['draw_quad'][2]*100:.2f}")
    print(f"  rating range seen: {m['rating_min']:.0f}..{m['rating_max']:.0f}")
    print("\n  rating -> (home / draw / away) fair odds:")
    for rt in [-10, -6, -3, 0, 3, 6, 10]:
        ph, pdw, pa = probs_from_rating(m, rt)
        print(f"   {rt:+3d}:  {ph*100:4.1f}% {pdw*100:4.1f}% {pa*100:4.1f}%   "
              f"odds {1/ph:4.2f} / {1/pdw:4.2f} / {1/pa:4.2f}")

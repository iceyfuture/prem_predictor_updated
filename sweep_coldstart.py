"""
Sweep K for the promoted-club shrinkage fix.

The bug this tests: promoted clubs have ~0 matches in the 8-year window. Rule 1 refits
after every matchday, so ONE result gives them a full-strength rating (Hull became the
league's best defence after beating Man United 2-0). apply_cold_start only helps clubs
MISSING from the fit, so the prior is discarded the moment they play once.

Fix under test:  w = n_eff/(n_eff + K);  rating = w*fitted + (1-w)*cold_prior
  K = 0   -> pure fitted = CURRENT SHIPPED BEHAVIOUR (the null)
  K -> inf-> pure prior, never learns

n_eff is the model's own TIME-WEIGHTED match count, so established clubs have n_eff in
the hundreds (w ~ 1, untouched) and the correction targets only thin-data clubs.

Walk-forward refits before EVERY matchday, exactly as Rule 1 does live, so promoted clubs
accumulate evidence within the season - which the season-level harness in validate.py
could not test. One fit per matchday serves every K (shrinkage is applied post-fit).
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import validate as V
import prem_dixon_coles as dc

COLD = (-0.12, -0.32)                      # build_dashboard.COLD_ATTACK / COLD_DEFENSE
HP = dict(window_years=dc.WINDOW_YEARS, decay_base=dc.DECAY_BASE,
          decay_span=dc.DECAY_SPAN, ridge=dc.RIDGE)     # SHIPPED hyperparameters
K_GRID = [0.0, 2.0, 4.0, 6.0, 10.0, 15.0, 25.0, 40.0, 80.0, 1e9]
THIN = 40.0                                # n_eff below this = "thin-data club"

_KS = np.arange(11)
_LG = np.array([float(np.sum(np.log(np.arange(1, k + 1)))) for k in _KS])


def _pois(lam):
    return np.exp(-lam + _KS * np.log(lam) - _LG)


def rating(m, team, k):
    """Shrink a club's fitted rating toward the promoted prior by evidence."""
    i = m["idx"].get(team)
    if i is None:
        return COLD, 0.0
    n = float(m["wmatches"][i])
    a, d = float(m["attack"][i]), float(m["defense"][i])
    if k <= 0:
        return (a, d), n
    w = n / (n + k)
    return (w * a + (1 - w) * COLD[0], w * d + (1 - w) * COLD[1]), n


def probs(m, hr, ar):
    lam = float(np.exp(hr[0] - ar[1] + m["home_adv"]))
    mu = float(np.exp(ar[0] - hr[1]))
    M = np.outer(_pois(lam), _pois(mu))
    rho = m["rho"]
    M[0, 0] *= 1.0 - lam * mu * rho; M[0, 1] *= 1.0 + lam * rho
    M[1, 0] *= 1.0 + mu * rho;       M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0, None); M /= M.sum()
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def main(path, first_season):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    seasons = sorted(df.season.unique())
    test = [s for s in seasons if s >= first_season]
    print(f"walk-forward: {len(test)} test seasons {test[0]} -> {test[-1]}")
    print(f"hyperparameters (shipped): {HP}\n")

    recs, t0, nfit = [], time.time(), 0
    for s in test:
        te = df[df.season == s]
        for date, day in te.groupby("date"):
            m = V.fit_dc(df, date, **HP)          # strictly pre-matchday data only
            nfit += 1
            for r in day.itertuples(index=False):
                for k in K_GRID:
                    hr, hn = rating(m, r.home_team, k)
                    ar, an = rating(m, r.away_team, k)
                    ph, pd_, pa = probs(m, hr, ar)
                    recs.append((s, k, r.home_score, r.away_score, ph, pd_, pa, min(hn, an)))
        print(f"  {s} done  ({nfit} fits, {time.time()-t0:.0f}s)", flush=True)

    o = pd.DataFrame(recs, columns=["season", "k", "hs", "as_", "ph", "pd", "pa", "nmin"])
    o.to_csv("outputs/coldstart_sweep.csv", index=False)

    def score(d):
        P = d[["ph", "pd", "pa"]].values
        O = V.onehot(d.hs, d.as_)
        return V.rps(P, O), V.logloss(P, O), V.acc(P, O), len(d)

    print("\n" + "=" * 78)
    print("ALL FIXTURES")
    print(f"  {'K':>8}{'RPS':>10}{'LogLoss':>10}{'Acc':>8}{'n':>7}")
    base = None
    for k in K_GRID:
        r, l, a, n = score(o[o.k == k])
        if k == 0: base = r
        lab = "0 (SHIPPED)" if k == 0 else ("inf" if k > 1e8 else f"{k:g}")
        print(f"  {lab:>8}{r:>10.4f}{l:>10.4f}{a:>8.3f}{n:>7}"
              + ("   <- null" if k == 0 else f"   {(base-r)/base*100:+.2f}%"))

    thin = o[o.nmin < THIN]
    print("\n" + "=" * 78)
    print(f"THIN-DATA FIXTURES ONLY  (a club with time-weighted n_eff < {THIN:g}) - where the bug lives")
    print(f"  {'K':>8}{'RPS':>10}{'LogLoss':>10}{'Acc':>8}{'n':>7}")
    base = None
    for k in K_GRID:
        d = thin[thin.k == k]
        if d.empty: continue
        r, l, a, n = score(d)
        if k == 0: base = r
        lab = "0 (SHIPPED)" if k == 0 else ("inf" if k > 1e8 else f"{k:g}")
        print(f"  {lab:>8}{r:>10.4f}{l:>10.4f}{a:>8.3f}{n:>7}"
              + ("   <- null" if k == 0 else f"   {(base-r)/base*100:+.2f}%"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../premier_league_history/results.csv",
         sys.argv[2] if len(sys.argv) > 2 else "2012-13")

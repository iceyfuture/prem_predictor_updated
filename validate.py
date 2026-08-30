"""
validate.py — out-of-sample validation for the Floodlit Dixon-Coles model.

This is the piece the bundle was missing. Everything in model/ fits the data; nothing
in the bundle ever asked how well it predicts data it has never seen. This does.

PROTOCOL (strictly walk-forward, no leakage):
    for each test season S:
        fit on matches with date < first kickoff of S   (window = WINDOW_YEARS back)
        predict every match in S with that frozen model
    No information from season S ever reaches the model that predicts season S.

METRICS
    RPS       Ranked Probability Score — the standard metric for ordered 1X2 outcomes.
              Lower is better. Published bookmaker closing lines on the EPL sit around
              0.19-0.20, so that is the number to chase.
    LogLoss   Multiclass log loss. Punishes confident mistakes hardest.
    Brier     Multiclass Brier score.
    Acc       Top-pick accuracy. Reported last on purpose: it is the least informative
              of the four and the easiest to accidentally brag about.

USAGE
    python validate.py results.csv                 # backtest + calibration + stability
    python validate.py results.csv --sweep         # also sweep decay / ridge
    python validate.py results.csv --from 2015-16  # change the first test season

results.csv needs: season, date, home_team, away_team, home_score, away_score
"""
import argparse
from math import lgamma

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MAXG = 10
_KS = np.arange(MAXG + 1)
_LG = np.array([lgamma(k + 1) for k in _KS])


# --------------------------------------------------------------------------- #
# model (exact replica of prem_dixon_coles.fit, hyperparameters exposed)
# --------------------------------------------------------------------------- #
def fit_dc(df_all, cutoff, window_years=8.0, decay_base=8.0, decay_span=4.0, ridge=2.0):
    cutoff = pd.Timestamp(cutoff)
    lo = cutoff - pd.Timedelta(days=365.25 * window_years)
    df = df_all[(df_all.date < cutoff) & (df_all.date >= lo)]
    teams = sorted(set(df.home_team) | set(df.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = df.home_team.map(idx).values
    ai = df.away_team.map(idx).values
    x = df.home_score.astype(int).values
    y = df.away_score.astype(int).values
    age = (cutoff - df.date).dt.days.values / 365.25
    w = decay_base ** (-age / decay_span)
    w = w / w.mean()

    m00 = (x == 0) & (y == 0); m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0); m11 = (x == 1) & (y == 1)

    def nll(p):
        a = p[:n]; a = a - a.mean()
        d = p[n:2 * n]; d = d - d.mean()
        h, rho = p[2 * n], p[2 * n + 1]
        loglam = a[hi] - d[ai] + h
        logmu = a[ai] - d[hi]
        lam, mu = np.exp(loglam), np.exp(logmu)
        ll = x * loglam - lam + y * logmu - mu
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        ll = ll + np.log(np.clip(tau, 1e-9, None))
        return -np.sum(w * ll) + ridge * (a @ a + d @ d)

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = 0.25
    x0[2 * n + 1] = -0.05
    bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "maxfun": 300000, "ftol": 1e-9, "gtol": 1e-6})
    a = res.x[:n]; a = a - a.mean()
    d = res.x[n:2 * n]; d = d - d.mean()
    wm = np.zeros(n)
    np.add.at(wm, hi, w); np.add.at(wm, ai, w)
    return dict(teams=teams, idx=idx, attack=a, defense=d, wmatches=wm,
                home_adv=float(res.x[2 * n]), rho=float(res.x[2 * n + 1]), n=len(df))


def _pois(lam):
    return np.exp(-lam + _KS * np.log(lam) - _LG)


def probs(model, home, away, cold=(0.0, 0.0)):
    i, j = model["idx"].get(home), model["idx"].get(away)
    ah, dh = (model["attack"][i], model["defense"][i]) if i is not None else cold
    aa, da = (model["attack"][j], model["defense"][j]) if j is not None else cold
    lam = float(np.exp(ah - da + model["home_adv"]))
    mu = float(np.exp(aa - dh))
    M = np.outer(_pois(lam), _pois(mu))
    rho = model["rho"]
    M[0, 0] *= 1.0 - lam * mu * rho
    M[0, 1] *= 1.0 + lam * rho
    M[1, 0] *= 1.0 + mu * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0, None); M /= M.sum()
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def onehot(hs, as_):
    gd = np.asarray(hs) - np.asarray(as_)
    O = np.zeros((len(gd), 3))
    O[gd > 0, 0] = 1; O[gd == 0, 1] = 1; O[gd < 0, 2] = 1
    return O


def rps(P, O):
    cp, co = np.cumsum(P, 1)[:, :2], np.cumsum(O, 1)[:, :2]
    return float(np.mean(np.sum((cp - co) ** 2, 1) / 2.0))


def logloss(P, O):
    return float(-np.mean(np.sum(O * np.log(np.clip(P, 1e-15, 1)), 1)))


def brier(P, O):
    return float(np.mean(np.sum((P - O) ** 2, 1)))


def acc(P, O):
    return float(np.mean(P.argmax(1) == O.argmax(1)))


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(df, test_seasons, cold_mode="bottom3", **hp):
    rows = []
    for s in test_seasons:
        te = df[df.season == s]
        if te.empty:
            continue
        m = fit_dc(df, te.date.min(), **hp)
        if cold_mode == "league_avg":
            cold = (0.0, 0.0)
        else:  # bottom3 — mirrors build_dashboard.apply_cold_start
            active = m["wmatches"] > 20
            order = np.argsort(m["attack"] + m["defense"])
            b3 = [k for k in order if active[k]][:3]
            cold = (float(m["attack"][b3].mean()), float(m["defense"][b3].mean()))
        for r in te.itertuples(index=False):
            ph, pd_, pa = probs(m, r.home_team, r.away_team, cold)
            rows.append(dict(season=s, date=r.date, home=r.home_team, away=r.away_team,
                             hs=r.home_score, as_=r.away_score, ph=ph, pd=pd_, pa=pa,
                             cold=(r.home_team not in m["idx"]) or (r.away_team not in m["idx"])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #
def report_headline(out):
    P = out[["ph", "pd", "pa"]].values
    O = onehot(out.hs, out.as_)
    base = np.tile(O.mean(0), (len(out), 1))
    rows = [
        dict(model="Dixon-Coles (walk-forward)", n=len(out), RPS=rps(P, O),
             LogLoss=logloss(P, O), Brier=brier(P, O), Acc=acc(P, O)),
        dict(model="baseline: 1X2 base rate", n=len(out), RPS=rps(base, O),
             LogLoss=logloss(base, O), Brier=brier(base, O), Acc=acc(base, O)),
    ]
    print("\n=== HEADLINE (out-of-sample) ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  reference: bookmaker closing lines on the EPL score roughly 0.19-0.20 RPS.")


def report_calibration(out):
    """Per-outcome bias is what matters for edge detection — pooling H/D/A hides it."""
    O = onehot(out.hs, out.as_)
    print("\n=== CALIBRATION: per-outcome bias ===")
    for k, (col, i) in {"home": ("ph", 0), "draw": ("pd", 1), "away": ("pa", 2)}.items():
        pred, act = out[col].mean(), O[:, i].mean()
        print(f"  {k:5s}  predicted {pred*100:5.2f}%   actual {act*100:5.2f}%   "
              f"bias {(pred-act)*100:+5.2f} pp")
    print("\n=== CALIBRATION: reliability by probability bucket ===")
    c = pd.DataFrame({
        "p": np.concatenate([out.ph, out.pd, out.pa]),
        "y": np.concatenate([O[:, 0], O[:, 1], O[:, 2]]),
    })
    c["bin"] = pd.cut(c.p, [0, .1, .2, .3, .4, .5, .6, .7, 1.01])
    g = c.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"),
                                            actual=("y", "mean"))
    g["bias_pp"] = (g.predicted - g.actual) * 100
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))


def report_stability(out):
    print("\n=== PER-SEASON STABILITY ===")
    rows = []
    for s, d in out.groupby("season"):
        P, O = d[["ph", "pd", "pa"]].values, onehot(d.hs, d.as_)
        rows.append(dict(season=s, n=len(d), RPS=rps(P, O), Acc=acc(P, O),
                         home_bias_pp=(d.ph.mean() - O[:, 0].mean()) * 100))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"  RPS spread across seasons: {t.RPS.min():.4f} - {t.RPS.max():.4f}   "
          f"(sd {t.RPS.std():.4f})")


def report_coldstart(out):
    print("\n=== COLD-START (promoted clubs with no rating history) ===")
    for lab, d in [("cold-start fixtures", out[out.cold]), ("rated fixtures", out[~out.cold])]:
        if d.empty:
            continue
        P, O = d[["ph", "pd", "pa"]].values, onehot(d.hs, d.as_)
        print(f"  {lab:22s} n={len(d):5d}  RPS {rps(P, O):.4f}  Acc {acc(P, O):.3f}")


def report_sweep(df, test_seasons):
    print("\n=== HYPERPARAMETER SWEEP (out-of-sample RPS) ===")
    print("  decay: w = base ** (-age_years / span);  shipped = 8 ** (-age/4), half-life 1.33y")
    grid = [("no decay", 1.0000001, 4.0), ("half-life 0.67y", 8.0, 2.0),
            ("half-life 1.00y", 8.0, 3.0), ("half-life 1.33y  <- SHIPPED", 8.0, 4.0),
            ("half-life 2.00y", 8.0, 6.0), ("half-life 3.00y", 8.0, 9.0),
            ("half-life 5.00y", 8.0, 15.0)]
    for lab, b, s in grid:
        o = walk_forward(df, test_seasons, decay_base=b, decay_span=s, ridge=2.0)
        P, O = o[["ph", "pd", "pa"]].values, onehot(o.hs, o.as_)
        print(f"  {lab:30s} RPS {rps(P, O):.4f}  LL {logloss(P, O):.4f}  Acc {acc(P, O):.3f}")
    print("\n  ridge (L2 shrink on attack/defense):")
    for ridge in [0.25, 1.0, 2.0, 5.0, 12.0]:
        o = walk_forward(df, test_seasons, ridge=ridge)
        P, O = o[["ph", "pd", "pa"]].values, onehot(o.hs, o.as_)
        tag = "  <- SHIPPED" if ridge == 2.0 else ""
        print(f"  ridge {ridge:<6}                    RPS {rps(P, O):.4f}  "
              f"LL {logloss(P, O):.4f}  Acc {acc(P, O):.3f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="data/results.csv")
    ap.add_argument("--from", dest="first", default="2012-13", help="first test season")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="oos_predictions.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.results)
    df = df[df.home_score.notna() & df.away_score.notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    seasons = sorted(df.season.unique())
    test = [s for s in seasons if s >= a.first]
    print(f"data:  {len(df)} matches, {seasons[0]}..{seasons[-1]}")
    print(f"test:  {len(test)} seasons out-of-sample, {test[0]}..{test[-1]}")

    out = walk_forward(df, test)
    report_headline(out)
    report_calibration(out)
    report_stability(out)
    report_coldstart(out)
    if a.sweep:
        report_sweep(df, test)

    out.to_csv(a.out, index=False)
    print(f"\nwrote per-match out-of-sample predictions -> {a.out} ({len(out)} rows)")
    print("Keep this file. It is the evidence behind every number you put on the site.")


if __name__ == "__main__":
    main()

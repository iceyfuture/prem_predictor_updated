"""
build_backtest.py — turns the out-of-sample walk-forward (validate.py) into backtest.json
for the dashboard's Backtest page. This is the evidence layer: edge signals are only
trustworthy to the extent these numbers hold up, so the site links every edge to this page.

Writes dashboard/backtest.json with:
  headline      RPS / LogLoss / Brier / Acc vs a base-rate baseline (+ market where priced)
  calibration   reliability buckets (predicted prob vs actual frequency), pooled H/D/A
  per_season    RPS + accuracy per season (stability)
  cold_start    rated vs cold-start fixtures (why promoted-team edges are downgraded)
  goals         predicted vs actual total goals, and over/under 2.5 calibration
  market        model vs bookmaker closing line on the priced subset
"""
import json, os, sys
import numpy as np
import pandas as pd

def _history(name):
    """Locate a file from the premier_league_history dataset.

    Repo layout (prem_predictor/ and premier_league_history/ as siblings) is tried FIRST so a
    clone works anywhere; the ~ location is kept as a fallback for the original local setup.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here)),
                 os.path.expanduser("~")):
        p = os.path.join(base, "premier_league_history", name)
        if os.path.exists(p):
            return p
    return os.path.expanduser(os.path.join("~/premier_league_history", name))


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import validate as V   # noqa: E402

RESULTS = os.path.join(ROOT, "..", "premier_league_history", "results.csv")
if not os.path.exists(RESULTS):
    RESULTS = _history("results.csv")
ODDS = os.path.join(ROOT, "odds.csv")
OUT = os.path.join(HERE, "backtest.json")
FIRST = "2016-17"


def walk_with_goals(df, test):
    """Like validate.walk_forward but also records model expected goals (lam, mu)."""
    rows = []
    for s in test:
        te = df[df.season == s]
        if te.empty:
            continue
        m = V.fit_dc(df, te.date.min())
        active = m["wmatches"] > 20
        order = np.argsort(m["attack"] + m["defense"])
        b3 = [k for k in order if active[k]][:3]
        cold = (float(m["attack"][b3].mean()), float(m["defense"][b3].mean()))
        for r in te.itertuples(index=False):
            i, j = m["idx"].get(r.home_team), m["idx"].get(r.away_team)
            ah, dh = (m["attack"][i], m["defense"][i]) if i is not None else cold
            aa, da = (m["attack"][j], m["defense"][j]) if j is not None else cold
            lam = float(np.exp(ah - da + m["home_adv"]))
            mu = float(np.exp(aa - dh))
            ph, pd_, pa = V.probs(m, r.home_team, r.away_team, cold)
            rows.append(dict(season=s, date=r.date, home=r.home_team, away=r.away_team,
                             hs=int(r.home_score), as_=int(r.away_score),
                             ph=ph, pd=pd_, pa=pa, lam=lam, mu=mu,
                             cold=(i is None or j is None)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RESULTS)
    df = df[df.home_score.notna() & df.away_score.notna()].copy()
    df["date"] = pd.to_datetime(df.date)
    df = df.sort_values("date").reset_index(drop=True)
    test = [s for s in sorted(df.season.unique()) if s >= FIRST]
    print(f"walk-forward over {len(test)} seasons ({test[0]}..{test[-1]})...")
    out = walk_with_goals(df, test)
    P = out[["ph", "pd", "pa"]].values
    O = V.onehot(out.hs, out.as_)
    base = np.tile(O.mean(0), (len(out), 1))

    headline = [
        {"model": "Dixon-Coles (out-of-sample)", "n": len(out), "rps": round(V.rps(P, O), 4),
         "logloss": round(V.logloss(P, O), 4), "brier": round(V.brier(P, O), 4), "acc": round(V.acc(P, O), 4)},
        {"model": "Base-rate baseline", "n": len(out), "rps": round(V.rps(base, O), 4),
         "logloss": round(V.logloss(base, O), 4), "brier": round(V.brier(base, O), 4), "acc": round(V.acc(base, O), 4)},
    ]

    # calibration reliability (pooled H/D/A)
    pp = np.concatenate([out.ph, out.pd, out.pa])
    yy = np.concatenate([O[:, 0], O[:, 1], O[:, 2]])
    edges = np.linspace(0, 1, 11)
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (pp >= lo) & (pp < hi if hi < 1 else pp <= hi)
        if sel.sum() >= 15:
            calib.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": int(sel.sum()),
                          "pred": round(float(pp[sel].mean()), 4), "act": round(float(yy[sel].mean()), 4)})

    # per-season stability
    per_season = []
    for s, d in out.groupby("season"):
        Ps, Os = d[["ph", "pd", "pa"]].values, V.onehot(d.hs, d.as_)
        per_season.append({"season": s, "n": len(d), "rps": round(V.rps(Ps, Os), 4),
                           "acc": round(V.acc(Ps, Os), 4)})

    # cold-start vs rated
    cold_rows = []
    for lab, d in [("rated", out[~out.cold]), ("cold_start", out[out.cold])]:
        if len(d):
            Ps, Os = d[["ph", "pd", "pa"]].values, V.onehot(d.hs, d.as_)
            cold_rows.append({"group": lab, "n": len(d), "rps": round(V.rps(Ps, Os), 4),
                              "acc": round(V.acc(Ps, Os), 4)})

    # predicted vs actual goals
    out["pred_goals"] = out.lam + out.mu
    out["act_goals"] = out.hs + out.as_
    gb = []
    for lo in np.arange(1.5, 4.0, 0.5):
        sel = (out.pred_goals >= lo) & (out.pred_goals < lo + 0.5)
        if sel.sum() >= 20:
            gb.append({"bucket": f"{lo:.1f}-{lo+0.5:.1f}", "n": int(sel.sum()),
                       "pred": round(float(out.pred_goals[sel].mean()), 2),
                       "act": round(float(out.act_goals[sel].mean()), 2)})
    goals = {"mean_pred": round(float(out.pred_goals.mean()), 2),
             "mean_act": round(float(out.act_goals.mean()), 2), "buckets": gb}

    # model vs market on the priced subset
    market = None
    if os.path.exists(ODDS):
        od = pd.read_csv(ODDS)
        od["date"] = pd.to_datetime(od.date)
        key = out.copy()
        key["d"] = key.date.dt.strftime("%Y-%m-%d")
        od["d"] = od.date.dt.strftime("%Y-%m-%d")
        mg = key.merge(od[["d", "home_team", "away_team", "p_h", "p_d", "p_a"]],
                       left_on=["d", "home", "away"], right_on=["d", "home_team", "away_team"], how="inner")
        if len(mg) > 100:
            Om = V.onehot(mg.hs, mg.as_)
            Pm = mg[["ph", "pd", "pa"]].values
            Mk = mg[["p_h", "p_d", "p_a"]].values
            market = {"n": len(mg),
                      "model": {"rps": round(V.rps(Pm, Om), 4), "logloss": round(V.logloss(Pm, Om), 4), "acc": round(V.acc(Pm, Om), 4)},
                      "market": {"rps": round(V.rps(Mk, Om), 4), "logloss": round(V.logloss(Mk, Om), 4), "acc": round(V.acc(Mk, Om), 4)}}

    data = {"meta": {"generated_at": pd.Timestamp.utcnow().isoformat(timespec="minutes"),
                     "seasons": f"{test[0]}..{test[-1]}", "n": len(out),
                     "protocol": "strictly walk-forward; each season predicted by a model fit only on earlier data"},
            "headline": headline, "calibration": calib, "per_season": per_season,
            "cold_start": cold_rows, "goals": goals, "market": market}
    json.dump(data, open(OUT, "w"), separators=(",", ":"))
    print(f"wrote {OUT}: RPS {headline[0]['rps']}, {len(calib)} calib bins, "
          f"{len(per_season)} seasons, market={'yes' if market else 'no'}")


if __name__ == "__main__":
    main()

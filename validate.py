"""
validate.py — out-of-sample validation for the Floodlit Dixon-Coles model.

This is the piece the bundle was missing. Everything in model/ fits the data; nothing
in the bundle ever asked how well it predicts data it has never seen. This does.

PROTOCOL (strictly walk-forward, no leakage):
    for each test season S:
        fit on matches with date < first kickoff of S   (window = WINDOW_YEARS back)
        predict every match in S with that frozen model
    No information from season S ever reaches the model that predicts season S.

RULE 49 (audit §1). This file used to carry its OWN Dixon-Coles -- `fit_dc` and `probs` --
and it had drifted from the deployed one: decay_span 4.0 against production's 3.0, ridge
2.0 against 8.0, no AWAY_CAL, a bottom-3 cold-start prior instead of the calibrated
promoted-club prior, and no supremacy blend at all. Production's own comment says span3/
ridge8 was "chosen by out-of-sample sweep in validate.py", and validate's defaults were
never moved to the winner -- so the sweep's loser went on publishing the evidence page.

Both copies are gone. Everything here now calls `forecast.py`, the one pipeline the live
dashboard uses. A sweep can still vary hyperparameters, because `forecast.fit` passes them
through to the production fit; it cannot accidentally run a different MODEL.

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

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast as F           # noqa: E402  the ONE pipeline
import supremacy_odds as so    # noqa: E402

MAXG = 10
_KS = np.arange(MAXG + 1)
_LG = np.array([lgamma(k + 1) for k in _KS])


# --------------------------------------------------------------------------- #
# model (exact replica of prem_dixon_coles.fit, hyperparameters exposed)
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
def walk_forward(df, test_seasons, mapping=None, form_by_match=None, **hp):
    """Fit before each test season, predict every match in it, with the DEPLOYED pipeline.

    `hp` is passed to the production fit and is only for sweeps -- pass nothing and you get
    exactly the model the site runs. `mapping` and `form_by_match` add the 75/25 supremacy
    blend; omit them and this measures Dixon-Coles alone, and every row is marked
    `blended=False` so a report cannot claim otherwise.
    """
    rows = []
    for s in test_seasons:
        te = df[df.season == s]
        if te.empty:
            continue
        m = F.fit(cutoff=te.date.min(), verbose=False, **hp)
        # production's cold start: the calibrated promoted-club prior with evidence
        # shrinkage, applied to THIS season's clubs -- not a bottom-3 mean
        clubs = sorted(set(te.home_team) | set(te.away_team))
        rated = set(m["teams"])
        m, provisional = F.apply_cold_start(m, clubs)
        for r in te.itertuples(index=False):
            sup = None
            if form_by_match is not None:
                sup = form_by_match.get((r.date, r.home_team, r.away_team))
            fc = F.forecast(m, r.home_team, r.away_team, mapping=mapping, supremacy=sup)
            ph, pd_, pa = fc["p"]
            rows.append(dict(season=s, date=r.date, home=r.home_team, away=r.away_team,
                             hs=r.home_score, as_=r.away_score, ph=ph, pd=pd_, pa=pa,
                             lam=fc["xg_h"], mu=fc["xg_a"], blended=fc["blended"],
                             cold=(r.home_team not in rated) or (r.away_team not in rated),
                             provisional=(r.home_team in provisional) or (r.away_team in provisional)))
    return pd.DataFrame(rows)


def supremacy_inputs(first_test_season_start):
    """(mapping, {(date, home, away): causal supremacy rating}) for the blend.

    The rating->odds mapping is fit ONLY on matches before the first test season, and each
    fixture's rating is its causal last-6 supremacy, so nothing here sees a result it is
    used to predict.
    """
    sd = so.rolling_form(so.load_results())
    mapping = so.fit_mapping(sd[sd.date < pd.Timestamp(first_test_season_start)])
    by_match = {(r.date, r.home_team, r.away_team): r.match_rating
                for r in sd.itertuples(index=False)}
    return mapping, by_match


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
    import prem_dixon_coles as dc
    print("\n=== HYPERPARAMETER SWEEP (out-of-sample RPS) ===")
    print(f"  decay: w = base ** (-age_years / span);  SHIPPED = {dc.DECAY_BASE:g} ** "
          f"(-age/{dc.DECAY_SPAN:g}), ridge {dc.RIDGE:g}")
    print("  the sweep varies ONE setting at a time; everything else stays at the deployed value.")
    grid = [("no decay", 1.0000001, 4.0), ("half-life 0.67y", 8.0, 2.0),
            ("half-life 1.00y", 8.0, 3.0), ("half-life 1.33y", 8.0, 4.0),
            ("half-life 2.00y", 8.0, 6.0), ("half-life 3.00y", 8.0, 9.0),
            ("half-life 5.00y", 8.0, 15.0)]
    for lab, b, sp in grid:
        o = walk_forward(df, test_seasons, decay_base=b, decay_span=sp)
        P, O = o[["ph", "pd", "pa"]].values, onehot(o.hs, o.as_)
        tag = "  <- SHIPPED" if (b == dc.DECAY_BASE and sp == dc.DECAY_SPAN) else ""
        print(f"  {lab:20s} RPS {rps(P, O):.4f}  LL {logloss(P, O):.4f}  "
              f"Acc {acc(P, O):.3f}{tag}")
    print("\n  ridge (L2 shrink on attack/defense):")
    for ridge in [0.25, 1.0, 2.0, 5.0, 8.0, 12.0]:
        o = walk_forward(df, test_seasons, ridge=ridge)
        P, O = o[["ph", "pd", "pa"]].values, onehot(o.hs, o.as_)
        tag = "  <- SHIPPED" if ridge == dc.RIDGE else ""
        print(f"  ridge {ridge:<6}              RPS {rps(P, O):.4f}  "
              f"LL {logloss(P, O):.4f}  Acc {acc(P, O):.3f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="data/results.csv")
    ap.add_argument("--from", dest="first", default="2012-13", help="first test season")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--dc-only", action="store_true",
                    help="measure Dixon-Coles alone instead of the deployed blend")
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

    # measure what is DEPLOYED: Dixon-Coles blended 75/25 with causal supremacy form.
    # --dc-only drops the blend to isolate the goals model.
    mapping = form_by_match = None
    if not a.dc_only:
        try:
            mapping, form_by_match = supremacy_inputs(df[df.season == test[0]].date.min())
            print("blend: Dixon-Coles x supremacy form, weights "
                  f"{F.blend_weights()[0]:g}/{F.blend_weights()[1]:g} (as deployed)")
        except Exception as e:
            print(f"! supremacy inputs unavailable ({e}); measuring Dixon-Coles alone")
    else:
        print("blend: OFF (--dc-only) — this is the goals model, not the deployed forecast")

    out = walk_forward(df, test, mapping=mapping, form_by_match=form_by_match)
    if len(out) and not out.blended.all():
        n = int((~out.blended).sum())
        print(f"! {n} of {len(out)} rows could not be blended and are pure Dixon-Coles")
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

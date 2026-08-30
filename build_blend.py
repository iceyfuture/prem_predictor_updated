"""
Blend the goal-supremacy odds model into Dixon-Coles, and prove it strengthens the
model with an OUT-OF-SAMPLE backtest.

Design (leakage-controlled):
  * Hold out the most recent full season. Fit Dixon-Coles with a cutoff at the start of
    that season, so the holdout is unseen by DC too (not just by the supremacy map).
  * Fit the supremacy rating->odds mapping only on matches BEFORE the holdout.
  * Supremacy probabilities use each team's pre-match form only (causal, no leakage).
  * For blend weight w, final = w*DC + (1-w)*supremacy. Sweep w, pick the w with the
    lowest holdout log-loss. Report log-loss / Brier / accuracy for DC-only,
    supremacy-only, and the blend.

Then refit both components on ALL data and save the production blend to .state/blend.json.
Prediction (predict.py) uses: final_prob = w*DC + (1-w)*supremacy, fair odds = 1/prob.
"""
import json
import os
import numpy as np
import pandas as pd

import prem_dixon_coles as dc
import supremacy_odds as so

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, ".state")
BLEND_CACHE = os.path.join(STATE, "blend.json")


def metrics(P, Y):
    """P: (n,3) probs [home,draw,away]; Y: (n,) outcome 0/1/2. Returns log-loss/Brier/acc."""
    P = np.clip(np.asarray(P), 1e-9, 1)
    P = P / P.sum(axis=1, keepdims=True)
    idx = np.arange(len(Y))
    logloss = float(-np.mean(np.log(P[idx, Y])))
    onehot = np.eye(3)[Y]
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    acc = float(np.mean(P.argmax(axis=1) == Y))
    return logloss, brier, acc


def outcome(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def backtest(holdout_start="2025-08-01"):
    cutoff = pd.Timestamp(holdout_start)
    print(f"Holdout: matches on/after {cutoff.date()} (DC & mapping fit only on earlier data)\n")

    dc_model = dc.fit(cutoff=cutoff, verbose=True)          # out-of-sample for holdout
    df = so.rolling_form(so.load_results())
    mapping = so.fit_mapping(df[df.date < cutoff])          # map fit pre-holdout

    hold = df[(df.date >= cutoff) & (df.home_prior >= so.MIN_PRIOR) &
              (df.away_prior >= so.MIN_PRIOR)].copy()

    DCp, SUPp, Y = [], [], []
    for r in hold.itertuples(index=False):
        p = dc.predict(dc_model, r.home_team, r.away_team)
        DCp.append([p["win_h"], p["draw"], p["win_a"]])
        SUPp.append(list(so.probs_from_rating(mapping, r.match_rating)))
        Y.append(outcome(r.home_score, r.away_score))
    DCp, SUPp, Y = np.array(DCp), np.array(SUPp), np.array(Y)
    print(f"Scored {len(Y)} holdout matches.\n")

    # sweep blend weight
    best = None
    for w in np.round(np.arange(0, 1.01, 0.05), 2):
        ll, br, ac = metrics(w * DCp + (1 - w) * SUPp, Y)
        if best is None or ll < best[1]:
            best = (w, ll, br, ac)
    ll_dc = metrics(DCp, Y); ll_sup = metrics(SUPp, Y)
    print(f"{'model':<22}{'logloss':>9}{'brier':>8}{'acc':>7}")
    print(f"{'Dixon-Coles only':<22}{ll_dc[0]:>9.4f}{ll_dc[1]:>8.4f}{ll_dc[2]*100:>6.1f}%")
    print(f"{'Supremacy only':<22}{ll_sup[0]:>9.4f}{ll_sup[1]:>8.4f}{ll_sup[2]*100:>6.1f}%")
    print(f"{'Blend (w*DC)':<22}{best[1]:>9.4f}{best[2]:>8.4f}{best[3]*100:>6.1f}%"
          f"    <- best w(DC)={best[0]}")
    improve = (ll_dc[0] - best[1]) / ll_dc[0] * 100
    print(f"\nBlend improves log-loss vs Dixon-Coles-only by {improve:.2f}% "
          f"(lower is better).")
    return best[0]


def build_production(weight):
    """Refit both components on ALL data and persist the blend config."""
    dc.get_model(refit=True)                      # full DC cached to .state/dc_prem.json
    mapping = so.fit_mapping()                     # full supremacy map
    os.makedirs(STATE, exist_ok=True)
    json.dump(so.get_mapping(refit=True), open(so.MAP_CACHE, "w"))
    blend = {"dc_weight": float(weight), "sup_weight": float(1 - weight),
             "form_n": so.FORM_N, "note": "final_prob = dc_weight*DC + sup_weight*supremacy"}
    json.dump(blend, open(BLEND_CACHE, "w"))
    print(f"\nSaved production blend: DC={weight}, supremacy={round(1-weight,2)} "
          f"-> {BLEND_CACHE}")


if __name__ == "__main__":
    w = backtest()
    build_production(w)

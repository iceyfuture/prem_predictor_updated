"""
Premier League Dixon-Coles prediction model  (standalone project).

This is an INDEPENDENT copy of the Dixon-Coles methodology used by the World Cup
predictor. It lives in its own project (~/prem_predictor) with its own virtualenv
and shares NO code or state with ~/world_cup_predictor — the two never interfere.

What the model does, in plain terms:
  1. Learns each PL team's ATTACK (goals scored) and DEFENSE (goals prevented) from
     real results, judged against the quality of the opponent.
  2. RECENCY: a match ~4 seasons old counts about 1/8 of a match today
     (w = 8 ** (-age_years / 4)), because the Prem churns squads and swaps teams
     every season. The last ~4 seasons dominate the ratings.
  3. HOME ADVANTAGE: a fitted global boost applied to every league match.
  4. GOAL DEPENDENCE: home and away goals are NOT independent — the Dixon-Coles
     tau(x,y; rho) correction ties them together for 0-0/1-0/0-1/1-1, which is why
     the model reproduces real football's draw rate.

Model:
    lambda (home goals) = exp(attack_home - defense_away + home_adv)
    mu     (away goals) = exp(attack_away - defense_home)
    P(x,y) = tau(x,y; rho) * Poisson(x; lambda) * Poisson(y; mu)
"""
import json
import os
from math import lgamma

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(*rel, default=None):
    """Resolve a data/state file across layouts (real project uses outputs/ + .state/ +
    ~/premier_league_history; a distributed bundle may use data/ + state/ as siblings)."""
    bases = [HERE, os.path.dirname(HERE), os.path.expanduser("~")]
    for b in bases:
        for r in rel:
            p = os.path.join(b, r)
            if os.path.exists(p):
                return p
    return os.path.join(HERE, default or rel[0])


DATA = _find("premier_league_history/results.csv", "data/results.csv", "../data/results.csv",
             default="data/results.csv")
STATE_DIR = os.path.dirname(_find(".state/dc_prem.json", "state/dc_prem.json",
                                  "../state/dc_prem.json", default=".state/dc_prem.json"))
MODEL_CACHE = os.path.join(STATE_DIR, "dc_prem.json")

WINDOW_YEARS = 8         # fit on the last ~8 seasons
DECAY_BASE = 8.0         # a recent game counts DECAY_BASE x ...
DECAY_SPAN = 3.0         # ... a game DECAY_SPAN seasons older -> w = 8 ** (-age/3)
MAXG = 10                # goal grid for score probabilities
RIDGE = 8.0              # L2 shrink on attack/defense (stabilises promoted/low-data teams)
# span/ridge chosen by out-of-sample sweep in validate.py (2016-26): span3+ridge8 gave
# RPS 0.2056 / LL 0.9888 vs the old span4+ridge2 (0.2061 / 0.9904).

COLS = ["date", "home_team", "away_team", "home_score", "away_score"]
# Shots are OPTIONAL extras: kept when the source has them, ignored when it does not, so a
# results file without them still loads. Until SHOT_BLEND > 0 they change nothing.
SHOT_COLS = ["home_shots", "away_shots"]

# RULE 37: blend shot-implied goals into the Poisson target.
#
# The model saw five columns - date, two teams, two scores - and threw away the shots,
# corners, cards and half-time scores sitting in the same file. Measured on 19,240 team-match
# observations, a team's SHOT difference predicts their next match's goal difference better
# than their GOAL difference does, at every window tested:
#
#     last 1 match    shots +0.1195   goals +0.0959
#     last 3 matches  shots +0.2089   goals +0.1747
#     last 5 matches  shots +0.2374   goals +0.2104
#
# and over a season, after 4 games, shot difference predicts rest-of-season points at r=0.618
# against goal difference's 0.553 (520 team-seasons, bootstrap 95% CI on the gap
# [+0.003, +0.127]).
#
# The blend is on the POISSON TARGET, not on the ratings: ll = x*log(lam) - lam has no
# factorial, so a non-integer x is a valid Poisson kernel. The Dixon-Coles tau correction
# still uses the REAL integer scores, because it is a correction to specific low-score cells
# and those cells are defined by what actually happened.
#
#     x_eff = (1 - SHOT_BLEND) * goals + SHOT_BLEND * shots * k
#
# where k = (total goals) / (total shots) over the fit window, so the blended target has the
# same mean as goals and only its DISTRIBUTION ACROSS TEAMS changes. 0.0 reproduces the old
# model exactly. Set from the walk-forward sweep in sweep_shotblend.py.
SHOT_BLEND = 0.0


def load_matches(cutoff=None, extra=None):
    df = pd.read_csv(DATA)
    keep = COLS + [c for c in SHOT_COLS if c in df.columns]
    df = df[df.home_score.notna() & df.away_score.notna()][keep].copy()
    df["date"] = pd.to_datetime(df.date)
    # RULE 1 (refit after each matchday): fold in freshly-finished results (e.g. the current
    # season's played games, from the live feed) so ratings update as the season unfolds.
    # De-duplicated against the CSV by (date, teams) so re-runs never double-count.
    if extra is not None and len(extra):
        extra = extra[[c for c in keep if c in extra.columns]].copy()
        extra["date"] = pd.to_datetime(extra["date"])
        have = {(d, frozenset((h, a))) for d, h, a in zip(df.date, df.home_team, df.away_team)}
        keep = [(r.date, frozenset((r.home_team, r.away_team))) not in have
                for r in extra.itertuples(index=False)]
        df = pd.concat([df, extra[pd.Series(keep, index=extra.index)]], ignore_index=True)
    if cutoff is None:
        cutoff = df.date.max() + pd.Timedelta(days=1)   # predict the "next" fixtures
    cutoff = pd.Timestamp(cutoff)
    lo = cutoff - pd.Timedelta(days=365.25 * WINDOW_YEARS)
    df = df[(df.date < cutoff) & (df.date >= lo)].reset_index(drop=True)
    return df, cutoff


def time_weights(dates, cutoff):
    age = (cutoff - dates).dt.days.values / 365.25
    return DECAY_BASE ** (-age / DECAY_SPAN)


def fit(cutoff=None, maxiter=500, verbose=True, extra=None):
    from scipy.optimize import minimize
    df, cutoff = load_matches(cutoff, extra=extra)
    teams = sorted(set(df.home_team) | set(df.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = df.home_team.map(idx).values
    ai = df.away_team.map(idx).values
    x = df.home_score.astype(int).values
    y = df.away_score.astype(int).values
    w = time_weights(df.date, cutoff)
    w = w / w.mean()

    # RULE 37: the Poisson target may blend in shot-implied goals. The tau correction below
    # keeps the integer scores - it corrects specific 0-0/1-0/0-1/1-1 cells, and those are
    # defined by the real result, not by a blend.
    xt, yt = x.astype(float), y.astype(float)
    shots_used = 0
    if SHOT_BLEND > 0 and {"home_shots", "away_shots"} <= set(df.columns):
        hs = pd.to_numeric(df.home_shots, errors="coerce").values
        as_ = pd.to_numeric(df.away_shots, errors="coerce").values
        ok = np.isfinite(hs) & np.isfinite(as_) & ((hs + as_) > 0)
        if ok.sum():
            # scale shots to goals so the blended target keeps the same mean; only the
            # distribution across teams moves
            k = (x[ok].sum() + y[ok].sum()) / (hs[ok].sum() + as_[ok].sum())
            b = SHOT_BLEND
            xt[ok] = (1 - b) * x[ok] + b * hs[ok] * k
            yt[ok] = (1 - b) * y[ok] + b * as_[ok] * k
            shots_used = int(ok.sum())

    # RULE 41: LIKELIHOOD WEIGHTING AND EVIDENCE MEASUREMENT ARE DIFFERENT THINGS.
    #
    # `w` above was divided by its own mean so the optimiser sees weights averaging 1. That is
    # correct for the fit and meaningless as a count: the divisor is 1/mean(w) = 5.87 on the
    # current window, so summing the NORMALISED weights multiplied every club's apparent
    # evidence by 5.87. Hull's five real matches measured as 28.47.
    #
    # That fed apply_cold_start's shrinkage, w = n/(n+K) with K=15, so a promoted club with
    # five games received 28.47/(28.47+15) = 0.655 of its fitted rating where the rule intends
    # 4.85/(4.85+15) = 0.244. The guardrail was running roughly 2.7x weaker than designed, and
    # it scaled with the window - a longer history would have weakened it further.
    #
    # Two separate numbers from here on:
    #   weighted_matches  raw, unnormalised recency weight - scale-invariant, and ~= the match
    #                     count when the games are recent (Hull: 5 games -> 4.85)
    #   kish_ess          Kish effective sample size, (sum w)^2 / sum(w^2) - how many
    #                     equally-weighted matches this evidence is worth once spread over
    #                     time is accounted for
    # Both are reported; shrinkage consumes `weighted_matches`.
    w_raw = time_weights(df.date, cutoff)          # the same weights, BEFORE normalisation
    wm = np.zeros(n)
    np.add.at(wm, hi, w_raw); np.add.at(wm, ai, w_raw)
    w_sq = np.zeros(n)
    np.add.at(w_sq, hi, w_raw ** 2); np.add.at(w_sq, ai, w_raw ** 2)
    kish = np.divide(wm ** 2, w_sq, out=np.zeros_like(wm), where=w_sq > 0)
    raw_n = np.zeros(n)
    np.add.at(raw_n, hi, 1.0); np.add.at(raw_n, ai, 1.0)

    m00 = (x == 0) & (y == 0); m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0); m11 = (x == 1) & (y == 1)

    def nll(p):
        a = p[:n]; a = a - a.mean()
        d = p[n:2 * n]; d = d - d.mean()
        h, rho = p[2 * n], p[2 * n + 1]
        loglam = a[hi] - d[ai] + h
        logmu = a[ai] - d[hi]
        lam, mu = np.exp(loglam), np.exp(logmu)
        ll = xt * loglam - lam + yt * logmu - mu
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        ll = ll + np.log(np.clip(tau, 1e-9, None))
        return -np.sum(w * ll) + RIDGE * (a @ a + d @ d)

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = 0.25
    x0[2 * n + 1] = -0.05
    bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]

    if verbose:
        if SHOT_BLEND > 0:
            print(f"  shot blend {SHOT_BLEND:.2f} on {shots_used} matches with shot data")
        print(f"Fitting Dixon-Coles on {len(df)} PL matches "
              f"({df.date.min().date()}..{df.date.max().date()}), {n} teams...")
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxfun": maxiter * 600,
                            "ftol": 1e-9, "gtol": 1e-6})
    a = res.x[:n]; a = a - a.mean()
    d = res.x[n:2 * n]; d = d - d.mean()
    model = dict(teams=teams, attack=a.tolist(), defense=d.tolist(),
                 weighted_matches=wm.tolist(),
                 kish_ess=kish.tolist(), raw_matches=raw_n.tolist(),
                 home_adv=float(res.x[2 * n]), rho=float(res.x[2 * n + 1]),
                 cutoff=str(cutoff.date()), n_matches=int(len(df)),
                 window_years=WINDOW_YEARS, decay_base=DECAY_BASE,
                 decay_span=DECAY_SPAN, ridge=RIDGE,
                 shot_blend=SHOT_BLEND, shot_rows=shots_used,
                 date_min=str(df.date.min().date()), date_max=str(df.date.max().date()))
    if verbose:
        print(f"  home_adv={model['home_adv']:.3f}  rho={model['rho']:.3f}  "
              f"(converged={res.success})")
    return model


def get_model(cutoff=None, refit=False):
    if not refit and os.path.exists(MODEL_CACHE):
        return json.load(open(MODEL_CACHE))
    m = fit(cutoff)
    os.makedirs(STATE_DIR, exist_ok=True)
    json.dump(m, open(MODEL_CACHE, "w"))
    return m


# Away-goal calibration. The MLE fit systematically UNDER-predicts away goals: walk-forward
# over 2,774 matches (2018-19..2025-26) the away rate came in +0.103 goals light, total goals
# +0.114 light, t = +3.63 - significant. Per season it is positive in 8 of the last 10
# (mean +0.099, sd 0.098), so it is a standing bias rather than one freak year.
#
# 1.08 is set to cancel the measured bias, not fitted to an outcome metric. Effect:
#   total-goals bias  +0.114 -> +0.017      RPS 0.2052 -> 0.2051 (no cost; 1X2 depends on the
#   Over-2.5 pricing  51.3% -> 53.6%        RATIO of the two rates, so this barely moves it)
# This matters for TOTALS and BTTS, not the moneyline - and it is the likely reason the prop
# ledger was full of losing "under"/"no" bets: an under-predicted goal rate makes every under
# look like value.
#
# Residual: even at zero mean bias, Over-2.5 still prices ~1pt under actual (55.1%). That is
# over-dispersion - real football scores have fatter tails than a Poisson - and a multiplier
# cannot fix it. The proper fix is a negative-binomial count model, not a bigger multiplier.
AWAY_CAL = 1.08


def _poisson_col(lam):
    ks = np.arange(MAXG + 1)
    return np.exp(-lam + ks * np.log(lam) - np.array([lgamma(k + 1) for k in ks]))


def score_matrix(model, home, away, neutral=False):
    idx = {t: i for i, t in enumerate(model["teams"])}
    a = np.array(model["attack"]); d = np.array(model["defense"])
    ah = a[idx[home]] if home in idx else 0.0
    dh = d[idx[home]] if home in idx else 0.0
    aa = a[idx[away]] if away in idx else 0.0
    da = d[idx[away]] if away in idx else 0.0
    h = model["home_adv"] * (0.0 if neutral else 1.0)
    rho = model["rho"]
    lam = float(np.exp(ah - da + h))
    mu = float(np.exp(aa - dh)) * AWAY_CAL
    M = np.outer(_poisson_col(lam), _poisson_col(mu))
    M[0, 0] *= 1.0 - lam * mu * rho
    M[0, 1] *= 1.0 + lam * rho
    M[1, 0] *= 1.0 + mu * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0, None)
    M /= M.sum()
    return M, lam, mu


def predict(model, home, away, neutral=False):
    M, lam, mu = score_matrix(model, home, away, neutral)
    i, j = np.unravel_index(M.argmax(), M.shape)
    return dict(home=home, away=away, xg_h=lam, xg_a=mu,
                win_h=float(np.tril(M, -1).sum()), draw=float(np.trace(M)),
                win_a=float(np.triu(M, 1).sum()),
                score=(int(i), int(j)), score_p=float(M[i, j]),
                known=(home in model["teams"] and away in model["teams"]))

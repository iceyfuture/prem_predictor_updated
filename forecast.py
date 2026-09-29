"""
forecast.py — THE forecasting pipeline. One implementation, called by production and by
every backtest.

RULE 49 (audit §1). Before this existed there were two Dixon-Coles implementations and
they had drifted apart:

  * `prem_dixon_coles.fit/predict` made the predictions the site displays, with
    DECAY_SPAN=3.0, RIDGE=8.0 and AWAY_CAL=1.08, blended 0.75/0.25 with supremacy form and
    with the promoted-club shrinkage of Rule 41 applied.
  * `validate.fit_dc/probs` -- a separate copy -- produced `backtest.json`, the page the
    site calls "the evidence behind every edge". It defaulted to decay_span=4.0, ridge=2.0,
    applied no AWAY_CAL, used a bottom-3 cold-start prior instead of the calibrated one,
    and never blended in supremacy form at all.

Production's own comment says span3/ridge8 was "chosen by out-of-sample sweep in
validate.py" -- and validate's defaults were never updated to the winner, so the sweep's
loser kept publishing the evidence. Measured over 3,800 fixtures and 10 seasons, the
headline barely moves (RPS 0.2061 -> 0.2057), but mean away-goal bias is reported as
-0.0884 when the deployed model's is -0.0008. AWAY_CAL exists precisely to remove that
bias; the published calibration chart was of a model that does not have it.

Everything the audit requires to be identical between production and backtest lives here:
the fit, the decay/ridge/window settings, the away-goal calibration, the promoted-club
shrinkage, the 75/25 blend, the recent-form limit, and the probability normalisation and
rounding. Nothing downstream re-implements any of it.
"""
import json
import os

import numpy as np

import prem_dixon_coles as dc
import supremacy_odds as so

HERE = os.path.dirname(os.path.abspath(__file__))
BLEND_PATH = os.path.join(HERE, ".state", "blend.json")

# ---- promoted-club prior (RULE 4 / RULE 41) --------------------------------------------
# Empirically calibrated on newly-promoted clubs, NOT a league average and NOT a bottom-3
# mean: promoted sides are worse than the bottom three of an established league.
COLD_ATTACK = -0.12
COLD_DEFENSE = -0.32
# Evidence shrinkage toward that prior: w = n/(n+K) on the model's own Kish effective match
# count, rescaled to reach exactly 1.0 at PROVISIONAL_N so there is no discontinuity.
COLD_K = 15.0
PROVISIONAL_N = 40.0

DEFAULT_BLEND = (0.75, 0.25)


def cold_start_weight(n):
    """How much of a club's own fitted rating to trust, given n effective matches.

    Continuous by construction: w(n) = [n/(n+K)] / [N/(N+K)] below N, 1.0 at and above it.
    0 matches -> 0.00, 5 -> 0.34, 17 -> 0.72, 40+ -> 1.00. Established clubs are untouched.

    A GUARDRAIL, not a validated accuracy gain -- see MODEL_RULES Rule 41.
    """
    if n <= 0:
        return 0.0
    if n >= PROVISIONAL_N:
        return 1.0
    ceiling = PROVISIONAL_N / (PROVISIONAL_N + COLD_K)
    return min(1.0, (n / (n + COLD_K)) / ceiling)


def apply_cold_start(model, clubs):
    """Shrink thin-data clubs toward the promoted prior. Returns (model, provisional set).

    Mutates `model` in place, as production always has. Callers that need the untouched
    fit should copy first.
    """
    idx = {t: i for i, t in enumerate(model["teams"])}
    provisional = set()

    # (1) never seen in the window -> pure prior
    for c in [c for c in clubs if c not in idx]:
        model["teams"].append(c)
        model["attack"].append(COLD_ATTACK)
        model["defense"].append(COLD_DEFENSE)
        if isinstance(model.get("weighted_matches"), list):
            model["weighted_matches"].append(0.0)
        provisional.add(c)

    # (2) rated but thin -> shrink by how much evidence there actually is
    idx = {t: i for i, t in enumerate(model["teams"])}
    wm = model.get("weighted_matches") or []
    for c in clubs:
        i = idx[c]
        n = float(wm[i]) if i < len(wm) else 0.0
        w = cold_start_weight(n)
        if w >= 1.0:
            continue                     # fully established: rating used as fitted
        if n < PROVISIONAL_N:
            provisional.add(c)           # the label is about evidence, not about shrinking
        model["attack"][i] = w * model["attack"][i] + (1 - w) * COLD_ATTACK
        model["defense"][i] = w * model["defense"][i] + (1 - w) * COLD_DEFENSE
    return model, provisional


def blend_weights():
    """(dc_weight, supremacy_weight). Fitted weights live in .state/blend.json and ARE
    committed; a fresh clone that has not run build_blend.py falls back to 0.75/0.25."""
    try:
        b = json.load(open(BLEND_PATH))
        return float(b["dc_weight"]), float(b["sup_weight"])
    except Exception:
        return DEFAULT_BLEND


def fit(cutoff=None, extra=None, verbose=False, **hp):
    """Fit the production Dixon-Coles at `cutoff`, using only matches strictly before it.

    Window, decay and ridge default to prem_dixon_coles' constants -- the deployed
    settings -- so a backtest that passes nothing cannot silently run a different
    configuration from the live model. `hp` forwards overrides for sweeps only
    (window_years, decay_base, decay_span, ridge); the returned model records whichever
    values were used, so a report can always state what it measured.
    """
    return dc.fit(cutoff=cutoff, verbose=verbose, extra=extra, **hp)


def to_pct(p):
    """Production's rounding, exactly: round each to a whole percent, then put the residual
    on the home side so the three always sum to 100. Backtests round the same way or they
    are not scoring the published numbers."""
    ph, pd_, pa = [int(round(float(x) * 100)) for x in p]
    ph += 100 - (ph + pd_ + pa)
    return ph, pd_, pa


def forecast(model, home, away, form=None, mapping=None, weights=None, neutral=False,
             supremacy=None):
    """The deployed 1X2 forecast for one fixture.

    The supremacy side can be given either way: `form` (club -> current form, as the live
    dashboard holds it) or `supremacy` (the already-differenced rating, as a walk-forward
    holds it per fixture). `mapping` is the rating->odds mapping from supremacy_odds.

    With neither, the blend degrades to pure Dixon-Coles -- which is what production does
    when form is unavailable. The caller is told which happened via `blended`, so a
    backtest can never report a blended number it did not actually compute.
    """
    dp = dc.predict(model, home, away, neutral=neutral)
    p = np.array([dp["win_h"], dp["draw"], dp["win_a"]], dtype=float)
    blended = False
    rating = supremacy
    if rating is None and form is not None:
        rating = form.get(home, 0) - form.get(away, 0)
    if rating is not None and mapping is not None:
        wdc, wsup = weights or blend_weights()
        sp = np.array(so.probs_from_rating(mapping, rating), dtype=float)
        p = wdc * p + wsup * sp
        blended = True
    p = p / p.sum()
    return {"p": p, "pct": to_pct(p), "xg_h": dp["xg_h"], "xg_a": dp["xg_a"],
            "score": dp["score"], "score_p": dp["score_p"], "known": dp["known"],
            "blended": blended}


# ---- forecast provenance (RULE 51, audit §2) -------------------------------------------
def season_of(dt):
    """Season label for a datetime/date/ISO string: '2026/27' for an August-2026 kickoff.

    A Premier League season spans two calendar years, so the year alone is ambiguous and
    July is the cut.
    """
    if isinstance(dt, str):
        s = dt.strip()
        if not s:
            return ""
        y, m = int(s[:4]), int(s[5:7])
    else:
        y, m = dt.year, dt.month
    start = y if m >= 7 else y - 1
    return f"{start}/{str(start + 1)[-2:]}"


def fixture_key(season, home, away, fixture_id=None):
    """A ledger key that survives a second season and a rescheduled kickoff.

    `home|away` alone is not a key: Arsenal play Chelsea every year, so the second season's
    row collides with the first and is silently skipped as "already locked". The ledger
    currently holds exactly one season, so nothing has been lost yet -- this is a latent
    bug, fixed before it fires rather than after.

    A provider fixture id is preferred when available: it survives a club renaming and a
    fixture moving date, which the team names and the kickoff do not.
    """
    fid = str(fixture_id or "").strip()
    if fid:
        return f"{season}|#{fid}"
    return f"{season}|{home}|{away}"


def code_version():
    """Short git commit of the working tree, or '' when it cannot be determined.

    Suffixed '+dirty' when there are uncommitted changes, because a forecast produced from
    an edited tree is not reproducible from the commit alone and should not claim to be.
    """
    import subprocess
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if h.returncode != 0:
            return ""
        rev = h.stdout.strip()
        d = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                           capture_output=True, text=True, timeout=10)
        return rev + ("+dirty" if d.returncode == 0 and d.stdout.strip() else "")
    except Exception:
        return ""


def model_config(model=None):
    """A short, stable string naming the configuration a forecast was produced with.

    Recorded per forecast so a row from before a settings change is never silently
    compared against one from after it.
    """
    wdc, wsup = blend_weights()
    m = model or {}
    parts = [f"dc:w{m.get('window_years', dc.WINDOW_YEARS):g}",
             f"b{m.get('decay_base', dc.DECAY_BASE):g}",
             f"s{m.get('decay_span', dc.DECAY_SPAN):g}",
             f"r{m.get('ridge', dc.RIDGE):g}",
             f"away{dc.AWAY_CAL:g}",
             f"blend{wdc:g}/{wsup:g}",
             f"cold{COLD_K:g}/{PROVISIONAL_N:g}"]
    return ",".join(parts)


def data_version(model=None):
    """A fingerprint of the training data: its last date and how many matches were used.

    Blank when there is no model to fingerprint. Returning a bare ":" would be worse than
    nothing: it looks like a recorded value and carries none.
    """
    m = model or {}
    last, n = m.get("date_max", ""), m.get("n_matches", "")
    return f"{last}:{n}" if (last or n != "") else ""


def provenance(model=None):
    """Everything a forecast needs to be reproducible later, as flat strings."""
    return {"code_commit": code_version(),
            "model_config": model_config(model),
            "training_cutoff": str((model or {}).get("cutoff", "")),
            "data_version": data_version(model)}

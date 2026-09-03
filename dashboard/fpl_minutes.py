"""
fpl_minutes.py — expected minutes, replacing the hard top-11 cut.

WHAT THIS REPLACES
`simulate_fpl.project_gw` used to build each club's "likely XI" by sorting available players on
availability-weighted `ep_next` and taking the top 1 GK + top 10 outfield. Everyone below that
line projected exactly **0.0** — not "less", zero. A player who is 100% fit, in form, and a
coin-flip to start was valued the same as one who is injured. That is what put Gibbs-White on
the bench in a week he returned 13, and it is why the cut had to go: the error is not a small
mis-ranking, it is a discontinuity at an arbitrary boundary.

Quantified on held-out seasons: the top-11 cut assigns 0 minutes to ~21,400 player-matches a
season, of which **1,813 actually started** and **4,188 played at all** (2025-26; 1,776 / 4,214
in 2024-25). That is roughly 1.8 real starters per team-match thrown away.

METHOD
Two logistic models, fit on the only pre-round information that also exists live:

    P(start)           given the player's own recent minutes, his price, and where that price
                       sits in his club's roster
    P(appear | !start) the same features, fit on the not-started rows

Features (all strictly leak-free — every one is shifted to PRIOR rounds only):
    ewm_start   exponentially-weighted mean of "started", alpha 0.35, over prior rounds
    ewm_min     the same over minutes/90
    last_start  started the previous round
    last_min    minutes/90 in the previous round
    w_hist      n_prior/(n_prior+W_K) — how much history there is, so a new signing's zeroed
                history features are inert rather than read as "never plays"
    price_z_club   price z-scored inside his club's roster
    price_pct_pos  price percentile inside his club x position — the role proxy that carries
                   GW1, before any minutes exist
    position dummies

W_K was swept out-of-sample (0.1 / 0.25 / 0.5 / 1 / 1.5 / 2 / 3 / 5) rather than assumed. The
first draft used 3.0, by analogy with the shrinkage constants elsewhere in this project, and it
was badly wrong here: at two prior rounds it predicted 0.67 for players who had started both
openers, against an actual 0.84. Minutes are far more persistent than per-90 rates are, so they
need far less shrinking. Brier at W_K=3.0 -> 0.25:  all rounds 0.0840 -> 0.0818,
GW<=6 0.1077 -> 0.0986 (-8.5%), and the started-both cell lands at 0.83 vs 0.84 actual.
Anything in 0.1-0.5 is equivalent; W_K=0 is undefined at n_prior=0.

Then the physical constraint the raw logistic does not know: **exactly 11 players start**.
P(start) is scaled inside each club to sum to 1 (GK) and 10 (outfield), water-filling the
residual when a scale would push someone over 1.0. Availability is applied BEFORE that scaling,
so an injured starter's minutes are redistributed to his own team-mates instead of vanishing.

Minutes then follow from the two probabilities and four empirical constants (2023-24..2025-26):

    started:  P(60+) 0.932   E[min|60+] 85.4   E[min|<60] 47.0   E[min] 82.8
    sub app:  P(60+) 0.013   E[min|60+] 70.8   E[min|<60] 17.5   E[min] 18.2
    sub appearances per team-match: 4.11

The 60-minute split is not cosmetic: FPL pays 1 appearance point under 60 and 2 at 60+, and
clean sheets only count for a player who reached 60. Consumers get the full state distribution,
not just a mean, so those thresholds are integrated properly rather than evaluated at E[min].

VALIDATION (walk-forward, train on earlier seasons only, never on the test season)

The baseline is deliberately CHARITABLE to the shipped code: the top-11 cut is credited with
the empirical P(60+|start)=0.932 and E[min|start]=82.8 rather than the certainty of 90 minutes
that `project_gw` actually assumed when it awarded a flat 2.0 appearance points and a full
clean-sheet term to every XI member. The cut still loses on every proper scoring rule.

    test 2024-25 (train 2023-24)          hard top-11 cut     minutes model
      P(start) Brier                            0.1302             0.0901
      3-state Brier  (dnp/1-59/60+)             0.4522             0.3198
      3-state log loss                          2.7718             0.5825
      minutes RMSE                             29.153             23.706
      appearance-points RMSE                    0.7109             0.5596

    test 2025-26 (train 2023-24 + 2024-25)
      P(start) Brier                            0.1218             0.0818
      3-state Brier                             0.4140             0.2902
      3-state log loss                          2.5464             0.5388
      minutes RMSE                             28.184             22.644
      appearance-points RMSE                    0.6841             0.5331

    GW <= 6 only — which is where the season is right now
      3-state Brier   2024-25   0.5490 -> 0.3592     2025-26   0.5299 -> 0.3376
      minutes RMSE    2024-25   33.08  -> 25.17      2025-26   33.54  -> 25.01

The gain is LARGEST in the opening weeks, when the cut has least to rank on.

Honest reading of the one metric that goes the other way: minutes **MAE** gets worse
(12.59 -> 15.11). That is expected and is not a defect — minutes are bimodal (0 or ~85), and
MAE is minimised by predicting the median, so a confident all-or-nothing guess beats a
calibrated expectation on MAE while being far worse as a forecast. Everything downstream multiplies the
expectation, so squared error and the proper scoring rules are the metrics that bind, and the
model wins all of them.

NOT DONE, and why: adding last season's minutes (name-matched from
premier_league_history/player_season_totals.csv) improved GW<=6 Brier from 0.1077 to 0.1054
— about 2%. It needs cross-source name matching, which has produced three separate bugs in
this project already (Bruno Fernandes, both Palmers, the two Wilsons). Not worth 2%.

Refit:  python fpl_minutes.py --fit     (writes ../.state/fpl_minutes.json)
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE = os.path.join(ROOT, ".state", "fpl_minutes.json")

ALPHA = 0.35          # EWM decay on prior rounds
W_K = 0.25            # history shrinkage: w = n/(n+W_K). SWEPT, not guessed - see below
FIT_SEASONS = ("2023-24", "2024-25", "2025-26")   # 2022-23 excluded: its `starts` column is
                                                  # only half-populated (FPL added the field
                                                  # mid-season) and reads 47% of subs as 60+.
FEATS = ["ewm_start", "ewm_min", "last_start", "last_min", "w_hist",
         "price_z_club", "price_pct_pos", "is_gk", "is_def", "is_mid"]

# Empirical minutes constants, measured over FIT_SEASONS. Refit rewrites them into the state
# file; these are the fallbacks if the state file is missing.
DEFAULTS = {
    "p60_start": 0.9316, "m_start_long": 85.40, "m_start_short": 47.05, "m_start": 82.75,
    "p60_sub": 0.0128, "m_sub_long": 70.79, "m_sub_short": 17.53, "m_sub": 18.18,
    "subs_per_team": 4.11,
}
START_SLOTS = {"GK": 1.0, "OUT": 10.0}


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def load():
    """Fitted coefficients + minutes constants. Falls back to an inert model if never fitted:
    p_start then comes only from the roster constraint, which is still better than a hard cut."""
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"b_start": None, "b_sub": None, "feats": FEATS, **DEFAULTS, "fitted": False}


# --------------------------------------------------------------------------------------
# live feature construction
# --------------------------------------------------------------------------------------

def season_history(path=None):
    """{player_id: [(gw, minutes, started), ...]} for FINISHED gameweeks, oldest first.

    `fpl_player_gameweek_2026_27.csv` only carries players who actually appeared, so a DNP is
    an ABSENT row, not a zero one. Absences are filled in per club: a club is taken to have
    played gameweek g when any of its players logged minutes in g, which needs no second data
    source and cannot mistake a blank gameweek for eleven benchings.

    Provisional (in-progress) gameweeks are skipped entirely - half its fixtures unplayed would
    read as a squad-wide benching for every club still to kick off.
    """
    import csv
    path = path or os.path.join(ROOT, "outputs", "fpl_player_gameweek_2026_27.csv")
    if not os.path.exists(path):
        return {}
    played = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if str(r.get("provisional", "")).lower() == "true":
                continue
            try:
                gw, pid = int(r["gw"]), int(r["id"])
                mins, st = float(r.get("minutes") or 0), float(r.get("starts") or 0)
            except (TypeError, ValueError):
                continue
            if mins > 0:
                played.setdefault(gw, {})[pid] = (min(90.0, mins), 1.0 if st >= 1 else 0.0)
    return played


def _ewm(vals):
    """pandas' ewm(alpha, adjust=True).mean() over `vals` given oldest-first: the most recent
    observation carries weight 1, the one before it (1-alpha), and so on."""
    num = den = 0.0
    for i, v in enumerate(reversed(vals)):
        w = (1.0 - ALPHA) ** i
        num += w * v
        den += w
    return num / den if den else 0.0


def _rank_pct(values):
    """Percentile rank with ties averaged, matching pandas rank(pct=True, method='average')."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0          # ranks are 1-based
        for k in range(i, j + 1):
            out[order[k]] = avg / n
        i = j + 1
    return out


def features(players, played, gw):
    """One feature vector per player, from prior finished gameweeks only."""
    prior = sorted(g for g in played if g < gw)
    seq = {}
    for g in prior:
        clubs_on = {p["ot"] for p in players if p["id"] in played[g]}
        for p in players:
            if p["ot"] not in clubs_on:
                continue                      # club had no fixture that gameweek
            m, s = played[g].get(p["id"], (0.0, 0.0))
            seq.setdefault(p["id"], []).append((m / 90.0, s))

    by_club, by_club_pos = {}, {}
    for p in players:
        by_club.setdefault(p["ot"], []).append(p)
        by_club_pos.setdefault((p["ot"], p["pos"]), []).append(p)
    zc, pp = {}, {}
    for club, ps in by_club.items():
        pr = [q["price"] for q in ps]
        mu = sum(pr) / len(pr)
        sd = (sum((x - mu) ** 2 for x in pr) / (len(pr) - 1)) ** 0.5 if len(pr) > 1 else 0.0
        for q in ps:
            zc[q["id"]] = (q["price"] - mu) / sd if sd > 0 else 0.0
    for _, ps in by_club_pos.items():
        for q, r in zip(ps, _rank_pct([x["price"] for x in ps])):
            pp[q["id"]] = r

    out = {}
    for p in players:
        h = seq.get(p["id"], [])
        n = float(len(h))
        w = n / (n + W_K)
        mins = [x[0] for x in h]
        starts = [x[1] for x in h]
        out[p["id"]] = {
            "ewm_start": (_ewm(starts) if h else 0.0) * w,
            "ewm_min": (_ewm(mins) if h else 0.0) * w,
            "last_start": (starts[-1] if h else 0.0) * w,
            "last_min": (mins[-1] if h else 0.0) * w,
            "w_hist": w,
            "price_z_club": zc.get(p["id"], 0.0),
            "price_pct_pos": pp.get(p["id"], 0.5),
            "is_gk": 1.0 if p["pos"] == "GK" else 0.0,
            "is_def": 1.0 if p["pos"] == "DEF" else 0.0,
            "is_mid": 1.0 if p["pos"] == "MID" else 0.0,
        }
    return out


def _scale_to(probs, target, cap=None):
    """Scale a club's probabilities to sum to `target` without letting any exceed its cap.
    Water-fills: whatever a capped player cannot absorb is pushed onto the rest."""
    n = len(probs)
    if n == 0:
        return []
    cap = cap if cap is not None else [1.0] * n
    out = list(probs)
    for _ in range(12):
        free = [i for i in range(n) if out[i] < cap[i] - 1e-9]
        fixed = sum(out[i] for i in range(n) if i not in free)
        need = target - fixed
        s = sum(out[i] for i in free)
        if not free or s <= 1e-12 or need <= 0:
            break
        k = need / s
        if abs(k - 1.0) < 1e-9:
            break
        for i in free:
            out[i] = min(cap[i], out[i] * k)
    return out


def predict(players, gw, played=None, model=None):
    """{player_id: {p_start, p_sub, p_long, p_short, exp_min, states}} for one gameweek.

    `states` is [(probability, minutes, reached_60)] over the four ways a player can feature,
    so a consumer integrates FPL's 60-minute thresholds instead of evaluating them at E[min].
    """
    model = model or load()
    played = season_history() if played is None else played
    F = features(players, played, gw)
    b_s, b_b = model.get("b_start"), model.get("b_sub")
    feats = model.get("feats", FEATS)

    raw_s, raw_b = {}, {}
    for p in players:
        f = F[p["id"]]
        x = [1.0] + [f[c] for c in feats]
        av = float(p.get("avail", 1.0) or 0.0)
        # availability multiplies BEFORE the roster constraint, so a club with an injured
        # starter re-distributes his minutes across his own team-mates rather than losing them
        raw_s[p["id"]] = (_sig(sum(a * b for a, b in zip(x, b_s))) if b_s else 0.3) * av
        raw_b[p["id"]] = (_sig(sum(a * b for a, b in zip(x, b_b))) if b_b else 0.2) * av

    p_start, p_sub = {}, {}
    clubs = {}
    for p in players:
        clubs.setdefault(p["ot"], []).append(p)
    for _, ps in clubs.items():
        for group, slots in (("GK", START_SLOTS["GK"]), ("OUT", START_SLOTS["OUT"])):
            g = [q for q in ps if (q["pos"] == "GK") == (group == "GK")]
            if not g:
                continue
            for q, v in zip(g, _scale_to([raw_s[q["id"]] for q in g], slots)):
                p_start[q["id"]] = v
        # P(sub appearance) = P(appears | did not start) x P(did not start) - the same
        # conditional form the validation harness scored, so live matches validated.
        headroom = [max(0.0, 1.0 - p_start[q["id"]]) for q in ps]
        raw = [raw_b[q["id"]] * headroom[i] for i, q in enumerate(ps)]
        for q, v in zip(ps, _scale_to(raw, model.get("subs_per_team", 4.11), cap=headroom)):
            p_sub[q["id"]] = v

    P60S, P60B = model["p60_start"], model["p60_sub"]
    MSL, MSS = model["m_start_long"], model["m_start_short"]
    MBL, MBS = model["m_sub_long"], model["m_sub_short"]
    out = {}
    for p in players:
        s, b = p_start[p["id"]], p_sub[p["id"]]
        states = [(s * P60S, MSL, True), (s * (1 - P60S), MSS, False),
                  (b * P60B, MBL, True), (b * (1 - P60B), MBS, False)]
        out[p["id"]] = {
            "p_start": s, "p_sub": b,
            "p_long": s * P60S + b * P60B,
            "p_short": s * (1 - P60S) + b * (1 - P60B),
            "exp_min": sum(pr * m for pr, m, _ in states),
            "states": states,
        }
    return out


# --------------------------------------------------------------------------------------
# fitting — offline, from premier_league_history. Not imported by the dashboard build.
# --------------------------------------------------------------------------------------

def _hist_path():
    for b in (ROOT, os.path.expanduser("~")):
        q = os.path.join(b, "premier_league_history", "player_gameweek.csv")
        if os.path.exists(q):
            return q
    raise SystemExit("player_gameweek.csv not found — need premier_league_history/")


def _frame():
    import numpy as np
    import pandas as pd
    d = pd.read_csv(_hist_path(), low_memory=False,
                    usecols=["season", "name", "element", "GW", "minutes", "starts",
                             "position", "value", "fixture", "team"])
    d = d[d.season.isin(FIT_SEASONS) & d.position.isin(["GK", "DEF", "MID", "FWD"])].copy()
    d["started"] = (d.starts == 1).astype(float)
    d["appeared"] = (d.minutes > 0).astype(float)
    d["price"] = d.value / 10.0
    d = d.sort_values(["season", "element", "GW", "fixture"]).reset_index(drop=True)

    parts = []
    for _, g in d.groupby(["season", "element"], sort=False):
        g = g.copy()
        st, mn = g.started.shift(1), (g.minutes.clip(upper=90) / 90.0).shift(1)
        g["ewm_start"] = st.ewm(alpha=ALPHA, ignore_na=True).mean()
        g["ewm_min"] = mn.ewm(alpha=ALPHA, ignore_na=True).mean()
        g["n_prior"] = st.notna().cumsum()
        g["last_start"], g["last_min"] = st, mn
        parts.append(g)
    d = pd.concat(parts).sort_index()

    grp = d.groupby(["season", "GW", "team"]).price
    d["price_z_club"] = ((d.price - grp.transform("mean"))
                         / grp.transform("std").replace(0, np.nan)).fillna(0.0)
    d["price_pct_pos"] = (d.groupby(["season", "GW", "team", "position"]).price
                          .rank(pct=True, method="average").fillna(0.5))
    return d


def _design(x):
    import numpy as np
    x = x.copy()
    w = x.n_prior / (x.n_prior + W_K)
    x["w_hist"] = w
    for c in ("ewm_start", "ewm_min", "last_start", "last_min"):
        x[c] = x[c].fillna(0.0) * w
    x["is_gk"] = (x.position == "GK").astype(float)
    x["is_def"] = (x.position == "DEF").astype(float)
    x["is_mid"] = (x.position == "MID").astype(float)
    return np.column_stack([np.ones(len(x))] + [x[c].values.astype(float) for c in FEATS])


def _logit(X, y, ridge=1e-3):
    import numpy as np
    from scipy.optimize import minimize as _min

    def nll(b):
        p = np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() + ridge * (b[1:] ** 2).sum()

    def grad(b):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (p - y) / len(y)
        g[1:] += 2 * ridge * b[1:]
        return g
    return _min(nll, np.zeros(X.shape[1]), jac=grad, method="L-BFGS-B").x


def _norm_club(te, p):
    """Same 11-starters constraint as `predict`, vectorised for the validation harness."""
    import numpy as np
    o = p.copy()
    for _, idx in te.groupby(["season", "fixture", "team"]).indices.items():
        for msk, t in ((te.position.values[idx] == "GK", 1.0),
                       (te.position.values[idx] != "GK", 10.0)):
            ii = idx[msk]
            if len(ii) == 0:
                continue
            s = o[ii].sum()
            if s > 0:
                o[ii] = np.clip(o[ii] * t / s, 0, 1)
    return o


def _cut11(te):
    """The shipped behaviour, as a baseline: hard top-11 per club, ranked on recent minutes
    (the live cut ranks on ep_next, which history does not carry — same discontinuity)."""
    import numpy as np
    c = np.zeros(len(te))
    for _, idx in te.groupby(["season", "fixture", "team"]).indices.items():
        for msk, n in ((te.position.values[idx] == "GK", 1),
                       (te.position.values[idx] != "GK", 10)):
            ii = idx[msk]
            if len(ii) == 0:
                continue
            c[ii[np.argsort(-np.nan_to_num(te.ewm_min.values[ii], nan=-1))][:n]] = 1.0
    return c


def fit(verbose=True):
    import numpy as np
    d = _frame().reset_index(drop=True)
    st = d[d.started == 1]
    sb = d[(d.started == 0) & (d.appeared == 1)]
    n_team_matches = d.groupby(["season", "fixture", "team"]).ngroups
    const = {
        "p60_start": float((st.minutes >= 60).mean()),
        "m_start_long": float(st[st.minutes >= 60].minutes.mean()),
        "m_start_short": float(st[st.minutes < 60].minutes.mean()),
        "m_start": float(st.minutes.mean()),
        "p60_sub": float((sb.minutes >= 60).mean()),
        "m_sub_long": float(sb[sb.minutes >= 60].minutes.mean()),
        "m_sub_short": float(sb[sb.minutes < 60].minutes.mean()),
        "m_sub": float(sb.minutes.mean()),
        "subs_per_team": len(sb) / n_team_matches,
    }

    def evaluate(tr, te):
        Xtr, Xte = _design(tr), _design(te)
        b_s = _logit(Xtr, tr.started.values)
        m = tr.started.values == 0
        b_b = _logit(Xtr[m], tr.appeared.values[m])
        p_s = _norm_club(te, 1 / (1 + np.exp(-np.clip(Xte @ b_s, -30, 30))))
        p_b = (1 / (1 + np.exp(-np.clip(Xte @ b_b, -30, 30)))) * (1 - p_s)
        cut = _cut11(te)
        mins, y = te.minutes.values, te.started.values
        YL = (mins >= 60).astype(float)
        YS = ((mins > 0) & (mins < 60)).astype(float)
        YD = (mins == 0).astype(float)

        def pack(ps, pb):
            return (ps * const["p60_start"] + pb * const["p60_sub"],
                    ps * (1 - const["p60_start"]) + pb * (1 - const["p60_sub"]),
                    ps * const["m_start"] + pb * const["m_sub"])
        res = {}
        for nm, (ps, pb) in (("cut", (cut, np.zeros(len(te)))), ("model", (p_s, p_b))):
            L, S, em = pack(ps, pb)
            if nm == "cut":
                em = cut * const["m_start"]
            D = 1 - L - S
            P = np.clip(np.column_stack([D, S, L]), 1e-6, 1)
            P /= P.sum(1, keepdims=True)
            Y = np.column_stack([YD, YS, YL])
            k = te.GW.values <= 6
            res[nm] = {
                "brier_start": float(np.mean((ps - y) ** 2)),
                "brier3": float(np.mean((L - YL) ** 2 + (S - YS) ** 2 + (D - YD) ** 2)),
                "ll3": float(-np.mean(np.log((P * Y).sum(1)))),
                "min_rmse": float(np.sqrt(np.mean((em - mins) ** 2))),
                "min_mae": float(np.mean(np.abs(em - mins))),
                "ap_rmse": float(np.sqrt(np.mean((S + 2 * L - (YS + 2 * YL)) ** 2))),
                "brier3_gw6": float(np.mean((L[k] - YL[k]) ** 2 + (S[k] - YS[k]) ** 2
                                            + (D[k] - YD[k]) ** 2)),
                "min_rmse_gw6": float(np.sqrt(np.mean((em[k] - mins[k]) ** 2))),
            }
        z = cut == 0
        res["zeroed_by_cut"] = {"n": int(z.sum()), "actually_started": int(y[z].sum()),
                                "actually_played": int((mins[z] > 0).sum()),
                                "their_mean_minutes": float(mins[z][mins[z] > 0].mean())}
        return res

    val = {}
    for ts in FIT_SEASONS[1:]:
        val[ts] = evaluate(d[d.season < ts], d[d.season == ts].reset_index(drop=True))
        if verbose:
            v = val[ts]
            print(f"\n  walk-forward: train {[s for s in FIT_SEASONS if s < ts]} -> test {ts}")
            print(f"    {'metric':<22}{'top-11 cut':>12}{'minutes model':>15}")
            for k, lab in (("brier_start", "P(start) Brier"), ("brier3", "3-state Brier"),
                           ("ll3", "3-state log loss"), ("min_rmse", "minutes RMSE"),
                           ("min_mae", "minutes MAE"), ("ap_rmse", "appearance-pts RMSE"),
                           ("brier3_gw6", "3-state Brier GW<=6"),
                           ("min_rmse_gw6", "minutes RMSE GW<=6")):
                print(f"    {lab:<22}{v['cut'][k]:>12.4f}{v['model'][k]:>15.4f}")
            z = v["zeroed_by_cut"]
            print(f"    cut zeroed {z['n']} player-matches: {z['actually_started']} started, "
                  f"{z['actually_played']} played (mean {z['their_mean_minutes']:.1f} min)")

    X = _design(d)
    b_s = _logit(X, d.started.values)
    m = d.started.values == 0
    b_b = _logit(X[m], d.appeared.values[m])
    out = {"b_start": [float(x) for x in b_s], "b_sub": [float(x) for x in b_b],
           "feats": FEATS, "alpha": ALPHA, "w_k": W_K, "fitted": True,
           "fit_seasons": list(FIT_SEASONS), "n_rows": int(len(d)),
           "validation": val, **const}
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"\n  fitted on {len(d)} player-rounds over {list(FIT_SEASONS)} -> {STATE}")
        print("    " + "  ".join(f"{n}={c:+.3f}" for n, c in zip(["int"] + FEATS, b_s)))
    return out


if __name__ == "__main__":
    import sys
    if "--fit" in sys.argv:
        fit()
    else:
        m = load()
        print(f"fitted={m.get('fitted')}  seasons={m.get('fit_seasons')}")
        print(json.dumps({k: v for k, v in m.items()
                          if k not in ("b_start", "b_sub", "validation")}, indent=1))

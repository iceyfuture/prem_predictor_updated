"""
Walk-forward backtest of the strengthened model, with RPS + calibration, benchmarked
against the bookmaker closing line.

Honest out-of-sample protocol:
  * Matches are grouped into rounds (date gaps > 4 days start a new round).
  * Before each round, Dixon-Coles is REFIT with a cutoff at the round's first date, so it
    has never seen that round or anything later. (~1.2s/fit.)
  * The supremacy rating->odds mapping is fit once on data BEFORE the backtest window;
    each team's form is its causal last-6 supremacy. Blend weight is fixed (0.75/0.25).
  * Every round's matches are then predicted and scored. Nothing in a prediction uses that
    match's result or any future result.

Scores: log-loss, Brier, RPS (ranked probability score, respects H<D<A ordering), accuracy.
Baselines: bookmaker closing odds, climatology (fixed base rates), home-always.
Also prints a home-win calibration table + ECE. Saves outputs/backtest_predictions.csv.

The market comparison is restricted to ONE odds provider (default: the primary provider
in odds.csv) so the benchmark cannot silently change source mid-window, and is reported
as a paired difference with a season-clustered CI -- predictions within a season share a
fitted model, so treating fixtures as independent understates the interval.

    ~/prem_predictor/.venv/bin/python backtest.py [START_DATE] [--provider NAME]
"""
import collections, csv, math, os, sys
import numpy as np
import pandas as pd
import prem_dixon_coles as dc
import supremacy_odds as so

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
ODDS = os.path.join(HERE, "odds.csv")


def rps(P, Y):
    P = np.asarray(P); cp = np.cumsum(P, axis=1)[:, :2]
    cy = np.cumsum(np.eye(3)[Y], axis=1)[:, :2]
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / 2.0))


def scores(P, Y):
    P = np.clip(np.asarray(P), 1e-9, 1); P = P / P.sum(1, keepdims=True)
    i = np.arange(len(Y))
    ll = float(-np.mean(np.log(P[i, Y])))
    br = float(np.mean(np.sum((P - np.eye(3)[Y]) ** 2, axis=1)))
    ac = float(np.mean(P.argmax(1) == Y))
    return ll, br, rps(P, Y), ac


def _season(d):
    """Season start year: August-May seasons are labelled by the year they begin."""
    return d.year if d.month >= 7 else d.year - 1


def outcome(h, a):
    return 0 if h > a else (1 if h == a else 2)


def paired_clustered(a, b, clusters):
    """Mean of (a - b) with a CI clustered on `clusters` (here: season).

    Fixtures in a season are predicted by models sharing the same training history, so
    they are not independent draws. Clustering on season widens the interval honestly.
    """
    d = np.asarray(a) - np.asarray(b)
    g = collections.defaultdict(list)
    for v, c in zip(d, clusters):
        g[c].append(v)
    means = np.array([np.mean(v) for v in g.values()])
    m = float(d.mean())
    if len(means) < 2:
        return m, float("nan"), float("nan"), len(means)
    se = float(means.std(ddof=1) / math.sqrt(len(means)))
    return m, se, (m / se if se > 0 else float("nan")), len(means)


def per_match(P, Y):
    """Per-fixture Brier and RPS, so differences can be paired before averaging."""
    P = np.clip(np.asarray(P), 1e-9, 1); P = P / P.sum(1, keepdims=True)
    oh = np.eye(3)[Y]
    br = np.sum((P - oh) ** 2, axis=1)
    cp = np.cumsum(P, axis=1)[:, :2]; cy = np.cumsum(oh, axis=1)[:, :2]
    rp = np.sum((cp - cy) ** 2, axis=1) / 2.0
    ll = -np.log(P[np.arange(len(Y)), Y])
    return br, rp, ll


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    start = args[0] if args else "2021-08-01"
    want_provider = None
    for i, a in enumerate(sys.argv):
        if a == "--provider" and i + 1 < len(sys.argv):
            want_provider = sys.argv[i + 1]
    df = so.rolling_form(so.load_results())              # causal form on full history
    mapping = so.fit_mapping(df[df.date < pd.Timestamp(start)])
    base = df[df.date < pd.Timestamp(start)]
    clim = np.array([(base.home_score > base.away_score).mean(),
                     (base.home_score == base.away_score).mean(),
                     (base.home_score < base.away_score).mean()])

    odds = pd.read_csv(ODDS)
    odds["date"] = pd.to_datetime(odds.date)
    if "provider" not in odds.columns:
        raise SystemExit("odds.csv has no provider column -- re-run ingest_odds.py "
                         "so the market benchmark carries its provenance.")
    provider = want_provider or odds.provider.mode().iat[0]
    dropped = int((odds.provider != provider).sum())
    odds = odds[odds.provider == provider]
    print(f"Market benchmark: provider={provider} ({len(odds)} priced fixtures); "
          f"dropped {dropped} rows from other providers.")
    okey = {(r.date, r.home_team, r.away_team): (r.p_h, r.p_d, r.p_a)
            for r in odds.itertuples(index=False)}

    bt = df[df.date >= pd.Timestamp(start)].sort_values("date").reset_index(drop=True)
    # group into rounds by date gap
    rounds, cur, last = [], [], None
    for r in bt.itertuples(index=False):
        if last is not None and (r.date - last).days > 4:
            rounds.append(cur); cur = []
        cur.append(r); last = r.date
    if cur:
        rounds.append(cur)

    recs = []
    print(f"Walk-forward from {start}: {len(rounds)} rounds, {len(bt)} matches. Refitting...")
    for k, rnd in enumerate(rounds):
        model = dc.fit(cutoff=rnd[0].date, verbose=False)
        known = set(model["teams"])
        for m in rnd:
            if m.home_team not in known or m.away_team not in known:
                continue
            dp = dc.predict(model, m.home_team, m.away_team)
            dcp = [dp["win_h"], dp["draw"], dp["win_a"]]
            sp = list(so.probs_from_rating(mapping, m.match_rating))
            bl = list(0.75 * np.array(dcp) + 0.25 * np.array(sp))
            mk = okey.get((m.date, m.home_team, m.away_team))
            recs.append({"date": m.date.date(), "home": m.home_team, "away": m.away_team,
                         "y": outcome(m.home_score, m.away_score),
                         "dc": dcp, "sup": sp, "blend": bl, "mkt": list(mk) if mk else None})
        if (k + 1) % 40 == 0:
            print(f"  ...{k+1}/{len(rounds)} rounds")

    Y = np.array([r["y"] for r in recs])
    DC = np.array([r["dc"] for r in recs]); SUP = np.array([r["sup"] for r in recs])
    BL = np.array([r["blend"] for r in recs])
    print(f"\nScored {len(Y)} matches (both teams rated).\n")
    hdr = f"{'model':<26}{'logloss':>9}{'brier':>8}{'rps':>8}{'acc':>8}"
    print(hdr); print("-" * len(hdr))
    for name, P in [("Dixon-Coles only", DC), ("Supremacy only", SUP), ("BLEND 0.75/0.25", BL),
                    ("climatology (base rate)", np.tile(clim, (len(Y), 1)))]:
        ll, br, rp, ac = scores(P, Y)
        print(f"{name:<26}{ll:>9.4f}{br:>8.4f}{rp:>8.4f}{ac*100:>7.1f}%")

    # ---- head-to-head vs market on the shared subset ----
    mask = [r["mkt"] is not None for r in recs]
    Ym = Y[mask]; BLm = BL[np.array(mask)]
    MK = np.array([r["mkt"] for r in recs if r["mkt"] is not None])
    seasons = [_season(r["date"]) for r in recs if r["mkt"] is not None]
    print(f"\n--- vs bookmaker closing line ({provider}), {len(Ym)} of {len(Y)} "
          f"matches priced ---")
    print(hdr); print("-" * len(hdr))
    for name, P in [("BLEND 0.75/0.25", BLm), (f"Market ({provider})", MK)]:
        ll, br, rp, ac = scores(P, Ym)
        print(f"{name:<26}{ll:>9.4f}{br:>8.4f}{rp:>8.4f}{ac*100:>7.1f}%")
    mb, mr, ml = per_match(BLm, Ym)
    kb, kr, kl = per_match(MK, Ym)
    print(f"\n  paired model - market, CI clustered by season "
          f"(n={len(Ym)} fixtures, {len(set(seasons))} seasons; positive = model worse):")
    for lab, a, b in (("Brier", mb, kb), ("RPS", mr, kr), ("logloss", ml, kl)):
        m, se, t, nc = paired_clustered(a, b, seasons)
        if se != se:
            print(f"    {lab:<8}{m:+.5f}   (only {nc} season, no interval)")
            continue
        lo, hi = m - 1.96 * se, m + 1.96 * se
        verdict = "model better" if hi < 0 else ("market better" if lo > 0 else "indistinguishable")
        print(f"    {lab:<8}{m:+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]   t={t:+.2f}   {verdict}")
    home_always = float(np.mean(Y == 0))
    print(f"{'home-always (acc only)':<26}{'':>9}{'':>8}{'':>8}{home_always*100:>7.1f}%")

    # ---- calibration on the blend's home-win probability ----
    print("\nCalibration — blend P(home win) vs actual home-win rate:")
    ph = BL[:, 0]; hw = (Y == 0).astype(float)
    edges = np.linspace(0, 1, 11); ece = 0.0
    print(f"  {'bin':>10}{'n':>6}{'pred':>8}{'actual':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ph >= lo) & (ph < hi if hi < 1 else ph <= hi)
        if sel.sum() == 0:
            continue
        pr, ac = ph[sel].mean(), hw[sel].mean()
        ece += abs(pr - ac) * sel.sum()
        print(f"  {int(lo*100):>3}-{int(hi*100):>3}% {sel.sum():>6}{pr*100:>7.1f}%{ac*100:>7.1f}%")
    print(f"  ECE(home) = {ece/len(ph):.4f}   (lower = better calibrated)")

    with open(os.path.join(OUT, "backtest_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "home", "away", "outcome",
                    "blend_pH", "blend_pD", "blend_pA", "mkt_pH", "mkt_pD", "mkt_pA"])
        for r in recs:
            mk = r["mkt"] or ["", "", ""]
            w.writerow([r["date"], r["home"], r["away"], r["y"],
                        *[round(x, 4) for x in r["blend"]], *mk])
    print(f"\nSaved per-match predictions -> outputs/backtest_predictions.csv")


if __name__ == "__main__":
    main()

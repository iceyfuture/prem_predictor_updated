"""
Walk-forward sweep of SHOT_BLEND. RULE 37's validation gate.

Refit before every matchday on matches strictly before it, price that matchday's fixtures,
score them. Nothing is tuned on the games it is scored against. Reports multiclass Brier, RPS
and log-loss per blend, plus a paired t-test against blend=0 clustered by matchday, because
matches on the same day share a fitted model.
"""
import sys, numpy as np, pandas as pd
import prem_dixon_coles as dc

SEASONS_BACK = int(sys.argv[1]) if len(sys.argv) > 1 else 3
BLENDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def outcome(h, a):
    return 0 if h > a else (1 if h == a else 2)


def rps(p, o):
    c = np.cumsum(p); y = np.cumsum([1 if i == o else 0 for i in range(3)])
    return float(np.sum((c - y) ** 2) / 2.0)


def main():
    df = pd.read_csv(dc.DATA)
    df = df[df.home_score.notna() & df.away_score.notna()].copy()
    df["date"] = pd.to_datetime(df.date)
    df = df.sort_values("date")
    last = df.date.max()
    start = last - pd.Timedelta(days=365.25 * SEASONS_BACK)
    days = sorted(d for d in df[df.date >= start].date.unique())
    print(f"walk-forward over {len(days)} matchdays "
          f"({pd.Timestamp(days[0]).date()}..{pd.Timestamp(days[-1]).date()})")

    res = {b: {"brier": [], "rps": [], "ll": [], "day": []} for b in BLENDS}
    for n, d in enumerate(days, 1):
        fx = df[df.date == d]
        if fx.empty:
            continue
        for b in BLENDS:
            dc.SHOT_BLEND = b
            try:
                m = dc.fit(cutoff=d, verbose=False)
            except Exception as e:
                print(f"  ! fit failed blend={b} on {pd.Timestamp(d).date()}: {e}")
                continue
            for r in fx.itertuples(index=False):
                try:
                    p = dc.predict(m, r.home_team, r.away_team)
                except Exception:
                    continue
                pv = np.array([p["win_h"], p["draw"], p["win_a"]], dtype=float)
                pv = np.clip(pv / pv.sum(), 1e-6, 1)
                o = outcome(r.home_score, r.away_score)
                res[b]["brier"].append(float(np.sum((pv - np.eye(3)[o]) ** 2)))
                res[b]["rps"].append(rps(pv, o))
                res[b]["ll"].append(float(-np.log(pv[o])))
                res[b]["day"].append(pd.Timestamp(d))
        if n % 20 == 0:
            print(f"  {n}/{len(days)} matchdays, {len(res[0.0]['brier'])} predictions")

    print(f"\n{'blend':>7}{'n':>7}{'Brier':>10}{'RPS':>10}{'logloss':>10}{'vs 0.0':>10}{'t (day)':>9}")
    base = np.array(res[0.0]["brier"])
    days0 = np.array(res[0.0]["day"])
    for b in BLENDS:
        v = np.array(res[b]["brier"])
        if len(v) != len(base):
            print(f"{b:>7.2f}{len(v):>7}   (length mismatch, skipped)")
            continue
        d_ = base - v                                    # +ve => blend better
        # cluster by matchday: one mean per day, t over days
        s = pd.Series(d_, index=days0).groupby(level=0).mean()
        t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std(ddof=1) > 0 else 0.0
        print(f"{b:>7.2f}{len(v):>7}{v.mean():>10.4f}{np.mean(res[b]['rps']):>10.4f}"
              f"{np.mean(res[b]['ll']):>10.4f}{d_.mean():>+10.4f}{t:>9.2f}")
    print("\n+ve 'vs 0.0' means the blend beats goals-only. |t| > 2 is significant.")


if __name__ == "__main__":
    main()

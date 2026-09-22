"""
Offline tests for the odds benchmark — audit finding §6. No network, no Claude.

The audit's claim was that Football-Data's Pinnacle prices went stale after 2025-07-23.
What is actually reproducible in .state/odds_raw is narrower and harder:

  * Pinnacle prices a full 380/380 played fixtures every season from 2018/19 to 2024/25,
    then 210/380 in 2025/26, with a CLEAN date break -- nothing from 2026-01-17 onward.
    Opening and closing Pinnacle columns die together, so it is a dead feed.
  * On the 210 fixtures of 2025/26 where both exist, Pinnacle scores worse than the
    average closing line (paired dBrier +0.0027). That flips the sign it held in all six
    prior seasons, but clustered by matchday the difference-in-differences is only
    t=+1.46. Degraded ACCURACY is therefore not claimed -- only the coverage collapse,
    which is not in doubt.

The fix is provenance, not a stale-date cutoff: every benchmark row records its provider
and price type, a season never mixes providers, and consumers compare within one.

Reads odds.csv and the cached raw downloads. Writes nothing.

    ./.venv/bin/python dashboard/test_odds_provenance.py
"""
import collections, csv, io, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import ingest_odds as IO  # noqa: E402

ODDS = os.path.join(ROOT, "odds.csv")
RAW = os.path.join(ROOT, ".state", "odds_raw")

PASS = FAIL = SKIP = 0


def check(name, fn, expect=None):
    global PASS, FAIL
    try:
        fn()
    except Exception as e:
        if expect and isinstance(e, expect):
            print(f"  ok   {name}"); PASS += 1
        else:
            print(f"  FAIL {name}: {type(e).__name__}: {e}"); FAIL += 1
        return
    if expect:
        print(f"  FAIL {name}: expected {expect.__name__}"); FAIL += 1
    else:
        print(f"  ok   {name}"); PASS += 1


def _assert(c, msg="assertion failed"):
    if not c:
        raise AssertionError(msg)


def season_of(date_str):
    y, m = int(date_str[:4]), int(date_str[5:7])
    return y if m >= 7 else y - 1


def main():
    global SKIP
    rows = list(csv.DictReader(open(ODDS)))

    print("=== every benchmark row carries its provenance ===")
    for col in ("provider", "price_type", "overround", "collected_at",
                "source_file", "source_sha8"):
        check(f"column `{col}` exists",
              lambda c=col: _assert(c in rows[0], f"missing {c}"))
    check("no row has a blank provider",
          lambda: _assert(all(r["provider"] for r in rows)))
    check("no row has a blank price type",
          lambda: _assert(all(r["price_type"] for r in rows)))
    check("every price type is opening or closing",
          lambda: _assert({r["price_type"] for r in rows} <= {"opening", "closing"}))
    check("every collection time is an ISO instant with a zone",
          lambda: _assert(all("T" in r["collected_at"] and
                              ("+" in r["collected_at"] or r["collected_at"].endswith("Z"))
                              for r in rows)))
    check("every row names the file it came from",
          lambda: _assert(all(r["source_file"].startswith("E0_") for r in rows)))
    check("every row carries a source hash",
          lambda: _assert(all(len(r["source_sha8"]) == 8 for r in rows)))

    print("\n=== a season never mixes providers ===")
    by_season = collections.defaultdict(set)
    for r in rows:
        by_season[season_of(r["date"])].add(r["provider"])
    mixed = {s: p for s, p in by_season.items() if len(p) > 1}
    check(f"no season draws on two providers ({len(mixed)} mixed)",
          lambda: _assert(not mixed, f"mixed seasons: {mixed}"))
    check("the primary provider covers most of the window",
          lambda: _assert(sum(1 for r in rows if r["provider"] == IO.PRIMARY) > 0.8 * len(rows)))

    print("\n=== probabilities are well-formed ===")
    check("every triple sums to 1",
          lambda: _assert(all(abs(float(r["p_h"]) + float(r["p_d"]) + float(r["p_a"]) - 1) < 5e-4
                              for r in rows)))
    check("every probability is inside (0,1)",
          lambda: _assert(all(0 < float(r[k]) < 1 for r in rows for k in ("p_h", "p_d", "p_a"))))
    check("every overround is >= 1 (vig is never negative)",
          lambda: _assert(all(float(r["overround"]) >= 1.0 for r in rows)))
    check("every overround is < 1.25 (sanity bound on a 1X2 book)",
          lambda: _assert(all(float(r["overround"]) < 1.25 for r in rows)))

    print("\n=== the alternate quote is genuinely a second source ===")
    alt = [r for r in rows if r["alt_provider"]]
    check(f"most rows carry an alternate provider ({len(alt)}/{len(rows)})",
          lambda: _assert(len(alt) > 0.5 * len(rows)))
    check("the alternate is never the same provider",
          lambda: _assert(all(r["alt_provider"] != r["provider"] for r in alt)))
    check("alternate probabilities also sum to 1",
          lambda: _assert(all(abs(float(r["alt_p_h"]) + float(r["alt_p_d"]) + float(r["alt_p_a"]) - 1) < 5e-4
                              for r in alt)))

    print("\n=== selection logic is explicit, not a silent fallback chain ===")
    row = {"AvgCH": "2.0", "AvgCD": "3.5", "AvgCA": "4.0",
           "PSCH": "2.1", "PSCD": "3.4", "PSCA": "3.9"}
    chosen, a = IO.pick(row)
    check("average closing is preferred over Pinnacle closing",
          lambda: _assert(chosen["provider"] == "avg_closing"))
    check("the alternate is the other provider",
          lambda: _assert(a["provider"] == "pinnacle_closing"))
    check("quotes() reports both, not just the winner",
          lambda: _assert(len(IO.quotes(row)) == 2))
    check("a fixture nobody priced yields nothing",
          lambda: _assert(IO.pick({"AvgCH": "", "PSCH": "x"}) == (None, None)))
    check("odds of 1.0 or less are rejected as impossible",
          lambda: _assert(IO.pick({"AvgCH": "1.0", "AvgCD": "3.5", "AvgCA": "4.0"}) == (None, None)))
    check("normalisation removes the overround",
          lambda: _assert(abs(sum(IO.quotes(row)[0]["probs"]) - 1) < 1e-12))
    check("the raw overround is preserved, not discarded",
          lambda: _assert(IO.quotes(row)[0]["overround"] > 1.0))

    print("\n=== season labelling ===")
    import datetime as dt
    check("an August date belongs to that year's season",
          lambda: _assert(IO.current_season(dt.date(2026, 8, 15)) == 2026))
    check("a January date belongs to the previous year's season",
          lambda: _assert(IO.current_season(dt.date(2026, 1, 15)) == 2025))
    check("ingest defaults forward to the current season, so it cannot freeze",
          lambda: _assert(IO.current_season(dt.date(2026, 9, 22)) == 2026))

    print("\n=== the Pinnacle coverage collapse is real (raw files) ===")
    raw_2526 = os.path.join(RAW, "E0_2526.csv")
    if not os.path.exists(raw_2526):
        print("  SKIP no cached raw download for 2025/26"); SKIP += 1
    else:
        def cov(path, cols):
            rs = [r for r in csv.DictReader(io.StringIO(
                open(path, encoding="utf-8-sig", errors="replace").read())) if r.get("FTR")]
            n = 0
            for r in rs:
                try:
                    o = [float(r[c]) for c in cols]
                except (KeyError, ValueError, TypeError):
                    continue
                if min(o) > 1:
                    n += 1
            return n, len(rs)
        psc, played = cov(raw_2526, ("PSCH", "PSCD", "PSCA"))
        avg, _ = cov(raw_2526, ("AvgCH", "AvgCD", "AvgCA"))
        check(f"Pinnacle is incomplete in 2025/26 ({psc}/{played})",
              lambda: _assert(psc < played))
        check(f"the average line is complete in 2025/26 ({avg}/{played})",
              lambda: _assert(avg == played))
        prior = os.path.join(RAW, "E0_2425.csv")
        if os.path.exists(prior):
            p24, n24 = cov(prior, ("PSCH", "PSCD", "PSCA"))
            check(f"Pinnacle WAS complete the season before ({p24}/{n24})",
                  lambda: _assert(p24 == n24))
        check("so the collapse is a broken feed, not a missing season",
              lambda: _assert(psc > 0 and psc < played))

    print("\n=== consumers compare within one provider ===")
    bt = open(os.path.join(ROOT, "backtest.py")).read()
    check("backtest.py filters odds by provider",
          lambda: _assert("odds.provider == provider" in bt))
    check("backtest.py refuses odds.csv with no provenance",
          lambda: _assert("has no provider column" in bt))
    check("backtest.py no longer hard-codes the provider name in its output",
          lambda: _assert("closing line (Pinnacle)" not in bt))
    check("backtest.py clusters its market CI",
          lambda: _assert("paired_clustered" in bt and "_season" in bt))
    bb = open(os.path.join(HERE, "build_backtest.py")).read()
    check("build_backtest.py filters odds by provider",
          lambda: _assert("od.provider == provider" in bb))
    check("build_backtest.py publishes the provider and sample size",
          lambda: _assert('"provider": provider' in bb and '"n": len(mg)' in bb))
    check("build_backtest.py publishes a clustered paired CI",
          lambda: _assert('"paired": paired' in bb))

    print("\n=== the published comparison reports uncertainty ===")
    import json
    bjson = os.path.join(HERE, "backtest.json")
    if not os.path.exists(bjson):
        print("  SKIP backtest.json not built"); SKIP += 1
    else:
        m = json.load(open(bjson)).get("market")
        check("backtest.json has a market block", lambda: _assert(m))
        check("it names the provider", lambda: _assert(m.get("provider")))
        check("it reports the matched sample size", lambda: _assert(m.get("n", 0) > 0))
        check("it says how many fixtures were modelled in total",
              lambda: _assert(m.get("n_model_fixtures", 0) >= m["n"]))
        check("it reports a paired CI for RPS",
              lambda: _assert(m["paired"]["rps"].get("ci")))
        check("the CI is an interval, not a point",
              lambda: _assert(m["paired"]["rps"]["ci"][0] < m["paired"]["rps"]["ci"][1]))
        check("it states a verdict rather than leaving the reader to guess",
              lambda: _assert(m["paired"]["rps"]["verdict"] in
                              ("model better", "market better", "indistinguishable")))
        check("no edge is claimed: the market is ahead on RPS",
              lambda: _assert(m["paired"]["rps"]["verdict"] == "market better"))

    print(f"\n  {PASS} passed, {FAIL} failed, {SKIP} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

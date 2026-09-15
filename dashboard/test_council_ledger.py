"""
Offline tests for council_ledger.py. NOTHING here contacts Claude, and NOTHING touches the
real dashboard/council_ledger.csv - every test writes to a throwaway file in a temp dir.

    ./.venv/bin/python dashboard/test_council_ledger.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import council as C
import council_ledger as L

PASS = FAIL = 0
REAL_LEDGER = L.PATH


def _fingerprint():
    """(size, mtime, contents) of the production ledger, or None when it does not exist."""
    if not os.path.exists(REAL_LEDGER):
        return None
    st = os.stat(REAL_LEDGER)
    with open(REAL_LEDGER) as f:
        return (st.st_size, st.st_mtime, f.read())


FINGERPRINT_BEFORE = _fingerprint()


def check(name, fn, expect=None):
    global PASS, FAIL
    try:
        fn()
    except Exception as e:
        if expect and isinstance(e, expect):
            print(f"  ok   {name} -> {type(e).__name__}"); PASS += 1
        else:
            print(f"  FAIL {name}: {type(e).__name__}: {e}"); FAIL += 1
        return
    if expect:
        print(f"  FAIL {name}: expected {expect.__name__}, nothing raised"); FAIL += 1
    else:
        print(f"  ok   {name}"); PASS += 1


def _assert(c):
    if not c:
        raise AssertionError("assertion failed")


def ctx(home="Brentford", away="Chelsea", kickoff="2026-09-18T19:00:00+00:00",
        market=(.3354, .2472, .4174)):
    return C.MatchContext(
        home, away, kickoff=kickoff,
        model_home=.42, model_draw=.26, model_away=.32,
        market_home=market[0] if market else None,
        market_draw=market[1] if market else None,
        market_away=market[2] if market else None)


def pred(h=.41, d=.28, a=.31, consensus=.4):
    three = [C.AnalystPrediction(n, .4, .3, .3, C.HOME, .5, ["e"], ["u"])
             for n in ("quant", "context", "market")]
    return C.CouncilPrediction(h, d, a, C.HOME, .25, consensus, True,
                               "mocked", three)


def main():
    tmp = tempfile.mkdtemp(prefix="council_ledger_test_")
    P = os.path.join(tmp, "council_ledger.csv")

    print("=== scoring formulas are IDENTICAL to build_dashboard's ===")
    import build_dashboard as B
    grid = [([.80, .14, .06], "H"), ([.19, .24, .57], "A"), ([.39, .29, .32], "D"),
            ([.33, .34, .33], "H"), ([1.0, 0.0, 0.0], "A"), ([0.0, 0.0, 1.0], "A"),
            ([.42, .26, .32], "D"), ([.5, .25, .25], "H")]
    check(f"brier matches on all {len(grid)} cases",
          lambda: _assert(all(L.brier(p, o) == B._brier(p, o) for p, o in grid)))
    check(f"rps matches on all {len(grid)} cases",
          lambda: _assert(all(L.rps(p, o) == B._rps(p, o) for p, o in grid)))
    check("a known ledger row reproduces exactly",
          lambda: _assert(L.brier([.80, .14, .06], "H") == 0.0632
                          and L.rps([.80, .14, .06], "H") == 0.0218))
    check("outcome_from_result parses", lambda: _assert(
        (L.outcome_from_result("2-1"), L.outcome_from_result("0-0"),
         L.outcome_from_result("1-3")) == ("H", "D", "A")))
    check("outcome_from_result refuses junk",
          lambda: _assert(L.outcome_from_result("abandoned") is None))

    print("\n=== locking: the first prediction wins ===")
    before = L.record(ctx(), pred(h=.41, d=.28, a=.31), now="2026-09-17T10:00:00+00:00", path=P)
    check("row written", lambda: _assert(os.path.exists(P)))
    check("fixture key is {home}|{away}",
          lambda: _assert(before["fixture_key"] == "Brentford|Chelsea"))
    check("not late (locked before kickoff)", lambda: _assert(before["late"] == ""))
    check("council probabilities stored 0-1",
          lambda: _assert(before["council_home"] == 0.41))
    check("model probabilities stored", lambda: _assert(before["model_home"] == 0.42))
    check("market probabilities stored", lambda: _assert(before["market_home"] == 0.3354))
    check("consensus stored", lambda: _assert(before["consensus_score"] == 0.4))

    again = L.record(ctx(), pred(h=.99, d=.005, a=.005, consensus=.99),
                     now="2026-09-17T18:00:00+00:00", path=P)
    check("a SECOND prediction does not replace the first",
          lambda: _assert(again["council_home"] == 0.41 and again["council_draw"] == 0.28))
    check("consensus not replaced either", lambda: _assert(again["consensus_score"] == 0.4))
    check("locked_at not restamped",
          lambda: _assert(again["locked_at"] == before["locked_at"]))
    check("still one row", lambda: _assert(len(L.load(P)) == 1))

    check("fresh write and re-read return the SAME TYPES",
          lambda: _assert({k: type(v).__name__ for k, v in before.items()}
                          == {k: type(v).__name__ for k, v in again.items()}))
    check("numeric columns come back as float, not str",
          lambda: _assert(isinstance(again["council_home"], float)
                          and isinstance(again["consensus_score"], float)))

    print("\n=== lookup: what the automation will ask ===")
    check("is_locked True", lambda: _assert(L.is_locked("Brentford|Chelsea", P)))
    check("is_locked False for unknown", lambda: _assert(not L.is_locked("A|B", P)))
    got = L.locked("Brentford|Chelsea", P)
    check("locked() returns the ORIGINAL prediction",
          lambda: _assert(got["council_home"] == 0.41))
    check("locked() returns None for unknown", lambda: _assert(L.locked("A|B", P) is None))
    check("locked() hands back a copy, not the stored dict",
          lambda: _assert((got.__setitem__("council_home", 9),
                           L.locked("Brentford|Chelsea", P)["council_home"] == 0.41)[1]))

    print("\n=== the key is stable when kickoff moves (TV picks) ===")
    moved = L.record(ctx(kickoff="2026-09-18T20:00:00+00:00"), pred(h=.10, d=.10, a=.80),
                     now="2026-09-17T12:00:00+00:00", path=P)
    check("same fixture, same row", lambda: _assert(len(L.load(P)) == 1))
    check("prediction still the original", lambda: _assert(moved["council_home"] == 0.41))

    print("\n=== late predictions are flagged, kept, and not scored ===")
    P2 = os.path.join(tmp, "late.csv")
    lt = L.record(ctx("Leeds", "Newcastle"), pred(), now="2026-09-19T09:00:00+00:00", path=P2)
    check("locked AFTER kickoff -> late=1", lambda: _assert(lt["late"] == "1"))
    L.settle({"Leeds|Newcastle": "2-1"}, path=P2)
    s = L.summary(P2)
    check("late row still stored", lambda: _assert(s["tracked"] == 1))
    check("late row counted as late", lambda: _assert(s["late"] == 1))
    check("late row NOT counted as valid", lambda: _assert(s["valid"] == 0))
    check("late row NOT graded in the scorecard", lambda: _assert(s["graded"] == 0))
    check("no council brier from a late row", lambda: _assert(s["council_brier"] is None))
    check("missing kickoff is not late",
          lambda: _assert(L.record(ctx("A", "B", kickoff=None), pred(),
                                   now="2030-01-01T00:00:00+00:00",
                                   path=os.path.join(tmp, "nk.csv"))["late"] == ""))
    check("unparseable kickoff is not late",
          lambda: _assert(L.record(ctx("C", "D", kickoff="Fri 18 Sep 19:00"), pred(),
                                   now="2030-01-01T00:00:00+00:00",
                                   path=os.path.join(tmp, "uk.csv"))["late"] == ""))

    print("\n=== grading ===")
    n = L.settle({"Brentford|Chelsea": "1-2"}, path=P)     # AWAY
    row = L.locked("Brentford|Chelsea", P)
    check("one fixture graded", lambda: _assert(n == 1))
    check("actual_result written", lambda: _assert(row["actual_result"] == "A"))
    check("council brier correct",
          lambda: _assert(row["council_brier"] == L.brier([.41, .28, .31], "A")))
    check("council rps correct",
          lambda: _assert(row["council_rps"] == L.rps([.41, .28, .31], "A")))
    check("model rps correct",
          lambda: _assert(row["model_rps"] == L.rps([.42, .26, .32], "A")))
    check("market rps correct",
          lambda: _assert(row["market_rps"] == L.rps([.3354, .2472, .4174], "A")))
    check("grading did NOT touch council probabilities",
          lambda: _assert(row["council_home"] == 0.41))
    check("grading did NOT touch model probabilities",
          lambda: _assert(row["model_home"] == 0.42))
    check("grading did not restamp locked_at",
          lambda: _assert(row["locked_at"] == before["locked_at"]))
    check("re-settling is a no-op", lambda: _assert(L.settle({"Brentford|Chelsea": "5-0"},
                                                             path=P) == 0))
    check("result not changed by the re-settle",
          lambda: _assert(L.locked("Brentford|Chelsea", P)["actual_result"] == "A"))
    check("unknown fixture ignored", lambda: _assert(L.settle({"X|Y": "1-0"}, path=P) == 0))
    check("unparseable result ignored",
          lambda: _assert(L.settle({"Brentford|Chelsea": "postponed"}, path=P) == 0))

    print("\n=== missing market stays missing ===")
    P3 = os.path.join(tmp, "nomarket.csv")
    r = L.record(ctx("Fulham", "Everton"), pred(), now="2026-09-17T10:00:00+00:00",
                 market=None, path=P3) if False else L.record(
        C.MatchContext("Fulham", "Everton", kickoff="2026-09-18T19:00:00+00:00",
                       model_home=.42, model_draw=.26, model_away=.32),
        pred(), now="2026-09-17T10:00:00+00:00", path=P3)
    check("market stored as missing, NOT zero", lambda: _assert(r["market_home"] is None))
    check("missing is not 0.0", lambda: _assert(r["market_home"] != 0.0))
    L.settle({"Fulham|Everton": "2-0"}, path=P3)
    rr = L.locked("Fulham|Everton", P3)
    check("market_rps stays ungraded", lambda: _assert(rr["market_rps"] is None))
    check("council still graded", lambda: _assert(rr["council_rps"] is not None))
    check("model still graded", lambda: _assert(rr["model_rps"] is not None))
    s3 = L.summary(P3)
    check("summary market_n is 0", lambda: _assert(s3["market_n"] == 0))
    check("summary market_rps is None, not 0", lambda: _assert(s3["market_rps"] is None))

    print("\n=== appends preserve existing rows ===")
    L.record(C.MatchContext("Arsenal", "Spurs", kickoff="2026-09-20T14:00:00+00:00",
                            model_home=.5, model_draw=.25, model_away=.25),
             pred(h=.5, d=.25, a=.25), now="2026-09-19T10:00:00+00:00", path=P)
    check("two rows now", lambda: _assert(len(L.load(P)) == 2))
    check("original row intact",
          lambda: _assert(L.locked("Brentford|Chelsea", P)["council_home"] == 0.41))
    check("original grading intact",
          lambda: _assert(L.locked("Brentford|Chelsea", P)["actual_result"] == "A"))

    print("\n=== type safety ===")
    check("rejects a non-MatchContext",
          lambda: L.record({"a": 1}, pred(), path=os.path.join(tmp, "x.csv")), TypeError)
    check("rejects a non-CouncilPrediction",
          lambda: L.record(ctx(), {"b": 2}, path=os.path.join(tmp, "x.csv")), TypeError)

    print("\n=== the REAL ledger was never touched ===")
    # The invariant is that the TESTS never touch the production ledger - not that it is
    # absent. Asserting absence encoded "the Council has never run", which stopped being true
    # the moment the first real forecast was locked.
    check("the real ledger is byte-identical to before these tests",
          lambda: _assert(_fingerprint() == FINGERPRINT_BEFORE))
    check("every test file lives in the temp dir",
          lambda: _assert(all(f.startswith(tmp) for f in
                              (P, P2, P3, os.path.join(tmp, "nk.csv")))))

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

"""
Offline tests for RULE 44 / audit §5 — data freshness and completeness guards.
No network, no Claude. Writes nothing.

The finding was that the main current-season store held 50 fixtures while the richer
team-stat store held 38, and the dashboard rendered tables from both as though each were
complete. A short source now says it is short, and a source the desk cannot function
without fails the build rather than publishing a plausible-looking blend of different ages.

    ./.venv/bin/python dashboard/test_source_health.py
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import build_dashboard as B

PASS = FAIL = 0


def check(name, fn, expect=None):
    global PASS, FAIL
    try:
        fn()
    except BaseException as e:      # SystemExit is the point of half these tests
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


def reset():
    B.SOURCE_CHECKS.clear()


def main():
    print("=== a complete source is ok ===")
    reset()
    r = B.check_source("a", "Results", 50, 50)
    check("status ok", lambda: _assert(r["status"] == "ok"))
    check("nothing missing", lambda: _assert(r["missing"] == 0))
    check("100 percent", lambda: _assert(r["complete_pct"] == 100.0))
    check("the report says nothing is incomplete",
          lambda: _assert(B.source_report()["any_incomplete"] is False))

    print("\n=== a short source says it is short ===")
    reset()
    r = B.check_source("b", "Team stats", 38, 50)
    check("12 missing counted", lambda: _assert(r["missing"] == 12))
    check("76% -> thin, not ok", lambda: _assert(r["status"] == "thin"))
    check("the exact audit case is not silently ok",
          lambda: _assert(r["status"] != "ok"))
    reset()
    r2 = B.check_source("c", "Sparse", 20, 50)
    check("40% -> incomplete", lambda: _assert(r2["status"] == "incomplete"))
    check("the 75% boundary is thin, not incomplete",
          lambda: (reset(), _assert(B.check_source("d", "Edge", 75, 100)["status"] == "thin")))
    check("just below the boundary is incomplete",
          lambda: (reset(), _assert(B.check_source("e", "Edge", 74, 100)["status"] == "incomplete")))

    print("\n=== a critical source that is short FAILS the build ===")
    reset()
    check("a short critical source aborts",
          lambda: B.check_source("f", "Fixtures", 38, 50, critical=True),
          expect=SystemExit)
    reset()
    check("a complete critical source does not",
          lambda: _assert(B.check_source("g", "Fixtures", 50, 50, critical=True)["status"] == "ok"))
    reset()
    msg = ""
    try:
        B.check_source("h", "Fixtures", 1, 50, critical=True)
    except SystemExit as e:
        msg = str(e)
    check("the abort message names the source and the shortfall",
          lambda: _assert("Fixtures" in msg and "1/50" in msg))
    check("and says why it refuses rather than just failing",
          lambda: _assert("look complete" in msg))

    print("\n=== degenerate inputs do not produce a false 'ok' ===")
    reset()
    check("expected 0 is treated as 100% (nothing was owed)",
          lambda: _assert(B.check_source("i", "Empty", 0, 0)["complete_pct"] == 100.0))
    reset()
    check("got 0 against a real expectation is incomplete",
          lambda: _assert(B.check_source("j", "Dead feed", 0, 50)["status"] == "incomplete"))
    reset()
    check("a feed returning MORE than expected is not negative-missing",
          lambda: _assert(B.check_source("k", "Over", 60, 50)["missing"] == 0))
    reset()
    check("None counts as zero, not as a crash",
          lambda: _assert(B.check_source("l", "None", None, 50)["status"] == "incomplete"))

    print("\n=== the report orders problems first ===")
    reset()
    B.check_source("m1", "Zebra ok", 10, 10)
    B.check_source("m2", "Alpha thin", 8, 10)
    B.check_source("m3", "Beta broken", 1, 10)
    rep = B.source_report()
    check("worst status surfaces",
          lambda: _assert(rep["worst"] == "incomplete"))
    check("incomplete is listed before thin, and thin before ok",
          lambda: _assert([r["status"] for r in rep["sources"]] ==
                          ["incomplete", "thin", "ok"]))
    check("any_incomplete is true when anything is short",
          lambda: _assert(rep["any_incomplete"] is True))
    check("every row carries the numbers a reader needs to check it",
          lambda: _assert(all(all(k in r for k in
                                  ("name", "rows", "expected", "missing", "complete_pct"))
                              for r in rep["sources"])))

    print("\n=== freshness is recorded, not inferred ===")
    reset()
    r = B.check_source("n", "FPL", 40, 50, as_of="2026-09-22T10:00+00:00",
                       note="FPL has not rolled over")
    check("as_of is stored", lambda: _assert(r["as_of"].startswith("2026-09-22")))
    check("the note explaining the shortfall is stored",
          lambda: _assert("rolled over" in r["note"]))
    check("a source with no as_of says so rather than claiming now",
          lambda: (reset(), _assert(B.check_source("o", "X", 1, 1)["as_of"] == "")))

    print("\n=== the page renders the warning ===")
    html = open(os.path.join(HERE, "index.html")).read()
    check("there is a container for it", lambda: _assert('id="sourceHealth"' in html))
    check("and a function that fills it",
          lambda: _assert("function renderSourceHealth()" in html))
    check("it is actually called on render",
          lambda: _assert("renderSourceHealth();" in html))
    check("it says plainly that figures are partial",
          lambda: _assert("Some data is incomplete" in html))
    check("it clears itself when everything is ok, rather than sticking",
          lambda: _assert("el.replaceChildren(); return;" in html))

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

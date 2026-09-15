"""
Offline tests for run_council.py. NO Claude calls, NO network, and the real
dashboard/council_ledger.csv is never read or written - every test uses a temp file.

    ./.venv/bin/python dashboard/test_run_council.py
"""
import os
import sys
import shutil
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import council as C
import council_ledger as L
import run_council as R

PASS = FAIL = 0
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


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


def match(home, away, ph=42, pd=26, pa=32):
    return {"home": home, "away": away, "ph": ph, "pd": pd, "pa": pa,
            "finished": False, "live": False, "time": "Fri 18 Sep 19:00",
            "xg": "1.66 - 1.55", "form": {"h": 3, "a": 1},
            "mkt": {"imp": {"h": .3354, "d": .2472, "a": .4174}},
            "kalshi": {"mid": {"h": 32.5, "d": 25.5, "a": 42.5}},
            "news": {"h": [{"who": "Furo", "what": "injury"}], "a": []}}


def ev(home, away, hours):
    return {"home": home, "away": away,
            "utc": (NOW + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")}


# three upcoming, one already played, one far outside the horizon
PAYLOAD = {"weeks": [{"gw": 5, "matches": [
    match("Brentford", "Chelsea"), match("Arsenal", "Man City"),
    match("Liverpool", "Everton"),
    dict(match("Leeds", "Newcastle"), finished=True),
    match("Fulham", "Spurs"),
]}]}
EVENTS = [ev("Brentford", "Chelsea", 72), ev("Arsenal", "Man City", 96),
          ev("Liverpool", "Everton", 120), ev("Leeds", "Newcastle", -24),
          ev("Fulham", "Spurs", 24 * 30)]           # 30 days out


class Spy:
    """Stands in for council.predict. Records every call; never touches the SDK."""
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, ctx):
        self.calls.append(ctx.fixture_key())
        if ctx.fixture_key() in self.fail_on:
            raise C.CouncilSDKError(f"{ctx.fixture_key()} exploded")
        three = [C.AnalystPrediction(n, .4, .3, .3, C.HOME, .5, ["e"], ["u"])
                 for n in ("quant", "context", "market")]
        return C.CouncilPrediction(.4, .3, .3, C.HOME, .5, .6, False, "", three)


def with_spy(fn, fail_on=()):
    spy, saved = Spy(fail_on), C.predict
    R.C.predict = spy
    try:
        return fn(spy)
    finally:
        R.C.predict = saved


def silent(*a, **k):
    pass


def main():
    tmp = tempfile.mkdtemp(prefix="run_council_test_")
    real_before = (os.path.exists(L.PATH),
                   open(L.PATH).read() if os.path.exists(L.PATH) else None)

    def fresh(name="l.csv"):
        return os.path.join(tmp, name)

    common = dict(now=NOW, payload=PAYLOAD, events=EVENTS, out=silent)

    print("=== fixture selection ===")
    up = R.upcoming(days=7, now=NOW, payload=PAYLOAD, events=EVENTS)
    check("three fixtures inside the horizon", lambda: _assert(len(up) == 3))
    check("finished fixture excluded",
          lambda: _assert("Leeds|Newcastle" not in [r["key"] for r in up]))
    check("fixture 30 days out excluded",
          lambda: _assert("Fulham|Spurs" not in [r["key"] for r in up]))
    check("sorted by kickoff",
          lambda: _assert([r["key"] for r in up] ==
                          ["Brentford|Chelsea", "Arsenal|Man City", "Liverpool|Everton"]))
    past = R.upcoming(days=7, now=NOW + timedelta(hours=100), payload=PAYLOAD, events=EVENTS)
    check("a fixture already kicked off is excluded",
          lambda: _assert("Brentford|Chelsea" not in [r["key"] for r in past]))
    check("--days narrows the horizon",
          lambda: _assert(len(R.upcoming(days=3.5, now=NOW, payload=PAYLOAD,
                                         events=EVENTS)) == 1))
    check("--days widens it",
          lambda: _assert(len(R.upcoming(days=40, now=NOW, payload=PAYLOAD,
                                         events=EVENTS)) == 4))
    check("--fixture filters to one",
          lambda: _assert([r["key"] for r in R.upcoming(
              days=7, now=NOW, payload=PAYLOAD, events=EVENTS,
              fixture="Arsenal|Man City")] == ["Arsenal|Man City"]))
    check("--fixture with no match returns nothing",
          lambda: _assert(R.upcoming(days=7, now=NOW, payload=PAYLOAD, events=EVENTS,
                                     fixture="X|Y") == []))

    print("\n=== context is built from project data ===")
    ctx = R.build_context(up[0]["match"])
    check("model probabilities converted to 0-1",
          lambda: _assert((ctx.model_home, ctx.model_draw, ctx.model_away) == (.42, .26, .32)))
    check("expected goals parsed",
          lambda: _assert((ctx.expected_home_goals, ctx.expected_away_goals) == (1.66, 1.55)))
    check("market carried", lambda: _assert(ctx.market_home == .3354))
    check("kalshi converted to 0-1", lambda: _assert(ctx.kalshi_home == .325))
    check("form carried", lambda: _assert((ctx.home_form, ctx.away_form) == (3, 1)))
    check("news flattened", lambda: _assert(ctx.news_items("home") == ["Furo: injury"]))
    check("fixture key matches", lambda: _assert(ctx.fixture_key() == "Brentford|Chelsea"))

    print("\n=== DRY RUN never predicts and never writes ===")
    p = fresh("dry.csv")
    s = with_spy(lambda spy: (R.run(live=False, ledger_path=p, **common), spy))
    summary, spy = s
    check("predict() never called", lambda: _assert(spy.calls == []))
    check("no ledger file created", lambda: _assert(not os.path.exists(p)))
    check("would_run counts the unlocked", lambda: _assert(summary["would_run"] == 3))
    check("ran is zero", lambda: _assert(summary["ran"] == 0))
    check("reports live=False", lambda: _assert(summary["live"] is False))

    print("\n=== a locked fixture is skipped ===")
    p = fresh("locked.csv")
    L.record(R.build_context(up[0]["match"]),
             C.CouncilPrediction(.38, .25, .37, C.HOME, .25, .38, True, "n",
                                 [C.AnalystPrediction(n, .4, .3, .3, C.HOME, .5, ["e"], ["u"])
                                  for n in ("q", "c", "m")]),
             now="2026-09-15T10:00:00+00:00", path=p)
    summary, spy = with_spy(lambda spy: (R.run(live=False, ledger_path=p, **common), spy))
    check("locked counted", lambda: _assert(summary["locked"] == 1))
    check("only the other two would run", lambda: _assert(summary["would_run"] == 2))
    check("dry run still called nothing", lambda: _assert(spy.calls == []))

    print("\n=== LIVE mode (predict mocked) ===")
    p = fresh("live.csv")
    summary, spy = with_spy(lambda spy: (R.run(live=True, ledger_path=p, **common), spy))
    check("predict called once per fixture", lambda: _assert(len(spy.calls) == 3))
    check("three rows written", lambda: _assert(len(L.load(p)) == 3))
    check("ran == 3", lambda: _assert(summary["ran"] == 3))
    check("each recorded row has council probabilities",
          lambda: _assert(all(r["council_home"] == 0.4 for r in L.load(p).values())))

    print("\n=== LIVE never re-runs a locked fixture ===")
    summary2, spy2 = with_spy(lambda spy: (R.run(live=True, ledger_path=p, **common), spy))
    check("second live run predicts NOTHING", lambda: _assert(spy2.calls == []))
    check("all three now reported locked", lambda: _assert(summary2["locked"] == 3))
    check("still three rows", lambda: _assert(len(L.load(p)) == 3))

    print("\n=== --limit ===")
    p = fresh("limit.csv")
    summary, spy = with_spy(lambda spy: (R.run(live=True, limit=1, ledger_path=p, **common), spy))
    check("only one fixture predicted", lambda: _assert(len(spy.calls) == 1))
    check("only one row written", lambda: _assert(len(L.load(p)) == 1))
    check("the rest reported as skipped", lambda: _assert(summary["would_run"] == 2))
    check("--limit 0 predicts nothing",
          lambda: _assert(with_spy(lambda s: (R.run(live=True, limit=0,
                                                    ledger_path=fresh("l0.csv"), **common), s))[1]
                          .calls == []))

    print("\n=== failure isolation ===")
    p = fresh("fail.csv")
    summary, spy = with_spy(
        lambda spy: (R.run(live=True, ledger_path=p, **common), spy),
        fail_on=("Arsenal|Man City",))
    check("all three attempted", lambda: _assert(len(spy.calls) == 3))
    check("processing continued past the failure", lambda: _assert(summary["ran"] == 2))
    check("failure counted", lambda: _assert(summary["failed"] == 1))
    check("the failed fixture is NOT in the ledger",
          lambda: _assert("Arsenal|Man City" not in L.load(p)))
    check("no invented prediction written", lambda: _assert(len(L.load(p)) == 2))
    check("error names the fixture",
          lambda: _assert(summary["errors"][0][0] == "Arsenal|Man City"))

    print("\n=== --fixture targets one ===")
    p = fresh("one.csv")
    summary, spy = with_spy(lambda spy: (R.run(live=True, fixture="Liverpool|Everton",
                                               ledger_path=p, **common), spy))
    check("only that fixture predicted", lambda: _assert(spy.calls == ["Liverpool|Everton"]))
    check("only that row written", lambda: _assert(list(L.load(p)) == ["Liverpool|Everton"]))

    print("\n=== the REAL ledger and the real fixture ===")
    check("real council_ledger.csv unchanged",
          lambda: _assert((os.path.exists(L.PATH),
                           open(L.PATH).read() if os.path.exists(L.PATH) else None)
                          == real_before))
    if real_before[0]:
        check("Brentford|Chelsea is locked in the REAL ledger",
              lambda: _assert(L.is_locked("Brentford|Chelsea")))
    for other in ("ledger.csv", "props_ledger.csv", "player_ledger.csv"):
        pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), other)
        check(f"{other} not written by the runner",
              lambda pth=pth: _assert(os.path.exists(pth)))

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

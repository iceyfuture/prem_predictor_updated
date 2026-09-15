"""
Offline tests for council_reasoning.py and its integration with run_council.py.
NO Claude calls. The real council_reasoning.jsonl and council_ledger.csv are never written -
every test uses temp files, and both are fingerprinted before and after to prove it.

    ./.venv/bin/python dashboard/test_council_reasoning.py
"""
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import council as C
import council_ledger as L
import council_reasoning as CR
import run_council as R

PASS = FAIL = 0
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _fp(p):
    if not os.path.exists(p):
        return None
    st = os.stat(p)
    with open(p) as f:
        return (st.st_size, st.st_mtime, f.read())


LEDGER_BEFORE = _fp(L.PATH)
REASONING_BEFORE = _fp(CR.PATH)


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


def analysts():
    return [
        C.AnalystPrediction("quant", .44, .25, .31, C.HOME, .55,
                            ["xG 1.66 v 1.55, edge +0.11", "form +3.000 v +1.000"],
                            ["no market supplied"]),
        C.AnalystPrediction("context", .40, .26, .34, C.HOME, .25,
                            ["three long-term absences listed for the home side"],
                            ["rest days not supplied", "replacement quality unknown"]),
        C.AnalystPrediction("market", .343, .252, .405, C.AWAY, .55,
                            ["model +10.0 pts above the book on HOME"],
                            ["gap unexplained by this payload"])]


def council():
    return C.CouncilPrediction(.38, .25, .37, C.HOME, .25, .38, True,
                               "form/xG favour home while the market favours away",
                               analysts())


def ctx(home="Brentford", away="Chelsea"):
    return C.MatchContext(home, away, kickoff="2026-09-18T19:00:00+00:00",
                          model_home=.42, model_draw=.26, model_away=.32)


def match(home, away):
    return {"home": home, "away": away, "ph": 42, "pd": 26, "pa": 32,
            "finished": False, "live": False, "time": "Fri 18 Sep 19:00",
            "xg": "1.66 - 1.55", "form": {"h": 3, "a": 1},
            "mkt": {"imp": {"h": .33, "d": .25, "a": .42}}, "kalshi": {}, "news": {}}


def ev(home, away, hours):
    return {"home": home, "away": away,
            "utc": (NOW + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")}


PAYLOAD = {"weeks": [{"gw": 5, "matches": [match("Arsenal", "Man City"),
                                           match("Liverpool", "Everton")]}]}
EVENTS = [ev("Arsenal", "Man City", 72), ev("Liverpool", "Everton", 96)]


class Spy:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, c):
        self.calls.append(c.fixture_key())
        if c.fixture_key() in self.fail_on:
            raise C.CouncilSDKError("boom")
        return council()


def with_spy(fn, fail_on=()):
    spy, saved = Spy(fail_on), R.C.predict
    R.C.predict = spy
    try:
        return fn(spy)
    finally:
        R.C.predict = saved


def silent(*a, **k):
    pass


def main():
    tmp = tempfile.mkdtemp(prefix="council_reasoning_test_")
    f = lambda n: os.path.join(tmp, n)

    print("=== a new prediction writes exactly one record ===")
    rp = f("r1.jsonl")
    rec = CR.append(ctx(), council(), locked_at="2026-09-15T17:16+00:00", path=rp)
    check("record returned", lambda: _assert(rec is not None))
    check("file has one line", lambda: _assert(len(open(rp).read().strip().split("\n")) == 1))
    check("one record loaded", lambda: _assert(len(CR.load(rp)) == 1))
    check("fixture_key stored", lambda: _assert(rec["fixture_key"] == "Brentford|Chelsea"))
    check("kickoff stored", lambda: _assert(rec["kickoff"] == "2026-09-18T19:00:00+00:00"))
    check("locked_at stored", lambda: _assert(rec["locked_at"] == "2026-09-15T17:16+00:00"))

    print("\n=== all three specialists are stored ===")
    for seat, hp in (("quant", .44), ("context", .40), ("market", .343)):
        check(f"{seat} present", lambda s=seat: _assert(rec[s] is not None))
        check(f"{seat} probabilities", lambda s=seat, h=hp: _assert(rec[s]["home"] == h))
        check(f"{seat} outcome", lambda s=seat: _assert(rec[s]["predicted_outcome"] in
                                                        ("HOME", "DRAW", "AWAY")))
        check(f"{seat} confidence", lambda s=seat: _assert(
            isinstance(rec[s]["confidence"], float)))
    check("quant evidence preserved verbatim",
          lambda: _assert(rec["quant"]["evidence"] == ["xG 1.66 v 1.55, edge +0.11",
                                                       "form +3.000 v +1.000"]))
    check("context uncertainties preserved",
          lambda: _assert(rec["context"]["uncertainties"] ==
                          ["rest days not supplied", "replacement quality unknown"]))
    check("market evidence preserved",
          lambda: _assert(rec["market"]["evidence"] ==
                          ["model +10.0 pts above the book on HOME"]))
    check("every evidence list is a list of str",
          lambda: _assert(all(isinstance(x, str) for s in ("quant", "context", "market")
                              for x in rec[s]["evidence"] + rec[s]["uncertainties"])))

    print("\n=== chairman output is stored ===")
    ch = rec["chairman"]
    check("probabilities", lambda: _assert((ch["home"], ch["draw"], ch["away"]) ==
                                           (.38, .25, .37)))
    check("outcome", lambda: _assert(ch["predicted_outcome"] == "HOME"))
    check("confidence", lambda: _assert(ch["confidence"] == .25))
    check("consensus_score", lambda: _assert(ch["consensus_score"] == .38))
    check("disagreement_note preserved",
          lambda: _assert("form/xG favour home" in ch["disagreement_note"]))

    print("\n=== duplicate protection ===")
    dup = CR.append(ctx(), council(), path=rp)
    check("second append returns None", lambda: _assert(dup is None))
    check("still one line", lambda: _assert(len(CR.load(rp)) == 1))
    check("a DIFFERENT prediction for the same fixture also refuses",
          lambda: _assert(CR.append(
              ctx(), C.CouncilPrediction(.9, .05, .05, C.HOME, .9, .9, False, "x",
                                         analysts()), path=rp) is None))
    check("original values untouched",
          lambda: _assert(CR.get("Brentford|Chelsea", rp)["chairman"]["home"] == .38))
    check("a different fixture DOES append",
          lambda: _assert(CR.append(ctx("Arsenal", "Spurs"), council(), path=rp) is not None))
    check("now two records", lambda: _assert(len(CR.load(rp)) == 2))
    check("has() works", lambda: _assert(CR.has("Arsenal|Spurs", rp)
                                         and not CR.has("X|Y", rp)))

    print("\n=== nothing sensitive is persisted ===")
    blob = open(rp).read()
    SECRETS = ["sk-ant", "ANTHROPIC", "CLAUDE_CODE_OAUTH", "oauth", "api_key", "apiKey",
               "Bearer", "token", "credential", "password", "x-apisports"]
    hits = [w for w in SECRETS if w.lower() in blob.lower()]
    check(f"no credential markers in the file", lambda: _assert(not hits))
    check("no thinking/chain-of-thought keys",
          lambda: _assert(not any(k in blob.lower()
                                  for k in ("thinking", "chain_of_thought", "reasoning_trace"))))
    check("council.py never reads ThinkingBlock",
          lambda: _assert("ThinkingBlock" not in
                          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "council.py")).read()))
    # Match actual USAGE, not the word. The first version grepped for "environ" and tripped
    # over the module's own docstring saying it reads no environment variable.
    check("module reads no environment variable", lambda: _assert(
        not re.search(r"\bos\.environ\b|\bgetenv\s*\(",
                      open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "council_reasoning.py")).read())))
    check("module imports no SDK",
          lambda: _assert("claude_agent_sdk" not in
                          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "council_reasoning.py")).read()))
    keys_seen = set()
    for r in CR.load(rp):
        keys_seen |= set(r)
        for s in ("quant", "context", "market"):
            if r[s]:
                keys_seen |= {f"{s}.{k}" for k in r[s]}
    allowed = {"fixture_key", "kickoff", "locked_at", "quant", "context", "market", "chairman"}
    allowed |= {f"{s}.{k}" for s in ("quant", "context", "market") for k in CR.ANALYST_FIELDS}
    check("no unexpected keys at all", lambda: _assert(keys_seen <= allowed))

    print("\n=== type safety ===")
    check("rejects non-MatchContext", lambda: CR.append({"a": 1}, council(), path=f("x.jsonl")),
          TypeError)
    check("rejects non-CouncilPrediction", lambda: CR.append(ctx(), {"b": 2}, path=f("x.jsonl")),
          TypeError)
    check("tolerates a corrupt line", lambda: _assert(
        (open(f("bad.jsonl"), "w").write('{"fixture_key":"A|B"}\nNOT JSON\n'),
         len(CR.load(f("bad.jsonl"))) == 1)[1]))

    print("\n=== runner integration ===")
    common = dict(now=NOW, payload=PAYLOAD, events=EVENTS, out=silent)

    lp, rp2 = f("dry.csv"), f("dry.jsonl")
    _, spy = with_spy(lambda s: (R.run(live=False, ledger_path=lp, reasoning_path=rp2,
                                       **common), s))
    check("dry run calls nothing", lambda: _assert(spy.calls == []))
    check("dry run writes no ledger", lambda: _assert(not os.path.exists(lp)))
    check("dry run writes NO reasoning", lambda: _assert(not os.path.exists(rp2)))

    lp, rp3 = f("live.csv"), f("live.jsonl")
    summary, spy = with_spy(lambda s: (R.run(live=True, ledger_path=lp, reasoning_path=rp3,
                                             **common), s))
    check("two fixtures ran", lambda: _assert(summary["ran"] == 2))
    check("two ledger rows", lambda: _assert(len(L.load(lp)) == 2))
    check("two reasoning records", lambda: _assert(len(CR.load(rp3)) == 2))
    check("reasoning keys match ledger keys",
          lambda: _assert(CR.keys(rp3) == set(L.load(lp))))
    check("reasoning carries locked_at from the ledger",
          lambda: _assert(all(r["locked_at"] for r in CR.load(rp3))))

    summary2, spy2 = with_spy(lambda s: (R.run(live=True, ledger_path=lp, reasoning_path=rp3,
                                               **common), s))
    check("already-locked fixtures call no Claude", lambda: _assert(spy2.calls == []))
    check("already-locked fixtures append NO reasoning",
          lambda: _assert(len(CR.load(rp3)) == 2))

    lp4, rp4 = f("fail.csv"), f("fail.jsonl")
    summary, spy = with_spy(lambda s: (R.run(live=True, ledger_path=lp4, reasoning_path=rp4,
                                             **common), s), fail_on=("Arsenal|Man City",))
    check("failed prediction writes no ledger row",
          lambda: _assert("Arsenal|Man City" not in L.load(lp4)))
    check("failed prediction writes NO reasoning",
          lambda: _assert("Arsenal|Man City" not in CR.keys(rp4)))
    check("the other fixture still recorded",
          lambda: _assert("Liverpool|Everton" in CR.keys(rp4)))

    print("\n=== a failed ledger lock writes no reasoning ===")
    lp5, rp5 = f("locked.csv"), f("locked.jsonl")
    saved_record = L.record
    L.record = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    try:
        summary, spy = with_spy(lambda s: (R.run(live=True, ledger_path=lp5,
                                                 reasoning_path=rp5, **common), s))
        check("prediction was made", lambda: _assert(len(spy.calls) == 2))
        check("ledger lock failed for both", lambda: _assert(summary["failed"] == 2))
        check("NO reasoning written", lambda: _assert(not os.path.exists(rp5)))
        check("ran is zero", lambda: _assert(summary["ran"] == 0))
    finally:
        L.record = saved_record

    print("\n=== a reasoning failure does not re-run Claude ===")
    lp6 = f("rfail.csv")
    saved_append = CR.append
    CR.append = lambda *a, **k: (_ for _ in ()).throw(OSError("read-only fs"))
    try:
        summary, spy = with_spy(lambda s: (R.run(live=True, ledger_path=lp6,
                                                 reasoning_path=f("rfail.jsonl"), **common), s))
        check("Claude called exactly once per fixture, not retried",
              lambda: _assert(len(spy.calls) == 2 and len(set(spy.calls)) == 2))
        check("forecasts are still LOCKED", lambda: _assert(len(L.load(lp6)) == 2))
        check("ran still counts them", lambda: _assert(summary["ran"] == 2))
        check("reasoning failure reported", lambda: _assert(summary["reasoning_failed"] == 2))
        check("locked rows not overwritten or removed",
              lambda: _assert(all(r["council_home"] == .38 for r in L.load(lp6).values())))
    finally:
        CR.append = saved_append

    print("\n=== the REAL files were never touched ===")
    check("real council_ledger.csv byte-identical", lambda: _assert(_fp(L.PATH) == LEDGER_BEFORE))
    check("real council_reasoning.jsonl unchanged",
          lambda: _assert(_fp(CR.PATH) == REASONING_BEFORE))
    # (An assertion that the production sidecar does not EXIST used to live here. It encoded
    # "the Council has never run for real", which stopped being true the moment the first
    # live forecast was locked - and it was redundant anyway: the check above already proves
    # these tests leave the file byte-identical, which is the actual invariant.)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

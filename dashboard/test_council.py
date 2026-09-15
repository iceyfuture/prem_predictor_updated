"""
Offline tests for the AI Council. NOTHING here contacts Claude.

Every test drives the parsing and validation path with a canned payload, so the contract can
be checked for free and in CI. The one function that would make a network call,
run_quant_analyst_async, is exercised only through the pieces around it.

    ./.venv/bin/python dashboard/test_council.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import council as C

PASS = FAIL = 0


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


GOOD = """{
  "analyst_name": "quant",
  "home_probability": 0.58,
  "draw_probability": 0.24,
  "away_probability": 0.18,
  "predicted_outcome": "HOME",
  "confidence": "medium",
  "evidence": ["model gives home 0.6200", "expected goals 1.90 v 1.05"],
  "uncertainties": ["no market price supplied"]
}"""

FULL = C.MatchContext(
    "Arsenal", "Chelsea", kickoff="2026-09-20T14:00Z",
    model_home=.62, model_draw=.23, model_away=.15,
    expected_home_goals=1.90, expected_away_goals=1.05,
    market_home=.58, market_draw=.24, market_away=.18,
    kalshi_home=.55, kalshi_draw=.245, kalshi_away=.205,
    home_form=.41, away_form=-.12,
    home_news="Saka fit", away_news="Palmer a doubt")
BARE = C.MatchContext("Coventry", "Hull")


def main():
    print("=== context serialization ===")
    txt = C.serialize_quant_context(FULL)
    check("includes the fixture", lambda: _assert("Arsenal (home) v Chelsea" in txt))
    check("includes model probabilities", lambda: _assert("0.6200" in txt))
    check("includes expected goals", lambda: _assert("1.90" in txt))
    check("includes form", lambda: _assert("+0.410" in txt))
    check("EXCLUDES team news (Saka)", lambda: _assert("Saka" not in txt))
    check("EXCLUDES team news (Palmer)", lambda: _assert("Palmer" not in txt))
    check("EXCLUDES market price", lambda: _assert("0.5800" not in txt))
    check("EXCLUDES kalshi price", lambda: _assert("0.2450" not in txt))
    bare = C.serialize_quant_context(BARE)
    check("bare context says MISSING", lambda: _assert(bare.count("MISSING") >= 3))
    check("bare context lists the gaps", lambda: _assert("model" in bare and "form" in bare))
    check("rejects a non-MatchContext", lambda: C.serialize_quant_context({"a": 1}),
          C.CouncilValidationError)

    print("\n=== JSON -> AnalystPrediction ===")
    p = C.analyst_from_payload(C._extract_json(GOOD))
    check("parses a good payload", lambda: _assert(isinstance(p, C.AnalystPrediction)))
    check("name carried", lambda: _assert(p.analyst_name == "quant"))
    check("probabilities carried", lambda: _assert(p.probabilities() == (.58, .24, .18)))
    check("outcome carried", lambda: _assert(p.predicted_outcome == C.HOME))
    check("confidence 'medium' -> 0.55", lambda: _assert(p.confidence == 0.55))
    check("evidence carried", lambda: _assert(len(p.evidence) == 2))
    check("uncertainties carried", lambda: _assert(len(p.uncertainties) == 1))
    check("outcome agrees with numbers", lambda: _assert(p.agrees_with_own_numbers()))
    check("```json fence tolerated",
          lambda: _assert(C.analyst_from_payload(
              C._extract_json("```json\n" + GOOD + "\n```")).confidence == 0.55))
    check("prose around the object tolerated",
          lambda: _assert(C.analyst_from_payload(
              C._extract_json("Here you go:\n" + GOOD + "\nHope that helps.")
          ).predicted_outcome == C.HOME))

    print("\n=== malformed input raises ===")
    check("not JSON", lambda: C._extract_json("I think Arsenal will win."), C.CouncilParseError)
    check("truncated JSON", lambda: C._extract_json('{"home_probability": 0.5'), C.CouncilParseError)
    check("empty reply", lambda: C._extract_json("   "), C.CouncilParseError)
    check("JSON array not object", lambda: C._extract_json("[1,2,3]"), C.CouncilParseError)
    check("missing required field",
          lambda: C.analyst_from_payload({"home_probability": .5, "draw_probability": .3}),
          C.CouncilParseError)
    check("unknown confidence label",
          lambda: C.analyst_from_payload({**C._extract_json(GOOD), "confidence": "certain"}),
          C.CouncilParseError)

    print("\n=== invalid probabilities are NOT repaired ===")
    bad_sum = {**C._extract_json(GOOD), "home_probability": .9, "draw_probability": .9,
               "away_probability": .9}
    check("simplex that does not sum to 1", lambda: C.analyst_from_payload(bad_sum),
          C.CouncilValidationError)
    check("probability above 1",
          lambda: C.analyst_from_payload({**C._extract_json(GOOD), "home_probability": 1.4}),
          C.CouncilValidationError)
    check("probability below 0",
          lambda: C.analyst_from_payload({**C._extract_json(GOOD), "away_probability": -0.2}),
          C.CouncilValidationError)
    check("bad outcome label",
          lambda: C.analyst_from_payload({**C._extract_json(GOOD), "predicted_outcome": "WIN"}),
          C.CouncilValidationError)

    print("\n=== the seat is locked down ===")
    o = C.quant_options()
    check("no tools allowed", lambda: _assert(list(o.allowed_tools) == []))
    check("web tools explicitly denied",
          lambda: _assert({"WebSearch", "WebFetch"} <= set(o.disallowed_tools)))
    check("bash explicitly denied", lambda: _assert("Bash" in o.disallowed_tools))
    check("max_turns is conservative", lambda: _assert(o.max_turns == 1))
    check("permissions not bypassed", lambda: _assert(o.permission_mode != "bypassPermissions"))
    check("external settings ignored", lambda: _assert(list(o.setting_sources) == []))
    check("system prompt is the AgentDefinition's",
          lambda: _assert(o.system_prompt == C.QUANT_ANALYST.prompt))
    check("prompt forbids outside knowledge",
          lambda: _assert("not contained in the input" in C.QUANT_ANALYST.prompt))
    check("prompt forbids inventing injuries",
          lambda: _assert("invent injuries" in C.QUANT_ANALYST.prompt))
    check("agent definition has no tools", lambda: _assert(C.QUANT_ANALYST.tools == []))

    print("\n=== model is configurable, not hard-coded (no network) ===")
    saved = os.environ.get(C.COUNCIL_MODEL_ENV)
    try:
        os.environ.pop(C.COUNCIL_MODEL_ENV, None)
        check("unset -> council_model() is None", lambda: _assert(C.council_model() is None))
        check("unset -> options.model is None (SDK default)",
              lambda: _assert(C.quant_options().model is None))
        check("unset -> registered agent has no model",
              lambda: _assert(C.quant_options().agents["quant"].model is None))
        check("the AgentDefinition itself pins no model",
              lambda: _assert(C.QUANT_ANALYST.model is None))
        check("no model string hard-coded in council.py", _no_hardcoded_model)

        os.environ[C.COUNCIL_MODEL_ENV] = "claude-haiku-4-5"
        check("env var overrides council_model()",
              lambda: _assert(C.council_model() == "claude-haiku-4-5"))
        check("env var reaches options.model",
              lambda: _assert(C.quant_options().model == "claude-haiku-4-5"))
        check("env var reaches the registered subagent",
              lambda: _assert(C.quant_options().agents["quant"].model == "claude-haiku-4-5"))
        check("explicit argument beats the env var",
              lambda: _assert(C.quant_options(model="claude-sonnet-5").model == "claude-sonnet-5"))

        os.environ[C.COUNCIL_MODEL_ENV] = "   "
        check("blank env var counts as unset",
              lambda: _assert(C.council_model() is None))

        os.environ[C.COUNCIL_MODEL_ENV] = "model-a"
        first = C.quant_options().model
        os.environ[C.COUNCIL_MODEL_ENV] = "model-b"
        check("resolved per call, not frozen at import",
              lambda: _assert(first == "model-a" and C.quant_options().model == "model-b"))
    finally:
        os.environ.pop(C.COUNCIL_MODEL_ENV, None)
        if saved is not None:
            os.environ[C.COUNCIL_MODEL_ENV] = saved

    print("\n=== wiring ===")
    check("sync wrapper exists", lambda: _assert(callable(C.run_quant_analyst)))
    check("async version is a coroutine function",
          lambda: _assert(asyncio.iscoroutinefunction(C.run_quant_analyst_async)))
    check("council predict() still NotImplemented", lambda: C.predict(FULL), NotImplementedError)
    check("sync wrapper refuses a running loop",
          lambda: asyncio.run(_call_from_loop()), C.CouncilSDKError)

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


async def _call_from_loop():
    return C.run_quant_analyst(BARE)   # must raise, not deadlock


def _no_hardcoded_model():
    """council.py must not name a Claude model anywhere - the point of AI_COUNCIL_MODEL."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "council.py")).read()
    import re as _re
    hits = _re.findall(r"claude-(?:opus|sonnet|haiku|fable|mythos)[\w.-]*", src)
    if hits:
        raise AssertionError(f"hard-coded model name(s) in council.py: {sorted(set(hits))}")


def _assert(cond):
    if not cond:
        raise AssertionError("assertion failed")


if __name__ == "__main__":
    sys.exit(main())

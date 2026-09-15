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

# the Context seat's two cases: news supplied, and no news at all
NEWSY = C.MatchContext(
    "Brentford", "Chelsea", kickoff="Fri 18 Sep 19:00",
    model_home=.42, model_draw=.26, model_away=.32,
    expected_home_goals=1.66, expected_away_goals=1.55,
    market_home=.3354, market_draw=.2472, market_away=.4174,
    kalshi_home=.55, home_form=+3.0, away_form=+1.0,
    home_news=["Flekken (GK) ruled out - confirmed", "Wissa suspended, one match"],
    away_news="Palmer a doubt, late fitness test")
NO_NEWS = C.MatchContext(
    "Brentford", "Chelsea", kickoff="Fri 18 Sep 19:00",
    model_home=.42, model_draw=.26, model_away=.32,
    expected_home_goals=1.66, expected_away_goals=1.55,
    home_form=+3.0, away_form=+1.0, home_news=[], away_news=[])

# the Market seat's three cases: both sources, one source, neither
BOTH_MKT = C.MatchContext(
    "Brentford", "Chelsea", kickoff="Fri 18 Sep 19:00",
    model_home=.60, model_draw=.23, model_away=.17,
    market_home=.50, market_draw=.27, market_away=.23,
    kalshi_home=.505, kalshi_draw=.265, kalshi_away=.23,
    expected_home_goals=1.66, expected_away_goals=1.55,
    home_form=+3.0, away_form=+1.0,
    home_news=["Starting goalkeeper confirmed unavailable"],
    away_news=["Key midfielder a doubt"])
BOOK_ONLY = C.MatchContext(
    "Brentford", "Chelsea", model_home=.42, model_draw=.26, model_away=.32,
    market_home=.3354, market_draw=.2472, market_away=.4174)
VIG_MKT = C.MatchContext(
    "Brentford", "Chelsea", model_home=.42, model_draw=.26, model_away=.32,
    market_home=.36, market_draw=.27, market_away=.44)          # sums to 1.07
NO_MKT = C.MatchContext(
    "Brentford", "Chelsea", model_home=.42, model_draw=.26, model_away=.32,
    expected_home_goals=1.66, home_form=+3.0, home_news=["something"])

MARKET_GOOD = """{
  "analyst_name": "market",
  "home_probability": 0.54,
  "draw_probability": 0.25,
  "away_probability": 0.21,
  "predicted_outcome": "HOME",
  "confidence": "medium",
  "evidence": ["model is +10.0 pts above the book on HOME"],
  "uncertainties": ["no exchange price supplied"]
}"""

CONTEXT_GOOD = """{
  "analyst_name": "context",
  "home_probability": 0.36,
  "draw_probability": 0.27,
  "away_probability": 0.37,
  "predicted_outcome": "AWAY",
  "confidence": "medium",
  "evidence": ["Flekken (GK) ruled out - confirmed, lowers home"],
  "uncertainties": ["no rest-day data supplied"]
}"""


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

    print("\n=== CONTEXT ANALYST: exists and is a distinct seat ===")
    check("CONTEXT_ANALYST defined",
          lambda: _assert(isinstance(C.CONTEXT_ANALYST, type(C.QUANT_ANALYST))))
    check("its name is 'context'", lambda: _assert(C.CONTEXT_ANALYST_NAME == "context"))
    check("different prompt from the Quant seat",
          lambda: _assert(C.CONTEXT_ANALYST.prompt != C.QUANT_ANALYST.prompt))
    check("no tools on the definition", lambda: _assert(C.CONTEXT_ANALYST.tools == []))
    check("prompt forbids general PL knowledge",
          lambda: _assert("general Premier League knowledge" in C.CONTEXT_ANALYST.prompt))
    check("prompt forbids inventing availability",
          lambda: _assert("invent player availability" in C.CONTEXT_ANALYST.prompt))
    check("prompt sets the baseline rule",
          lambda: _assert("NEUTRAL STARTING POINT" in C.CONTEXT_ANALYST.prompt))
    check("prompt allows returning the baseline unchanged",
          lambda: _assert("Returning the baseline is a valid and correct answer"
                          in C.CONTEXT_ANALYST.prompt))
    check("async + sync entry points exist",
          lambda: _assert(asyncio.iscoroutinefunction(C.run_context_analyst_async)
                          and callable(C.run_context_analyst)))

    print("\n=== CONTEXT ANALYST: what it is shown ===")
    t = C.serialize_team_context(NEWSY)
    check("teams present", lambda: _assert("Brentford (home) v Chelsea" in t))
    check("kickoff present", lambda: _assert("Fri 18 Sep 19:00" in t))
    check("baseline model probabilities present", lambda: _assert("0.4200" in t and "0.2600" in t))
    check("supplied home news appears", lambda: _assert("Flekken" in t))
    check("second home news item appears", lambda: _assert("Wissa suspended" in t))
    check("away news (a plain string) appears", lambda: _assert("Palmer" in t))
    check("EXCLUDES expected goals", lambda: _assert("1.66" not in t and "1.55" not in t))
    check("EXCLUDES form ratings", lambda: _assert("+3.000" not in t))
    check("EXCLUDES market odds", lambda: _assert("0.3354" not in t))
    check("EXCLUDES kalshi", lambda: _assert("0.5500" not in t))
    check("says what is withheld", lambda: _assert("WITHHELD FROM THIS SEAT" in t))
    check("lists unsupplied categories", lambda: _assert("rest days" in t and "fixture congestion" in t))
    check("rejects a non-MatchContext", lambda: C.serialize_team_context(None),
          C.CouncilValidationError)

    print("\n=== CONTEXT ANALYST: missing news is explicit, not silent ===")
    n = C.serialize_team_context(NO_NEWS)
    check("says NONE SUPPLIED", lambda: _assert(n.count("NONE SUPPLIED") == 2))
    check("warns that absence != full strength",
          lambda: _assert("not 'full-strength squad'" in n))
    check("baseline still shown with no news", lambda: _assert("0.4200" in n))
    check("news_items normalises a list", lambda: _assert(len(NEWSY.news_items("home")) == 2))
    check("news_items normalises a string", lambda: _assert(NEWSY.news_items("away") ==
                                                           ["Palmer a doubt, late fitness test"]))
    check("news_items on empty list is []", lambda: _assert(NO_NEWS.news_items("home") == []))
    check("news_items on None is []", lambda: _assert(BARE.news_items("home") == []))
    check("dict-shaped news is flattened",
          lambda: _assert(C.MatchContext("A", "B", home_news=[{"what": "Doe injured"}])
                          .news_items("home") == ["Doe injured"]))
    check("has_news false when empty", lambda: _assert(NO_NEWS.has_news() is False))
    check("missing() reports news gap", lambda: _assert("news" in NO_NEWS.missing()))

    print("\n=== CONTEXT ANALYST: parsing and validation (shared with Quant) ===")
    cp = C.analyst_from_payload(C._extract_json(CONTEXT_GOOD), C.CONTEXT_ANALYST_NAME)
    check("valid payload parses", lambda: _assert(cp.analyst_name == "context"))
    check("probabilities carried", lambda: _assert(cp.probabilities() == (.36, .27, .37)))
    check("confidence mapped", lambda: _assert(cp.confidence == 0.55))
    check("malformed JSON raises",
          lambda: C.analyst_from_payload(C._extract_json("Chelsea look strong"), "context"),
          C.CouncilParseError)
    check("bad simplex raises",
          lambda: C.analyst_from_payload({**C._extract_json(CONTEXT_GOOD),
                                          "home_probability": .8}, "context"),
          C.CouncilValidationError)
    check("bad outcome raises",
          lambda: C.analyst_from_payload({**C._extract_json(CONTEXT_GOOD),
                                          "predicted_outcome": "DRAWN"}, "context"),
          C.CouncilValidationError)

    print("\n=== CONTEXT ANALYST: same lockdown as the Quant seat ===")
    co = C.context_options()
    check("no tools allowed", lambda: _assert(list(co.allowed_tools) == []))
    check("web tools denied", lambda: _assert({"WebSearch", "WebFetch"} <= set(co.disallowed_tools)))
    check("max_turns is 1", lambda: _assert(co.max_turns == 1))
    check("permissions not bypassed", lambda: _assert(co.permission_mode != "bypassPermissions"))
    check("external settings ignored", lambda: _assert(list(co.setting_sources) == []))
    check("system prompt is the definition's",
          lambda: _assert(co.system_prompt == C.CONTEXT_ANALYST.prompt))
    check("lockdown identical to the Quant seat", lambda: _assert(
        (list(co.allowed_tools), list(co.disallowed_tools), co.max_turns,
         co.permission_mode, list(co.setting_sources)) ==
        (list(C.quant_options().allowed_tools), list(C.quant_options().disallowed_tools),
         C.quant_options().max_turns, C.quant_options().permission_mode,
         list(C.quant_options().setting_sources))))

    print("\n=== MARKET SKEPTIC: exists and is a distinct seat ===")
    check("MARKET_SKEPTIC defined",
          lambda: _assert(isinstance(C.MARKET_SKEPTIC, type(C.QUANT_ANALYST))))
    check("its name is 'market'", lambda: _assert(C.MARKET_SKEPTIC_NAME == "market"))
    check("prompt differs from both other seats",
          lambda: _assert(C.MARKET_SKEPTIC.prompt not in
                          (C.QUANT_ANALYST.prompt, C.CONTEXT_ANALYST.prompt)))
    check("no tools on the definition", lambda: _assert(C.MARKET_SKEPTIC.tools == []))
    check("prompt forbids inventing explanations",
          lambda: _assert("invent explanations for why the market differs"
                          in C.MARKET_SKEPTIC.prompt))
    check("prompt forbids using team news",
          lambda: _assert("use team news" in C.MARKET_SKEPTIC.prompt))
    check("prompt says do not just copy the market",
          lambda: _assert("not here to repeat the market" in C.MARKET_SKEPTIC.prompt))
    # the prompt wraps this instruction across a line break, so match the contiguous fragment
    check("prompt says do not average the two sources",
          lambda: _assert("simply average the two" in C.MARKET_SKEPTIC.prompt))
    check("async + sync entry points exist",
          lambda: _assert(asyncio.iscoroutinefunction(C.run_market_skeptic_async)
                          and callable(C.run_market_skeptic)))

    print("\n=== MARKET SKEPTIC: what it is shown ===")
    mk = C.serialize_market_context(BOTH_MKT)
    check("teams present", lambda: _assert("Brentford (home) v Chelsea" in mk))
    check("model probabilities present", lambda: _assert("0.6000" in mk and "0.2300" in mk))
    check("bookmaker probabilities present", lambda: _assert("0.5000" in mk))
    check("kalshi probabilities present", lambda: _assert("0.5050" in mk))
    check("EXCLUDES expected goals", lambda: _assert("1.66" not in mk and "1.55" not in mk))
    check("EXCLUDES form ratings", lambda: _assert("+3.000" not in mk))
    check("EXCLUDES team news", lambda: _assert("goalkeeper" not in mk.lower()))
    check("EXCLUDES injuries/context", lambda: _assert("doubt" not in mk.lower()))
    check("says what is withheld", lambda: _assert("WITHHELD FROM THIS SEAT" in mk))
    check("quantifies disagreement vs book",
          lambda: _assert("+10.0 pts" in mk))
    check("quantifies disagreement vs exchange", lambda: _assert("vs exchange:" in mk))
    check("compares the two sources to each other",
          lambda: _assert("bookmaker vs exchange" in mk))
    check("rejects a non-MatchContext", lambda: C.serialize_market_context("x"),
          C.CouncilValidationError)

    print("\n=== MARKET SKEPTIC: missing sources are explicit ===")
    b = C.serialize_market_context(BOOK_ONLY)
    check("book present when only book supplied", lambda: _assert("0.3354" in b))
    check("missing kalshi stated", lambda: _assert("NOT SUPPLIED" in b and "Kalshi lists EPL" in b))
    check("missing sources listed", lambda: _assert("MARKET SOURCES NOT SUPPLIED: kalshi" in b))
    check("no source-vs-source line with one source",
          lambda: _assert("bookmaker vs exchange" not in b))
    n = C.serialize_market_context(NO_MKT)
    check("both sources reported missing",
          lambda: _assert(n.count("NOT SUPPLIED") >= 2))
    check("baseline still shown with no market", lambda: _assert("0.4200" in n))
    check("no disagreement block without a market",
          lambda: _assert("DISAGREEMENT" not in n))
    check("no-market payload still hides xG/form/news",
          lambda: _assert("1.66" not in n and "+3.000" not in n and "something" not in n))

    print("\n=== MARKET SKEPTIC: overround is surfaced ===")
    v = C.serialize_market_context(VIG_MKT)
    check("flags un-normalised book prices", lambda: _assert("overround" in v))
    check("reports the sum", lambda: _assert("sum to 1.0700" in v))
    check("normalised book is labelled so",
          lambda: _assert("already normalised" in C.serialize_market_context(BOOK_ONLY)))

    print("\n=== MARKET SKEPTIC: parsing and validation (shared) ===")
    mp = C.analyst_from_payload(C._extract_json(MARKET_GOOD), C.MARKET_SKEPTIC_NAME)
    check("valid payload parses", lambda: _assert(mp.analyst_name == "market"))
    check("probabilities carried", lambda: _assert(mp.probabilities() == (.54, .25, .21)))
    check("confidence mapped", lambda: _assert(mp.confidence == 0.55))
    check("malformed JSON raises",
          lambda: C.analyst_from_payload(C._extract_json("the market knows best"), "market"),
          C.CouncilParseError)
    check("bad simplex raises",
          lambda: C.analyst_from_payload({**C._extract_json(MARKET_GOOD),
                                          "away_probability": .9}, "market"),
          C.CouncilValidationError)
    check("bad outcome raises",
          lambda: C.analyst_from_payload({**C._extract_json(MARKET_GOOD),
                                          "predicted_outcome": "H"}, "market"),
          C.CouncilValidationError)

    print("\n=== MARKET SKEPTIC: lockdown matches the other seats ===")
    mo = C.market_options()
    check("no tools allowed", lambda: _assert(list(mo.allowed_tools) == []))
    check("web tools denied", lambda: _assert({"WebSearch", "WebFetch"} <= set(mo.disallowed_tools)))
    check("max_turns is 1", lambda: _assert(mo.max_turns == 1))
    check("permissions not bypassed", lambda: _assert(mo.permission_mode != "bypassPermissions"))
    check("external settings ignored", lambda: _assert(list(mo.setting_sources) == []))
    check("system prompt is the definition's",
          lambda: _assert(mo.system_prompt == C.MARKET_SKEPTIC.prompt))
    check("all three seats share one lockdown", lambda: _assert(
        len({(tuple(o.allowed_tools), tuple(o.disallowed_tools), o.max_turns,
              o.permission_mode, tuple(o.setting_sources))
             for o in (C.quant_options(), C.context_options(), C.market_options())}) == 1))
    check("three distinct seats registered", lambda: _assert(
        {list(C.quant_options().agents)[0], list(C.context_options().agents)[0],
         list(C.market_options().agents)[0]} == {"quant", "context", "market"}))

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
        check("context seat honours the env var",
              lambda: _assert(C.context_options().model == "claude-haiku-4-5"))
        check("context seat's subagent honours it",
              lambda: _assert(C.context_options().agents["context"].model == "claude-haiku-4-5"))
        check("context AgentDefinition pins no model",
              lambda: _assert(C.CONTEXT_ANALYST.model is None))
        check("market seat honours the env var",
              lambda: _assert(C.market_options().model == "claude-haiku-4-5"))
        check("market seat's subagent honours it",
              lambda: _assert(C.market_options().agents["market"].model == "claude-haiku-4-5"))
        check("market AgentDefinition pins no model",
              lambda: _assert(C.MARKET_SKEPTIC.model is None))

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

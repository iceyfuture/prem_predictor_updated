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
import time

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

# three specialist opinions to hand the Chairman, and a rich context it must NOT see through
A_QUANT = C.AnalystPrediction(analyst_name="quant", home_probability=.44,
    draw_probability=.24, away_probability=.32, predicted_outcome=C.HOME, confidence=.55,
    evidence=["expected goals 1.66 v 1.55, an edge of +0.11"],
    uncertainties=["no market price supplied"])
A_CONTEXT = C.AnalystPrediction(analyst_name="context", home_probability=.34,
    draw_probability=.28, away_probability=.38, predicted_outcome=C.AWAY, confidence=.55,
    evidence=["starting goalkeeper confirmed unavailable"],
    uncertainties=["rest days not supplied"])
A_MARKET = C.AnalystPrediction(analyst_name="market", home_probability=.52,
    draw_probability=.26, away_probability=.22, predicted_outcome=C.HOME, confidence=.55,
    evidence=["model is +10.0 pts above the book on HOME"],
    uncertainties=["gap unexplained by anything in this payload"])
THREE = [A_QUANT, A_CONTEXT, A_MARKET]

RICH = C.MatchContext(
    "Brentford", "Chelsea", kickoff="Fri 18 Sep 19:00",
    model_home=.42, model_draw=.26, model_away=.32,
    expected_home_goals=1.66, expected_away_goals=1.55,
    market_home=.3354, market_draw=.2472, market_away=.4174,
    kalshi_home=.5050, kalshi_draw=.2650, kalshi_away=.2300,
    home_form=+3.0, away_form=+1.0,
    home_news=["Flekken ruled out"], away_news=["Palmer a doubt"])

CHAIR_GOOD = """{
  "home_probability": 0.43,
  "draw_probability": 0.26,
  "away_probability": 0.31,
  "predicted_outcome": "HOME",
  "confidence": "medium",
  "consensus_score": 0.45,
  "major_disagreement": "one analyst calls AWAY on a confirmed absence while two call HOME"
}"""

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

    print("\n=== CHAIRMAN: exists, and requires exactly three analysts ===")
    check("CHAIRMAN defined", lambda: _assert(isinstance(C.CHAIRMAN, type(C.QUANT_ANALYST))))
    check("prompt differs from all three seats",
          lambda: _assert(C.CHAIRMAN.prompt not in (C.QUANT_ANALYST.prompt,
                          C.CONTEXT_ANALYST.prompt, C.MARKET_SKEPTIC.prompt)))
    check("no tools on the definition", lambda: _assert(C.CHAIRMAN.tools == []))
    check("prompt forbids new football facts",
          lambda: _assert("may not introduce any new football facts" in C.CHAIRMAN.prompt))
    check("prompt separates consensus from confidence",
          lambda: _assert("CONSENSUS IS NOT CONFIDENCE" in C.CHAIRMAN.prompt))
    check("prompt carries the consensus bands",
          lambda: _assert("0.31-0.60" in C.CHAIRMAN.prompt and "0.81-1.00" in C.CHAIRMAN.prompt))
    check("prompt says do not force consensus",
          lambda: _assert("Do not force consensus" in C.CHAIRMAN.prompt))
    check("async + sync entry points exist",
          lambda: _assert(asyncio.iscoroutinefunction(C.run_chairman_async)
                          and callable(C.run_chairman)))
    check("two analysts rejected",
          lambda: C.serialize_chairman_context(RICH, THREE[:2]), C.CouncilValidationError)
    check("four analysts rejected",
          lambda: C.serialize_chairman_context(RICH, THREE + [A_QUANT]), C.CouncilValidationError)
    check("zero analysts rejected",
          lambda: C.serialize_chairman_context(RICH, []), C.CouncilValidationError)
    check("non-list rejected",
          lambda: C.serialize_chairman_context(RICH, A_QUANT), C.CouncilValidationError)
    check("wrong element type rejected",
          lambda: C.serialize_chairman_context(RICH, [A_QUANT, A_CONTEXT, {"p": 1}]),
          C.CouncilValidationError)

    print("\n=== CHAIRMAN: analysts are anonymised ===")
    ch = C.serialize_chairman_context(RICH, THREE)
    check("labelled Analyst A/B/C",
          lambda: _assert(all(l in ch for l in ("Analyst A", "Analyst B", "Analyst C"))))
    check("seat name 'quant' never appears", lambda: _assert("quant" not in ch.lower()))
    check("seat name 'context' never appears as a label",
          lambda: _assert("analyst_name" not in ch and "Context Analyst" not in ch))
    check("seat name 'market' not used as a label",
          lambda: _assert("Market Skeptic" not in ch))
    check("each analyst's probabilities included",
          lambda: _assert("0.4400" in ch and "0.3400" in ch and "0.5200" in ch))
    check("each analyst's outcome included",
          lambda: _assert(ch.count("calls it") == 3))
    check("each analyst's confidence included", lambda: _assert(ch.count("stated confidence") == 3))
    check("evidence included",
          lambda: _assert("an edge of +0.11" in ch and "goalkeeper confirmed unavailable" in ch))
    check("uncertainties included",
          lambda: _assert("rest days not supplied" in ch and "gap unexplained" in ch))

    print("\n=== CHAIRMAN: raw specialist-only data is NOT leaked ===")
    check("raw team news NOT independently exposed", lambda: _assert("Flekken" not in ch))
    check("raw away news NOT exposed", lambda: _assert("Palmer" not in ch))
    check("raw expected goals NOT exposed (1.66/1.55 only via quoted evidence)",
          lambda: _assert(ch.count("1.66") <= 1 and "EXPECTED GOALS" not in ch))
    check("raw form ratings NOT exposed", lambda: _assert("+3.000" not in ch))
    check("raw bookmaker price NOT exposed", lambda: _assert("0.3354" not in ch))
    check("raw kalshi price NOT exposed", lambda: _assert("0.2650" not in ch))
    check("no BOOKMAKER section", lambda: _assert("BOOKMAKER" not in ch))
    check("no TEAM NEWS section", lambda: _assert("TEAM NEWS" not in ch))
    check("baseline model probabilities ARE included",
          lambda: _assert("0.4200" in ch and "BASELINE" in ch))
    check("states it has nothing else",
          lambda: _assert("no information beyond the above" in ch))

    print("\n=== CHAIRMAN: payload -> CouncilPrediction ===")
    cpred = C.council_from_payload(C._extract_json(CHAIR_GOOD), THREE)
    check("becomes a CouncilPrediction",
          lambda: _assert(isinstance(cpred, C.CouncilPrediction)))
    check("probabilities carried", lambda: _assert(cpred.probabilities() == (.43, .26, .31)))
    check("consensus carried", lambda: _assert(cpred.consensus_score == 0.45))
    check("confidence mapped", lambda: _assert(cpred.confidence == 0.55))
    check("disagreement prose -> bool True",
          lambda: _assert(cpred.major_disagreement is True))
    check("disagreement prose preserved",
          lambda: _assert("confirmed absence" in cpred.disagreement_note))
    check("'none' -> bool False", lambda: _assert(
        C.council_from_payload({**C._extract_json(CHAIR_GOOD),
                                "major_disagreement": "none"}, THREE).major_disagreement is False))
    check("empty string -> bool False", lambda: _assert(
        C.council_from_payload({**C._extract_json(CHAIR_GOOD),
                                "major_disagreement": ""}, THREE).major_disagreement is False))
    check("the original three analysts are carried",
          lambda: _assert(cpred.analyst_predictions == THREE))
    check("carries them by identity, not copies",
          lambda: _assert(cpred.analyst_predictions[0] is A_QUANT))

    print("\n=== CHAIRMAN: invalid payloads are rejected, not repaired ===")
    check("consensus below 0", lambda: C.council_from_payload(
        {**C._extract_json(CHAIR_GOOD), "consensus_score": -0.2}, THREE),
        C.CouncilValidationError)
    check("consensus above 1", lambda: C.council_from_payload(
        {**C._extract_json(CHAIR_GOOD), "consensus_score": 1.4}, THREE),
        C.CouncilValidationError)
    check("simplex that does not sum to 1", lambda: C.council_from_payload(
        {**C._extract_json(CHAIR_GOOD), "home_probability": .9}, THREE),
        C.CouncilValidationError)
    check("probability above 1", lambda: C.council_from_payload(
        {**C._extract_json(CHAIR_GOOD), "draw_probability": 1.2}, THREE),
        C.CouncilValidationError)
    check("bad outcome", lambda: C.council_from_payload(
        {**C._extract_json(CHAIR_GOOD), "predicted_outcome": "HOME WIN"}, THREE),
        C.CouncilValidationError)
    check("missing consensus_score", lambda: C.council_from_payload(
        {k: v for k, v in C._extract_json(CHAIR_GOOD).items() if k != "consensus_score"},
        THREE), C.CouncilParseError)
    check("unknown confidence label", lambda: C.council_from_payload(
        {**C._extract_json(CHAIR_GOOD), "confidence": "very high"}, THREE),
        C.CouncilParseError)
    check("malformed JSON", lambda: C._extract_json("The council has decided."),
          C.CouncilParseError)
    check("payload with wrong analyst count", lambda: C.council_from_payload(
        C._extract_json(CHAIR_GOOD), THREE[:2]), C.CouncilValidationError)

    print("\n=== CHAIRMAN: lockdown matches every other seat ===")
    cho = C.chairman_options()
    check("no tools allowed", lambda: _assert(list(cho.allowed_tools) == []))
    check("web tools denied",
          lambda: _assert({"WebSearch", "WebFetch"} <= set(cho.disallowed_tools)))
    check("max_turns is 1", lambda: _assert(cho.max_turns == 1))
    check("permissions not bypassed", lambda: _assert(cho.permission_mode != "bypassPermissions"))
    check("external settings ignored", lambda: _assert(list(cho.setting_sources) == []))
    check("all FOUR seats share one lockdown", lambda: _assert(
        len({(tuple(o.allowed_tools), tuple(o.disallowed_tools), o.max_turns,
              o.permission_mode, tuple(o.setting_sources))
             for o in (C.quant_options(), C.context_options(),
                       C.market_options(), C.chairman_options())}) == 1))

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
        check("chairman honours the env var",
              lambda: _assert(C.chairman_options().model == "claude-haiku-4-5"))
        check("chairman's subagent honours it",
              lambda: _assert(C.chairman_options().agents["chairman"].model == "claude-haiku-4-5"))
        check("chairman AgentDefinition pins no model",
              lambda: _assert(C.CHAIRMAN.model is None))

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
    check("predict / predict_async exist",
          lambda: _assert(callable(C.predict) and asyncio.iscoroutinefunction(C.predict_async)))
    check("predict rejects a non-MatchContext",
          lambda: asyncio.run(C.predict_async("nope")), C.CouncilValidationError)
    check("sync wrapper refuses a running loop",
          lambda: asyncio.run(_call_from_loop()), C.CouncilSDKError)

    orchestration_tests()

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


async def _call_from_loop():
    return C.run_quant_analyst(BARE)   # must raise, not deadlock


# ---------------------------------------------------------------- orchestration, fully mocked
class _Spy:
    """Stands in for the four seats. Records call order, timing and arguments.

    NOTHING here touches the Agent SDK - every 'seat' is a local coroutine, so the whole
    orchestration suite runs offline and for free.
    """
    def __init__(self):
        self.events = []          # (event, name, monotonic)
        self.ctx_seen = []
        self.chair_args = None
        self.fail = {}            # name -> exception to raise
        self.delay = 0.05

    def _stamp(self, ev, name):
        self.events.append((ev, name, time.monotonic()))

    def seat(self, name):
        async def run(context):
            self._stamp("start", name)
            self.ctx_seen.append((name, context))
            await asyncio.sleep(self.delay)      # simulate a subprocess round trip
            if name in self.fail:
                self._stamp("fail", name)
                raise self.fail[name]
            self._stamp("end", name)
            return C.AnalystPrediction(
                analyst_name=name, home_probability=.4, draw_probability=.3,
                away_probability=.3, predicted_outcome=C.HOME, confidence=.5,
                evidence=[f"{name} evidence"], uncertainties=[f"{name} uncertainty"])
        return run

    def chairman(self):
        async def run(context, analysts):
            self._stamp("start", "chairman")
            self.chair_args = (context, list(analysts))
            await asyncio.sleep(0.01)
            if "chairman" in self.fail:
                raise self.fail["chairman"]
            self._stamp("end", "chairman")
            return C.CouncilPrediction(
                home_probability=.41, draw_probability=.28, away_probability=.31,
                predicted_outcome=C.HOME, confidence=.25, consensus_score=.4,
                major_disagreement=True, disagreement_note="mocked",
                analyst_predictions=list(analysts))
        return run

    def install(self):
        self.saved = (C.run_quant_analyst_async, C.run_context_analyst_async,
                      C.run_market_skeptic_async, C.run_chairman_async)
        C.run_quant_analyst_async = self.seat("quant")
        C.run_context_analyst_async = self.seat("context")
        C.run_market_skeptic_async = self.seat("market")
        C.run_chairman_async = self.chairman()
        return self

    def restore(self):
        (C.run_quant_analyst_async, C.run_context_analyst_async,
         C.run_market_skeptic_async, C.run_chairman_async) = self.saved

    def at(self, ev, name):
        return next(t for e, n, t in self.events if e == ev and n == name)

    def names(self, ev):
        return [n for e, n, _ in self.events if e == ev]


def orchestration_tests():
    CTX = C.MatchContext("Brentford", "Chelsea", model_home=.42, model_draw=.26, model_away=.32)

    print("\n=== ORCHESTRATION: the happy path (all mocked, no network) ===")
    spy = _Spy().install()
    try:
        res = asyncio.run(C.predict_async(CTX))
        check("returns a CouncilPrediction",
              lambda: _assert(isinstance(res, C.CouncilPrediction)))
        check("all three specialists ran",
              lambda: _assert(set(spy.names("start")) >= {"quant", "context", "market"}))
        check("chairman ran once", lambda: _assert(spy.names("start").count("chairman") == 1))
        check("each specialist called exactly once",
              lambda: _assert(all(spy.names("start").count(n) == 1
                                  for n in ("quant", "context", "market"))))
        check("all three got the SAME context object",
              lambda: _assert(all(c is CTX for _, c in spy.ctx_seen) and len(spy.ctx_seen) == 3))
        check("chairman got the same context too",
              lambda: _assert(spy.chair_args[0] is CTX))
        check("chairman got exactly three analysts",
              lambda: _assert(len(spy.chair_args[1]) == 3))
        check("chairman got quant/context/market ORDER",
              lambda: _assert([a.analyst_name for a in spy.chair_args[1]]
                              == ["quant", "context", "market"]))
        check("result carries the three analysts",
              lambda: _assert(len(res.analyst_predictions) == 3))

        print("\n=== ORCHESTRATION: concurrency, not sequence ===")
        last_start = max(spy.at("start", n) for n in ("quant", "context", "market"))
        first_end = min(spy.at("end", n) for n in ("quant", "context", "market"))
        check("every specialist STARTED before any FINISHED (truly concurrent)",
              lambda: _assert(last_start < first_end))
        span = max(spy.at("end", n) for n in ("quant", "context", "market")) - \
            min(spy.at("start", n) for n in ("quant", "context", "market"))
        check(f"wall time ~one delay, not three ({span*1000:.0f}ms vs {spy.delay*3*1000:.0f}ms serial)",
              lambda: _assert(span < spy.delay * 2))
        check("chairman started AFTER the last specialist finished",
              lambda: _assert(spy.at("start", "chairman") >=
                              max(spy.at("end", n) for n in ("quant", "context", "market"))))
    finally:
        spy.restore()

    print("\n=== ORCHESTRATION: a specialist failure stops everything ===")
    for bad in ("quant", "context", "market"):
        spy = _Spy().install()
        spy.fail[bad] = C.CouncilSDKError(f"{bad} exploded")
        try:
            try:
                asyncio.run(C.predict_async(CTX))
                print(f"  FAIL {bad} failure not raised"); globals()['FAIL'] = FAIL + 1
            except C.CouncilSDKError as e:
                msg = str(e)
                ok = (bad in msg and "Chairman was NOT run" in msg
                      and "chairman" not in spy.names("start"))
                print(f"  {'ok  ' if ok else 'FAIL'} {bad} failure: names the seat, "
                      f"chairman skipped")
                globals()['PASS' if ok else 'FAIL'] = globals()['PASS' if ok else 'FAIL'] + 1
                if bad == "market":
                    print(f"       message: {msg[:110]}...")
        finally:
            spy.restore()

    spy = _Spy().install()
    spy.fail["quant"] = RuntimeError("boom")
    try:
        try:
            asyncio.run(C.predict_async(CTX))
        except C.CouncilSDKError as e:
            check("original error preserved as __cause__",
                  lambda: _assert(isinstance(e.__cause__, RuntimeError)))
            check("no averaging / no substitution mentioned",
                  lambda: _assert("no substitute was used" in str(e)))
    finally:
        spy.restore()

    print("\n=== ORCHESTRATION: chairman failure propagates ===")
    spy = _Spy().install()
    spy.fail["chairman"] = RuntimeError("chair exploded")
    try:
        try:
            asyncio.run(C.predict_async(CTX))
            check("chairman failure raised", lambda: _assert(False))
        except C.CouncilSDKError as e:
            check("chairman failure is a CouncilSDKError", lambda: _assert(True))
            check("names the chairman", lambda: _assert("Chairman failed" in str(e)))
            check("cause preserved", lambda: _assert(isinstance(e.__cause__, RuntimeError)))
            check("all three specialists still ran first",
                  lambda: _assert(len(spy.names("end")) >= 3))
    finally:
        spy.restore()

    print("\n=== ORCHESTRATION: no retries, no writes ===")
    spy = _Spy().install()
    spy.fail["context"] = C.CouncilSDKError("once")
    try:
        try:
            asyncio.run(C.predict_async(CTX))
        except C.CouncilSDKError:
            pass
        check("failing seat was attempted exactly ONCE (no silent retry)",
              lambda: _assert(spy.names("start").count("context") == 1))
    finally:
        spy.restore()

    spy = _Spy().install()
    before = set(os.listdir(os.path.dirname(os.path.abspath(__file__))))
    try:
        asyncio.run(C.predict_async(CTX))
        after = set(os.listdir(os.path.dirname(os.path.abspath(__file__))))
        check("predict() created NO files", lambda: _assert(before == after))
        check("no council_ledger.csv written", lambda: _assert(
            not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "council_ledger.csv"))))
    finally:
        spy.restore()

    print("\n=== ORCHESTRATION: sync wrapper ===")
    spy = _Spy().install()
    try:
        r = C.predict(CTX)
        check("predict() returns a CouncilPrediction",
              lambda: _assert(isinstance(r, C.CouncilPrediction)))
        check("predict() refuses a running loop",
              lambda: asyncio.run(_predict_from_loop(CTX)), C.CouncilSDKError)
    finally:
        spy.restore()


async def _predict_from_loop(ctx):
    return C.predict(ctx)


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

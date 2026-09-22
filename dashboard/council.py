"""
council.py — the AI Council: an INDEPENDENT experimental prediction layer.

WHAT THIS IS
A second opinion on a fixture, produced by a panel of analysts reasoning over the same
evidence the desk already has. It is deliberately NOT part of the priced model.

WHAT THIS IS NOT
It does not touch Dixon-Coles, the 0.75/0.25 supremacy blend, or any existing ledger. The
production probability is still

    p = wdc * dixon_coles + wsup * supremacy_form            (build_dashboard.py)

and nothing in this file may change it. The Council produces its own vector, carried beside
the model in the payload, and is graded on its own ledger with the same Brier and RPS the
model is scored with. If it ever beats the model out-of-sample, that is a finding to argue
about with evidence - not a reason for this module to reach into the blend.

WHY THE SEPARATION IS STRICT
This project has killed three plausible ideas at the validation gate: the draw correction
(t=-0.86 over 10 seasons), the 60-75% confidence band (z=-1.59, n=29, and it turned out to be
five corner props), and cold-start K as an accuracy gain (t=0.83). Each looked right before it
was measured. A layer that can quietly alter the baseline it is measured against cannot be
measured at all, so the Council stays outside the blend until its own ledger says otherwise.

STATUS: all four seats live. `predict()` runs the full in-memory pipeline - three specialists
concurrently, then the Chairman. It writes nothing: no files, no ledger, no cache.
"""
import asyncio
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultMessage,
    TextBlock,
    query,
)

# Outcome labels. These are the Council's own vocabulary; the existing ledger uses the
# single letters "H"/"D"/"A" and the two are mapped at the ledger boundary, not here.
HOME, DRAW, AWAY = "HOME", "DRAW", "AWAY"
OUTCOMES = (HOME, DRAW, AWAY)

# How far a probability triple may drift from summing to 1 before it is rejected. Wide enough
# for a model that rounds to whole percentages (build_dashboard stores 47/26/27), tight enough
# that a genuinely malformed vector is caught.
SUM_TOLERANCE = 0.02


# --------------------------------------------------------------------------- validation
class CouncilValidationError(ValueError):
    """A prediction that cannot be scored. Raised early, on construction."""


def check_probability(value, label):
    """A single probability must be a real number in [0, 1]."""
    if value is None:
        raise CouncilValidationError(f"{label} is missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CouncilValidationError(f"{label} must be a number, got {type(value).__name__}")
    if not (0.0 <= float(value) <= 1.0):
        raise CouncilValidationError(f"{label} must be between 0 and 1, got {value!r}")
    return float(value)


def check_simplex(home, draw, away, who="prediction"):
    """H/D/A must each be a probability AND sum to ~1.

    Checked together rather than separately because three individually-valid numbers can still
    be an unusable forecast: 0.9/0.9/0.9 passes every per-field check and scores as nonsense.
    """
    h = check_probability(home, f"{who}.home_probability")
    d = check_probability(draw, f"{who}.draw_probability")
    a = check_probability(away, f"{who}.away_probability")
    total = h + d + a
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise CouncilValidationError(
            f"{who} probabilities must sum to ~1 (tolerance {SUM_TOLERANCE}), got "
            f"{h:.4f} + {d:.4f} + {a:.4f} = {total:.4f}")
    return h, d, a


def check_outcome(value, who="prediction"):
    """predicted_outcome must be exactly one of HOME / DRAW / AWAY."""
    if value not in OUTCOMES:
        raise CouncilValidationError(
            f"{who}.predicted_outcome must be one of {OUTCOMES}, got {value!r}")
    return value


def implied_outcome(home, draw, away):
    """The outcome the probabilities themselves point at - ties break HOME > DRAW > AWAY.

    Useful for checking that a stated outcome agrees with the numbers beside it, which is a
    real failure mode for a language model asked to produce both.
    """
    return max(zip((home, draw, away), OUTCOMES), key=lambda t: t[0])[1]


# --------------------------------------------------------------------------- the context
@dataclass
class MatchContext:
    """Everything the Council is shown about one fixture.

    Only the two team names are required. EVERY other field is Optional and defaults to None,
    because the desk genuinely does not always have them: books open about a week out (13 of
    380 fixtures were priced at the time of writing), Kalshi lists EPL later still, and team
    news is empty until the pressers. An analyst must be able to reason from a partial context
    and say what it could not see - a context that refuses to exist without a market price
    would simply never be built for most fixtures.
    """
    home_team: str
    away_team: str
    kickoff: Optional[str] = None

    # the production model's own 1X2, as probabilities in [0, 1]
    model_home: Optional[float] = None
    model_draw: Optional[float] = None
    model_away: Optional[float] = None

    # the Dixon-Coles expected goals behind those probabilities
    expected_home_goals: Optional[float] = None
    expected_away_goals: Optional[float] = None

    # sportsbook implied probabilities (overround NOT removed unless the caller did it)
    market_home: Optional[float] = None
    market_draw: Optional[float] = None
    market_away: Optional[float] = None

    # Kalshi exchange mid-prices; cents ARE the implied probability, so /100 before storing
    kalshi_home: Optional[float] = None
    kalshi_draw: Optional[float] = None
    kalshi_away: Optional[float] = None

    # recent-form ratings (the supremacy input) and free-text team news
    home_form: Optional[float] = None
    away_form: Optional[float] = None
    # News arrives from the desk as a LIST of items, and as prose when a human writes it, so
    # both are accepted. `[]` and `None` both mean "no news supplied" - see news_items().
    home_news: Optional[Union[str, List[str]]] = None
    away_news: Optional[Union[str, List[str]]] = None

    def has_model(self):
        return None not in (self.model_home, self.model_draw, self.model_away)

    def has_market(self):
        return None not in (self.market_home, self.market_draw, self.market_away)

    def news_items(self, side):
        """Team news for 'home'/'away' as a list of strings. Empty list when none was supplied.

        Tolerates the three shapes the desk actually produces: a sentence, a list of strings,
        or a list of {"what": ...} rows straight out of the FPL feed. An empty list and None
        mean the same thing - nothing was supplied - and neither is allowed to read as
        "confirmed no absences", which is a different and much stronger claim.
        """
        raw = self.home_news if side == "home" else self.away_news
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw.strip()] if raw.strip() else []
        out = []
        for item in raw:
            if isinstance(item, dict):
                item = item.get("what") or item.get("text") or item.get("note") or ""
            item = str(item).strip()
            if item:
                out.append(item)
        return out

    def has_news(self):
        return bool(self.news_items("home") or self.news_items("away"))

    def has_kalshi(self):
        return None not in (self.kalshi_home, self.kalshi_draw, self.kalshi_away)

    def fixture_key(self):
        """Matches the existing ledger convention: stable across kickoff changes.

        RULE 5 keys on "{home}|{away}" precisely because TV picks move kickoff times, and a
        key containing the time would split one fixture into two rows.
        """
        return f"{self.home_team}|{self.away_team}"

    def missing(self):
        """Which optional blocks are absent - so a prompt can say so rather than imply zero."""
        gaps = []
        if not self.has_model():
            gaps.append("model")
        if not self.has_market():
            gaps.append("market")
        if not self.has_kalshi():
            gaps.append("kalshi")
        if self.home_form is None and self.away_form is None:
            gaps.append("form")
        if not self.has_news():
            gaps.append("news")
        return gaps


# --------------------------------------------------------------------- one analyst's view
@dataclass
class AnalystPrediction:
    """One panel member's forecast, validated on construction.

    `evidence` and `uncertainties` are lists rather than prose on purpose: the point of a
    panel is to see WHERE members disagree, and that is far easier to compare as itemised
    claims than as paragraphs. `uncertainties` is not optional politeness - an analyst shown a
    fixture with no market price and no team news needs somewhere to say so.
    """
    analyst_name: str
    home_probability: float
    draw_probability: float
    away_probability: float
    predicted_outcome: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not str(self.analyst_name or "").strip():
            raise CouncilValidationError("analyst_name must not be empty")
        who = f"analyst[{self.analyst_name}]"
        self.home_probability, self.draw_probability, self.away_probability = check_simplex(
            self.home_probability, self.draw_probability, self.away_probability, who)
        self.predicted_outcome = check_outcome(self.predicted_outcome, who)
        self.confidence = check_probability(self.confidence, f"{who}.confidence")

    def probabilities(self):
        return (self.home_probability, self.draw_probability, self.away_probability)

    def agrees_with_own_numbers(self):
        """Does the stated outcome match the highest probability? Not enforced - an analyst is
        allowed to argue its pick against its own spread - but the disagreement is worth
        surfacing rather than silently averaging away."""
        return self.predicted_outcome == implied_outcome(*self.probabilities())


# ------------------------------------------------------------------- the council's verdict
@dataclass
class CouncilPrediction:
    """The panel's aggregate forecast plus the individual views it came from.

    `consensus_score` and `major_disagreement` are kept as fields rather than computed here so
    that whatever aggregation is eventually chosen owns its own definition of agreement. This
    skeleton takes no position on how to combine analysts - averaging probabilities, taking a
    median, or weighting by track record are all different models, and picking one now would
    bake in an untested assumption.
    """
    home_probability: float
    draw_probability: float
    away_probability: float
    predicted_outcome: str
    confidence: float
    consensus_score: float
    major_disagreement: bool
    # The Chairman reports the disagreement as PROSE ("the market seat's 10pt gap is
    # unexplained"), while this field has always been a bool that a ledger can group on. Both
    # are worth having, so the text is kept rather than collapsed away - throwing it out would
    # leave "True" with no record of what the panel actually split over.
    disagreement_note: str = ""
    analyst_predictions: List[AnalystPrediction] = field(default_factory=list)

    def __post_init__(self):
        self.home_probability, self.draw_probability, self.away_probability = check_simplex(
            self.home_probability, self.draw_probability, self.away_probability, "council")
        self.predicted_outcome = check_outcome(self.predicted_outcome, "council")
        self.confidence = check_probability(self.confidence, "council.confidence")
        self.consensus_score = check_probability(self.consensus_score, "council.consensus_score")
        if not isinstance(self.major_disagreement, bool):
            raise CouncilValidationError("council.major_disagreement must be a bool")
        for a in self.analyst_predictions:
            if not isinstance(a, AnalystPrediction):
                raise CouncilValidationError(
                    f"analyst_predictions must contain AnalystPrediction, got "
                    f"{type(a).__name__}")

    def probabilities(self):
        return (self.home_probability, self.draw_probability, self.away_probability)

    def as_row(self):
        """The flat shape council_ledger.py stores. Kept here so the ledger never has to know
        the dataclass internals."""
        return {"council_home": round(self.home_probability, 6),
                "council_draw": round(self.draw_probability, 6),
                "council_away": round(self.away_probability, 6),
                "consensus_score": round(self.consensus_score, 6)}


# The Council's public entry point - predict() / predict_async() - is defined at the END
# of this module, after every seat it orchestrates exists.
# ===========================================================================================
# THE QUANT ANALYST — the first runtime Council member.
#
# One panel seat, deliberately narrow: it sees NUMBERS ONLY. No team news, no tactics, no
# recollection of who Arsenal signed. That restriction is the point - a panel is only worth
# more than one opinion if its members are actually looking at different things, and the
# cheapest way to lose that is to let every seat fall back on the same diffuse football
# knowledge. The Context Analyst (news, absences) and the Market Skeptic (prices) are separate
# seats precisely so this one cannot quietly do their jobs badly.
#
# It is also the seat most at risk of hallucinated authority: asked about a fixture, a model
# will happily produce an injury list. So the prompt forbids it, the tool list is empty, and
# the context handed over contains only quantitative fields.
# ===========================================================================================

# The JSON contract says confidence is "low"/"medium"/"high"; AnalystPrediction requires a
# probability. Both are deliberate - a categorical label is what a language model reports
# reliably, a number is what a ledger can score - so they are mapped HERE, explicitly, at the
# boundary. An unrecognised label raises rather than defaulting to something plausible: a
# silent fallback would put an invented confidence into the evidence log.
CONFIDENCE_LEVELS = {"low": 0.25, "medium": 0.55, "high": 0.85}


class CouncilSDKError(RuntimeError):
    """The Agent SDK failed to produce a usable response. Always raised `from` the cause."""


class CouncilParseError(CouncilSDKError):
    """Claude replied, but not with the JSON object the contract requires."""


# Which model the panel runs on is a deployment decision, not a property of the seat, so it is
# NOT baked into the AgentDefinition. A pinned model would also be a quiet cost decision: the
# Council will eventually be four seats across ten fixtures every refresh, and that multiplies.
#
# Resolved at CALL time rather than import time - a module-level lookup would freeze whatever
# the environment happened to be when council.py was first imported, which is exactly the kind
# of thing that makes a test pass and production behave differently.
COUNCIL_MODEL_ENV = "AI_COUNCIL_MODEL"


def council_model():
    """The model every Council seat should use, or None to accept the SDK's own default.

    An empty or whitespace-only value counts as unset: `AI_COUNCIL_MODEL=` in a shell profile
    should mean "I have not chosen one", not "use a model named empty string".
    """
    return (os.environ.get(COUNCIL_MODEL_ENV) or "").strip() or None


QUANT_ANALYST_NAME = "quant"

QUANT_SYSTEM_PROMPT = """\
You are a Premier League quantitative forecasting analyst.

You may ONLY use quantitative information supplied in MatchContext.

Focus on:
- existing model H/D/A probabilities
- expected goals
- home advantage if supplied
- recent quantitative form if supplied
- other numerical team-strength information explicitly present

You must NOT:
- invent injuries
- invent tactical information
- invent team news
- browse the web
- rely on general football knowledge not contained in the input

If information is missing, state that explicitly.

Return a calibrated Home/Draw/Away probability vector.

OUTPUT CONTRACT
Reply with a single JSON object and nothing else - no prose before or after, no markdown
fence. Exactly these fields:

{
  "analyst_name": "quant",
  "home_probability": float,
  "draw_probability": float,
  "away_probability": float,
  "predicted_outcome": "HOME" | "DRAW" | "AWAY",
  "confidence": "low" | "medium" | "high",
  "evidence": [string, ...],
  "uncertainties": [string, ...]
}

Rules for those fields:
- the three probabilities are decimals in [0, 1] and must sum to 1.00
- predicted_outcome is upper-case and is one of HOME, DRAW, AWAY
- every entry in "evidence" must cite a number that appears in the context you were given
- every context field listed as missing belongs in "uncertainties"
- do not round a probability to 0 or 1; a football match is never certain
"""

# The canonical definition of this panel seat. Its `prompt` is the single source of truth for
# the system prompt below, so the two can never drift apart. Registering it as an
# AgentDefinition is what lets a future Chairman DELEGATE to this seat; for a single analyst
# we drive it directly, which costs one turn instead of two.
QUANT_ANALYST = AgentDefinition(
    description=(
        "Quantitative Premier League forecaster. Reads only the numbers in MatchContext - "
        "model probabilities, expected goals, form ratings - and is forbidden from using team "
        "news, tactics, or outside football knowledge."
    ),
    prompt=QUANT_SYSTEM_PROMPT,
    tools=[],                       # no tools at all: this seat reads numbers and returns JSON
    model=None,                     # resolved per call - see council_model()
)

# Belt and braces. `tools=[]` on the definition and `allowed_tools=[]` on the options should
# already mean nothing is callable; naming the tools that could reach outside the context
# makes the intent explicit and survives a future change to either default.
FORBIDDEN_TOOLS = ["WebSearch", "WebFetch", "Bash", "Read", "Write", "Edit",
                   "Glob", "Grep", "NotebookEdit", "Task"]


def has_usable_market(context) -> bool:
    """Does this fixture have ANY market price for the Market Skeptic to work from?

    RULE 40. The Market seat's whole job is to compare the model against a price and quantify
    the gap. With neither a bookmaker nor an exchange quote it has nothing to be skeptical
    ABOUT: it falls back to the baseline, reports low confidence, and the panel silently
    becomes two opinions rather than three - while still costing four Claude calls.

    That is not hypothetical. Tottenham|Aston Villa was locked with no bookmaker price, so its
    row carries no market_rps and is excluded from every Council-vs-market comparison. One of
    three graded forecasts is unusable for that question.

    Either source is enough - they are independent, and the seat handles having only one
    (it caps its own confidence and says so). Only the total absence of both is disqualifying.
    """
    if not isinstance(context, MatchContext):
        raise CouncilValidationError(
            f"expected MatchContext, got {type(context).__name__}")
    return context.has_market() or context.has_kalshi()


def market_sources(context) -> list:
    """Which market sources this fixture actually has, for reporting."""
    out = []
    if context.has_market():
        out.append("bookmaker")
    if context.has_kalshi():
        out.append("exchange")
    return out


def serialize_quant_context(context: MatchContext) -> str:
    """The quantitative half of a MatchContext, as text for the prompt.

    News fields are NOT included - not merely unused, absent. An analyst told to ignore team
    news while being shown it is being asked to prove a negative; leaving it out of the
    payload makes the restriction structural instead of aspirational.

    Fields the desk does not have are listed as missing rather than omitted, so the analyst
    can put them in `uncertainties` instead of quietly assuming a value.
    """
    if not isinstance(context, MatchContext):
        raise CouncilValidationError(
            f"expected MatchContext, got {type(context).__name__}")

    L = [f"FIXTURE: {context.home_team} (home) v {context.away_team} (away)"]
    if context.kickoff:
        L.append(f"KICKOFF: {context.kickoff}")

    L.append("\nEXISTING MODEL (Dixon-Coles blended with recent-form supremacy):")
    if context.has_model():
        L.append(f"  P(home) {context.model_home:.4f}   "
                 f"P(draw) {context.model_draw:.4f}   P(away) {context.model_away:.4f}")
    else:
        L.append("  MISSING - no model probabilities supplied")

    L.append("\nEXPECTED GOALS:")
    if context.expected_home_goals is not None or context.expected_away_goals is not None:
        h = ("?" if context.expected_home_goals is None
             else f"{context.expected_home_goals:.2f}")
        a = ("?" if context.expected_away_goals is None
             else f"{context.expected_away_goals:.2f}")
        L.append(f"  home {h}   away {a}")
    else:
        L.append("  MISSING - no expected goals supplied")

    L.append("\nRECENT FORM (goal-supremacy rating; positive favours that side):")
    if context.home_form is not None or context.away_form is not None:
        h = "?" if context.home_form is None else f"{context.home_form:+.3f}"
        a = "?" if context.away_form is None else f"{context.away_form:+.3f}"
        L.append(f"  {context.home_team} {h}   {context.away_team} {a}")
    else:
        L.append("  MISSING - no form ratings supplied")

    gaps = [g for g in context.missing() if g != "news"]
    L.append("\nMISSING FROM THIS CONTEXT: " + (", ".join(gaps) if gaps else "nothing"))
    L.append("Market and exchange prices are deliberately withheld from this seat - another "
             "analyst covers them. Do not guess at them.")
    return "\n".join(L)


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse the one JSON object the contract asks for.

    A ```json fence is stripped if present. That is parsing, not repair: the fence is a
    transport artifact around an otherwise conforming object. Nothing inside the object is
    touched - a malformed or incomplete body raises, and the caller sees it.
    """
    if not isinstance(text, str) or not text.strip():
        raise CouncilParseError("the analyst returned no text at all")
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    if not body.startswith("{"):
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            raise CouncilParseError(
                f"no JSON object in the analyst's reply; got: {text.strip()[:200]!r}")
        body = body[start:end + 1]
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise CouncilParseError(
            f"the analyst's reply is not valid JSON ({e}); got: {body[:200]!r}") from e
    if not isinstance(obj, dict):
        raise CouncilParseError(
            f"expected a JSON object, got {type(obj).__name__}: {body[:200]!r}")
    return obj


def analyst_from_payload(obj: Dict[str, Any], name: str = QUANT_ANALYST_NAME
                         ) -> AnalystPrediction:
    """Turn a parsed contract object into a validated AnalystPrediction.

    Deliberately does NOT normalise: a probability vector that fails the simplex check raises
    CouncilValidationError from AnalystPrediction's own __post_init__ and the caller sees
    exactly what the model said. Rescaling a bad vector to sum to 1 would hide the one signal
    that says this answer should not be trusted.
    """
    missing = [k for k in ("home_probability", "draw_probability", "away_probability",
                           "predicted_outcome", "confidence") if k not in obj]
    if missing:
        raise CouncilParseError(
            f"analyst payload is missing required field(s): {', '.join(missing)}")

    raw_conf = obj["confidence"]
    if isinstance(raw_conf, str):
        key = raw_conf.strip().lower()
        if key not in CONFIDENCE_LEVELS:
            raise CouncilParseError(
                f"confidence must be one of {sorted(CONFIDENCE_LEVELS)}, got {raw_conf!r}")
        conf = CONFIDENCE_LEVELS[key]
    else:
        conf = raw_conf            # a number goes straight to check_probability, which judges it

    def _listify(key):
        v = obj.get(key) or []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            raise CouncilParseError(f"{key} must be a list of strings, got {type(v).__name__}")
        return [str(x) for x in v]

    # AnalystPrediction.__post_init__ validates; any failure propagates unchanged.
    return AnalystPrediction(
        analyst_name=str(obj.get("analyst_name") or name),
        home_probability=obj["home_probability"],
        draw_probability=obj["draw_probability"],
        away_probability=obj["away_probability"],
        predicted_outcome=obj["predicted_outcome"],
        confidence=conf,
        evidence=_listify("evidence"),
        uncertainties=_listify("uncertainties"),
    )


def quant_options(model: Optional[str] = None) -> ClaudeAgentOptions:
    """Locked-down options for this seat. Separated so a test can assert on them without
    running anything.

    `model` falls back to AI_COUNCIL_MODEL, and then to None - which hands the choice to the
    Agent SDK's own default rather than this file guessing one.
    """
    chosen = model or council_model()
    return ClaudeAgentOptions(
        model=chosen,
        system_prompt=QUANT_ANALYST.prompt,   # single source of truth - see QUANT_ANALYST
        allowed_tools=[],                     # nothing callable
        disallowed_tools=list(FORBIDDEN_TOOLS),
        permission_mode="default",            # never bypassPermissions for a panel seat
        max_turns=1,                          # one question, one JSON answer
        setting_sources=[],                   # ignore user/project settings - no tool leakage
        # the registered seat runs on the same model the query does
        agents={QUANT_ANALYST_NAME: replace(QUANT_ANALYST, model=chosen)},
    )


async def _ask_analyst_async(prompt: str, options: ClaudeAgentOptions,
                             fixture: str, name: str) -> AnalystPrediction:
    """One panel seat, one question, one validated answer.

    Shared by every analyst ON PURPOSE. The tool lockdown, the single turn, the JSON parsing
    and the refusal to repair a bad probability vector are the safeguards that make a seat
    trustworthy; giving each seat its own copy is how they quietly drift apart until one of
    them is the weak one. Seats differ in their PROMPT and their CONTEXT, not in their rules.

    Makes a real Claude request via the Agent SDK, which spawns the bundled Claude Code CLI as
    a subprocess. Errors are wrapped with the fixture attached, because a bare transport
    failure three layers down says nothing about which match it was.
    """
    chunks: List[str] = []
    result: Optional[ResultMessage] = None
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                result = message
    except ClaudeSDKError as e:
        raise CouncilSDKError(f"Agent SDK failed for {fixture} ({name}): {e}") from e

    if result is not None and getattr(result, "is_error", False):
        raise CouncilSDKError(
            f"the {name} analyst errored on {fixture}: "
            f"{getattr(result, 'result', None) or 'no detail returned'}")
    if not chunks:
        raise CouncilParseError(f"the {name} analyst returned no text for {fixture}")

    return analyst_from_payload(_extract_json("\n".join(chunks)), name)


def _sync(coro, name: str) -> AnalystPrediction:
    """Own the asyncio boundary so every caller stays synchronous.

    build_dashboard.py and the rest of the pipeline are synchronous top to bottom, and the
    Agent SDK's query() is an async generator. That boundary lives HERE, once, rather than
    turning the desk async to accommodate an experimental layer.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise CouncilSDKError(
        f"run_{name}_analyst() was called from inside a running event loop; "
        f"await run_{name}_analyst_async(context) instead")


async def run_quant_analyst_async(context: MatchContext) -> AnalystPrediction:
    """Ask the Quant Analyst about one fixture. Numbers only - see QUANT_SYSTEM_PROMPT."""
    return await _ask_analyst_async(
        serialize_quant_context(context), quant_options(),
        context.fixture_key(), QUANT_ANALYST_NAME)


def run_quant_analyst(context: MatchContext) -> AnalystPrediction:
    """Synchronous entry point for the Quant Analyst."""
    return _sync(run_quant_analyst_async(context), QUANT_ANALYST_NAME)


# ===========================================================================================
# THE CONTEXT ANALYST — the second panel seat.
#
# Where the Quant Analyst sees numbers and no words, this one sees words and almost no
# numbers. It gets the existing model's H/D/A as a BASELINE to move away from, and the team
# news, and nothing else: no expected goals, no form ratings, no market, no exchange. Two
# seats reading the same evidence would be one seat with extra billing.
#
# The failure mode here is worse than the Quant seat's. Asked about a Premier League fixture
# with no news attached, a model will reach for what it remembers - a manager, a suspension, a
# "traditionally tough away trip". All of that is untethered from the desk's data and unfalsifiable
# against it. So: the prompt forbids it, the tools are empty, and the payload states outright
# which contextual categories were NOT supplied, so silence reads as "unknown" rather than
# "nothing to report".
# ===========================================================================================

CONTEXT_ANALYST_NAME = "context"

CONTEXT_SYSTEM_PROMPT = """\
You are a Premier League team-context forecasting analyst.

You may ONLY use verified contextual information explicitly supplied in MatchContext.

Focus on:
- injuries
- suspensions
- player availability
- fixture congestion
- rest days
- rotation indicators
- recent squad changes
- manager changes if explicitly supplied
- verified team news
- other contextual information explicitly provided

You must NOT:
- browse the web
- use general Premier League knowledge
- invent injuries
- invent player availability
- invent tactical information
- assume a player is important unless the supplied context says so
- infer news that was not supplied

If contextual information is missing, say so explicitly.

HOW TO USE THE BASELINE
You are given the existing statistical model's Home/Draw/Away probabilities. Treat them as a
NEUTRAL STARTING POINT, not as something to second-guess. You are not being asked to re-rate
these teams - another analyst does that from the numbers, and you cannot see them.

- Move a probability away from the baseline ONLY where a supplied piece of context justifies
  it, and say in your evidence which item justified which direction.
- The size of the move should match the weight of the evidence. A confirmed absence of a
  first-choice goalkeeper is worth more than an unspecified knock to an unnamed squad player.
- If NO contextual information was supplied, return the baseline probabilities essentially
  unchanged, set confidence to "low", and state in uncertainties that there was insufficient
  context to justify an adjustment. Returning the baseline is a valid and correct answer -
  inventing a reason to move is not.
- A supplied item that does not bear on the result (a returning player already expected to
  start) may leave the baseline alone. Say so rather than manufacturing a nudge.
- Absence of news is NOT evidence of a full-strength squad. Treat it as unknown.

OUTPUT CONTRACT
Reply with a single JSON object and nothing else - no prose before or after, no markdown
fence. Exactly these fields:

{
  "analyst_name": "context",
  "home_probability": float,
  "draw_probability": float,
  "away_probability": float,
  "predicted_outcome": "HOME" | "DRAW" | "AWAY",
  "confidence": "low" | "medium" | "high",
  "evidence": [string, ...],
  "uncertainties": [string, ...]
}

Rules for those fields:
- the three probabilities are decimals in [0, 1] and must sum to 1.00
- predicted_outcome is upper-case and is one of HOME, DRAW, AWAY
- every entry in "evidence" must quote a context item you were actually given, or state that
  you are returning the baseline because none was given
- every contextual category listed as not supplied belongs in "uncertainties"
- do not round a probability to 0 or 1; a football match is never certain
"""

CONTEXT_ANALYST = AgentDefinition(
    description=(
        "Team-context Premier League forecaster. Reads only verified team news supplied in "
        "MatchContext and adjusts the existing model's baseline probabilities where that news "
        "justifies it. Sees no expected goals, form, market or exchange prices, and is "
        "forbidden from using remembered football knowledge."
    ),
    prompt=CONTEXT_SYSTEM_PROMPT,
    tools=[],
    model=None,                     # resolved per call - see council_model()
)

# Categories the prompt tells this seat to look for. The desk does not currently supply most
# of them, and the payload says so explicitly rather than staying silent - silence would let
# "no rest-day data" be read as "both sides are well rested", which is a much stronger claim
# than anything the desk knows. Adding a MatchContext field later removes a line from here.
CONTEXT_CATEGORIES = [
    "injuries", "suspensions", "player availability", "fixture congestion", "rest days",
    "rotation indicators", "recent squad changes", "manager changes",
]
SUPPLIED_CATEGORIES = {"injuries", "suspensions", "player availability", "verified team news"}


def serialize_team_context(context: MatchContext) -> str:
    """The contextual half of a MatchContext, as text for the prompt.

    Expected goals, form ratings, market prices and exchange quotes are all ABSENT - not
    merely unused. The Quant Analyst covers the numbers and the Market Skeptic will cover the
    prices; a seat shown everything is not a second opinion. The model's H/D/A is the one
    number that crosses over, and only as the baseline this seat is asked to move away from.
    """
    if not isinstance(context, MatchContext):
        raise CouncilValidationError(
            f"expected MatchContext, got {type(context).__name__}")

    L = [f"FIXTURE: {context.home_team} (home) v {context.away_team} (away)"]
    if context.kickoff:
        L.append(f"KICKOFF: {context.kickoff}")

    L.append("\nBASELINE FROM THE EXISTING STATISTICAL MODEL:")
    if context.has_model():
        L.append(f"  P(home) {context.model_home:.4f}   "
                 f"P(draw) {context.model_draw:.4f}   P(away) {context.model_away:.4f}")
        L.append("  Start from these. Move them only where a context item below justifies it.")
    else:
        L.append("  MISSING - no baseline probabilities supplied. Say so, and do not invent one.")

    for side, label in (("home", context.home_team), ("away", context.away_team)):
        items = context.news_items(side)
        L.append(f"\nTEAM NEWS - {label} ({'home' if side == 'home' else 'away'}):")
        if items:
            L.extend(f"  - {i}" for i in items)
        else:
            L.append("  NONE SUPPLIED. This means UNKNOWN, not 'full-strength squad'.")

    absent = [c for c in CONTEXT_CATEGORIES if c not in SUPPLIED_CATEGORIES]
    L.append("\nCONTEXT CATEGORIES NOT SUPPLIED BY THIS DESK: " + ", ".join(absent))
    L.append("Do not estimate them. List the ones that matter in your uncertainties.")
    L.append("\nWITHHELD FROM THIS SEAT: expected goals, form ratings, bookmaker odds, "
             "exchange prices. Other analysts cover those. Do not guess at them.")
    return "\n".join(L)


def context_options(model: Optional[str] = None) -> ClaudeAgentOptions:
    """Locked-down options for the Context seat - identical safeguards to the Quant seat."""
    chosen = model or council_model()
    return ClaudeAgentOptions(
        model=chosen,
        system_prompt=CONTEXT_ANALYST.prompt,
        allowed_tools=[],
        disallowed_tools=list(FORBIDDEN_TOOLS),
        permission_mode="default",
        max_turns=1,
        setting_sources=[],
        agents={CONTEXT_ANALYST_NAME: replace(CONTEXT_ANALYST, model=chosen)},
    )


async def run_context_analyst_async(context: MatchContext) -> AnalystPrediction:
    """Ask the Context Analyst about one fixture. Team news only - see CONTEXT_SYSTEM_PROMPT."""
    return await _ask_analyst_async(
        serialize_team_context(context), context_options(),
        context.fixture_key(), CONTEXT_ANALYST_NAME)


def run_context_analyst(context: MatchContext) -> AnalystPrediction:
    """Synchronous entry point for the Context Analyst."""
    return _sync(run_context_analyst_async(context), CONTEXT_ANALYST_NAME)


# ===========================================================================================
# THE MARKET SKEPTIC — the third panel seat.
#
# This seat exists because of a finding this project has now reproduced four times: the model
# does NOT beat the closing line. Blending it toward the market improved the score at every
# weight over 1,893 matches, all the way to 100% market. Every apparent edge collapsed once
# its two luckiest tickets were removed. So a panel that only ever asks "what do our numbers
# say?" is missing the one voice that has consistently been right.
#
# But it is a SKEPTIC, not a mirror. Copying the market is worthless - the desk already has
# the market. Its job is to say how far the model has wandered, and whether anything in front
# of it justifies the distance. It sees no expected goals, no form, no team news: those are
# other seats' evidence, and "the market disagrees because of an injury" is exactly the kind
# of invented explanation the role forbids.
#
# One trap this seat must not fall into: bookmaker implied probabilities usually carry
# overround and sum to MORE than 1. Treating them as probabilities without noticing is a
# calibration error, so the payload states the sum and lets the analyst judge.
# ===========================================================================================

MARKET_SKEPTIC_NAME = "market"

MARKET_SYSTEM_PROMPT = """\
You are a skeptical Premier League forecasting and calibration analyst.

Your job is to compare the existing model forecast against supplied market probabilities.

The market is not automatically correct, but a large disagreement requires strong evidence.

Focus on:
- model Home/Draw/Away probabilities
- bookmaker implied probabilities when supplied
- Kalshi/exchange probabilities when supplied
- size and direction of disagreement
- calibration and uncertainty

You must NOT:
- browse the web
- use team news
- use injuries
- use expected goals
- use form ratings
- use general Premier League knowledge
- invent explanations for why the market differs

If the market and model disagree, quantify the disagreement.
If there is no supplied evidence explaining the disagreement, be skeptical of large departures
from the market.

HOW TO FORM YOUR ANSWER
You are not here to repeat the market. The desk already has the market price; a seat that
copies it adds nothing. You are also not here to defend the model. Your output is a calibrated
third number, and you must justify where it sits using only the figures in front of you.

- State the disagreement in percentage points, per outcome, before you adjust anything.
- Landing between the model and the market is usually reasonable. Say WHY your answer sits
  where it does between them - closer to the market when the gap is large and unexplained,
  closer to the model when the gap is small or the market sources disagree with each other.
- You have NO information about why the market differs. Do not speculate about injuries, form,
  team news, or anything else you were not given. "The gap is unexplained by anything I can
  see" is the correct and complete observation.
- If bookmaker and exchange prices are BOTH supplied, compare them to each other first. Broad
  agreement between two independent sources makes a large model departure harder to justify.
  Disagreement between them is itself a reason for caution - put it in uncertainties. Do not
  simply average the two.
- If NO market prices were supplied, return the baseline model probabilities essentially
  unchanged, set confidence to "low", and state plainly that there is insufficient market
  evidence to challenge the baseline. That is the correct answer, not a failure.
- Bookmaker implied probabilities may include overround and sum to more than 1. If the payload
  says they do, note it - an un-normalised price is not a probability.

OUTPUT CONTRACT
Reply with a single JSON object and nothing else - no prose before or after, no markdown
fence. Exactly these fields:

{
  "analyst_name": "market",
  "home_probability": float,
  "draw_probability": float,
  "away_probability": float,
  "predicted_outcome": "HOME" | "DRAW" | "AWAY",
  "confidence": "low" | "medium" | "high",
  "evidence": [string, ...],
  "uncertainties": [string, ...]
}

Rules for those fields:
- the three probabilities are decimals in [0, 1] and must sum to 1.00
- predicted_outcome is upper-case and is one of HOME, DRAW, AWAY
- every entry in "evidence" must cite a number that appears in the payload you were given, or
  a difference computed from two of them
- any market source that was not supplied belongs in "uncertainties"
- do not round a probability to 0 or 1; a football match is never certain
"""

MARKET_SKEPTIC = AgentDefinition(
    description=(
        "Skeptical calibration analyst. Compares the existing model's probabilities against "
        "supplied bookmaker and exchange prices, quantifies the disagreement, and resists "
        "large unexplained departures from the market. Sees no expected goals, form, or team "
        "news, and may not invent reasons for why the market differs."
    ),
    prompt=MARKET_SYSTEM_PROMPT,
    tools=[],
    model=None,                     # resolved per call - see council_model()
)


def _delta_line(label, model_p, other_p):
    """One '+4.2 pts' row, or None when the other source is absent."""
    if other_p is None or model_p is None:
        return None
    return f"    {label:<6} model {model_p:.4f}   source {other_p:.4f}   " \
           f"model is {(model_p - other_p) * 100:+.1f} pts"


def serialize_market_context(context: MatchContext) -> str:
    """The market half of a MatchContext, as text for the prompt.

    Expected goals, form ratings and team news are ABSENT - not merely unused. This seat's
    whole value is that it CANNOT explain a price gap, so it has to report the gap honestly
    instead of narrating a cause. Handing it an injury list would let it do exactly what its
    prompt forbids.

    Differences are pre-computed. That is arithmetic on numbers already in the payload, not
    new evidence, and doing it here removes a class of error that would otherwise look like
    an opinion.
    """
    if not isinstance(context, MatchContext):
        raise CouncilValidationError(
            f"expected MatchContext, got {type(context).__name__}")

    L = [f"FIXTURE: {context.home_team} (home) v {context.away_team} (away)"]
    if context.kickoff:
        L.append(f"KICKOFF: {context.kickoff}")

    L.append("\nEXISTING MODEL (the forecast you are asked to scrutinise):")
    if context.has_model():
        L.append(f"  P(home) {context.model_home:.4f}   "
                 f"P(draw) {context.model_draw:.4f}   P(away) {context.model_away:.4f}")
    else:
        L.append("  MISSING - no model probabilities supplied. Say so; do not invent one.")

    L.append("\nBOOKMAKER IMPLIED PROBABILITIES:")
    if context.has_market():
        tot = context.market_home + context.market_draw + context.market_away
        L.append(f"  P(home) {context.market_home:.4f}   "
                 f"P(draw) {context.market_draw:.4f}   P(away) {context.market_away:.4f}")
        L.append(f"  these sum to {tot:.4f}"
                 + (f" - they carry ~{(tot - 1) * 100:.1f}% overround and are NOT normalised "
                    f"probabilities" if abs(tot - 1) > 0.005
                    else " - already normalised"))
    else:
        L.append("  NOT SUPPLIED - books often open only about a week before kickoff.")

    L.append("\nEXCHANGE (KALSHI) PROBABILITIES:")
    if context.has_kalshi():
        tot = context.kalshi_home + context.kalshi_draw + context.kalshi_away
        L.append(f"  P(home) {context.kalshi_home:.4f}   "
                 f"P(draw) {context.kalshi_draw:.4f}   P(away) {context.kalshi_away:.4f}")
        L.append(f"  these sum to {tot:.4f}")
    else:
        L.append("  NOT SUPPLIED - Kalshi lists EPL only shortly before kickoff.")

    if context.has_model() and (context.has_market() or context.has_kalshi()):
        L.append("\nDISAGREEMENT (positive = the model is HIGHER than that source):")
        if context.has_market():
            L.append("  vs bookmaker:")
            for lab, m, o in (("HOME", context.model_home, context.market_home),
                              ("DRAW", context.model_draw, context.market_draw),
                              ("AWAY", context.model_away, context.market_away)):
                line = _delta_line(lab, m, o)
                if line:
                    L.append(line)
        if context.has_kalshi():
            L.append("  vs exchange:")
            for lab, m, o in (("HOME", context.model_home, context.kalshi_home),
                              ("DRAW", context.model_draw, context.kalshi_draw),
                              ("AWAY", context.model_away, context.kalshi_away)):
                line = _delta_line(lab, m, o)
                if line:
                    L.append(line)
        if context.has_market() and context.has_kalshi():
            spread = max(abs(context.market_home - context.kalshi_home),
                         abs(context.market_draw - context.kalshi_draw),
                         abs(context.market_away - context.kalshi_away))
            L.append(f"  bookmaker vs exchange: largest gap between the two sources is "
                     f"{spread * 100:.1f} pts")

    gaps = [g for g in ("market", "kalshi") if g in context.missing()]
    L.append("\nMARKET SOURCES NOT SUPPLIED: " + (", ".join(gaps) if gaps else "none"))
    L.append("WITHHELD FROM THIS SEAT: expected goals, form ratings, team news, injuries. "
             "Other analysts cover those. You cannot explain a price gap - report it.")
    return "\n".join(L)


def market_options(model: Optional[str] = None) -> ClaudeAgentOptions:
    """Locked-down options for the Market seat - identical safeguards to the other seats."""
    chosen = model or council_model()
    return ClaudeAgentOptions(
        model=chosen,
        system_prompt=MARKET_SKEPTIC.prompt,
        allowed_tools=[],
        disallowed_tools=list(FORBIDDEN_TOOLS),
        permission_mode="default",
        max_turns=1,
        setting_sources=[],
        agents={MARKET_SKEPTIC_NAME: replace(MARKET_SKEPTIC, model=chosen)},
    )


async def run_market_skeptic_async(context: MatchContext) -> AnalystPrediction:
    """Ask the Market Skeptic about one fixture. Prices only - see MARKET_SYSTEM_PROMPT."""
    return await _ask_analyst_async(
        serialize_market_context(context), market_options(),
        context.fixture_key(), MARKET_SKEPTIC_NAME)


def run_market_skeptic(context: MatchContext) -> AnalystPrediction:
    """Synchronous entry point for the Market Skeptic."""
    return _sync(run_market_skeptic_async(context), MARKET_SKEPTIC_NAME)


# ===========================================================================================
# THE CHAIRMAN — the seat that decides, and the only one that sees the others.
#
# Its input is three opinions plus the baseline, and NOTHING else. It does not get a fourth,
# unrestricted look at the raw data: if the Chairman could read the xG, the news and the
# market itself, the three specialists would be decoration and the panel would collapse into
# one opinion wearing four hats.
#
# The analysts arrive as "Analyst A / B / C" with their names stripped. A role label is an
# invitation to weight by reputation rather than by evidence - "the market seat is usually
# right" is exactly the prior this desk has spent months trying to replace with measurement.
#
# HONEST LIMIT: anonymity here is positional, not total. A seat that writes about bookmaker
# overround is identifiable from its evidence no matter what it is called. What the labels
# buy is the removal of an automatic hierarchy, not genuine blindness - and the ledger, not
# this prompt, is what will eventually say which seat deserves weight.
# ===========================================================================================

CHAIRMAN_NAME = "chairman"
REQUIRED_ANALYSTS = 3
ANALYST_LABELS = ("Analyst A", "Analyst B", "Analyst C")

CHAIRMAN_SYSTEM_PROMPT = """\
You are the chairman of a Premier League forecasting council.

Three specialist analysts have independently evaluated the fixture.

Your task is to synthesize their forecasts into one final calibrated
Home/Draw/Away probability vector.

You must evaluate the QUALITY OF THEIR EVIDENCE, not their writing style,
confidence, or verbosity.

You may not introduce any new football facts.

You may not browse the web.

You may not use general Premier League knowledge.

Do not force consensus.

If analysts materially disagree, preserve that uncertainty in the final
forecast and report a lower consensus score.

If analysts broadly agree for independent reasons, consensus may be higher.

Do not simply average the three probability vectors unless the evidence
actually supports treating them equally.

CONSENSUS IS NOT CONFIDENCE
These are two different measurements and you must not let one drive the other.

  consensus_score = how much the three analysts AGREE WITH EACH OTHER
    0.00-0.30  strong disagreement
    0.31-0.60  meaningful disagreement
    0.61-0.80  moderate agreement
    0.81-1.00  strong agreement

  confidence = how much the PANEL'S EVIDENCE supports the final answer

All three analysts can agree that HOME is most likely while all three say their evidence is
thin. That is HIGH consensus and LOW confidence, and reporting it that way is correct. The
reverse also happens: two analysts with strong, specific evidence pulling in opposite
directions is LOW consensus, and the confidence you report should reflect which evidence
survives scrutiny, not the fact that they differed.

HOW TO WEIGH THE ANALYSTS
- Weight by the SPECIFICITY and VERIFIABILITY of what each analyst cites. An analyst quoting
  a confirmed fact it was given outranks one reasoning from an absence.
- An analyst that reports it had insufficient information, and stayed near the baseline, is
  being honest, not useless. Do not penalise it, and do not treat its baseline-like answer as
  independent corroboration of another analyst who moved.
- Stated confidence is a claim, not evidence. A "high" backed by one vague sentence is worth
  less than a "low" backed by two specific figures.
- Length is not weight. Ignore verbosity entirely.
- Where an analyst's own uncertainties undercut its conclusion, say so.
- You may land outside the range of the three analysts if the evidence justifies it, but say
  why. Usually the answer sits inside their range.

OUTPUT CONTRACT
Reply with a single JSON object and nothing else - no prose before or after, no markdown
fence. Exactly these fields:

{
  "home_probability": float,
  "draw_probability": float,
  "away_probability": float,
  "predicted_outcome": "HOME" | "DRAW" | "AWAY",
  "confidence": "low" | "medium" | "high",
  "consensus_score": float,
  "major_disagreement": string
}

Rules for those fields:
- the three probabilities are decimals in [0, 1] and must sum to 1.00
- predicted_outcome is upper-case and is one of HOME, DRAW, AWAY
- consensus_score is a decimal between 0 and 1, read against the bands above
- major_disagreement is a SHORT SENTENCE naming what the panel actually split over, or the
  exact string "none" when they did not materially disagree
- every claim you make must trace to something an analyst wrote or to the baseline you were
  given; you have no other information
- do not round a probability to 0 or 1; a football match is never certain
"""

CHAIRMAN = AgentDefinition(
    description=(
        "Chairman of the forecasting council. Synthesises three anonymised specialist "
        "forecasts into one calibrated probability vector, weighting by evidence quality "
        "rather than stated confidence. Sees only the analysts' output and the baseline - "
        "never the raw expected goals, team news or market prices the specialists saw."
    ),
    prompt=CHAIRMAN_SYSTEM_PROMPT,
    tools=[],
    model=None,
)


def _require_three(analysts):
    if not isinstance(analysts, (list, tuple)):
        raise CouncilValidationError(
            f"analysts must be a list of AnalystPrediction, got {type(analysts).__name__}")
    if len(analysts) != REQUIRED_ANALYSTS:
        raise CouncilValidationError(
            f"the Chairman requires exactly {REQUIRED_ANALYSTS} analyst predictions, "
            f"got {len(analysts)}")
    for i, a in enumerate(analysts):
        if not isinstance(a, AnalystPrediction):
            raise CouncilValidationError(
                f"analysts[{i}] must be an AnalystPrediction, got {type(a).__name__}")
    return list(analysts)


def serialize_chairman_context(context: MatchContext,
                               analysts: List[AnalystPrediction]) -> str:
    """The three opinions, anonymised, plus the baseline. Nothing else.

    Deliberately does NOT include expected goals, form ratings, team news or market prices.
    Those were each given to one specialist; handing them all to the Chairman would make it a
    fourth analyst with a better view than the other three, and its job is to judge THEIR
    reasoning, not to redo it.

    Analyst names are stripped. A role label invites weighting by reputation - see the module
    comment for why that is the one prior this desk is trying to avoid.
    """
    if not isinstance(context, MatchContext):
        raise CouncilValidationError(
            f"expected MatchContext, got {type(context).__name__}")
    analysts = _require_three(analysts)

    L = [f"FIXTURE: {context.home_team} (home) v {context.away_team} (away)"]
    if context.kickoff:
        L.append(f"KICKOFF: {context.kickoff}")

    L.append("\nBASELINE FROM THE EXISTING STATISTICAL MODEL (for reference only):")
    if context.has_model():
        L.append(f"  P(home) {context.model_home:.4f}   "
                 f"P(draw) {context.model_draw:.4f}   P(away) {context.model_away:.4f}")
    else:
        L.append("  MISSING - no baseline supplied.")

    L.append("\nTHE THREE SPECIALIST FORECASTS")
    L.append("Each analyst saw a DIFFERENT slice of the evidence, and none saw all of it.")
    L.append("They are unlabelled on purpose: judge them on what they cite, not on who wrote it.")
    for label, a in zip(ANALYST_LABELS, analysts):
        h, d, aw = a.probabilities()
        L.append(f"\n{label}")
        L.append(f"  P(home) {h:.4f}   P(draw) {d:.4f}   P(away) {aw:.4f}")
        L.append(f"  calls it {a.predicted_outcome}, stated confidence {a.confidence:.2f}")
        L.append("  evidence:")
        L.extend(f"    - {e}" for e in (a.evidence or ["(none given)"]))
        L.append("  uncertainties:")
        L.extend(f"    - {u}" for u in (a.uncertainties or ["(none given)"]))

    L.append("\nYou have no information beyond the above. Do not add football facts.")
    return "\n".join(L)


def chairman_options(model: Optional[str] = None) -> ClaudeAgentOptions:
    """Locked-down options for the Chairman - identical safeguards to every other seat."""
    chosen = model or council_model()
    return ClaudeAgentOptions(
        model=chosen,
        system_prompt=CHAIRMAN.prompt,
        allowed_tools=[],
        disallowed_tools=list(FORBIDDEN_TOOLS),
        permission_mode="default",
        max_turns=1,
        setting_sources=[],
        agents={CHAIRMAN_NAME: replace(CHAIRMAN, model=chosen)},
    )


def council_from_payload(obj: Dict[str, Any],
                         analysts: List[AnalystPrediction]) -> CouncilPrediction:
    """Turn the Chairman's parsed JSON into a validated CouncilPrediction.

    Like analyst_from_payload, this does NOT normalise: a vector that fails the simplex check
    raises, and the caller sees exactly what the Chairman said.

    `major_disagreement` arrives as prose and the dataclass field is a bool, so the text is
    kept in `disagreement_note` and the flag derived from it. "none" (any casing), an empty
    string and a literal false all mean no material disagreement.
    """
    analysts = _require_three(analysts)
    missing = [k for k in ("home_probability", "draw_probability", "away_probability",
                           "predicted_outcome", "confidence", "consensus_score")
               if k not in obj]
    if missing:
        raise CouncilParseError(
            f"chairman payload is missing required field(s): {', '.join(missing)}")

    raw_conf = obj["confidence"]
    if isinstance(raw_conf, str):
        key = raw_conf.strip().lower()
        if key not in CONFIDENCE_LEVELS:
            raise CouncilParseError(
                f"confidence must be one of {sorted(CONFIDENCE_LEVELS)}, got {raw_conf!r}")
        conf = CONFIDENCE_LEVELS[key]
    else:
        conf = raw_conf

    raw_dis = obj.get("major_disagreement", "")
    if isinstance(raw_dis, bool):
        note, flag = ("", raw_dis)
    else:
        note = str(raw_dis or "").strip()
        flag = bool(note) and note.lower().rstrip(".") not in ("none", "no", "n/a", "false")

    return CouncilPrediction(
        home_probability=obj["home_probability"],
        draw_probability=obj["draw_probability"],
        away_probability=obj["away_probability"],
        predicted_outcome=obj["predicted_outcome"],
        confidence=conf,
        consensus_score=obj["consensus_score"],
        major_disagreement=flag,
        disagreement_note=note,
        analyst_predictions=analysts,
    )


async def run_chairman_async(context: MatchContext,
                             analysts: List[AnalystPrediction]) -> CouncilPrediction:
    """Ask the Chairman to synthesise three specialist forecasts into one."""
    analysts = _require_three(analysts)
    prompt = serialize_chairman_context(context, analysts)
    options = chairman_options()
    fixture = context.fixture_key()

    chunks: List[str] = []
    result = None
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                result = message
    except ClaudeSDKError as e:
        raise CouncilSDKError(f"Agent SDK failed for {fixture} (chairman): {e}") from e

    if result is not None and getattr(result, "is_error", False):
        raise CouncilSDKError(
            f"the chairman errored on {fixture}: "
            f"{getattr(result, 'result', None) or 'no detail returned'}")
    if not chunks:
        raise CouncilParseError(f"the chairman returned no text for {fixture}")

    return council_from_payload(_extract_json("\n".join(chunks)), analysts)


def run_chairman(context: MatchContext,
                 analysts: List[AnalystPrediction]) -> CouncilPrediction:
    """Synchronous entry point for the Chairman."""
    return _sync(run_chairman_async(context, analysts), CHAIRMAN_NAME)


# ===========================================================================================
# ORCHESTRATION — the whole Council, in memory, for one fixture.
#
#     MatchContext ──┬─► Quant Analyst  ─┐
#                    ├─► Context Analyst ├─► Chairman ─► CouncilPrediction
#                    └─► Market Skeptic ─┘
#
# The three specialists run CONCURRENTLY and the Chairman runs strictly after all three have
# returned - it cannot synthesise opinions that do not exist yet. Concurrency is not a
# micro-optimisation here: each seat spawns its own Claude Code subprocess and the smoke tests
# measured 9-19s per specialist, so sequential execution would put a single fixture near two
# minutes before the Chairman even starts.
#
# WHAT THIS DELIBERATELY DOES NOT DO
#   * no retries - a flaky answer silently retried is an answer you cannot reason about
#   * no fallback to two analysts - a panel that quietly shrinks is not the panel you tested
#   * no averaging as a substitute for a missing seat - that is a different, untested model
#   * no caching, no file writes, no ledger - orchestration only, by design
# Every one of those is a place where a failure could become invisible, and this project has
# already been bitten by numbers that looked fine because something upstream failed quietly
# (the empty-feed build that rewrote 164 transfers; the ledger graded on retrodictions).
# ===========================================================================================

# Stable order in, stable order out. The Chairman anonymises these as Analyst A/B/C, but the
# ORDER is fixed so two runs on identical inputs present the panel identically - the only way
# a disagreement between runs can be attributed to the model rather than to shuffling.
SPECIALIST_ORDER = (QUANT_ANALYST_NAME, CONTEXT_ANALYST_NAME, MARKET_SKEPTIC_NAME)


async def predict_async(context: MatchContext) -> CouncilPrediction:
    """Run the full Council over one fixture and return the Chairman's synthesis.

    Raises CouncilSDKError naming the seat that failed, or CouncilParseError /
    CouncilValidationError if a seat answered but answered badly. Nothing is retried and
    nothing is written.
    """
    if not isinstance(context, MatchContext):
        raise CouncilValidationError(
            f"expected MatchContext, got {type(context).__name__}")

    # every seat gets the SAME context object - each serialiser decides what it may show
    runners = (run_quant_analyst_async, run_context_analyst_async, run_market_skeptic_async)
    results = await asyncio.gather(*(r(context) for r in runners), return_exceptions=True)

    # return_exceptions=True so a failure does not cancel the siblings mid-subprocess AND so
    # the failing seat can be named. gather preserves input order, so index == seat.
    failures = [(name, r) for name, r in zip(SPECIALIST_ORDER, results)
                if isinstance(r, BaseException)]
    if failures:
        names = ", ".join(n for n, _ in failures)
        first_name, first_err = failures[0]
        raise CouncilSDKError(
            f"Council aborted for {context.fixture_key()}: specialist(s) failed: {names}. "
            f"The Chairman was NOT run and no substitute was used. "
            f"First failure ({first_name}): {type(first_err).__name__}: {first_err}"
        ) from first_err

    analysts = list(results)            # [quant, context, market] - SPECIALIST_ORDER
    try:
        return await run_chairman_async(context, analysts)
    except (CouncilSDKError, CouncilParseError, CouncilValidationError):
        raise                           # already specific; re-raising loses nothing
    except Exception as e:
        raise CouncilSDKError(
            f"the Chairman failed for {context.fixture_key()} after all three specialists "
            f"returned: {type(e).__name__}: {e}") from e


def predict(context: MatchContext) -> CouncilPrediction:
    """Run the Council synchronously. The public entry point.

    Owns the asyncio boundary, exactly like the per-seat wrappers, so build_dashboard.py and
    the rest of the pipeline stay synchronous. Returns a validated CouncilPrediction with the
    three specialist opinions attached; writes nothing anywhere.
    """
    return _sync(predict_async(context), "council")

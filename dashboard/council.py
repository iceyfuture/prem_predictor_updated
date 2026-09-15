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

STATUS: skeleton. `predict()` raises NotImplementedError - see the note there.
"""
import asyncio
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

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
    home_news: Optional[str] = None
    away_news: Optional[str] = None

    def has_model(self):
        return None not in (self.model_home, self.model_draw, self.model_away)

    def has_market(self):
        return None not in (self.market_home, self.market_draw, self.market_away)

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
        if not (self.home_news or self.away_news):
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


# ----------------------------------------------------------------------------- entry point
def predict(context: MatchContext) -> CouncilPrediction:
    """Run the Council over one fixture.

    NOT IMPLEMENTED. The panel needs the Claude Agent SDK to actually reason over the context,
    and that is a later step by design: the dataclasses, their validation and the ledger
    columns are worth settling while they are cheap to change, and before anything can be
    graded on retrodictions.

    When it is implemented it must obey two constraints from this module's docstring:
      * it may not modify the Dixon-Coles probabilities or the production blend
      * its output must be locked BEFORE kickoff to count - see council_ledger.late
    """
    raise NotImplementedError(
        "AI Council prediction is not implemented yet. This skeleton defines the data "
        "contract (MatchContext, AnalystPrediction, CouncilPrediction) and its validation; "
        "the Claude Agent SDK panel that fills it in is a later step. Nothing calls this "
        "function yet - it is deliberately not wired into build_dashboard.py.")


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


async def run_quant_analyst_async(context: MatchContext) -> AnalystPrediction:
    """Ask the Quant Analyst about one fixture.

    Makes a real Claude request via the Agent SDK, which spawns the bundled Claude Code CLI as
    a subprocess. Errors are wrapped in CouncilSDKError with the fixture attached, because a
    bare transport failure three layers down says nothing about which match it was.
    """
    prompt = serialize_quant_context(context)
    options = quant_options()

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
        raise CouncilSDKError(
            f"Agent SDK failed for {context.fixture_key()}: {e}") from e

    if result is not None and getattr(result, "is_error", False):
        raise CouncilSDKError(
            f"the Quant Analyst errored on {context.fixture_key()}: "
            f"{getattr(result, 'result', None) or 'no detail returned'}")
    if not chunks:
        raise CouncilParseError(
            f"the Quant Analyst returned no text for {context.fixture_key()}")

    return analyst_from_payload(_extract_json("\n".join(chunks)))


def run_quant_analyst(context: MatchContext) -> AnalystPrediction:
    """Synchronous entry point - owns the asyncio boundary so every caller stays synchronous.

    build_dashboard.py and the rest of the pipeline are synchronous top to bottom, and the
    Agent SDK's query() is an async generator. That boundary lives HERE, once, rather than
    turning the desk async to accommodate one experimental layer.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_quant_analyst_async(context))
    raise CouncilSDKError(
        "run_quant_analyst() was called from inside a running event loop; "
        "await run_quant_analyst_async(context) instead")

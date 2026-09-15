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
from dataclasses import dataclass, field
from typing import List, Optional

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

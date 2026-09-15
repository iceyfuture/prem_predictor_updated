"""
council_ledger.py — append-only forward test for the AI Council.

STATUS: column definitions and file layout only. Nothing writes yet, nothing calls this, and
no CSV is created until the first real lock. Built now rather than later on purpose - see
RULE 5 and the bug it caught in player_ledger.py.

WHY IT EXISTS BEFORE THE PREDICTOR DOES
The Council's whole claim is that it might forecast better than the model. That claim is only
testable against predictions written down before kickoff. When player_ledger.py was first
run it locked 485 GW4 rows a DAY AFTER GW4 was played; had there been no guard, those
retrodictions would have been scored as a forward test and the minutes model would have looked
clairvoyant. The `late` column is that guard, and it is here from the start so the Council can
never be graded on a game it already knows the result of.

DESIGN, FOLLOWING player_ledger.py
  * one row per fixture, keyed "{home}|{away}" - the same key record_ledger uses, stable when
    TV picks move kickoff times
  * a row is written ONCE at lock time and its prediction fields are never restated
  * grading fills the result columns later, in place
  * `late=1` marks a row locked at or after kickoff; excluded from every score

WHAT IT STORES THAT THE OTHER LEDGERS DO NOT
The model's and the market's numbers are copied in beside the Council's. That looks redundant
against ledger.csv, but it is the point: this ledger answers "was the Council closer than the
model, ON THE SAME FIXTURE" without a join, and a join across two append-only files written at
different moments is exactly how a denominator mismatch gets in. RULE 33 was that bug - the
props Brier averaged the model over 440 markets and Kalshi over 387, and ~88% of the apparent
edge was the mismatch rather than skill. Three RPS columns stored side by side cannot drift
apart that way.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "council_ledger.csv")

# Identity and provenance. `late` is the forward-test guard, not a diagnostic.
ID_COLS = [
    "fixture_key",      # "{home}|{away}" - matches record_ledger's convention
    "kickoff",          # ISO kickoff as known at lock time
    "locked_at",        # when this row was written
    "late",             # "1" if locked at/after kickoff -> a retrodiction, never scored
    "home_team",
    "away_team",
]

# What the production model said, copied in at lock time so scoring needs no join.
MODEL_COLS = ["model_home", "model_draw", "model_away"]

# Sportsbook implied probabilities at lock time. May be blank: books open ~1 week out.
MARKET_COLS = ["market_home", "market_draw", "market_away"]

# The Council's own forecast, plus how much its analysts agreed.
COUNCIL_COLS = ["council_home", "council_draw", "council_away", "consensus_score"]

# Filled in after the match. Every forecaster is scored on the SAME row, same denominator.
RESULT_COLS = [
    "actual_result",    # "H" / "D" / "A" - the letters the existing ledger already uses
    "council_brier",
    "council_rps",
    "model_rps",
    "market_rps",
]

COLS = ID_COLS + MODEL_COLS + MARKET_COLS + COUNCIL_COLS + RESULT_COLS

# Written once at lock time and never restated. Anything outside this set may be filled in
# later by grading; anything inside it is frozen the moment the row exists.
LOCKED_COLS = frozenset(ID_COLS + MODEL_COLS + MARKET_COLS + COUNCIL_COLS)

# Grading writes only these.
GRADED_COLS = frozenset(RESULT_COLS)

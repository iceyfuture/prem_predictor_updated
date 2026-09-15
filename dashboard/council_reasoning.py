"""
council_reasoning.py — the Council's reasoning sidecar.

WHY THIS EXISTS
council_ledger.csv records what the Council CONCLUDED and nothing about why. The first real
forecast (Brentford|Chelsea) demonstrated the cost: asked afterwards what the analysts had
argued, the answer was that it was unrecoverable - the ledger has no column for it, the Agent
SDK left no transcript, and the process that held the objects had exited.

That is a real hole in a system whose entire premise is that the Chairman "evaluates the
QUALITY OF THEIR EVIDENCE". Storing the verdict and discarding the evidence means that when a
forecast turns out wrong you can see THAT it was wrong but never WHICH seat's reasoning
failed - which is the only thing that would tell you what to fix. The rest of this project
already meets that standard: ledger.csv keeps the closing line beside the prediction,
props_ledger.csv keeps the Kalshi quote.

WHY A SIDECAR RATHER THAN MORE COLUMNS
Four seats x two prose lists inside a CSV cell is exactly the shape that breaks quoting and
makes a file unreadable. JSONL keeps the ledger narrow and greppable, appends cleanly, and
survives a partially-written final line (every complete line before it still parses).

WHAT IS NEVER STORED
Only the explicit fields our own dataclasses hold - the JSON each analyst returned under its
output contract. Specifically NOT:
  * credentials of any kind - this module reads no environment variable and imports no SDK
  * hidden chain-of-thought - council.py collects TextBlock only and never imports
    ThinkingBlock, so reasoning traces never enter the application in the first place
  * anything from the transport layer
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "council_reasoning.jsonl")

# Exactly what a stored record may contain. Anything not in this shape is not written.
ANALYST_FIELDS = ("home", "draw", "away", "predicted_outcome", "confidence",
                  "evidence", "uncertainties")
CHAIRMAN_FIELDS = ("home", "draw", "away", "predicted_outcome", "confidence",
                   "consensus_score", "disagreement_note")


def _analyst(a):
    h, d, aw = a.probabilities()
    return {"home": round(h, 6), "draw": round(d, 6), "away": round(aw, 6),
            "predicted_outcome": a.predicted_outcome,
            "confidence": a.confidence,
            "evidence": list(a.evidence or []),
            "uncertainties": list(a.uncertainties or [])}


def load(path=None):
    """[record, ...] in file order. A malformed line is skipped, not fatal.

    Tolerating a bad line matters: this file is appended to after a ledger row is already
    locked, so a truncated write must never make the whole history unreadable.
    """
    p = path or PATH
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("fixture_key"):
                out.append(obj)
    return out


def keys(path=None):
    return {r["fixture_key"] for r in load(path)}


def has(fixture_key, path=None):
    return fixture_key in keys(path)


def get(fixture_key, path=None):
    """The stored reasoning for one fixture, or None."""
    for r in load(path):
        if r["fixture_key"] == fixture_key:
            return r
    return None


def append(context, prediction, *, locked_at=None, path=None):
    """Write one reasoning record for a newly locked fixture. Returns the record, or None.

    IDEMPOTENT BY FIXTURE: if this fixture already has a record, nothing is written and None
    is returned. One fixture, one record - the same rule the ledger enforces, enforced here
    independently so it holds no matter which caller is involved.

    Must be called ONLY after council_ledger.record() has successfully locked a NEW row. This
    module does not check the ledger itself: doing so would couple two append-only stores into
    a single point of failure, and the caller is the one that knows whether the lock was new.
    """
    from council import CouncilPrediction, MatchContext          # local: avoid import cycles
    if not isinstance(context, MatchContext):
        raise TypeError(f"context must be a MatchContext, got {type(context).__name__}")
    if not isinstance(prediction, CouncilPrediction):
        raise TypeError(
            f"prediction must be a CouncilPrediction, got {type(prediction).__name__}")

    p = path or PATH
    key = context.fixture_key()
    if has(key, p):
        return None                       # already recorded - never a second line

    by_name = {a.analyst_name: a for a in prediction.analyst_predictions}
    h, d, aw = prediction.probabilities()
    record = {
        "fixture_key": key,
        "kickoff": context.kickoff or "",
        "locked_at": locked_at or "",
        "quant": _analyst(by_name["quant"]) if "quant" in by_name else None,
        "context": _analyst(by_name["context"]) if "context" in by_name else None,
        "market": _analyst(by_name["market"]) if "market" in by_name else None,
        "chairman": {
            "home": round(h, 6), "draw": round(d, 6), "away": round(aw, 6),
            "predicted_outcome": prediction.predicted_outcome,
            "confidence": prediction.confidence,
            "consensus_score": prediction.consensus_score,
            "disagreement_note": prediction.disagreement_note or "",
        },
    }
    with open(p, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

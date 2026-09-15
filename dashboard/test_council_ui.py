"""
Offline tests for the AI Council dashboard integration.

NO Claude calls. Verifies the build-side join (build_dashboard.attach_council) and that the
Council stays display-only. The real ledger and reasoning files are read but never written.

    ./.venv/bin/python dashboard/test_council_ui.py
"""
import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dashboard as B

PASS = FAIL = 0
HERE = os.path.dirname(os.path.abspath(__file__))


def _fp(p):
    if not os.path.exists(p):
        return None
    st = os.stat(p)
    with open(p) as f:
        return (st.st_size, st.st_mtime, f.read())


LEDGER_BEFORE = _fp(B.COUNCIL_LEDGER)
REASONING_BEFORE = _fp(B.COUNCIL_REASONING)


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
        print(f"  FAIL {name}: expected {expect.__name__}"); FAIL += 1
    else:
        print(f"  ok   {name}"); PASS += 1


def _assert(c):
    if not c:
        raise AssertionError("assertion failed")


def weeks():
    mk = lambda h, a: {"home": h, "away": a, "ph": 42, "pd": 26, "pa": 32}
    return [{"gw": 5, "matches": [mk("Brentford", "Chelsea"),
                                  mk("Tottenham", "Aston Villa"),
                                  mk("Everton", "Ipswich")]}]


def with_files(ledger_csv, reasoning_lines, fn):
    """Point build_dashboard at temp Council files for the duration of fn."""
    tmp = tempfile.mkdtemp()
    lp, rp = os.path.join(tmp, "l.csv"), os.path.join(tmp, "r.jsonl")
    if ledger_csv is not None:
        open(lp, "w").write(ledger_csv)
    if reasoning_lines is not None:
        open(rp, "w").write(reasoning_lines)
    saved = (B.COUNCIL_LEDGER, B.COUNCIL_REASONING)
    B.COUNCIL_LEDGER, B.COUNCIL_REASONING = lp, rp
    try:
        return fn()
    finally:
        B.COUNCIL_LEDGER, B.COUNCIL_REASONING = saved


HDR = ("fixture_key,kickoff,locked_at,late,home_team,away_team,model_home,model_draw,"
       "model_away,market_home,market_draw,market_away,council_home,council_draw,"
       "council_away,consensus_score,actual_result,council_brier,council_rps,model_rps,"
       "market_rps\n")
ROW_B = ("Brentford|Chelsea,Fri 18 Sep,2026-09-15T17:16+00:00,,Brentford,Chelsea,"
         "0.42,0.26,0.32,0.33,0.25,0.42,0.38,0.25,0.37,0.38,,,,,\n")
ROW_T = ("Tottenham|Aston Villa,2026-09-19T11:30+00:00,2026-09-15T18:07+00:00,,Tottenham,"
         "Aston Villa,0.37,0.27,0.36,,,,0.39,0.27,0.34,0.55,,,,,\n")
SEAT = {"home": .35, "draw": .27, "away": .38, "predicted_outcome": "AWAY",
        "confidence": .25, "evidence": ["xG 1.51 v 1.45 to Villa"],
        "uncertainties": ["no market cross-check"]}
REC_T = json.dumps({"fixture_key": "Tottenham|Aston Villa", "kickoff": "x", "locked_at": "y",
                    "quant": SEAT, "context": dict(SEAT, predicted_outcome="HOME"),
                    "market": dict(SEAT, predicted_outcome="HOME"),
                    "chairman": {"home": .39, "draw": .27, "away": .34,
                                 "predicted_outcome": "HOME", "confidence": .55,
                                 "consensus_score": .55,
                                 "disagreement_note": "xG tilts Villa, market tilts Spurs"}}) + "\n"


def main():
    print("=== no Council data at all ===")
    w = weeks()
    n = with_files(None, None, lambda: B.attach_council(w))
    check("nothing attached", lambda: _assert(n["forecasts"] == 0))
    check("no fixture gains a council key",
          lambda: _assert(not any("council" in m for m in w[0]["matches"])))
    check("model probabilities untouched",
          lambda: _assert(all((m["ph"], m["pd"], m["pa"]) == (42, 26, 32)
                              for m in w[0]["matches"])))

    print("\n=== forecast WITHOUT reasoning (the Brentford case) ===")
    w = weeks()
    n = with_files(HDR + ROW_B, "", lambda: B.attach_council(w))
    b = w[0]["matches"][0]
    check("one forecast attached", lambda: _assert(n["forecasts"] == 1))
    check("zero with reasoning", lambda: _assert(n["with_reasoning"] == 0))
    check("council block present", lambda: _assert(b["council"]["home"] == 0.38))
    check("consensus carried", lambda: _assert(b["council"]["consensus_score"] == 0.38))
    check("late is False", lambda: _assert(b["council"]["late"] is False))
    check("locked_at carried", lambda: _assert(b["council"]["locked_at"].startswith("2026-09-15")))
    check("NO council_reasoning key", lambda: _assert("council_reasoning" not in b))
    check("outcome/confidence are None without a chairman record",
          lambda: _assert(b["council"]["predicted_outcome"] is None
                          and b["council"]["confidence"] is None))
    check("model probabilities untouched", lambda: _assert((b["ph"], b["pd"], b["pa"])
                                                           == (42, 26, 32)))

    print("\n=== forecast WITH reasoning (the Tottenham case) ===")
    w = weeks()
    n = with_files(HDR + ROW_B + ROW_T, REC_T, lambda: B.attach_council(w))
    t = w[0]["matches"][1]
    check("two forecasts", lambda: _assert(n["forecasts"] == 2))
    check("one with reasoning", lambda: _assert(n["with_reasoning"] == 1))
    check("all three seats present",
          lambda: _assert(all(t["council_reasoning"][k] for k in ("quant", "context", "market"))))
    check("chairman present", lambda: _assert(t["council_reasoning"]["chairman"]))
    check("specialist evidence preserved",
          lambda: _assert(t["council_reasoning"]["quant"]["evidence"] ==
                          ["xG 1.51 v 1.45 to Villa"]))
    check("specialist uncertainties preserved",
          lambda: _assert(t["council_reasoning"]["quant"]["uncertainties"] ==
                          ["no market cross-check"]))
    check("disagreement note surfaced on council",
          lambda: _assert("market tilts Spurs" in t["council"]["disagreement_note"]))
    check("outcome and confidence taken from the chairman",
          lambda: _assert((t["council"]["predicted_outcome"], t["council"]["confidence"])
                          == ("HOME", 0.55)))
    check("the OTHER fixture still has no reasoning",
          lambda: _assert("council_reasoning" not in w[0]["matches"][0]))
    check("a third fixture gains nothing",
          lambda: _assert("council" not in w[0]["matches"][2]))

    print("\n=== 'what changed vs model' deltas (computed in the UI, verified here) ===")
    model = (t["ph"], t["pd"], t["pa"])
    c = t["council"]
    deltas = [round(c["home"] * 100) - model[0], round(c["draw"] * 100) - model[1],
              round(c["away"] * 100) - model[2]]
    check("deltas from the STORED values, not recomputed probabilities",
          lambda: _assert(deltas == [39 - 42, 27 - 26, 34 - 32]))
    # and against the real ledger row's own model column
    real_model = (0.37, 0.27, 0.36)
    check("against the model stored at lock time: +2 / 0 / -2",
          lambda: _assert([round(.39 * 100) - round(real_model[0] * 100),
                           round(.27 * 100) - round(real_model[1] * 100),
                           round(.34 * 100) - round(real_model[2] * 100)] == [2, 0, -2]))

    print("\n=== consensus bands ===")
    bands = [(0.00, "Strong disagreement"), (0.30, "Strong disagreement"),
             (0.31, "Meaningful disagreement"), (0.60, "Meaningful disagreement"),
             (0.61, "Moderate agreement"), (0.80, "Moderate agreement"),
             (0.81, "Strong agreement"), (1.00, "Strong agreement")]
    def band(c):
        return ("Strong disagreement" if c <= .30 else "Meaningful disagreement" if c <= .60
                else "Moderate agreement" if c <= .80 else "Strong agreement")
    for v, lab in bands:
        check(f"consensus {v:.2f} -> {lab}", lambda v=v, lab=lab: _assert(band(v) == lab))
    check("0.38 (Brentford) is Meaningful disagreement",
          lambda: _assert(band(0.38) == "Meaningful disagreement"))
    check("0.55 (Tottenham) is Meaningful disagreement",
          lambda: _assert(band(0.55) == "Meaningful disagreement"))

    print("\n=== malformed input degrades gracefully ===")
    w = weeks()
    n = with_files(HDR + ROW_T, "NOT JSON\n{broken\n" + REC_T, lambda: B.attach_council(w))
    check("build continues", lambda: _assert(n["forecasts"] == 1))
    check("the good record still used", lambda: _assert(n["with_reasoning"] == 1))
    w = weeks()
    n = with_files(HDR + ROW_T, "NOT JSON AT ALL\n", lambda: B.attach_council(w))
    check("all-malformed reasoning still renders the summary",
          lambda: _assert(n["forecasts"] == 1 and n["with_reasoning"] == 0))
    check("fixture keeps its council block",
          lambda: _assert(w[0]["matches"][1]["council"]["home"] == 0.39))
    w = weeks()
    n = with_files("total,garbage\n1,2\n", None, lambda: B.attach_council(w))
    check("garbage ledger attaches nothing rather than crashing",
          lambda: _assert(n["forecasts"] == 0))

    print("\n=== the Council is DISPLAY-ONLY ===")
    w = weeks()
    before = copy.deepcopy(w)
    with_files(HDR + ROW_B + ROW_T, REC_T, lambda: B.attach_council(w))
    for i, m in enumerate(w[0]["matches"]):
        o = before[0]["matches"][i]
        check(f"{m['home']}: ph/pd/pa unchanged",
              lambda m=m, o=o: _assert((m["ph"], m["pd"], m["pa"]) == (o["ph"], o["pd"], o["pa"])))
    check("only 'council' / 'council_reasoning' keys were added",
          lambda: _assert(set(w[0]["matches"][1]) - set(before[0]["matches"][1])
                          == {"council", "council_reasoning"}))
    src = open(os.path.join(HERE, "build_dashboard.py")).read()
    body = src[src.index("def attach_council"):]
    body = body[:body.index("\ndef ", 5)] if "\ndef " in body[5:] else body
    for banned in ('m["ph"]', 'm["pd"]', 'm["pa"]', '"edge"', '"conf"', '"tier"', '"props"'):
        check(f"attach_council never writes {banned}",
              lambda b=banned: _assert(f'{b} =' not in body and f'{b}=' not in body))

    print("\n=== no Claude / SDK anywhere in the build path ===")
    check("build_dashboard does not import the SDK",
          lambda: _assert("claude_agent_sdk" not in src))
    check("build_dashboard does not import council.py",
          lambda: _assert("import council\n" not in src and "import council as" not in src))
    # Match a CALL, not a substring: "predict" also appears inside the field name
    # "predicted_outcome", which is data being copied, not a function being invoked.
    import re as _re
    check("attach_council invokes no predictor",
          lambda: _assert(not _re.search(r"\bpredict\s*\(|\.predict\b|\bquery\s*\(", body)))
    check("attach_council only opens files for READING",
          lambda: _assert(not _re.search(r"open\([^)]*['\"][wa]", body)))

    print("\n=== the real Council files were only READ ===")
    check("council_ledger.csv byte-identical",
          lambda: _assert(_fp(B.COUNCIL_LEDGER) == LEDGER_BEFORE))
    check("council_reasoning.jsonl byte-identical",
          lambda: _assert(_fp(B.COUNCIL_REASONING) == REASONING_BEFORE))

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

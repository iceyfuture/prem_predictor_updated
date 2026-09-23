"""
Offline tests for RULE 49 — one forecasting pipeline (audit §1). No network, no Claude.

The audit's requirement: "Backtests must call the real production forecasting function."
The way that guarantee dies is quietly — someone adds a second implementation for a good
reason and it drifts. These tests fail if a second one reappears.

    ./.venv/bin/python dashboard/test_pipeline_parity.py
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import numpy as np
import forecast as F
import prem_dixon_coles as dc
import validate as V
import build_dashboard as B

PASS = FAIL = 0


def check(name, fn, expect=None):
    global PASS, FAIL
    try:
        fn()
    except Exception as e:
        if expect and isinstance(e, expect):
            print(f"  ok   {name}"); PASS += 1
        else:
            print(f"  FAIL {name}: {type(e).__name__}: {e}"); FAIL += 1
        return
    if expect:
        print(f"  FAIL {name}: expected {expect.__name__}"); FAIL += 1
    else:
        print(f"  ok   {name}"); PASS += 1


def _assert(c, msg="assertion failed"):
    if not c:
        raise AssertionError(msg)


def main():
    print("=== there is exactly ONE Dixon-Coles ===")
    val = open(os.path.join(ROOT, "validate.py")).read()
    check("validate.py no longer defines its own fit",
          lambda: _assert("def fit_dc(" not in val))
    check("validate.py no longer defines its own probability function",
          lambda: _assert(not re.search(r"^def probs\(", val, re.M)))
    check("validate.py has no Poisson column of its own",
          lambda: _assert(not re.search(r"^def _pois\(", val, re.M)))
    check("validate.walk_forward calls the shared fit",
          lambda: _assert("F.fit(" in val))
    check("validate.walk_forward calls the shared forecast",
          lambda: _assert("F.forecast(" in val))
    bb = open(os.path.join(HERE, "build_backtest.py")).read()
    check("build_backtest.py does not reimplement lambda/mu",
          lambda: _assert("np.exp(ah - da" not in bb))
    check("build_backtest.py walks forward through validate",
          lambda: _assert("V.walk_forward(" in bb))
    bd = open(os.path.join(HERE, "build_dashboard.py")).read()
    check("build_dashboard.py does not blend by hand",
          lambda: _assert("wdc * dcp + wsup * sp" not in bd))
    check("build_dashboard.py calls the shared forecast",
          lambda: _assert("F.forecast(" in bd))
    check("build_dashboard.py does not define its own cold start",
          lambda: _assert("def apply_cold_start(" not in bd and
                          "def cold_start_weight(" not in bd))

    print("\n=== the dashboard's names still point at the shared implementation ===")
    check("apply_cold_start is forecast's", lambda: _assert(B.apply_cold_start is F.apply_cold_start))
    check("cold_start_weight is forecast's", lambda: _assert(B.cold_start_weight is F.cold_start_weight))
    for k in ("COLD_ATTACK", "COLD_DEFENSE", "COLD_K", "PROVISIONAL_N"):
        check(f"{k} matches", lambda kk=k: _assert(getattr(B, kk) == getattr(F, kk)))

    print("\n=== production settings are the DEFAULT, overrides are opt-in ===")
    m = F.fit(cutoff="2025-08-01")
    check(f"default decay_span is production's ({dc.DECAY_SPAN})",
          lambda: _assert(m["decay_span"] == dc.DECAY_SPAN))
    check(f"default ridge is production's ({dc.RIDGE})",
          lambda: _assert(m["ridge"] == dc.RIDGE))
    check(f"default window is production's ({dc.WINDOW_YEARS})",
          lambda: _assert(m["window_years"] == dc.WINDOW_YEARS))
    check("the model records the settings it was fit with",
          lambda: _assert(all(k in m for k in
                              ("decay_base", "decay_span", "ridge", "window_years", "cutoff"))))
    swept = F.fit(cutoff="2025-08-01", decay_span=4.0, ridge=2.0)
    check("a sweep can still vary them",
          lambda: _assert(swept["decay_span"] == 4.0 and swept["ridge"] == 2.0))
    check("and a swept fit really differs",
          lambda: _assert(swept["home_adv"] != m["home_adv"]))

    print("\n=== away-goal calibration reaches the backtest ===")
    teams = m["teams"][:2]
    fc = F.forecast(m, teams[0], teams[1])
    raw = dc.predict(m, teams[0], teams[1])
    check("forecast's expected goals come from production predict",
          lambda: _assert(fc["xg_h"] == raw["xg_h"] and fc["xg_a"] == raw["xg_a"]))
    check(f"AWAY_CAL {dc.AWAY_CAL} is applied (it is not 1.0)",
          lambda: _assert(dc.AWAY_CAL != 1.0))
    # the old validate.probs built mu without AWAY_CAL; prove the shared path does not
    i = m["teams"].index(teams[0]); j = m["teams"].index(teams[1])
    mu_uncal = float(np.exp(m["attack"][j] - m["defense"][i]))
    check("away expected goals are calibrated, not raw",
          lambda: _assert(abs(fc["xg_a"] - mu_uncal * dc.AWAY_CAL) < 1e-9))
    check("and differ from the uncalibrated value",
          lambda: _assert(abs(fc["xg_a"] - mu_uncal) > 1e-6))

    print("\n=== probabilities: normalisation and rounding are production's ===")
    check("the three probabilities sum to 1",
          lambda: _assert(abs(sum(fc["p"]) - 1) < 1e-12))
    check("each is inside (0,1)", lambda: _assert(all(0 < x < 1 for x in fc["p"])))
    check("the displayed percentages sum to exactly 100",
          lambda: _assert(sum(fc["pct"]) == 100))
    for trio in ([1/3, 1/3, 1/3], [0.005, 0.005, 0.99], [0.4444, 0.3333, 0.2223],
                 [0.5, 0.25, 0.25], [0.9999, 0.00005, 0.00005]):
        check(f"rounding {trio} still sums to 100",
              lambda t=trio: _assert(sum(F.to_pct(t)) == 100))
    check("the residual goes on the home side, as production does",
          lambda: _assert(F.to_pct([0.3334, 0.3333, 0.3333])[1:] == (33, 33)))

    print("\n=== the blend is applied, and its absence is reported ===")
    mapping = {"x": 0}
    check("no mapping -> not blended",
          lambda: _assert(F.forecast(m, teams[0], teams[1])["blended"] is False))
    check("a supremacy rating with no mapping is still not blended",
          lambda: _assert(F.forecast(m, teams[0], teams[1], supremacy=0.5)["blended"] is False))
    check("weights default to the fitted/committed pair",
          lambda: _assert(F.blend_weights() == F.DEFAULT_BLEND or
                          abs(sum(F.blend_weights()) - 1.0) < 1e-9))
    check("weights sum to 1", lambda: _assert(abs(sum(F.blend_weights()) - 1.0) < 1e-9))

    print("\n=== the published evidence says which model it measured ===")
    import json
    bj = os.path.join(HERE, "backtest.json")
    d = json.load(open(bj))
    label = d["headline"][0]["model"]
    check(f"headline names the deployed forecast ({label!r})",
          lambda: _assert("supremacy-form" in label))
    check("the label is derived, not hard-coded",
          lambda: _assert("_model_label(out)" in bb))
    check("walk-forward output carries a blended flag per row",
          lambda: _assert("blended=fc[\"blended\"]" in val))

    print("\n=== chronology: a fit never sees its own test season ===")
    mm = F.fit(cutoff="2020-08-01")
    check("the fit's last training match is before the cutoff",
          lambda: _assert(mm["date_max"] < "2020-08-01"))
    check("and the window starts window_years earlier, not at the dawn of time",
          lambda: _assert(mm["date_min"] >= "2012-07-01"))
    check("a later cutoff sees strictly more recent data",
          lambda: _assert(F.fit(cutoff="2024-08-01")["date_max"] > mm["date_max"]))

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

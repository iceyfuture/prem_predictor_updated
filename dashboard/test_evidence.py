"""
Offline tests for RULE 41 — evidence measurement and continuous shrinkage.
No network, no Claude. Audit finding §3.

    ./.venv/bin/python dashboard/test_evidence.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_dashboard as B

PASS = FAIL = 0
def check(name, fn, expect=None):
    global PASS, FAIL
    try: fn()
    except Exception as e:
        if expect and isinstance(e, expect): print(f"  ok   {name}"); PASS += 1
        else: print(f"  FAIL {name}: {type(e).__name__}: {e}"); FAIL += 1
        return
    if expect: print(f"  FAIL {name}: expected {expect.__name__}"); FAIL += 1
    else: print(f"  ok   {name}"); PASS += 1
def _assert(c):
    if not c: raise AssertionError("assertion failed")

W = B.cold_start_weight
K, N = B.COLD_K, B.PROVISIONAL_N

def main():
    print("=== the weight curve is well-formed ===")
    check("0 matches -> 0", lambda: _assert(W(0) == 0.0))
    check("negative -> 0", lambda: _assert(W(-5) == 0.0))
    check("at the threshold -> exactly 1", lambda: _assert(W(N) == 1.0))
    check("beyond the threshold -> 1", lambda: _assert(W(N*10) == 1.0))
    check("always within [0,1]",
          lambda: _assert(all(0.0 <= W(n) <= 1.0 for n in range(0, 500))))
    check("monotonically non-decreasing",
          lambda: _assert(all(W(n) <= W(n+1) for n in range(0, 200))))

    print("\n=== CONTINUOUS: no jump at the old cliff ===")
    step = W(N) - W(N - 0.1)
    check(f"step across the threshold is tiny ({step:.4f})", lambda: _assert(step < 0.01))
    old = lambda n: 1.0 if n >= N else (n/(n+K) if n > 0 else 0.0)
    old_step = old(N) - old(N - 0.1)
    check(f"the OLD rule jumped {old_step:.3f} there", lambda: _assert(old_step > 0.25))
    check("no single-unit jump anywhere exceeds 0.25",
          lambda: _assert(max(W(n+1) - W(n) for n in range(0, 200)) < 0.25))

    print("\n=== thin clubs are shrunk, established are not ===")
    for n, lo, hi in [(0,0,0.01),(1,0.05,0.15),(5,0.25,0.45),(15,0.60,0.75),
                      (40,1.0,1.0),(100,1.0,1.0),(304,1.0,1.0)]:
        check(f"{n:>3} matches -> {lo}..{hi}", lambda n=n,lo=lo,hi=hi: _assert(lo <= W(n) <= hi))
    # At a FIXED n the new curve is gentler - it is the same shape rescaled to reach 1.0 at
    # the threshold. The harder shrinkage on a real promoted club comes from the MEASUREMENT
    # fix, not the curve: Hull was measured at 28.47 (-> 0.655) and is now 4.85 (-> 0.344).
    check("at a fixed n the rescaled curve is >= the old one",
          lambda: _assert(all(W(n) >= old(n) - 1e-9 for n in range(0, int(N)))))
    check("a real promoted club ends up shrunk far harder than before",
          lambda: _assert(W(4.85) < old(28.47) - 0.25))

    print("\n=== fit() reports evidence on the RAW scale ===")
    import csv, pandas as pd, prem_dixon_coles as dc
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = [{"date": r['date'], "home_team": r['team'], "away_team": r['opponent'],
             "home_score": float(r['gf']), "away_score": float(r['ga'])}
            for r in csv.DictReader(open(os.path.join(root,'outputs/team_match_2026_27.csv')))
            if r['venue'] == 'H']
    m = dc.fit(extra=pd.DataFrame(rows), verbose=False)
    check("weighted_matches reported", lambda: _assert("weighted_matches" in m))
    check("kish_ess reported", lambda: _assert("kish_ess" in m))
    check("raw_matches reported", lambda: _assert("raw_matches" in m))
    i = {t: j for j, t in enumerate(m["teams"])}
    if "Hull" in i:
        h = i["Hull"]
        check("Hull: 5 real matches",
              lambda: _assert(m["raw_matches"][h] == 5))
        check("Hull: ~5 units of evidence, NOT ~28",
              lambda: _assert(4.0 < m["weighted_matches"][h] < 6.0))
        check("Hull: Kish ESS ~= 5",
              lambda: _assert(4.5 < m["kish_ess"][h] < 5.5))
        check("Hull: shrink weight ~0.34, not ~0.66",
              lambda: _assert(0.25 < W(m["weighted_matches"][h]) < 0.45))
    big = max(range(len(m["teams"])), key=lambda j: m["raw_matches"][j])
    check("an established club is NOT shrunk",
          lambda: _assert(W(m["weighted_matches"][big]) == 1.0))
    check("evidence never exceeds real match count",
          lambda: _assert(all(m["weighted_matches"][j] <= m["raw_matches"][j] + 1e-6
                              for j in range(len(m["teams"])))))

    print("\n=== apply_cold_start behaviour ===")
    fake = {"teams": ["Thin", "Fat"], "attack": [0.5, 0.5], "defense": [0.5, 0.5],
            "weighted_matches": [5.0, 100.0], "home_adv": 0.25, "rho": -0.05}
    out, prov = B.apply_cold_start(dict(fake, attack=list(fake["attack"]),
                                        defense=list(fake["defense"])), ["Thin", "Fat"])
    check("thin club is flagged provisional", lambda: _assert("Thin" in prov))
    check("established club is not", lambda: _assert("Fat" not in prov))
    check("thin club's rating moved toward the prior",
          lambda: _assert(out["attack"][0] < 0.5))
    check("established club's rating untouched",
          lambda: _assert(out["attack"][1] == 0.5))
    unseen = {"teams": [], "attack": [], "defense": [], "weighted_matches": [],
              "home_adv": .25, "rho": -.05}
    out2, prov2 = B.apply_cold_start(unseen, ["Newbie"])
    check("never-seen club gets the pure prior",
          lambda: _assert(out2["attack"][0] == B.COLD_ATTACK))
    check("and is provisional", lambda: _assert("Newbie" in prov2))

    print(f"\n  {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0

if __name__ == "__main__":
    sys.exit(main())

"""
Offline tests for the scorer model — audit finding §7 (RULE 48). No network, no Claude.
Reads the real ledger to check its schema; writes nothing.

    ./.venv/bin/python dashboard/test_scorer.py
"""
import csv, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import pandas as pd
import prem_scorer as ps
import player_ledger as PL

PASS = FAIL = SKIP = 0


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
    global SKIP
    shares = ps.load_shares()

    print("=== nobody is declared incapable of scoring ===")
    zeros = [(t, pl) for t, s in shares.items() for pl, v in s.items() if v <= 0]
    check(f"no player in any squad has share 0 ({len(zeros)} found)",
          lambda: _assert(not zeros, f"zero shares: {zeros[:3]}"))
    df = pd.read_csv(ps.LINKED)
    df["w_goals"] = pd.to_numeric(df.w_goals, errors="coerce").fillna(0.0)
    nohist = df[df.w_goals <= 0]
    check(f"the {len(nohist)} players with no PL goals are still priced",
          lambda: _assert(all(ps._surname(r.player) in {ps._surname(x) for x in shares.get(ps.canon_team(r.team_2026_27), {}).index}
                              for _, r in nohist.head(40).iterrows())))
    check("every club's shares sum to 1",
          lambda: _assert(all(abs(s.sum() - 1) < 1e-9 for s in shares.values())))
    check("all 20 clubs have scorer data",
          lambda: _assert(len(shares) == 20, f"{len(shares)} clubs"))

    print("\n=== the rate is a proper posterior ===")
    R = ps.player_rate
    check("no PL minutes -> exactly the position prior",
          lambda: _assert(abs(R(0, 0, "FWD") - ps.PRIOR_G90["FWD"]) < 1e-12))
    check("a goalkeeper's prior is tiny but not zero",
          lambda: _assert(0 < R(0, 0, "GKP") < 0.001))
    check("an unknown position falls back to midfield, not to zero",
          lambda: _assert(R(0, 0, "WINGER") == R(0, 0, ps.DEFAULT_POS)))
    check("more goals at the same minutes -> a higher rate",
          lambda: _assert(R(10, 2000, "FWD") > R(2, 2000, "FWD")))
    check("the same goals over more minutes -> a lower rate",
          lambda: _assert(R(10, 4000, "FWD") < R(10, 2000, "FWD")))
    check("a long record dominates the prior",
          lambda: _assert(abs(R(90, 9000, "FWD") - 0.9) < 0.06))
    # A posterior's steps are naturally largest where the denominator is smallest, so a
    # flat bound on step size is the wrong test -- it fails on smooth curves. What matters
    # is that there is no THRESHOLD: the steps must shrink monotonically, never spike.
    steps = [abs(R(5, m, "FWD") - R(5, m + 90, "FWD")) for m in range(90, 9000, 90)]
    check("the rate has no threshold jump (steps shrink monotonically)",
          lambda: _assert(all(a >= b - 1e-12 for a, b in zip(steps, steps[1:])),
                          "a step grew, which means a cliff"))
    check("steps become negligible once a record exists",
          lambda: _assert(steps[-1] < 1e-3))   # <0.001 goals/90 per extra match
    check("the rate is always positive",
          lambda: _assert(all(R(g, m, p) > 0 for g in (0, 1, 30)
                              for m in (0, 500, 9000) for p in ps.PRIOR_G90)))

    print("\n=== a little data never scores worse than none ===")
    # the McBurnie inversion: a proven forward ranked below a team-mate who has never
    # played in the Premier League, because his own PL minutes were used as his minutes
    # expectation. An estimator must never punish a player for having some evidence.
    for pos in ("FWD", "MID", "DEF"):
        base = ps.PRIOR_G90[pos]
        check(f"{pos}: a player scoring at the prior rate ties a newcomer",
              lambda b=base, p=pos: _assert(abs(R(b * 10, 900, p) - R(0, 0, p)) < 0.02))
        check(f"{pos}: a player scoring ABOVE the prior beats a newcomer",
              lambda b=base, p=pos: _assert(R(b * 30, 900, p) > R(0, 0, p)))

    print("\n=== probabilities are well-formed ===")
    t = sorted(shares)[0]
    p = ps.scorer_probs(t, 1.5, shares)
    check("every probability is strictly inside (0,1)",
          lambda: _assert(all(0 < v < 1 for v in p.values())))
    check("a higher team lambda raises every player's chance",
          lambda: _assert(all(ps.scorer_probs(t, 2.5, shares)[k] > v for k, v in p.items())))
    check("lambda 0 gives everyone 0",
          lambda: _assert(all(v == 0 for v in ps.scorer_probs(t, 0.0, shares).values())))
    check("match_scorers returns the top n, highest first",
          lambda: _assert([v for _, v in ps.match_scorers(t, 1.5, shares, n=5)] ==
                          sorted((v for _, v in ps.match_scorers(t, 1.5, shares, n=5)), reverse=True)))
    check("an unknown club yields nothing rather than raising",
          lambda: _assert(ps.scorer_probs("Not A Club", 1.5, shares) == {}))

    print("\n=== availability and minutes ===")
    top = max(p, key=p.get)
    out = {(t, ps._surname(top)): 0.0}
    after = ps.scorer_probs(t, 1.5, shares, avail=out)
    check("a player ruled out drops to 0",
          lambda: _assert(after[top] == 0.0))
    check("his share goes to team-mates, who all rise",
          lambda: _assert(all(after[k] > p[k] for k in p if k != top)))
    # a realistic map: every squad member present, the tail on zero minutes
    mins = {(t, ps._surname(pl)): (90.0 if i < 10 else 0.0) for i, pl in enumerate(p)}
    m_after = ps.scorer_probs(t, 1.5, shares, minutes=mins)
    check("expected minutes of 0 is equivalent to being out",
          lambda: _assert(all(m_after[pl] == 0.0 for pl in list(p)[10:])))
    check("and the players who do play still carry the whole team's goals",
          lambda: _assert(all(m_after[pl] > 0 for pl in list(p)[:10])))

    print("\n=== a player the minutes map does not list is NOT a starter ===")
    # The bug this replaces: absence from the map fell through to avail's default of 1.0,
    # so a player FPL does not list at all was weighted as a nailed-on 90 minutes. Brighton's
    # Mark O'Mahony -- 7.6 weighted PL minutes, not in FPL's squad -- was published as the
    # club's most likely scorer at 20%.
    everyone = list(p)
    partial = {(t, ps._surname(pl)): 90.0 for pl in everyone[:8]}
    got = ps.scorer_probs(t, 1.5, shares, minutes=partial)
    check("a covered club's unlisted players drop to zero",
          lambda: _assert(all(got[pl] == 0.0 for pl in everyone[8:])))
    check("the listed ones are still priced",
          lambda: _assert(all(got[pl] > 0 for pl in everyone[:8])))
    check("an unlisted player cannot outrank a listed one",
          lambda: _assert(max(got[pl] for pl in everyone[8:]) <=
                          min(got[pl] for pl in everyone[:8])))
    check("availability's 1.0 default no longer rescues an unlisted player",
          lambda: _assert(ps.scorer_probs(t, 1.5, shares, avail={}, minutes=partial)
                          [everyone[-1]] == 0.0))
    # but a club the map does not cover at all must not be wiped out
    other = {(  "Not A Club", "someone"): 90.0}
    safe = ps.scorer_probs(t, 1.5, shares, minutes=other)
    check("a club absent from the map entirely is left alone, not zeroed",
          lambda: _assert(all(v > 0 for v in safe.values())))
    check("an empty minutes map is treated as no information",
          lambda: _assert(all(v > 0 for v in ps.scorer_probs(t, 1.5, shares, minutes={}).values())))
    check("expected minutes take precedence over availability",
          lambda: _assert(ps.scorer_probs(t, 1.5, shares, avail={(t, ps._surname(top)): 0.0},
                                          minutes={(t, ps._surname(top)): 90.0})[top] > 0))

    print("\n=== club names are canonical, so the join cannot silently fail ===")
    for a, b in (("Man Utd", "Man United"), ("Spurs", "Tottenham"), ("Hull City", "Hull"),
                 ("Coventry City", "Coventry"), ("Ipswich Town", "Ipswich")):
        check(f"{a!r} resolves to {b!r}", lambda x=a, y=b: _assert(ps.canon_team(x) == y))
    check("an unknown club passes through unchanged",
          lambda: _assert(ps.canon_team("Barnsley") == "Barnsley"))
    check("canon_team is idempotent",
          lambda: _assert(all(ps.canon_team(ps.canon_team(k)) == ps.canon_team(k)
                              for k in ps.TEAM_ALIAS)))
    check("every club in the ledger resolves to one with scorer data",
          lambda: _assert({ps.canon_team(r["team"]) for r in
                           csv.DictReader(open(PL.PATH))} <= set(shares)))

    print("\n=== the ledger grades what the dashboard displays ===")
    check("p_score_shown is a stored column",
          lambda: _assert("p_score_shown" in PL.COLS))
    check("brier_score_shown is a stored column",
          lambda: _assert("brier_score_shown" in PL.COLS))
    check("the new columns are appended, so old rows migrate to blank",
          lambda: _assert(PL.COLS[-2:] == ["p_score_shown", "brier_score_shown"]))
    src = open(os.path.join(HERE, "simulate_fpl.py")).read()
    check("project_gw captures the displayed probability at build time",
          lambda: _assert("ps.scorer_probs(" in src and "p_score_shown" in src))
    check("it is emitted blank, not zero, when the player was never priced",
          lambda: _assert('if p["id"] in p_shown else ""' in src))
    bd = open(os.path.join(HERE, "build_dashboard.py")).read()
    check("the fixture card and the projection share one minutes map",
          lambda: _assert("scorer_minutes" in bd and "minutes=scorer_minutes" in bd))
    # `sf` is imported INSIDE the build function, which makes it function-local for the whole
    # body. The first version of this block used sf.fpl_players() above that import, raised
    # UnboundLocalError, and was swallowed by the block's own except -- so the build printed
    # a one-line warning, fell back to availability weights, and shipped the diluted shares
    # this feature exists to prevent. Nothing failed; the feature was simply off.
    first_use = bd.find("sf.fpl_players()")
    first_import = bd.find("import simulate_fpl as sf")
    check("simulate_fpl is imported before its first use in the build",
          lambda: _assert(first_import != -1 and first_use != -1 and first_import < first_use))
    led = open(os.path.join(HERE, "player_ledger.py")).read()
    check("the ledger no longer claims xG/90 is the displayed pipeline",
          lambda: _assert("so it grades the same pipeline the dashboard displays" not in led))
    check("a blank shown-probability is excluded from the score, not read as 0",
          lambda: _assert('_f(rec.get("p_score_shown")) is not None' in led))

    print("\n=== INVARIANT: nobody is published who is not in a club's FPL squad ===")
    # The end-to-end version of the same check. This is the one that would have caught
    # O'Mahony on the live desk rather than in a unit test.
    import json as _json
    _dj = os.path.join(HERE, "dashboard.json")
    if not os.path.exists(_dj):
        print("  SKIP dashboard.json not built"); SKIP += 1
    else:
        try:
            import simulate_fpl as _sf
            _players = _sf.fpl_players()
        except Exception as _e:
            print(f"  SKIP FPL squads unavailable offline ({type(_e).__name__})"); SKIP += 1
            _players = None
        if _players:
            _fpl = {(ps.canon_team(q["ot"]), ps._surname(q["name"])) for q in _players}
            from datetime import datetime as _dt, timezone as _tz
            _now = _dt.now(_tz.utc)
            _bad, _tot = [], 0
            for _w in _json.load(open(_dj)).get("weeks", []):
                for _m in _w.get("matches", []):
                    _u = _m.get("utc")
                    if not _u or _dt.fromisoformat(_u.replace("Z", "+00:00")) <= _now:
                        continue
                    for _side, _club in (("h", _m["home"]), ("a", _m["away"])):
                        for _s in (_m.get("scorers", {}).get(_side) or []):
                            _tot += 1
                            if (ps.canon_team(_club), ps._surname(_s["n"])) not in _fpl:
                                _bad.append((_club, _s["n"]))
            check(f"every published scorer is in an FPL squad ({_tot} checked, {len(_bad)} not)",
                  lambda: _assert(not _bad))

    print("\n=== summary reports the displayed model on its own rows ===")
    s = PL.summary()
    for k in ("n_score_shown", "brier_score_shown", "cal_score_shown"):
        check(f"summary exposes {k}", lambda kk=k: _assert(kk in s))
    check("the displayed model is not averaged into the xG/90 figure",
          lambda: _assert("brier_score" in s and "brier_score_shown" in s))
    rows = list(csv.DictReader(open(PL.PATH)))
    shown = [r for r in rows if (r.get("p_score_shown") or "").strip()]
    if not shown:
        print(f"  SKIP no rows carry p_score_shown yet ({len(rows)} rows predate RULE 48)")
        print("       the column grades prospectively from the next build; it is not backfilled,")
        print("       because reconstructing a displayed number after the fact is retrodiction")
        SKIP += 1
    else:
        check("every stored shown-probability is a valid probability",
              lambda: _assert(all(0 <= float(r["p_score_shown"]) <= 1 for r in shown)))

    print(f"\n  {PASS} passed, {FAIL} failed, {SKIP} skipped")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

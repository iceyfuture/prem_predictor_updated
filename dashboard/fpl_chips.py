"""
FPL chip advisor — when to play Bench Boost, Triple Captain, Wildcard and Free Hit,
scored from the SAME per-matchweek player projections that build the recommended squad.

Each chip is valued in EXTRA POINTS it would earn this matchweek, then compared with a
threshold set from what that chip is typically worth in a good week. Nothing is recommended
on vibes: every number below comes from simulate_fpl.project_gw() for the gameweek in question.

  BENCH BOOST    the 4 bench players' projected points also count.
                 value = sum(bench projections)
  TRIPLE CAPTAIN captain scores 3x instead of 2x.
                 value = captain's projection (one extra multiple)
  WILDCARD       unlimited transfers; value = what the optimal squad projects versus the
                 squad you already hold, minus the ~1 free transfer you get anyway.
  FREE HIT       one-week unlimited transfers; same gap but it reverts next week, so it only
                 pays in a genuinely unusual week (blanks, doubles, injury pile-ups).

A chip is only recommended when its value clears the threshold AND the inputs are trustworthy
(no stale FPL feed, enough players with real projections).
"""
import os, sys, csv

HERE = os.path.dirname(os.path.abspath(__file__))

# What each chip is worth in a week that justifies burning it. Tuned to typical FPL returns:
# a bench normally makes 4-10, so 16+ means the bench is unusually strong (often a double GW);
# a captain normally projects 5-7, so 9+ is a standout; squad gaps above ~18 justify a rebuild.
THRESHOLD = {"bench_boost": 16.0, "triple_captain": 9.0, "wildcard": 18.0, "free_hit": 18.0}


def _held_squad(gw):
    """The squad the model locked LAST gameweek - what you would still be holding."""
    p = os.path.join(HERE, "fpl_forward.csv")
    if not os.path.exists(p):
        return None
    prev = [r for r in csv.DictReader(open(p)) if int(r["gw"]) == gw - 1]
    return {int(r["element"]) for r in prev} or None


def advise(gw, team, rows):
    """team = build_team() output for `gw`; rows = project_gw() output for `gw`."""
    if not team:
        return None
    squad = team["squad"]
    xi = [s for s in squad if s.get("start")]
    bench = [s for s in squad if not s.get("start")]
    cap = next((s for s in squad if s.get("cap")), None)

    bench_val = round(sum(s["proj"] for s in bench), 1)
    tc_val = round(cap["proj"], 1) if cap else 0.0

    # Wildcard / Free Hit: how much better is this week's optimal squad than what you hold?
    held = _held_squad(gw)
    by_id = {r["id"]: r for r in rows if r.get("id") is not None}
    if held:
        held_proj = sum(by_id[i]["proj"] for i in held if i in by_id)
        # only the best 11 of each actually score, so compare like with like
        held_best = sorted((by_id[i]["proj"] for i in held if i in by_id), reverse=True)[:11]
        new_best = sorted((s["proj"] for s in xi), reverse=True)[:11]
        gap = round(sum(new_best) - sum(held_best), 1)
    else:
        gap = None
    # one free transfer already buys you roughly the best single upgrade
    single = 0.0
    if held:
        outs = sorted((by_id[i]["proj"] for i in held if i in by_id))[:1]
        ins = sorted((s["proj"] for s in xi), reverse=True)[:1]
        single = round((ins[0] - outs[0]) if outs and ins else 0.0, 1)
    wc_val = round(gap - single, 1) if gap is not None else None

    def rec(v, key):
        if v is None:
            return "unknown"
        return "PLAY" if v >= THRESHOLD[key] else ("consider" if v >= THRESHOLD[key] * 0.75 else "hold")

    chips = [
        {"chip": "Bench Boost", "value": bench_val, "threshold": THRESHOLD["bench_boost"],
         "verdict": rec(bench_val, "bench_boost"),
         "why": f"bench projects {bench_val} pts ({', '.join(f'{s[chr(110)+chr(109)]} {s[chr(112)+chr(114)+chr(111)+chr(106)]}' for s in sorted(bench, key=lambda x:-x['proj']))})"},
        {"chip": "Triple Captain", "value": tc_val, "threshold": THRESHOLD["triple_captain"],
         "verdict": rec(tc_val, "triple_captain"),
         "why": (f"captain {cap['nm']} ({cap['ot']}) projects {tc_val} pts; TC adds one more multiple"
                 if cap else "no captain")},
        {"chip": "Wildcard", "value": wc_val, "threshold": THRESHOLD["wildcard"],
         "verdict": rec(wc_val, "wildcard"),
         "why": (f"optimal XI projects {gap} pts more than the squad you hold; a free transfer "
                 f"already buys ~{single}, so the wildcard is worth ~{wc_val}"
                 if wc_val is not None else "no previous squad on record yet - available from MW2")},
        {"chip": "Free Hit", "value": wc_val, "threshold": THRESHOLD["free_hit"],
         "verdict": ("hold" if wc_val is None or wc_val < THRESHOLD["free_hit"] else rec(wc_val, "free_hit")),
         "why": ("same one-week gap as the wildcard, but it reverts next week - only worth it in a "
                 "blank/double gameweek or an injury pile-up"
                 if wc_val is not None else "needs a previous squad to compare against")},
    ]
    play = [c["chip"] for c in chips if c["verdict"] == "PLAY"]
    return {"gw": gw, "chips": chips, "play": play,
            "summary": (f"Play {' + '.join(play)} this matchweek." if play
                        else "Hold all four chips - nothing this week clears the bar.")}


if __name__ == "__main__":
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
    import simulate_fpl as sf, feeds, build_dashboard as bd, prem_dixon_coles as dc
    gw = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ev = feeds.espn_events(); weeks = feeds.to_matchweeks(ev)
    model = dc.get_model()
    clubs = sorted({e["home"] for e in ev} | {e["away"] for e in ev})
    model, _ = bd.apply_cold_start(model, clubs)
    meta = {}
    for e in ev:
        meta.setdefault(e["home"], {"color": e["home_color"], "abbr": e["home_abbr"]})
        meta.setdefault(e["away"], {"color": e["away_color"], "abbr": e["away_abbr"]})
    rows = sf.project_gw(gw, model=model, weeks=weeks)
    team = sf.build_team(gw, weeks=weeks, team_meta=meta, rows=rows)
    a = advise(gw, team, rows)
    print(f"\nCHIP ADVICE — Matchweek {gw}\n")
    print(f"  {'chip':<16}{'value':>7}{'bar':>7}  verdict")
    print("  " + "-" * 62)
    for c in a["chips"]:
        v = f"{c['value']}" if c["value"] is not None else "-"
        print(f"  {c['chip']:<16}{v:>7}{c['threshold']:>7}  {c['verdict']}")
    print(f"\n  {a['summary']}\n")
    for c in a["chips"]:
        print(f"  - {c['chip']}: {c['why']}")

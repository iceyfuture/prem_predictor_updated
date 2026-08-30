"""
Compile every FPL player's individual statistics to CSV, including the expected-stats family
(xG, xA, xGI, xGC) the FPL /statistics page exposes one column at a time.

Source is `bootstrap-static`, which is what that page renders from - verified field-by-field
against the live page (price and total points matched on 8/8 distinct players checked). Using
the API gets all 616 players and every column in one request instead of paginating a table.

Writes two files:
  fpl_player_stats_2026_27.csv     season totals + per-90 + derived over/under-performance
  fpl_player_gameweek_2026_27.csv  per-player per-gameweek, from event/{gw}/live
"""
import csv, json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feeds

# repo-relative so a clone works from any directory, on any machine
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
API = "https://fantasy.premierleague.com/api"


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def season_totals():
    b = feeds._get(feeds.FPL, "fpl_bootstrap.json", max_age_min=60)
    tm = {t["id"]: t["name"] for t in b["teams"]}
    rows = []
    for p in b["elements"]:
        xg, xa = _f(p.get("expected_goals")), _f(p.get("expected_assists"))
        g, a = p.get("goals_scored", 0), p.get("assists", 0)
        mins = p.get("minutes", 0)
        rows.append({
            "id": p["id"], "name": p["web_name"],
            "full_name": f"{p.get('first_name','')} {p.get('second_name','')}".strip(),
            "team": tm[p["team"]], "pos": POS[p["element_type"]],
            "price": p["now_cost"] / 10.0,
            "status": p.get("status"), "chance_next": p.get("chance_of_playing_next_round"),
            "news": (p.get("news") or "").strip(),
            "minutes": mins, "starts": p.get("starts", 0),
            "total_points": p.get("total_points", 0), "points_per_game": _f(p.get("points_per_game")),
            "form": _f(p.get("form")), "selected_by_pct": _f(p.get("selected_by_percent")),
            "ep_next": _f(p.get("ep_next")),
            # --- the expected-stats family ---
            "xG": xg, "xA": xa,
            "xGI": _f(p.get("expected_goal_involvements")),
            "xGC": _f(p.get("expected_goals_conceded")),
            "xG_per90": _f(p.get("expected_goals_per_90")),
            "xA_per90": _f(p.get("expected_assists_per_90")),
            "xGI_per90": _f(p.get("expected_goal_involvements_per_90")),
            "xGC_per90": _f(p.get("expected_goals_conceded_per_90")),
            # --- actuals, so over/under-performance is computable ---
            "goals": g, "assists": a,
            "goals_minus_xG": round(g - xg, 2), "assists_minus_xA": round(a - xa, 2),
            "clean_sheets": p.get("clean_sheets", 0), "goals_conceded": p.get("goals_conceded", 0),
            "goals_conceded_per90": _f(p.get("goals_conceded_per_90")),
            "clean_sheets_per90": _f(p.get("clean_sheets_per_90")),
            "saves": p.get("saves", 0), "saves_per90": _f(p.get("saves_per_90")),
            "own_goals": p.get("own_goals", 0),
            "pens_saved": p.get("penalties_saved", 0), "pens_missed": p.get("penalties_missed", 0),
            "yellow": p.get("yellow_cards", 0), "red": p.get("red_cards", 0),
            # --- defensive contribution (new-style FPL scoring) ---
            "tackles": p.get("tackles", 0), "cbi": p.get("clearances_blocks_interceptions", 0),
            "recoveries": p.get("recoveries", 0),
            "defensive_contribution": p.get("defensive_contribution", 0),
            "def_contrib_per90": _f(p.get("defensive_contribution_per_90")),
            # --- ICT ---
            "influence": _f(p.get("influence")), "creativity": _f(p.get("creativity")),
            "threat": _f(p.get("threat")), "ict_index": _f(p.get("ict_index")),
            "bonus": p.get("bonus", 0), "bps": p.get("bps", 0),
        })
    rows.sort(key=lambda r: -r["total_points"])
    return rows


def gameweeks(upto, current=None):
    """Per-player per-gameweek rows from event/{gw}/live - 29 stats per player per week.

    `current` is an IN-PROGRESS gameweek: pulled too, flagged `provisional=True`. Without this
    a gameweek with 9 of 10 games played is invisible to every downstream consumer until the
    last match ends, which is exactly when form data is most wanted.
    """
    out, failed = [], []
    span = list(range(1, upto + 1))
    if current and current not in span:
        span.append(current)
    for gw in span:
        # Retry: an unattended run hit "Connection reset by peer" and silently dropped a whole
        # gameweek of player data. A dropped gameweek is invisible in the output, so it must not
        # be a single-attempt fetch.
        d, err = None, None
        for attempt in range(4):
            try:
                r = urllib.request.Request(f"{API}/event/{gw}/live/",
                                           headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(r, timeout=30) as f:
                    d = json.load(f)
                break
            except Exception as e:
                err = e
                if attempt < 3:
                    time.sleep(2 ** attempt)
        if d is None:
            print(f"  GW{gw}: FAILED after 4 attempts ({err})")
            failed.append(gw)
            continue
        for el in d.get("elements", []):
            s = el.get("stats", {})
            if not s.get("minutes"):
                continue
            out.append({"gw": gw, "provisional": bool(current and gw == current),
                        "id": el["id"], **{k: s.get(k) for k in sorted(s)}})
        print(f"  GW{gw}: {sum(1 for x in out if x['gw'] == gw)} players with minutes"
              + ("  [PROVISIONAL - gameweek still in progress]" if current and gw == current else ""))
    if failed:
        # loud, and non-zero exit, so the scheduler does not record a silent partial success
        print(f"  ERROR: gameweeks {failed} could not be fetched - per-gameweek file is INCOMPLETE")
    return out, failed


def append_history(rows, gw):
    """Append a dated snapshot so the season ROLLS FORWARD instead of being overwritten.

    The season-totals file is a point-in-time snapshot; rewriting it each week would lose the
    trajectory - how a player's xG accumulated, when his price moved, how ownership drifted,
    whether his form was rising or falling. This keeps one row per player per snapshot date,
    de-duplicated on (date, id) so re-running the same day is idempotent.
    """
    path = os.path.join(OUT, "fpl_player_history.csv")
    import datetime
    today = datetime.date.today().isoformat()
    keep = ["id", "name", "team", "pos", "price", "minutes", "starts", "total_points", "form",
            "selected_by_pct", "xG", "xA", "xGI", "xGC", "goals", "assists",
            "goals_minus_xG", "assists_minus_xA", "clean_sheets", "goals_conceded",
            "defensive_contribution", "ict_index", "bps"]
    cols = ["snapshot_date", "gw"] + keep
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r.get("snapshot_date") != today]
    new = [{"snapshot_date": today, "gw": gw, **{k: r[k] for k in keep}} for r in rows]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(old + new)
    dates = sorted({r["snapshot_date"] for r in old} | {today})
    print(f"wrote {path}: {len(old)+len(new)} rows across {len(dates)} snapshots "
          f"({dates[0]} -> {dates[-1]})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = season_totals()
    p1 = os.path.join(OUT, "fpl_player_stats_2026_27.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {p1}: {len(rows)} players x {len(rows[0])} columns")

    b = feeds._get(feeds.FPL, "fpl_bootstrap.json", max_age_min=60)
    cur = max([e["id"] for e in b["events"] if e.get("finished")] or [0])
    live = next((e["id"] for e in b["events"] if not e.get("finished") and e.get("is_current")), None)
    if live is None:
        live = next((e["id"] for e in b["events"]
                     if not e.get("finished") and e.get("deadline_time_epoch", 0)
                     and e["id"] == cur + 1), None)
    append_history(rows, cur or (live or 1))
    gws, failed_gws = gameweeks(cur, current=live)
    if gws:
        p2 = os.path.join(OUT, "fpl_player_gameweek_2026_27.csv")
        keys = sorted({k for r in gws for k in r})
        keys = ["gw", "provisional", "id"] + [k for k in keys if k not in ("gw", "provisional", "id")]
        with open(p2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(gws)
        print(f"wrote {p2}: {len(gws)} rows x {len(keys)} columns")
    if failed_gws:
        sys.exit(1)

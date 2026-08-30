"""
build_dashboard.py  —  LIVE edition
===================================
Merges three sources into dashboard.json for the Floodlit terminal:

  1. ESPN  (public, key-less)  -> real 2026/27 fixtures, kickoff, venue, status, live/final
                                  scores, club colours, and DraftKings moneyline odds.
  2. FPL   (public, key-less)  -> team news, injuries, chance-of-playing, prices, xP.
  3. YOUR MODEL (local state)  -> win/draw/win probabilities, xG, scorelines, scorers.

Every ESPN/FPL field used here was verified against a live response (see feeds.verify_feeds).
Team names are mapped explicitly in feeds.py; anything unmapped is PRINTED, never guessed.

EDGES: where a book has priced a game, edge = model EV against the vig-free line,
       EV% = (model_prob * decimal_odds - 1) * 100. Flagged at >= EDGE_MIN.

RUN:  ~/prem_predictor/.venv/bin/python dashboard/build_dashboard.py
"""
import csv, json, os, sys
from datetime import datetime, timezone, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import prem_dixon_coles as dc          # noqa: E402
import supremacy_odds as so            # noqa: E402
import prem_scorer as ps               # noqa: E402
import feeds                            # noqa: E402
import kalshi                           # noqa: E402
import prem_corners as pcorn           # noqa: E402
import fpl_chips
import fpl_form
import fpl_transfers                        # noqa: E402

OUT = os.path.join(HERE, "dashboard.json")
EDGE_MIN = 3.0          # flag an edge at >= +3% expected value


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


# Cold-start prior for newly-promoted clubs, calibrated on 95 promoted sides (1994-2026):
# their first 5 games average GF 1.01 / GA 1.56 per game and they lose 49% of them. The old
# "bottom-3 of last season" prior was far too generous (bottom-3 are relegation-quality but
# PL-experienced) and let promoted teams look like value plays vs strong sides. These attack/
# defense values reproduce that GF/GA against average opposition given home_adv~0.24.
COLD_ATTACK = -0.12
COLD_DEFENSE = -0.32

# Evidence shrinkage for thin-data clubs: w = n_eff/(n_eff + COLD_K) toward the COLD_* prior.
# Swept walk-forward with per-matchday refits over 14 seasons (sweep_coldstart.py). Overall
# RPS is unchanged to 4dp at every K (established clubs have n_eff in the hundreds, so they
# never move); on thin-data fixtures K=15-25 was best but only n=163 such fixtures exist and
# the paired t is 0.83 — NOT significant. Shipped as a guardrail against the Hull pathology,
# NOT as an accuracy improvement. PROVISIONAL_N keeps Rule 4's flags on until a club has
# roughly a season of weighted evidence.
COLD_K = 15.0
PROVISIONAL_N = 40.0


def apply_cold_start(model, clubs):
    """Promoted clubs have no rating in the window. Give them the empirically-calibrated
    newly-promoted prior (see COLD_* above), NOT a league-average or bottom-3 prior — and
    keep shrinking them toward it until they have actually earned a rating.

    Why the shrink exists: Rule 1 refits after every matchday, so a club with no history in
    the 8-year window gets a FULL-STRENGTH rating off its first result. In 26/27 Hull beat
    Man United 2-0 in MW1 and came out with the best defence in the league, rated 4th
    overall on one game — and because it was no longer "missing" it also lost its
    provisional flag, switching off every Rule 4 guardrail exactly when they mattered most.

    Shrinkage: w = n_eff/(n_eff + COLD_K) on the model's own time-weighted match count, so
    established clubs (n_eff in the hundreds) are untouched and only thin-data clubs move.
    See MODEL_RULES.md — this is a GUARDRAIL, not a validated accuracy gain.
    """
    idx = {t: i for i, t in enumerate(model["teams"])}
    provisional = set()

    # (1) never seen in the window -> pure prior
    for c in [c for c in clubs if c not in idx]:
        model["teams"].append(c)
        model["attack"].append(COLD_ATTACK)
        model["defense"].append(COLD_DEFENSE)
        if isinstance(model.get("weighted_matches"), list):
            model["weighted_matches"].append(0.0)
        provisional.add(c)

    # (2) rated but thin -> shrink toward the prior by how much evidence there actually is
    idx = {t: i for i, t in enumerate(model["teams"])}
    wm = model.get("weighted_matches") or []
    for c in clubs:
        i = idx[c]
        n = float(wm[i]) if i < len(wm) else 0.0
        if n >= PROVISIONAL_N:
            continue
        provisional.add(c)
        if n > 0:
            w = n / (n + COLD_K)
            model["attack"][i] = w * model["attack"][i] + (1 - w) * COLD_ATTACK
            model["defense"][i] = w * model["defense"][i] + (1 - w) * COLD_DEFENSE
    return model, provisional


def top_scores(model, h, a, k=3):
    M, _, _ = dc.score_matrix(model, h, a)
    flat = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
    return [{"s": f"{int(i)}-{int(j)}", "p": round(float(M[i, j]) * 100, 1)} for i, j in flat[:k]]


def build_strength_2627(model, clubs, cold):
    """Strength index for THIS season's 20 clubs (not last season's): simulate a full
    double round-robin among them with the cold-started model and score 3*P(win)+1*P(draw).
    Promoted clubs run on their cold-start prior and are marked provisional — their number
    will move once they actually play and the model retrains on real results."""
    idx = {t: i for i, t in enumerate(model["teams"])}
    a = np.array(model["attack"]); d = np.array(model["defense"])
    rows = []
    for t in clubs:
        pts = 0.0
        for o in clubs:
            if o == t:
                continue
            ph = dc.predict(model, t, o)      # t home
            pa = dc.predict(model, o, t)      # t away
            pts += 3 * ph["win_h"] + ph["draw"] + 3 * pa["win_a"] + pa["draw"]
        i = idx[t]
        rows.append({"team": t, "attack": round(float(a[i]), 3), "defense": round(float(d[i]), 3),
                     "net": round(float(a[i] + d[i]), 3), "proj_points": round(pts, 1),
                     "provisional": t in cold})
    rows.sort(key=lambda r: -r["proj_points"])
    top = rows[0]["proj_points"]
    for k, r in enumerate(rows, 1):
        r["rank"] = k
        r["idx"] = round(100 * r["proj_points"] / top, 1)
    return rows


def forward_test(active_gw, team, weeks, events, built_at):
    """Snapshot the active gameweek's recommended squad (projected points, locked pre-deadline),
    and grade any past snapshotted week against the ACTUAL FPL points players scored. This is the
    always-on forward test: how well did the model's picks really do, matchweek by matchweek."""
    path = os.path.join(HERE, "fpl_forward.csv")
    rows = {}
    if os.path.exists(path):
        for r in read_csv(path):
            rows[(int(r["gw"]), int(r["element"]))] = r
    # LOCK ONCE PER GAMEWEEK. The squad is regenerated every build (by design - the model
    # re-learns), so without this guard each rebuild ADDS its new picks to the same gameweek
    # and the "XI" silently grows past 11. Once a gameweek has any snapshot, it is sealed.
    # Snapshot at REVEAL, not during the locked period: before reveal the squad is only a
    # preview built on stale prices/availability, and grading that is not the product promise.
    # HARD GUARD against leakage: never snapshot a gameweek whose first ball has been kicked.
    # (Clearing a bad snapshot once let GW1 re-lock after 9 of its 10 games had finished, by
    # which point the model had already refit on those results - a squad picked with hindsight.)
    ko = team.get("first_kickoff")
    before_ko = True
    if ko:
        try:
            before_ko = datetime.now(timezone.utc) < datetime.fromisoformat(ko.replace("Z", "+00:00"))
        except ValueError:
            before_ko = True
    already = {k[0] for k in rows}
    if active_gw not in already and not team.get("locked") and before_ko:
        for s in team.get("squad", []):
            if s.get("id") is None:
                continue
            rows[(active_gw, int(s["id"]))] = {
                "gw": active_gw, "element": s["id"], "player": s["nm"], "pos": s["pos"],
                "start": str(bool(s.get("start"))), "cap": str(bool(s.get("cap"))),
                "proj": s.get("proj", 0),
                "snapshot_at": built_at, "actual": "", "graded": ""}
    finished_gws = {wk["gw"] for wk in weeks
                    if all(fx["id"] in {e["id"] for e in events if e["finished"]} for fx in wk["fixtures"])}
    for gw in sorted({k[0] for k in rows}):
        if gw in finished_gws and any(not rows[k].get("graded") for k in rows if k[0] == gw):
            live = feeds.fpl_event_live(gw)
            for k in rows:
                if k[0] == gw and not rows[k].get("graded"):
                    rows[k]["actual"] = live.get(k[1], 0); rows[k]["graded"] = built_at
    cols = ["gw", "element", "player", "pos", "start", "cap", "proj", "snapshot_at", "actual", "graded"]
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows.values():
            w.writerow(r)
    summary = []
    for gw in sorted({k[0] for k in rows}):
        xi = [rows[k] for k in rows if k[0] == gw and str(rows[k]["start"]) == "True"]
        graded = bool(xi) and all(rows[(gw, int(r["element"]))].get("graded") for r in xi)
        # FPL doubles the captain's score - count it on both the projection and the actual
        def tot(field):
            return sum(float(r.get(field) or 0) * (2 if str(r.get("cap")) == "True" else 1) for r in xi)
        snap = min((r.get("snapshot_at") or "") for r in xi) if xi else ""
        ko_gw = next((min(fx["utc"] for fx in wk["fixtures"]) for wk in weeks if wk["gw"] == gw), "")
        tainted = bool(snap and ko_gw and snap > ko_gw)   # snapshot taken after kickoff
        summary.append({"gw": gw, "proj": round(tot("proj"), 1),
                        "actual": round(tot("actual"), 1) if graded else None,
                        "captain": next((r["player"] for r in xi if str(r.get("cap")) == "True"), None),
                        "graded": graded, "tainted": tainted})
    ng = sum(1 for s in summary if s["graded"])
    print(f"  forward-test: {len(rows)} snapshotted picks, {ng} graded gameweeks")
    return {"current_gw": active_gw, "weeks": summary}


def _brier(p, outcome):
    """Multiclass Brier for a 3-way (H,D,A) forecast. Lower = better."""
    y = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}[outcome]
    return round(sum((pi - yi) ** 2 for pi, yi in zip(p, y)), 4)


def _rps(p, outcome):
    """Ranked probability score for ordered H>D>A. Lower = better."""
    y = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}[outcome]
    cp = [p[0], p[0] + p[1]]
    cy = [y[0], y[0] + y[1]]
    return round(sum((a - b) ** 2 for a, b in zip(cp, cy)) / 2.0, 4)


def refresh_clubs(fpl_bootstrap):
    """Transfers keep happening after the squad list was scraped. The FPL API is live and
    authoritative for who plays where TODAY, so re-derive current club from it on every build
    and rewrite it into the derived CSVs. Without this, a transferred player keeps contributing
    his goal share to his OLD club in the scorer model (e.g. Morgan Rogers still counting for
    Aston Villa after moving to Chelsea)."""
    import re, unicodedata
    def norm(x):
        x = unicodedata.normalize("NFKD", x or "").encode("ascii", "ignore").decode()
        return re.sub(r"[.\-' ]", "", x).lower()
    teams = {t["id"]: feeds.FPL_TEAMS.get(t["name"], t["name"]) for t in fpl_bootstrap.get("teams", [])}
    truth = {}
    for pl in fpl_bootstrap.get("elements", []):
        club = teams.get(pl.get("team"))
        if not club:
            continue
        truth[norm(pl["first_name"] + pl["second_name"])] = club
        truth.setdefault(norm(pl["web_name"]), club)
    changed = []
    for path, namecol, teamcol in (
            (os.path.join(ROOT, "outputs", "squad_2026_27_linked.csv"), "player", "team_2026_27"),
            (os.path.join(ROOT, "outputs", "player_rankings_2026_27.csv"), "player", "latest_team"),
            (os.path.join(ROOT, "outputs", "top50_players.csv"), "player", "latest_team")):
        if not os.path.exists(path):
            continue
        rows = read_csv(path)
        if not rows or teamcol not in rows[0]:
            continue
        for r in rows:
            cur = truth.get(norm(r.get(namecol, "")))
            if cur and cur != r.get(teamcol):
                changed.append((r.get(namecol), r.get(teamcol), cur))
                r[teamcol] = cur
        import csv as _csv
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    if changed:
        uniq = sorted({c[0] for c in changed})
        print(f"  transfers applied from FPL: {len(uniq)} players moved club "
              f"({', '.join(uniq[:6])}{'...' if len(uniq) > 6 else ''})")
    return changed


def record_props_ledger(weeks, built_at):
    """Prop forward test. One row per (fixture, market): the model's price and Kalshi's last
    pre-kickoff quote are locked, then settled from the final score. Binary Brier = (p-y)^2.
    Markets: BTTS, Over N.5 goals, win-by->N.5 spreads."""
    path = os.path.join(HERE, "props_ledger.csv")
    rows = {}
    if os.path.exists(path):
        for r in read_csv(path):
            rows[r["key"]] = r
    for w in weeks:
        for m in w["matches"]:
            pr = m.get("props")
            if not pr:
                continue
            started = m["finished"] or m["live"]
            for p_ in pr["rows"]:
                key = f"{m['home']}|{m['away']}|{p_['kind']}|{p_['side'] or '-'}|{p_['line'] or '-'}"
                rec = rows.get(key, {})
                if not rec:
                    rec = {"key": key, "gw": w["gw"], "home": m["home"], "away": m["away"],
                           "kind": p_["kind"], "side": p_["side"] or "", "line": p_["line"] or "",
                           "label": p_["label"], "locked_at": built_at,
                           "result": "", "settled": "", "graded": ""}
                if not started:      # refresh both closing prices until kickoff, then freeze
                    rec["model"] = p_["model"]
                    rec["close_at"] = built_at
                    if p_["mid"] is not None and not p_["thin"]:
                        rec["kal_mid"], rec["kal_ask"] = p_["mid"], p_["ask"]
                        rec["kal_at"] = built_at
                if m["finished"] and m["result"] and not rec.get("graded"):
                    hs, as_ = (int(x) for x in m["result"].split("-"))
                    line = float(rec["line"]) if rec["line"] not in ("", None) else 0.0
                    y = kalshi.settle_prop(rec["kind"], rec["side"], line, hs, as_)
                    rec["result"] = m["result"]; rec["settled"] = "YES" if y else "NO"
                    rec["graded"] = built_at
                    try:
                        mp = float(rec["model"]) / 100.0
                        rec["brier_model"] = round((mp - y) ** 2, 4)
                    except (KeyError, TypeError, ValueError):
                        pass
                    if rec.get("kal_mid") not in (None, ""):
                        kpv = float(rec["kal_mid"]) / 100.0
                        rec["brier_kalshi"] = round((kpv - y) ** 2, 4)
                        if "brier_model" in rec:
                            rec["closer"] = ("model" if rec["brier_model"] < rec["brier_kalshi"]
                                             else "kalshi" if rec["brier_kalshi"] < rec["brier_model"] else "tie")
                rows[key] = rec
    cols = ["key", "gw", "home", "away", "kind", "side", "line", "label", "locked_at",
            "close_at", "model", "kal_at", "kal_mid", "kal_ask",
            "result", "settled", "graded", "brier_model", "brier_kalshi", "closer"]
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows.values():
            w.writerow(r)
    settled = [r for r in rows.values() if r.get("graded")]
    quoted = sum(1 for r in rows.values() if r.get("kal_mid") not in (None, ""))
    def avg(k, src):
        v = [float(x[k]) for x in src if x.get(k) not in (None, "")]
        return round(sum(v) / len(v), 4) if v else None
    by_kind = {}
    for kind in ("btts", "total", "spread"):
        sub = [r for r in settled if r["kind"] == kind]
        if sub:
            by_kind[kind] = {"n": len(sub), "brier_model": avg("brier_model", sub),
                             "brier_kalshi": avg("brier_kalshi", sub)}
    print(f"  props ledger: {len(rows)} markets locked, {quoted} with a real Kalshi quote, "
          f"{len(settled)} settled")
    return {"tracked": len(rows), "quoted": quoted, "settled": len(settled),
            "brier_model": avg("brier_model", settled), "brier_kalshi": avg("brier_kalshi", settled),
            "model_wins": sum(1 for r in settled if r.get("closer") == "model"),
            "kalshi_wins": sum(1 for r in settled if r.get("closer") == "kalshi"),
            "by_kind": by_kind,
            "board": [{"gw": int(r["gw"]), "fixture": f"{r['home']} v {r['away']}",
                       "label": r["label"], "model": float(r["model"]) if r.get("model") else None,
                       "kalshi": float(r["kal_mid"]) if r.get("kal_mid") not in (None, "") else None,
                       "settled": r["settled"], "result": r["result"],
                       "brier_model": float(r["brier_model"]) if r.get("brier_model") else None,
                       "brier_kalshi": float(r["brier_kalshi"]) if r.get("brier_kalshi") else None,
                       "closer": r.get("closer", "")}
                      for r in sorted(settled, key=lambda x: (-int(x["gw"]), x["home"]))[:60]]}


def record_ledger(weeks, built_at):
    """RULE 5 + Kalshi forward test.

    Per fixture we log THREE forecasts and freeze them at kickoff:
      pred_*        the model's FIRST locked prediction (made far ahead - the strict test)
      close_*       the model's LAST pre-kickoff prediction (its closing number)
      kal_*         Kalshi's LAST pre-kickoff mid  (the exchange's closing price)
      line_*        the sportsbook's last pre-kickoff implied probs
    Kalshi lists EPL only ~a week ahead, so its price is refreshed each build until the game
    starts, then never touched again - that is the closing price, and it is always recorded
    BEFORE kickoff. Once the result lands we score every forecast with Brier + RPS and record
    which was closer. That answers: was the model or Kalshi right?"""
    path = os.path.join(HERE, "ledger.csv")
    rows = {}
    if os.path.exists(path):
        for r in read_csv(path):
            rows[r["key"]] = r
    for w in weeks:
        for m in w["matches"]:
            key = f"{m['home']}|{m['away']}"   # stable: kickoff times move with TV picks
            rec = rows.get(key, {})
            k = m.get("kalshi")
            if not rec.get("pred_at"):        # first sighting -> lock the opening prediction
                rec = {"key": key, "gw": w["gw"], "home": m["home"], "away": m["away"],
                       "kickoff": m["time"], "pred_at": built_at,
                       "pred_h": m["ph"], "pred_d": m["pd"], "pred_a": m["pa"],
                       "result": "", "outcome": "", "graded": ""}
            rec["kickoff"] = m["time"]          # may move with TV picks; key stays stable
            started = m["finished"] or m["live"]
            if not started:
                # refresh the CLOSING numbers on every build until the game starts
                rec["close_h"], rec["close_d"], rec["close_a"] = m["ph"], m["pd"], m["pa"]
                rec["close_at"] = built_at
                if m["mkt"]:
                    rec["line_h"] = round(m["mkt"]["imp"]["h"] * 100, 1)
                    rec["line_d"] = round(m["mkt"]["imp"]["d"] * 100, 1)
                    rec["line_a"] = round(m["mkt"]["imp"]["a"] * 100, 1)
                if k and not k.get("thin") and k["mid"]["h"] is not None:
                    rec["kal_h"], rec["kal_d"], rec["kal_a"] = k["mid"]["h"], k["mid"]["d"], k["mid"]["a"]
                    rec["kal_vig"] = k["vig"]; rec["kal_at"] = built_at; rec["kal_url"] = k["url"]
            if m["finished"] and m["result"] and not rec.get("graded"):
                hs, as_ = (int(x) for x in m["result"].split("-"))
                out = "H" if hs > as_ else ("D" if hs == as_ else "A")
                rec["result"] = m["result"]; rec["outcome"] = out; rec["graded"] = built_at

                def trio(pre):
                    try:
                        v = [float(rec[f"{pre}_h"]), float(rec[f"{pre}_d"]), float(rec[f"{pre}_a"])]
                    except (KeyError, TypeError, ValueError):
                        return None
                    s = sum(v)
                    return [x / s for x in v] if s > 0 else None
                mp, kp, dp = trio("close") or trio("pred"), trio("kal"), trio("line")
                if mp:
                    rec["brier_model"] = _brier(mp, out); rec["rps_model"] = _rps(mp, out)
                if kp:
                    rec["brier_kalshi"] = _brier(kp, out); rec["rps_kalshi"] = _rps(kp, out)
                if dp:
                    rec["brier_book"] = _brier(dp, out)
                if mp and kp:
                    rec["closer"] = ("model" if rec["brier_model"] < rec["brier_kalshi"]
                                     else "kalshi" if rec["brier_kalshi"] < rec["brier_model"] else "tie")
            rows[key] = rec
    cols = ["key", "gw", "home", "away", "kickoff", "pred_at", "pred_h", "pred_d", "pred_a",
            "close_at", "close_h", "close_d", "close_a",
            "kal_at", "kal_h", "kal_d", "kal_a", "kal_vig", "kal_url",
            "line_h", "line_d", "line_a", "result", "outcome", "graded",
            "brier_model", "brier_kalshi", "brier_book", "rps_model", "rps_kalshi", "closer"]
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows.values():
            w.writerow(r)

    # ---- settled-market scorecard for the Model Desk ----
    settled = [r for r in rows.values() if r.get("graded") and r.get("brier_kalshi")]
    tracked = sum(1 for r in rows.values() if r.get("kal_h"))
    def avg(key, src):
        v = [float(r[key]) for r in src if r.get(key) not in (None, "")]
        return round(sum(v) / len(v), 4) if v else None
    board = [{
        "gw": int(r["gw"]), "home": r["home"], "away": r["away"], "result": r["result"],
        "outcome": r["outcome"], "url": r.get("kal_url", ""),
        "model": [float(r.get("close_h") or r["pred_h"]), float(r.get("close_d") or r["pred_d"]),
                  float(r.get("close_a") or r["pred_a"])],
        "kalshi": [float(r["kal_h"]), float(r["kal_d"]), float(r["kal_a"])],
        "brier_model": float(r["brier_model"]), "brier_kalshi": float(r["brier_kalshi"]),
        "closer": r.get("closer", ""),
    } for r in settled]
    board.sort(key=lambda x: (-x["gw"], x["home"]))
    wins = sum(1 for r in settled if r.get("closer") == "model")
    losses = sum(1 for r in settled if r.get("closer") == "kalshi")
    scorecard = {
        "tracked": tracked, "settled": len(settled),
        "model_wins": wins, "kalshi_wins": losses,
        "ties": len(settled) - wins - losses,
        "brier_model": avg("brier_model", settled), "brier_kalshi": avg("brier_kalshi", settled),
        "brier_book": avg("brier_book", settled),
        "rps_model": avg("rps_model", settled), "rps_kalshi": avg("rps_kalshi", settled),
        "board": board[:40],
    }
    graded = sum(1 for r in rows.values() if r.get("graded"))
    print(f"  RULE 5: ledger {len(rows)} locked, {graded} graded | Kalshi tracked on {tracked}, "
          f"{len(settled)} settled (model {wins} - {losses} kalshi)")
    return scorecard


def build():
    print("fetching ESPN fixtures + odds ...")
    events = feeds.espn_events()
    weeks_raw = feeds.to_matchweeks(events)
    print(f"  {len(events)} fixtures over {len(weeks_raw)} matchweeks")

    print("fetching FPL team news ...")
    fpl = feeds.fpl_feed()
    print(f"  FPL serving {fpl['season']} ({fpl['current_gw']}), {len(fpl['news'])} news rows")

    refresh_clubs(feeds._get(feeds.FPL, "fpl_bootstrap.json", max_age_min=180))

    print("fetching Kalshi EPL markets ...")
    # DISPLAY/COMPARISON LAYER ONLY — Kalshi does not feed the model's probabilities or the
    # existing edge calc. It is a second, sharper market to cross-reference against.
    kmk = kalshi.fetch_markets()
    kprops = kalshi.fetch_props()
    print(f"  Kalshi: {len(kmk)} fixtures with a 3-way market, {len(kprops)} with props")

    # RULE 1: refit the ratings every build, folding in any 26/27 results that have finished
    # (from the live feed) so the model updates after each matchday. RULE 3: where a finished
    # game exposes shot data we blend goals with a shots-on-target xG proxy to damp luck.
    finished = [e for e in events if e["finished"] and e["hs"] is not None]
    extra = None
    if finished:
        import pandas as pd
        rec = []
        for e in finished:
            hs, as_ = e["hs"], e["as"]
            # RULE 3 WAS VALIDATED AND TURNED OFF. Blending a shots-on-target xG proxy into
            # the training data made out-of-sample RPS monotonically WORSE (goals-only 0.2092,
            # 45% proxy 0.2102, 100% proxy 0.2141) across 2018-26. SoT x conversion throws away
            # shot quality entirely, so it is a noisier target than the goals it replaces.
            # Real per-shot xG (now available per player from FPL each gameweek) is worth
            # re-testing once enough of it has accumulated; the crude proxy is not.
            rec.append({"date": e["utc"][:10], "home_team": e["home"], "away_team": e["away"],
                        "home_score": hs, "away_score": as_})
        extra = pd.DataFrame(rec)
        print(f"  RULE 1: folding {len(extra)} finished 26/27 results into the refit")
    model = dc.fit(extra=extra, verbose=False)

    # ---- CORNERS: fold this season's real corner counts (ESPN) into the rolling rates ----
    crows = []
    for e in finished:
        st = feeds.espn_match_stats(e["id"])
        if not st:
            continue
        try:
            ch = int(st[0]["stats"].get("wonCorners", 0)); ca = int(st[1]["stats"].get("wonCorners", 0))
        except (TypeError, ValueError):
            continue
        crows.append({"date": e["utc"][:10], "home_team": e["home"], "away_team": e["away"],
                      "home_corners": ch, "away_corners": ca})
    import pandas as _pd
    cmodel = pcorn.build(extra=_pd.DataFrame(crows) if crows else None)
    print(f"  corners model: {cmodel['matches']} matches through {cmodel['asof']} "
          f"(+{len(crows)} from this season)")
    mapping = so.get_mapping()
    # Fitted blend weights. These live in .state/ and ARE committed - they are model
    # parameters, not cache. Fall back to the documented production split rather than
    # killing the whole build: a fresh clone that has not run build_blend.py yet should
    # still produce a dashboard.
    _bp = os.path.join(ROOT, ".state", "blend.json")
    if os.path.exists(_bp):
        blend = json.load(open(_bp))
        wdc, wsup = blend["dc_weight"], blend["sup_weight"]
    else:
        wdc, wsup = 0.75, 0.25
        print("  NOTE: .state/blend.json missing - using the default 0.75/0.25 split. "
              "Run build_blend.py to refit it.")

    # club colour + abbreviation per team, for the FPL pitch
    team_meta = {}
    for e in events:
        team_meta.setdefault(e["home"], {"color": e["home_color"], "abbr": e["home_abbr"]})
        team_meta.setdefault(e["away"], {"color": e["away_color"], "abbr": e["away_abbr"]})

    clubs = sorted({e["home"] for e in events} | {e["away"] for e in events})
    model, cold = apply_cold_start(model, clubs)
    if cold:
        print(f"  cold-start prior -> {sorted(cold)}")

    form = so.current_form()
    shares = ps.load_shares()
    strength = read_csv(os.path.join(ROOT, "outputs", "team_strength_index.csv"))
    players = read_csv(os.path.join(ROOT, "outputs", "player_rankings_2026_27.csv"))
    aidx = {t: i for i, t in enumerate(model["teams"])}

    # availability weights for the scorer model, from FPL status/chance-of-playing
    avail = {}
    for n in fpl["news"]:
        w = 0.0 if n["flag"] == "out" else ((n.get("chance") or 50) / 100.0 if n["flag"] == "doubt" else 1.0)
        avail[(n["team"], ps._surname(n["who"]))] = w
    # news indexed by club so each card can show its own absentees
    news_by_team = {}
    for n in fpl["news"]:
        news_by_team.setdefault(n["team"], []).append(n)

    def confidence(h, a, priced, vig, cold_pair):
        """0-100 reliability score for a fixture's prediction, with reasons. Cold-start is
        the dominant penalty (backtest: cold RPS 0.2315 vs 0.2036 rated); stale news and
        thin/absent market lines reduce it further."""
        c, why = 100, []
        if cold_pair:
            c -= 50; why.append("newly-promoted team, no rating history (priors are guesses)")
        if fpl["season"] != "2026/27":
            c -= 10; why.append("team news is last-season (FPL not rolled over)")
        if not priced:
            c -= 6; why.append("no market line yet to cross-check")
        elif vig is not None and vig > 8:
            c -= 8; why.append(f"soft/thin line (vig {vig:.1f}%)")
        c = max(5, min(100, c))
        tier = "high" if c >= 75 else ("medium" if c >= 55 else "low")
        if cold_pair:
            tier = "low"          # promoted-team predictions are never trustworthy enough to act on
        return c, tier, why

    weeks, n_edges, n_odds = [], 0, 0
    for wk in weeks_raw:
        matches = []
        for e in wk["fixtures"]:
            h, a = e["home"], e["away"]
            dp = dc.predict(model, h, a)
            dcp = np.array([dp["win_h"], dp["draw"], dp["win_a"]])
            hf, af = form.get(h, 0), form.get(a, 0)
            sp = np.array(so.probs_from_rating(mapping, hf - af))
            p = wdc * dcp + wsup * sp
            p = p / p.sum()
            ph, pd_, pa = [round(float(x) * 100) for x in p]
            ph += 100 - (ph + pd_ + pa)
            cold_pair = (h in cold or a in cold)
            priced = bool(e["odds"])
            vig = e["odds"]["overround"] if priced else None
            conf, tier, conf_why = confidence(h, a, priced, vig, cold_pair)

            edge = None
            if priced:
                n_odds += 1
                dec = e["odds"]["dec"]
                evs = {k: (float(p[i]) * dec[k] - 1) * 100 for i, k in enumerate(("h", "d", "a"))}
                side, best = max(evs.items(), key=lambda kv: kv[1])
                if best >= EDGE_MIN:
                    # grade the edge by confidence. Cold-start (promoted) fixtures can NEVER be
                    # actionable or watch — the "edge" is just the prior being wrong.
                    grade = ("low" if cold_pair
                             else ("actionable" if tier == "high"
                                   else ("watch" if tier == "medium" else "low")))
                    if grade == "actionable":
                        n_edges += 1
                    edge = {"side": {"h": h, "d": "Draw", "a": a}[side],
                            "ev": round(best, 1), "odds": dec[side], "grade": grade}

            # "why this pick" factors
            ih, ia = aidx.get(h), aidx.get(a)
            ah = round(float(model["attack"][ih]), 2) if ih is not None else None
            da = round(float(model["defense"][ia]), 2) if ia is not None else None
            fav = h if ph >= pa and ph >= pd_ else (a if pa >= pd_ else "Draw")
            why = {
                "fav": fav, "confidence": conf, "tier": tier, "conf_reasons": conf_why,
                "factors": [
                    {"k": "Long-run strength", "v": f"{h} net {round(float(model['attack'][ih]+model['defense'][ih]),2) if ih is not None else 'n/a'} vs {a} net {round(float(model['attack'][ia]+model['defense'][ia]),2) if ia is not None else 'n/a'}"},
                    {"k": "Recent form (last 6)", "v": f"{h} {hf:+d} vs {a} {af:+d}"},
                    {"k": "Expected goals", "v": f"{dp['xg_h']:.2f} - {dp['xg_a']:.2f}"},
                    {"k": "vs market", "v": (f"{edge['side']} +{edge['ev']}% ({edge['grade']})" if edge else ("in line with market" if priced else "no market yet"))},
                ],
            }

            # Kalshi cross-reference (display only — never feeds the model or the edge above)
            kro = kmk.get((h, a))
            kal = None
            if kro:
                cmp_ = kalshi.compare((float(p[0]), float(p[1]), float(p[2])), kro)
                b = cmp_["best"]
                nm = {"h": h, "d": "Draw", "a": a}
                kal = {
                    "url": kro["url"], "event": kro["event"], "vig": kro["vig"],
                    "vol": kro["vol"], "thin": kro["thin"], "rows": cmp_["rows"],
                    "mid": {k2: (round(kro["legs"][k2]["mid"] * 100, 1)
                                 if kro["legs"][k2]["mid"] is not None else None)
                            for k2 in ("h", "d", "a")},
                    "entry": ({"side": nm[b["leg"]], "ev": b["ev"], "ask": b["ask"],
                               "model": b["model"], "kalshi": b["kalshi_mid"],
                               # an entry is only "candidate" when the market is liquid AND the
                               # fixture isn't cold-start; otherwise it's flagged, never promoted
                               "grade": ("candidate" if (b["ev"] >= 5 and not kro["thin"]
                                                         and not cold_pair and tier == "high")
                                         else "watch" if b["ev"] >= 5 else "none")}
                              if b and b["ev"] is not None else None),
                }

            # ---- PROPS: BTTS / totals / spreads, priced off the SAME score matrix ----
            Mx, _, _ = dc.score_matrix(model, h, a)
            pm = kalshi.price_props(Mx.tolist())
            kp = kprops.get((h, a))
            plist = []
            def add_prop(kind, label, line, side, mp, leg):
                ask = leg["ask"] if leg else None
                bid = leg["bid"] if leg else None
                mid = leg["mid"] if leg else None
                ev = round((mp / ask - 1) * 100, 1) if ask and ask > 0 else None
                plist.append({
                    "kind": kind, "label": label, "line": line, "side": side,
                    "model": round(mp * 100, 1),
                    "bid": round(bid * 100, 1) if bid is not None else None,
                    "ask": round(ask * 100, 1) if ask is not None else None,
                    "mid": round(mid * 100, 1) if mid is not None else None,
                    "thin": (leg or {}).get("thin", True), "ev": ev,
                    "url": (leg or {}).get("url", ""),
                })
            # always publish the model's prop prices, with or without a Kalshi quote
            add_prop("btts", "Both teams to score", None, None, pm["btts"],
                     (kp or {}).get("btts"))
            kt_tot = {t["line"]: t["leg"] for t in (kp or {}).get("total", [])}
            for ln in (1.5, 2.5, 3.5):
                add_prop("total", f"Over {ln} goals", ln, None, pm["over"][int(ln)], kt_tot.get(ln))
            kt_sp = {(s["side"], s["line"]): s["leg"] for s in (kp or {}).get("spread", [])}
            for ln in (1.5, 2.5):
                add_prop("spread", f"{h} wins by >{ln}", ln, "h", pm["home_by"][int(ln)], kt_sp.get(("h", ln)))
                add_prop("spread", f"{a} wins by >{ln}", ln, "a", pm["away_by"][int(ln)], kt_sp.get(("a", ln)))
            # a prop entry is only worth showing when the quote is real AND the fixture is sound
            best_prop = None
            live_props = [p_ for p_ in plist if p_["ev"] is not None and not p_["thin"]]
            if live_props and not cold_pair and tier == "high":
                b2 = max(live_props, key=lambda x: x["ev"])
                if b2["ev"] >= 5:
                    best_prop = {"label": b2["label"], "ev": b2["ev"], "ask": b2["ask"]}
            props = {"rows": plist, "quoted": sum(1 for p_ in plist if p_["ask"] is not None),
                     # `.get("btts", {})` returns None when the key EXISTS but is null, which
                     # Kalshi does when a prop series has no live market - the {} default never
                     # applies. Crashed the whole build the first time it happened unattended.
                     "best": best_prop, "url": ((kp or {}).get("btts") or {}).get("url", "")}

            # corners: team markets carry real signal, totals validated at only +0.4% vs naive
            cp = pcorn.price(cmodel, h, a)
            cactual = None
            if e["finished"]:
                _st = feeds.espn_match_stats(e["id"])
                if _st:
                    try:
                        cactual = {"h": int(_st[0]["stats"].get("wonCorners", 0)),
                                   "a": int(_st[1]["stats"].get("wonCorners", 0))}
                    except (TypeError, ValueError):
                        cactual = None
            corners = {"xc_home": cp["xc_home"], "xc_away": cp["xc_away"], "xc_total": cp["xc_total"],
                       "home_at_least": cp["home_at_least"], "away_at_least": cp["away_at_least"],
                       "total_at_least": cp["total_at_least"], "total_weak": True,
                       "actual": cactual}

            kt = datetime.fromisoformat(e["utc"].replace("Z", "+00:00"))
            slug = f"{e['home_abbr']}-{e['away_abbr']}".lower()
            matches.append({
                "id": slug, "home": h, "away": a,
                "home_abbr": e["home_abbr"], "away_abbr": e["away_abbr"],
                "home_color": e["home_color"], "away_color": e["away_color"],
                "venue": e["venue"], "time": kt.strftime("%a %d %b %H:%M"),
                "status": e["status"], "finished": e["finished"], "live": e["live"],
                "result": (f"{e['hs']}-{e['as']}" if e["finished"] and e["hs"] is not None else None),
                "ph": ph, "pd": pd_, "pa": pa,
                "fair": {k: round(1 / float(p[i]), 2) for i, k in enumerate(("h", "d", "a"))},
                "mkt": e["odds"], "edge": edge, "conf": conf, "tier": tier, "why": why,
                "kalshi": kal, "props": props, "corners": corners,
                "score": top_scores(model, h, a)[0]["s"],
                "scorelines": top_scores(model, h, a),
                "xg": f"{dp['xg_h']:.2f} - {dp['xg_a']:.2f}",
                "form": {"h": hf, "a": af},
                "scorers": {"h": [{"n": n, "p": round(v * 100)} for n, v in ps.match_scorers(h, dp["xg_h"], shares, avail, n=3)],
                            "a": [{"n": n, "p": round(v * 100)} for n, v in ps.match_scorers(a, dp["xg_a"], shares, avail, n=3)]},
                "news": {"h": news_by_team.get(h, [])[:3], "a": news_by_team.get(a, [])[:3]},
                "cold_start": cold_pair,
            })
        weeks.append({"gw": wk["gw"], "matches": matches,
                      "start": wk["fixtures"][0]["utc"][:10]})

    # ---- FPL WEEK-BY-WEEK: only ever simulate the ACTIVE gameweek (never all 38), and only
    # reveal it 2 days before its first kickoff. The team is regenerated from the freshest data
    # each build, so the model effectively re-learns each matchweek. ----
    import simulate_fpl as sf
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    # last fully-finished gameweek -> the active week is the next one
    last_finished = max([wk["gw"] for wk in weeks_raw
                         if all(fx["id"] in {e["id"] for e in events if e["finished"]}
                                for fx in wk["fixtures"])] or [0])
    active_gw = min(last_finished + 1, weeks_raw[-1]["gw"])
    active_wk = next(w for w in weeks_raw if w["gw"] == active_gw)
    first_ko = min(fx["utc"] for fx in active_wk["fixtures"])
    reveal_at = datetime.fromisoformat(first_ko.replace("Z", "+00:00")) - timedelta(days=2)
    locked = now_dt < reveal_at
    fpl_players = sf.fpl_players()
    active_rows = sf.project_gw(active_gw, players=fpl_players, model=model, shares=shares, weeks=weeks_raw)
    active_team = sf.build_team(active_gw, weeks=weeks_raw, team_meta=team_meta, rows=active_rows)
    # ---- TRANSFER REALITY: you hold last week's squad and get ONE free transfer (extras -4).
    # build_team drafts from scratch, which is only reachable in GW1 or on a wildcard, so we
    # also compute what you can actually GET TO from the squad you hold. ----
    # Source of truth is the FPL API - the squad you ACTUALLY hold, with your real bank.
    # (The model's own GW1 snapshot was tainted, and using it as the starting squad produced
    # advice to buy a player the user already owned.) Falls back to the snapshot only if no
    # entry id is configured in fpl_config.json.
    real = fpl_transfers.real_squad(active_gw)
    if real:
        held = real["ids"]
        by_price = {p["id"]: p["price"] for p in fpl_players}
        budget = round(sum(by_price.get(i, 0.0) for i in held) + real["bank"], 1)
        src = f"FPL entry {real['entry']} (picks as of GW{real['src_gw']})"
    else:
        held = fpl_transfers.held_squad(active_gw)
        budget, src = 100.0, "model snapshot (no FPL entry id configured)"
    transfer_plan = (fpl_transfers.plan(active_rows, held, budget=budget, free_transfers=1,
                                        all_players=fpl_players) if held else None)
    if transfer_plan:
        transfer_plan["source"] = src
        transfer_plan["budget"] = budget
    if transfer_plan:
        print(f"  transfers: {transfer_plan['verdict']}"
              f" | held XI {transfer_plan['base_xi']} pts, bank {transfer_plan['bank']}m")
    elif held:
        print("  transfers: held squad could not be resolved from this week's player pool")
    else:
        print("  transfers: no held squad yet - the from-scratch draft IS the advice this week")
    # The squad that gets graded must be the one you could actually FIELD: last week's squad
    # plus the recommended transfer(s). Grading the from-scratch draft measures a team no one
    # could have owned. In GW1 (or after a wildcard) there is no held squad and the draft IS
    # the advice, so it is graded as-is.
    draft_team = active_team
    if transfer_plan:
        reach = fpl_transfers.decorate(transfer_plan["best"]["squad"], active_gw,
                                       weeks=weeks_raw, team_meta=team_meta)
        if reach:
            reach["draft"] = {k: draft_team[k] for k in ("cost", "formation", "xi_proj",
                                                         "captain", "squad", "bench_order")}
            active_team = reach
    active_team["transfers"] = transfer_plan
    active_team["draft_only"] = held is None
    active_team["locked"] = locked
    active_team["reveal_at"] = reveal_at.isoformat(timespec="minutes")
    active_team["first_kickoff"] = first_ko
    picks_gw = [{"nm": r["name"], "team": r["team"], "pos": r["pos"], "price": r["price"],
                 "proj": r["proj"], "own": r["own"]} for r in active_rows[:16]]
    print(f"  active GW={active_gw} (last finished {last_finished}); "
          f"reveal {reveal_at.date()}; locked={locked}")

    # forward-test: snapshot the active team's projections, grade past weeks vs actual FPL points
    chips = fpl_chips.advise(active_gw, active_team, active_rows)
    if chips: print(f"  chips: {chips[chr(39)+chr(115)+chr(117)+chr(109)+chr(109)+chr(97)+chr(114)+chr(121)+chr(39)]}" if False else f"  chips: {chips['summary']}")
    forward = forward_test(active_gw, active_team, weeks_raw, events, now_dt.isoformat(timespec="minutes"))

    # strength index over THIS SEASON'S clubs (relegated sides gone, promoted sides in)
    ABBR = {m["home"]: m["home_abbr"] for w in weeks for m in w["matches"]}
    COL = {m["home"]: m["home_color"] for w in weeks for m in w["matches"]}
    strength_rows = build_strength_2627(model, clubs, cold)
    for r in strength_rows:
        r["abbr"] = ABBR.get(r["team"], r["team"][:3].upper())
        r["color"] = COL.get(r["team"], "#5B7A72")
    # attach each player's current club news count is not needed; players list stays model-based
    players_2627 = [{"pos": p["position"], "nm": p["player"], "team": p["latest_team"],
                     "rating": float(p["rating"]),
                     "g": float(p["w_goals"]), "a": float(p["w_assists"])} for p in players[:15]]

    # ---- per-source freshness (each feed carries its own "as of" + rule) ----
    now = datetime.now(timezone.utc)
    news_dates = [n["added"] for n in fpl["news"] if n.get("added")]
    model_asof = dc.get_model().get("date_max", "")
    bt = {}
    bt_path = os.path.join(HERE, "backtest.json")
    if os.path.exists(bt_path):
        bt = json.load(open(bt_path))
    sources = [
        {"key": "fixtures", "name": "Fixtures", "src": "ESPN", "as_of": now.isoformat(timespec="minutes"),
         "fresh": True, "rule": "refetched every build", "note": f"{sum(len(w['matches']) for w in weeks)} fixtures"},
        {"key": "odds", "name": "Market odds", "src": "ESPN / DraftKings", "as_of": now.isoformat(timespec="minutes"),
         "fresh": n_odds > 0, "rule": "priced fixtures only; books open ~1 week out",
         "note": f"{n_odds} of {sum(len(w['matches']) for w in weeks)} priced"},
        {"key": "news", "name": "Team news", "src": "FPL", "as_of": (max(news_dates) if news_dates else ""),
         "fresh": fpl["season"] == "2026/27", "rule": "stale until FPL rolls the new season on",
         "note": f"serving {fpl['season']} ({fpl['current_gw']})"},
        {"key": "model", "name": "Model ratings", "src": "Dixon-Coles", "as_of": model_asof,
         "fresh": True, "rule": "refit on daily rebuild", "note": f"trained through {model_asof}"},
        {"key": "backtest", "name": "Validation", "src": "walk-forward", "as_of": (bt.get("meta") or {}).get("generated_at", ""),
         "fresh": bool(bt), "rule": "out-of-sample evidence behind every edge",
         "note": (f"RPS {bt['headline'][0]['rps']}" if bt else "not run")},
    ]

    data = {
        "meta": {
            "generated_at": now.isoformat(timespec="minutes"),
            "season": "2026/27", "weeks": len(weeks),
            "fixtures": sum(len(w["matches"]) for w in weeks),
            "sources_line": "ESPN (fixtures, odds) + FPL (team news) + local Dixon-Coles model",
            "fpl_season": fpl["season"], "fpl_gw": fpl["current_gw"],
            "fpl_stale": fpl["season"] != "2026/27",
            "cold_start": sorted(cold),
            "model": f"Dixon-Coles {wdc:g} x supremacy-form {wsup:g}",
            "priced": n_odds, "edges": n_edges,
        },
        "sources": sources,
        "weeks": weeks, "strength": strength_rows,
        "news": fpl["news"], "ticker": fpl["news"][:16],
        "team": active_team, "active_gw": active_gw, "picks_gw": picks_gw,
        "forward": forward, "chips": chips,
        "players": players_2627, "backtest": bt,
    }
    data["meta"]["active_gw"] = active_gw
    data["meta"]["team_locked"] = active_team["locked"]
    data["transfers"] = transfer_plan
    # FORM vs QUALITY, kept separate on purpose - see fpl_form.py. Conflating them is what
    # makes raw "form" tables recommend the player about to regress.
    try:
        form_rows, _ = fpl_form.build()
        for fr in form_rows:
            fr["quadrant"] = fpl_form.quadrant(fr)
        data["form"] = form_rows[:60]
        hot = [f for f in form_rows if f["quadrant"] == "hot elite"][:5]
        print(f"  form ratings: {len(form_rows)} players; hot elite -> "
              + ", ".join(f["name"] for f in hot))
    except Exception as e:
        data["form"] = []
        print(f"  form ratings: FAILED ({e})")
    data["meta"]["reveal_at"] = active_team["reveal_at"]

    # RULE 5: record every prediction (with the closing line) BEFORE kickoff, then grade it
    # against the result + closing odds once the game is played. This is the live evidence log.
    data["settled"] = record_ledger(weeks, now.isoformat(timespec="minutes"))
    data["settled_props"] = record_props_ledger(weeks, now.isoformat(timespec="minutes"))
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT}: {data['meta']['fixtures']} fixtures, {n_odds} priced, "
          f"{n_edges} edges, {len(data['news'])} news rows "
          f"({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    build()

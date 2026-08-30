"""
Form vs quality, kept as two SEPARATE numbers.

The question this answers: "who is playing well right now, and is that real?"

Conflating form and quality is the classic fantasy trap. A fringe player with one lucky haul
outranks an elite player who blanked, and you buy the wrong one. So this reports:

  QUALITY  the player's established level, from recency-weighted historical output per 90
  FORM     this season's underlying output per 90, z-scored within position
  HEAT     FORM - QUALITY: is he above or below his OWN baseline right now?

The pairing is what you act on:
  high quality + high form   -> genuinely hot, the strongest hold/buy
  high quality + low form    -> buy the dip; elite players mean-revert upward
  low quality  + high form   -> fade; this is the one that regresses, and the one
                                 raw "form" tables tell you to buy
  low quality  + low form    -> ignore

Form is measured on UNDERLYING output (xG + xA per 90), not points. Points are lumpy - bonus,
clean sheets and hauls swamp the signal over a handful of games - while xGI accumulates from
every chance and stabilises far sooner. Points per 90 is carried alongside as a cross-check,
never as the primary.

CONFIDENCE is minutes-based, w = m/(m+K). Early in a season nothing is significant, and the
rating says so rather than pretending. `form_adj` is the shrunk value used for ranking;
`form_raw` is what the player has actually done.
"""
import csv, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
# repo-relative so a clone works from any directory, on any machine
OUT = os.path.join(os.path.dirname(HERE), "outputs")
MIN_K = 900.0            # ~10 full matches before this season's rate is half-trusted
MIN_MINUTES = 45         # below this, a per-90 rate is meaningless noise


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _z(vals):
    """z-score helper that degrades safely when everything is identical."""
    if len(vals) < 2:
        return [0.0] * len(vals)
    m = st.mean(vals)
    sd = st.pstdev(vals)
    return [0.0] * len(vals) if sd < 1e-9 else [(v - m) / sd for v in vals]


# FPL's short names collide: 16 surnames in the ranking file map to two different players,
# usually an outfielder and a keeper at the same club (Chelsea has both a MID Palmer with
# 2895 weighted minutes and a GKP Palmer with 414). Keying a dict on the bare name silently
# kept whichever row came last - which rated Cole Palmer on the goalkeeper's zero xGI and
# scored him -2.02 for quality. Resolve on POSITION first, then most minutes.
_POSMAP = {"GKP": "GK", "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def load_baseline():
    """Established quality: recency-weighted historical output per 90, keyed name -> candidates."""
    p = os.path.join(OUT, "player_rankings_2026_27.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        mins = _f(r.get("w_minutes"))
        if mins < 90:
            continue
        out.setdefault(r["player"].strip().lower(), []).append({
            "xgi90": _f(r.get("w_xgi")) / mins * 90.0,
            "gi90": _f(r.get("w_goal_involvements")) / mins * 90.0,
            "pts90": _f(r.get("w_points")) / mins * 90.0,
            "pos": _POSMAP.get((r.get("position") or "").strip().upper(), ""),
            "team": (r.get("latest_team") or "").strip(),
            "mins": mins,
        })
    return out


def _match(base, name, full_name, pos, team):
    """Pick the right candidate for a name: position must agree, then most minutes."""
    for key in (name.strip().lower(), full_name.strip().lower()):
        cands = base.get(key)
        if not cands:
            continue
        same_pos = [c for c in cands if c["pos"] == pos]
        pool = same_pos or ([] if len(cands) > 1 else cands)
        if not pool:
            continue
        return max(pool, key=lambda c: c["mins"])
    return None


def build():
    stats = list(csv.DictReader(open(os.path.join(OUT, "fpl_player_stats_2026_27.csv"))))
    base = load_baseline()
    rows = [r for r in stats if int(_f(r["minutes"])) >= MIN_MINUTES]

    # --- current-season per-90 rates ---
    for r in rows:
        m = _f(r["minutes"])
        r["_xgi90"] = _f(r["xGI"]) / m * 90.0
        r["_pts90"] = _f(r["total_points"]) / m * 90.0
        r["_dc90"] = _f(r["def_contrib_per90"])
        b = _match(base, r["name"], r["full_name"], r["pos"], r["team"])
        r["_b_xgi90"] = b["xgi90"] if b else None
        r["_b_pts90"] = b["pts90"] if b else None

    out = []
    for pos in ("GK", "DEF", "MID", "FWD"):
        grp = [r for r in rows if r["pos"] == pos]
        if not grp:
            continue
        # form: underlying first, points as a secondary cross-check
        zf_u = _z([r["_xgi90"] for r in grp])
        zf_p = _z([r["_pts90"] for r in grp])
        zf_d = _z([r["_dc90"] for r in grp])
        # defenders and keepers earn far more from defensive work than from xGI, so the
        # blend is position-aware rather than one-size-fits-all
        wu, wp, wd = ((0.15, 0.55, 0.30) if pos in ("GK", "DEF") else (0.55, 0.30, 0.15))
        # quality: historical baseline, z-scored inside the same position
        have = [r for r in grp if r["_b_xgi90"] is not None]
        zq = dict(zip((id(r) for r in have), _z([r["_b_xgi90"] for r in have]))) if have else {}
        for i, r in enumerate(grp):
            form_raw = wu * zf_u[i] + wp * zf_p[i] + wd * zf_d[i]
            conf = _f(r["minutes"]) / (_f(r["minutes"]) + MIN_K)
            qual = zq.get(id(r))
            out.append({
                "id": r["id"], "name": r["name"], "team": r["team"], "pos": pos,
                "price": _f(r["price"]), "minutes": int(_f(r["minutes"])),
                "own_pct": _f(r["selected_by_pct"]),
                "form_raw": round(form_raw, 2),
                "form_adj": round(form_raw * conf, 3),
                "confidence": round(conf, 3),
                "quality": round(qual, 2) if qual is not None else "",
                "heat": round(form_raw - qual, 2) if qual is not None else "",
                "xGI90": round(r["_xgi90"], 3), "pts90": round(r["_pts90"], 2),
                "dc90": round(r["_dc90"], 2),
                "base_xGI90": round(r["_b_xgi90"], 3) if r["_b_xgi90"] is not None else "",
                "goals": r["goals"], "assists": r["assists"],
                "xG": _f(r["xG"]), "xA": _f(r["xA"]),
                "status": r["status"], "chance_next": r["chance_next"],
            })
    out.sort(key=lambda r: -r["form_adj"])
    p = os.path.join(OUT, "fpl_form_ratings.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    return out, p


def quadrant(r):
    """Where a player sits on the quality/form grid - the thing you actually act on."""
    if r["quality"] == "":
        return "unrated"
    q, f = float(r["quality"]), r["form_raw"]
    if q >= 0.5 and f >= 0.5:
        return "hot elite"
    if q >= 0.5 and f < -0.25:
        return "elite, cold (buy the dip)"
    if q < 0 and f >= 0.75:
        return "overperforming (fade)"
    return "steady"


if __name__ == "__main__":
    rows, path = build()
    print(f"wrote {path}: {len(rows)} players\n")
    print("TOP 20 BY FORM (minutes-shrunk)")
    print(f"  {'player':<16}{'club':<14}{'pos':<5}{'form':>6}{'qual':>6}{'heat':>6}"
          f"{'xGI90':>7}{'pts90':>7}{'mins':>6}  quadrant")
    for r in rows[:20]:
        print(f"  {r['name'][:15]:<16}{r['team'][:13]:<14}{r['pos']:<5}{r['form_raw']:>6.2f}"
              f"{(r['quality'] if r['quality']!='' else 0):>6}{(r['heat'] if r['heat']!='' else 0):>6}"
              f"{r['xGI90']:>7.2f}{r['pts90']:>7.1f}{r['minutes']:>6}  {quadrant(r)}")

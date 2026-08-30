"""
Team strength index (last season's 20 PL clubs) + top-50 player list.

Team strength index is derived from the Dixon-Coles ratings:
  * attack / defense / net rating (the fitted parameters)
  * expected goals for/against a league-average opponent
  * projected season points: simulate a full double round-robin among the 20 clubs with
    the model (home & away vs every other side), scoring 3*P(win)+1*P(draw) each game
  * Strength Index = projected points scaled so the strongest club = 100
  * recent-form supremacy shown alongside (transient, not part of the index)

Writes outputs/team_strength_index.csv and outputs/top50_players.csv.
"""
import csv, os
import numpy as np
import pandas as pd
import prem_dixon_coles as dc
import supremacy_odds as so

def _history(name):
    """Locate a file from the premier_league_history dataset.

    Repo layout (prem_predictor/ and premier_league_history/ as siblings) is tried FIRST so a
    clone works anywhere; the ~ location is kept as a fallback for the original local setup.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here)),
                 os.path.expanduser("~")):
        p = os.path.join(base, "premier_league_history", name)
        if os.path.exists(p):
            return p
    return os.path.expanduser(os.path.join("~/premier_league_history", name))


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
RESULTS = _history("results.csv")


def last_season_teams():
    df = pd.read_csv(RESULTS)
    s = df[df.season == "2025-26"]
    return sorted(set(s.home_team) | set(s.away_team))


def build_team_index():
    model = dc.get_model()
    teams = last_season_teams()
    ai = {t: i for i, t in enumerate(model["teams"])}
    a = np.array(model["attack"]); d = np.array(model["defense"])
    form = so.current_form()

    rows = []
    for t in teams:
        if t not in ai:
            continue
        i = ai[t]
        # projected points over a full double round-robin vs the other 19
        pts = gf = ga = 0.0
        for o in teams:
            if o == t or o not in ai:
                continue
            ph = dc.predict(model, t, o)      # t at home
            pa = dc.predict(model, o, t)      # t away
            pts += 3 * ph["win_h"] + ph["draw"] + 3 * pa["win_a"] + pa["draw"]
            gf += ph["xg_h"] + pa["xg_a"]; ga += ph["xg_a"] + pa["xg_h"]
        rows.append({"team": t, "attack": round(a[i], 3), "defense": round(d[i], 3),
                     "net_strength": round(a[i] + d[i], 3),
                     "proj_points": round(pts, 1),
                     "proj_gf": round(gf, 1), "proj_ga": round(ga, 1),
                     "proj_gd": round(gf - ga, 1),
                     "form_last6": int(form.get(t, 0))})
    rows.sort(key=lambda r: -r["proj_points"])
    top = rows[0]["proj_points"]
    for k, r in enumerate(rows, 1):
        r["rank"] = k
        r["strength_index"] = round(100 * r["proj_points"] / top, 1)
    cols = ["rank", "team", "strength_index", "proj_points", "proj_gd",
            "attack", "defense", "net_strength", "proj_gf", "proj_ga", "form_last6"]
    with open(os.path.join(OUT, "team_strength_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    return rows


def top_players(n=50):
    src = os.path.join(OUT, "player_rankings_2026_27.csv")
    rows = list(csv.DictReader(open(src)))[:n]
    cols = ["rank", "player", "position", "latest_team", "rating", "w_goals",
            "w_assists", "w_goal_involvements", "w_xgi", "gi_per90", "seasons"]
    with open(os.path.join(OUT, "top50_players.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    teams = build_team_index()
    print("TEAM STRENGTH INDEX (last season's 20 clubs, Dixon-Coles)\n")
    print(f"{'#':>2}  {'team':<15}{'index':>6}{'proj pts':>9}{'proj GD':>8}"
          f"{'atk':>7}{'def':>7}{'form':>6}")
    for r in teams:
        print(f"{r['rank']:>2}  {r['team']:<15}{r['strength_index']:>6.1f}"
              f"{r['proj_points']:>9.1f}{r['proj_gd']:>+8.1f}{r['attack']:>+7.2f}"
              f"{r['defense']:>+7.2f}{r['form_last6']:>+6d}")
    pl = top_players(50)
    print(f"\nTOP {len(pl)} PLAYERS (recency-weighted, currently in the Prem)\n")
    for r in pl:
        print(f"{r['rank']:>2}. {r['player']:<18}{r['position']:<4}{r['latest_team']:<14}"
              f"rat {r['rating']:>6}  G {r['w_goals']:>5}  A {r['w_assists']:>5}")

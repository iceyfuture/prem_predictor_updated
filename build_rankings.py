"""
Fit the PL Dixon-Coles model, rank teams and players, and export everything to CSV
so you can see exactly HOW each ranking was produced.

Writes into ~/prem_predictor/outputs/ :
  team_rankings.csv    one row per team: the fitted Dixon-Coles attack/defense ratings,
                       net strength, model-expected goals for/against a league-average
                       opponent (home & away), plus the model settings that produced
                       them (home_adv, rho, decay, window, cutoff, weighted matches).
  player_rankings.csv  one row per player: a recency-weighted rating built from the
                       historical player stats with the SAME decay as the team model
                       (recent seasons count ~8x a season 4 years ago), with every
                       component column shown so the ranking is fully transparent.
  model_meta.csv       one row: the global model parameters and settings.

Run:  ~/prem_predictor/.venv/bin/python ~/prem_predictor/build_rankings.py
"""
import os
import numpy as np
import pandas as pd

import prem_dixon_coles as dc

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
os.makedirs(OUT, exist_ok=True)
PLAYER_TOTALS = _history("player_season_totals.csv")

DECAY_BASE, DECAY_SPAN = dc.DECAY_BASE, dc.DECAY_SPAN   # reuse the team model's recency


# ------------------------------------------------------------------ TEAM RANKINGS
def build_team_rankings(model):
    teams = model["teams"]
    a = np.array(model["attack"]); d = np.array(model["defense"])
    wm = np.array(model["weighted_matches"])
    hadv, rho = model["home_adv"], model["rho"]

    # league-average opponent has attack=defense=0 (ratings are mean-centered), so
    # expected goals vs an average side are a clean, comparable readout of each rating:
    xg_home = np.exp(a + hadv)      # goals this team scores at home vs avg opponent
    xg_away = np.exp(a)             # ... away
    xga_home = np.exp(-d)           # goals conceded at home vs avg opponent (their attack=0)
    xga_away = np.exp(-d + hadv)    # ... away (opponent gets the home boost)
    net = a + d

    df = pd.DataFrame({
        "team": teams,
        "attack_rating": a.round(4),
        "defense_rating": d.round(4),
        "net_strength": net.round(4),
        "exp_goals_for_home": xg_home.round(3),
        "exp_goals_against_home": xga_home.round(3),
        "exp_goals_for_away": xg_away.round(3),
        "exp_goals_against_away": xga_away.round(3),
        "exp_goal_diff_per_game": ((xg_home - xga_home + xg_away - xga_away) / 2).round(3),
        "weighted_matches": wm.round(1),
    })
    df = df.sort_values("net_strength", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    # stamp the settings that produced these numbers onto every row (self-documenting)
    df["model"] = "dixon-coles"
    df["home_adv"] = round(hadv, 4)
    df["rho"] = round(rho, 4)
    df["decay"] = f"{int(DECAY_BASE)}**(-age/{int(DECAY_SPAN)})"
    df["window_years"] = model["window_years"]
    df["train_from"] = model["date_min"]
    df["train_to"] = model["date_max"]
    df["cutoff"] = model["cutoff"]
    path = os.path.join(OUT, "team_rankings.csv")
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} teams)")
    return df


# ---------------------------------------------------------------- PLAYER RANKINGS
def season_start_year(s):        # "2023-24" -> 2023
    return int(str(s)[:4])


def build_player_rankings(latest_season_year):
    df = pd.read_csv(PLAYER_TOTALS)
    for c in ["total_points", "minutes", "goals_scored", "assists", "bonus",
              "expected_goals", "expected_assists", "ict_index"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    # same recency curve as the team model: age in seasons -> weight 8**(-age/4)
    df["age"] = latest_season_year - df["season"].map(season_start_year)
    df["w"] = DECAY_BASE ** (-df["age"] / DECAY_SPAN)
    df["gi"] = df["goals_scored"] + df["assists"]
    df["xgi"] = df["expected_goals"] + df["expected_assists"]

    # identity = full name (web_name alone merges e.g. the two Wilsons / Hendersons)
    key = ["first_name", "second_name"]
    # weighted contributions, summed per player (vectorised — no groupby.apply)
    for base in ["total_points", "goals_scored", "assists", "gi", "xgi", "bonus", "minutes"]:
        df["_w_" + base] = df[base] * df["w"]
    sums = (df.groupby(key, sort=False)[["_w_total_points", "_w_goals_scored",
            "_w_assists", "_w_gi", "_w_xgi", "_w_bonus", "_w_minutes"]].sum())
    seasons = df.groupby(key, sort=False).size().rename("seasons")
    # identity/team/position from each player's most recent season (smallest age)
    recent = (df.loc[df.groupby(key, sort=False)["age"].idxmin()]
                .set_index(key)[["web_name", "position", "team", "season"]])

    r = sums.join(seasons).join(recent).reset_index()   # first_name/second_name -> columns
    r = r.rename(columns={
        "web_name": "player", "team": "latest_team", "season": "latest_season",
        "_w_total_points": "w_points", "_w_goals_scored": "w_goals",
        "_w_assists": "w_assists", "_w_gi": "w_goal_involvements",
        "_w_xgi": "w_xgi", "_w_bonus": "w_bonus", "_w_minutes": "w_minutes"})
    r["gi_per90"] = np.where(r["w_minutes"] > 0,
                             r["w_goal_involvements"] / (r["w_minutes"] / 90), 0.0)
    # rating = recency-weighted FPL points (availability + performance), the game's own
    # scoring of overall value; component columns let you re-rank however you like.
    r["rating"] = r["w_points"].round(2)
    r = r.sort_values("rating", ascending=False).reset_index(drop=True)
    r.insert(0, "rank", np.arange(1, len(r) + 1))
    for c in ["w_points", "w_goals", "w_assists", "w_goal_involvements", "w_xgi",
              "w_bonus", "w_minutes", "gi_per90"]:
        r[c] = r[c].round(2)
    r["ranking_method"] = f"recency-weighted FPL points, decay {int(DECAY_BASE)}**(-age/{int(DECAY_SPAN)})"
    path = os.path.join(OUT, "player_rankings.csv")
    r.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(r)} players)")
    return r


def write_meta(model):
    meta = pd.DataFrame([{
        "model": "dixon-coles",
        "home_adv": round(model["home_adv"], 4),
        "rho": round(model["rho"], 4),
        "decay": f"{int(DECAY_BASE)}**(-age/{int(DECAY_SPAN)})",
        "window_years": model["window_years"],
        "ridge": model["ridge"],
        "n_matches_weighted_in": model["n_matches"],
        "train_from": model["date_min"],
        "train_to": model["date_max"],
        "cutoff": model["cutoff"],
        "n_teams": len(model["teams"]),
    }])
    path = os.path.join(OUT, "model_meta.csv")
    meta.to_csv(path, index=False)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print("Fitting model...")
    model = dc.get_model(refit=True)
    print("\nExporting rankings + model detail to outputs/ ...")
    teams = build_team_rankings(model)
    write_meta(model)
    latest_year = season_start_year(model["date_max"][:4] + "-00")  # e.g. 2026 -> use data year
    latest_year = int(model["date_max"][:4])
    players = build_player_rankings(latest_year)

    print("\nTop 10 teams by net strength:")
    print(teams.head(10)[["rank", "team", "attack_rating", "defense_rating",
                          "net_strength", "exp_goals_for_home"]].to_string(index=False))
    print("\nTop 10 players by recency-weighted rating:")
    print(players.head(10)[["rank", "player", "position", "latest_team",
                           "rating", "w_goals", "w_assists"]].to_string(index=False))
    print("\nDone. See ~/prem_predictor/outputs/")

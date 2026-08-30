# Premier League History

A flat, long-history dataset of English Premier League club football — built in the same
spirit as the [martj42 international results](https://github.com/martj42/international_results)
folder, but for club football. One row per match going back to the league's first season,
plus per-player historical stats.

All files are plain CSV, sorted chronologically, safe to load directly into pandas/Excel.

## Files

| File | Grain | Rows | Coverage |
|------|-------|------|----------|
| `results.csv` | one row per match | 12,704 | **1993-94 → 2025-26** (every PL match) |
| `teams.csv` | one row per club | 51 | every club to appear in the PL, with season span |
| `player_season_totals.csv` | player × season | 7,358 | **2016-17 → 2025-26** (season totals) |
| `player_gameweek.csv` | player × gameweek | 253,900 | **2016-17 → 2025-26** (per-gameweek) |

## `results.csv`

The backbone: results **and** team performance stats for every Premier League match.

- `season`, `date`, `home_team`, `away_team`
- `home_score`, `away_score`, `result` (H / D / A)
- `ht_home_score`, `ht_away_score`, `ht_result` — half-time
- `referee`
- Team match stats: `home_shots`/`away_shots`, `home_shots_on_target`/`away_shots_on_target`,
  `home_fouls`/`away_fouls`, `home_corners`/`away_corners`,
  `home_yellows`/`away_yellows`, `home_reds`/`away_reds`

Notes:
- 1993-94 and 1994-95 have **462 matches** each (the league had 22 clubs then); 380 from 1995-96 on.
- The match-stat columns (shots, corners, cards, referee) begin around **2000-01**;
  earlier seasons carry scores/results only. Missing values are left blank.

## `player_season_totals.csv`

One row per player per season — compact, model-ready.

`season`, `player_id`, `first_name`, `second_name`, `web_name`, `team`, `position`
(GKP/DEF/MID/FWD), `total_points`, `minutes`, `goals_scored`, `assists`, `clean_sheets`,
`goals_conceded`, `bonus`, `bps`, `influence`, `creativity`, `threat`, `ict_index`,
`expected_goals`, `expected_assists`, `now_cost`, `selected_by_percent`, `starts`.

> `expected_goals` / `expected_assists` are only populated from 2022-23 onward (when FPL began
> publishing them). `player_id` is the FPL element id and is **not stable across seasons** —
> join on name/team within a season.

## `player_gameweek.csv`

The full gameweek-by-gameweek history (form / time-series modelling). ~74 columns as published
by FPL each week (minutes, goals, assists, bonus, bps, xG/xA, ICT, transfers, price, fixture,
opponent, was_home, etc.), with a `season` column prepended. Column set is the union across
seasons, so newer stat columns are blank in older seasons.

## Sources

- **Match results & team stats** — [football-data.co.uk](https://www.football-data.co.uk/englandm.php) (E0 / Premier League).
- **Player stats** — [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) (official FPL data, 2016-17 onward), with season/team names from its `master_team_list.csv`.

Rebuild with `build_pl_history.py` (re-downloads from source and regenerates every file).

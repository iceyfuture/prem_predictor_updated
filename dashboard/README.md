# Floodlit — Model Desk (Premier League 2026/27)

Quant terminal: real fixtures + market odds, team news, and your Dixon-Coles model,
in the Premier League palette.

## Feeds (both public, key-less — every field verified against a live response)
| Source | Gives us |
|---|---|
| **ESPN** `site.api.espn.com/.../soccer/eng.1/scoreboard?dates=` | real 26/27 fixtures, kickoff, venue, status, live/final scores, club colours + abbreviations, **DraftKings moneyline / totals** |
| **FPL** `fantasy.premierleague.com/api/bootstrap-static/` | player status, injury news, chance-of-playing, price, form, expected points |

`python feeds.py` runs `verify_feeds()` and prints exactly what it found — run it any time
the shapes look wrong instead of trusting field names.

## Run
```bash
cd ~/prem_predictor
.venv/bin/python dashboard/feeds.py             # verify the feeds
.venv/bin/python dashboard/build_dashboard.py   # fetch + merge -> dashboard.json
cd dashboard && python3 -m http.server          # open localhost:8000
```
`standalone.html` has the data inlined — open it directly, no server.

## Edges
Where a book has priced a game: `EV% = (model_prob x decimal_odds - 1) x 100`, flagged at
>= +3%. Vig is removed from the book's implied probabilities before comparison.

## Honest state (all surfaced in the UI, never hidden)
- **380 fixtures / 38 matchweeks are real** (ESPN). Matchweeks are reconstructed by grouping
  in date order — ESPN carries no round number — verified as exactly 38 x 10.
- **Only ~10 fixtures are priced.** Books open later weeks nearer the time; unpriced cards
  show fair odds and say "awaiting book".
- **FPL is still serving 2025/26** (Gameweek 38). Its news is end-of-last-season. The UI
  shows an amber warning and it will refresh automatically when FPL rolls over — no code change.
- **Coventry & Hull are cold-started** (no top-flight history in the rating window). Any edge
  on their games is demoted to amber "low confidence" — model uncertainty, not value.


## Structure (v2 — three sections, one tab bar)
- **Games** — matchweek cards only (predictions, market, edges, scorers, per-match injuries).
  The model-performance strip was removed on request.
- **Team News** — the full FPL injury/doubt/suspension list with Out / Doubtful filters.
- **Strength & Fantasy** — this season's power ranking + who to pick (FPL xP) + top players.

## Strength index = THIS season's clubs
`build_strength_2627()` simulates a double round-robin among the 20 clubs in the ESPN
fixture list, so relegated sides (West Ham, Wolves, Burnley) are gone and promoted sides
(Coventry, Hull, Ipswich) are in. Coventry & Hull have no rating history, so they carry a
provisional prior and a "PROV" flag; their number updates once real results retrain the model.

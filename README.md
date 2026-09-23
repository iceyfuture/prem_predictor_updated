# Prem Predictor

A Premier League prediction model and analysis desk: a Dixon-Coles goals model refit after
every matchday, a Fantasy Premier League planner that works from the squad you actually own,
and a running evidence log that scores every prediction against the market and the result.

Everything here is validated walk-forward before it ships. Several features were tested and
**removed** because they made the model worse - `dashboard/MODEL_RULES.md` records what was
tried, what the numbers said, and what was retired. Read that file before changing the model.

There is exactly **one** forecasting pipeline, `forecast.py`. The live dashboard and every
backtest call it. This was not always true, and the consequence was that the site's evidence
page measured a model nobody was running - see Rule 49. `dashboard/test_pipeline_parity.py`
fails if a second implementation reappears.

`AUDIT_RESPONSE.md` answers an independent audit finding by finding, including the three
places where the audit turned out to be wrong. `DATA_SOURCES.md` records where every number
comes from and what may be redistributed.

## What it does

| Piece | File | What it is |
|---|---|---|
| **The pipeline** | `forecast.py` | The one forecasting function. Production and every backtest call it |
| Goals model | `prem_dixon_coles.py` | Dixon-Coles bivariate Poisson, 8-year window, recency decay, refit every matchday |
| Odds blend | `supremacy_odds.py` | Goal-supremacy to 1X2 mapping, fit on our own data |
| Corners | `prem_corners.py` | Rolling 30-match rates (the Dixon-Coles version failed validation and was dropped) |
| Scorers | `prem_scorer.py` | Each player's shrunk goal rate times his expected minutes, scaled to the team's expected goals |
| Market benchmark | `ingest_odds.py` | Bookmaker closing odds with provider, price type and collection time on every row |
| Dashboard | `dashboard/build_dashboard.py` | Builds the whole desk: fixtures, edges, ratings, news, ledgers |
| FPL squad | `dashboard/simulate_fpl.py` | Per-gameweek projections and an optimal 15 |
| FPL transfers | `dashboard/fpl_transfers.py` | What your ONE free transfer actually buys, net of -4 hits |
| Form ratings | `dashboard/fpl_form.py` | Form and quality as separate numbers |
| Player stats | `dashboard/build_player_stats.py` | Every FPL player's xG/xA/xGI/xGC to CSV |
| Forward test | `dashboard/player_ledger.py` | Every player projection locked before kickoff and graded after |
| AI Council | `dashboard/council.py` | Experimental LLM panel. Display only — it cannot move a probability |
| Validation | `validate.py`, `backtest.py`, `sweep_coldstart.py` | Walk-forward harnesses used to accept or reject changes |

## Setup

Needs Python 3.12+ and an internet connection (it pulls live data from ESPN, the FPL API and
Kalshi - no API keys, no accounts).

```
git clone <your-repo-url> prem_predictor
cd prem_predictor
./setup.sh
```

Then point it at your own FPL team - either edit `dashboard/fpl_config.json` (created by
setup) or set the environment variable, which takes priority:

```
export FPL_ENTRY_ID=1234567
```

Your team id is the number in the URL when you view your team:
`fantasy.premierleague.com/entry/`**`1234567`**`/event/1`

## Running it

```
./dashboard/refresh.sh
```

That chains the three build steps and writes `dashboard/standalone.html` - open it in any
browser. It is a single self-contained file, so you can send it to someone.

Safe to run as often as you like: every step is idempotent. Predictions lock once per fixture
before kickoff and are never rewritten, the fantasy snapshot seals once per gameweek and
refuses to touch a gameweek that has already started, and stat snapshots de-duplicate on date.

To keep it current automatically on a Mac:

```
./dashboard/install_schedule.sh
```

That installs a launchd agent that runs the refresh daily at 06:30. Daily rather than weekly
because gameweeks do not land on a fixed weekday, and prices and injury news move nightly.
Remove it with `./dashboard/install_schedule.sh --uninstall`.

## Data

`premier_league_history/` holds the results dataset (every Premier League match since 1993 with
shots, corners and cards). `outputs/` holds derived CSVs - player ratings, squads, form ratings,
per-gameweek player stats.

`premier_league_history/player_gameweek.csv` (50MB) is **not** in the repo - no model code reads
it. Regenerate it with `premier_league_history/build_pl_history.py` if you want it.

## The evidence log

These are the point of the project, and they are committed so the history survives:

- `dashboard/ledger.csv` - every fixture's prediction, locked before kickoff, alongside the
  closing line and Kalshi's closing price, then graded on Brier and RPS
- `dashboard/props_ledger.csv` - the same for BTTS, totals and spreads
- `dashboard/player_ledger.csv` - every player's start probability, expected minutes and
  anytime-scorer probability, locked per gameweek and graded on what he actually did
- `dashboard/council_ledger.csv` - the experimental AI Council's forecasts, graded the same way
- `dashboard/fpl_forward.csv` - the fantasy squad locked each gameweek, graded on real points

Three rules make these worth anything. A prediction made at or after kickoff is stamped
`late=1` and excluded from every score (Rule 42). Keys carry the season and the provider's
fixture id, so next season's Arsenal-Chelsea is not mistaken for this one's (Rule 51). And
every forecast records the commit, model configuration, training cutoff and data version
that produced it, so rows from either side of a settings change are never silently averaged.

## Honest status

Read this before betting anything.

- **On match outcomes the model does not beat the market.** Measured on 2,660 fixtures priced
  by the average closing line, paired and clustered by season: RPS +0.00952, 95% CI
  [+0.00659, +0.01246]. The interval excludes zero. Blending toward the closing line improves
  the score at every weight, all the way to 100% market. It carries no information the market
  has not already priced.
- **The prop "profit" is two lucky tickets.** Strip the two best winners from the settled bets
  and a $27.51 profit becomes a $16.33 loss.
- **The AI Council has three graded forecasts** and is currently behind the statistical model.
  Neither fact means anything at that sample size, and the page says so.
- Where it does earn its keep is FPL selection, where there is no market to beat.

Figures on the page are read from the build, not typed into it. Three frozen numbers were
found and removed during the audit response; a number typed into a template is a claim that
nothing will ever re-check.

## License / sharing

Private project. See **`DATA_SOURCES.md`** for every source, what it is used for, and what
may be redistributed. The short version: do not republish `odds.csv`, do not repackage FPL
data as a feed, and do not build anything commercial on the FotMob dependency.

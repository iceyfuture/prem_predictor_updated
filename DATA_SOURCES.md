# Data sources, provenance and redistribution limits

Audit finding §5. Every external source this project reads, what it is used for, what its
terms allow, and what that means for anything published from it.

**Nothing here is legal advice.** It is a record of what was checked and when, so the next
person does not have to re-derive it — and so a licence problem is noticed before something
is redistributed, not after.

Last reviewed: **2026-09-23**.

---

## The sources

| Source | Endpoint | Used for | Access | Status |
|---|---|---|---|---|
| Fantasy Premier League | `fantasy.premierleague.com/api/` | Player prices, ownership, status/injury news, per-gameweek points, official xG/xA/xGI | Public, unauthenticated | Primary player source |
| Football-Data.co.uk | `football-data.co.uk/mmz4281/` | Historical results and bookmaker closing odds (the market benchmark) | Public CSV download | Primary odds source |
| ESPN | `site.api.espn.com/apis/site/v2/sports/soccer/eng.1/` | Fixtures, live scores, match stats, bookmaker lines | Undocumented public JSON | Fallback fixture source |
| FotMob | `fotmob.com/api/` | Fixtures and richer per-match team stats (xG, shots, possession) | Undocumented, scraped | **Being reduced — see below** |
| Kalshi | `api.elections.kalshi.com/trade-api/v2/` | Exchange prices for EPL markets, used as a second market benchmark | Public, unauthenticated read | Secondary market source |
| API-Football | `v3.football.api-sports.io` | Licensed fixture/stat API | API key, free tier (100 req/day) | Available, largely unused |
| GitHub raw | `raw.githubusercontent.com` | A pinned reference dataset | Public | Static |

## Provenance recorded per row

Anything used as evidence carries where it came from:

- **`odds.csv`** — `provider`, `price_type` (opening/closing), `overround`, `collected_at`,
  `source_file`, `source_sha8`, plus a second provider's prices in `alt_*`. `collected_at`
  is the mtime of our cached download: an upper bound on when *we* observed the price, not
  when the bookmaker posted it. Football-Data does not timestamp individual quotes, and the
  field is documented as the weaker thing it actually is. See Rule 46.
- **`dashboard/ledger.csv`** — `code_commit`, `model_config`, `training_cutoff`,
  `data_version` on every forecast. See Rule 51.
- **Every source, every build** — expected rows, received rows, completeness percent,
  `as_of` and a note, via `check_source()`. A source the desk cannot function without fails
  the build rather than publishing a plausible-looking blend of different ages. See Rule 44.

## Redistribution limits

These are the constraints that actually bite:

**Football-Data.co.uk** is free for personal use and explicitly asks that the data not be
republished commercially or passed off as your own. The odds here are a *benchmark for
measuring this model*, not a product. Do not publish `odds.csv` as a dataset. Derived
summary statistics (a Brier score, an RPS difference) are fine; the price table is not.

**Fantasy Premier League** is Premier League data served for the official game. There is no
public licence granting redistribution. Player names, prices and points are shown to the
person running their own desk; they should not be repackaged as a feed for others.

**FotMob** has no public API and no terms permitting programmatic access. The audit's
instruction was explicit: *"Do not build a commercial dependency on undocumented FotMob
scraping."* That is correct and it is a real exposure — it can break without notice and
using it at volume is not defensible. **Current status: FotMob is the fixture source with
ESPN as fallback, and supplies per-match team stats that FPL does not.** The migration path
is documented below and is not finished.

**ESPN**'s scoreboard JSON is undocumented and unversioned. It has already broken twice in
this project's short life — a 403 on every browser User-Agent (Rule 24) and a 400 on date
ranges (Rule 46's predecessor). Treat it as a convenience, never as a guarantee.

**Kalshi** allows unauthenticated reads of public market data. Prices are shown for
comparison against this model. No orders are ever placed by this code, and none should be.

**API-Football** is properly licensed but the free tier is 100 requests/day, which does not
cover a full daily refresh across fixtures, stats and players. It is the right destination
for anything that must be dependable.

## Known gap: the FotMob dependency

Audit §5 asks to prefer licensed API-Football data or already-collected official FPL xG over
undocumented scraping. Partially addressed, honestly incomplete:

- **Done** — FPL's own xG/xA/xGI now carries most of the player-level load (Rule 35a). The
  scorer model and player ledger read FPL, not FotMob.
- **Not done** — per-match *team* stats (team xG, shots, possession) still come from FotMob,
  and fixtures still prefer it with ESPN as fallback. FPL does not publish per-match team
  xG, so this is not a straight swap; API-Football does, within a request budget that would
  need managing.

Until that is closed, anything derived from FotMob should be treated as best-effort: it may
vanish, and it must not be redistributed.

## What is in the repository

`premier_league_history/` holds the results dataset (Premier League matches since 1993 with
shots, corners and cards). `outputs/` holds derived CSVs. `odds.csv` holds the market
benchmark. The ledgers under `dashboard/` hold this desk's own forecasts and their grades —
those are the project's own output and carry no third-party restriction.

`premier_league_history/player_gameweek.csv` (50MB) is deliberately **not** committed; no
model code reads it. Regenerate with `premier_league_history/build_pl_history.py`.

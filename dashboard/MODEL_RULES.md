# Model rules (enforced every build)

These are hardcoded into the pipeline so they are always followed. Each rebuild
(`build_dashboard.py`) applies all five.

### 1. Refit team attack/defence after each matchday
`prem_dixon_coles.fit(extra=...)` accepts freshly-finished results, and `build_dashboard`
calls it **every run** (not the cache), folding in any 26/27 games that have finished from
the live ESPN feed (de-duplicated by date+teams). So ratings update as the season unfolds.
*Preseason:* no results yet, so the refit uses full history; the fold activates automatically
once games are played.

### 2. Weight new matches strongly, but not excessively
Recency decay `w = 8^(-age/DECAY_SPAN)` with `DECAY_SPAN = 3` (a game ~3 seasons old counts
~1/8), plus `RIDGE = 8` shrinkage. These weren't guessed — they won the out-of-sample sweep in
`validate.py` (RPS 0.2056 vs 0.2061 for the old settings). Strong recency without overreacting.

### 3. Update from xG where possible, not only final scores
When a folded finished game exposes shot data, `build_dashboard` blends the final score with a
shots-on-target xG proxy (`0.55·goals + 0.45·(0.31·SoT)`) before refitting, so a lucky 1-0 with
few chances doesn't spike a team's rating. Damps luck-driven "form". (Uses final scores when no
shot data is present.)

### 4. Keep early-season promoted-team predictions visibly provisional
Newly-promoted clubs get an **empirically-calibrated** cold-start prior — attack −0.12,
defence −0.32 — derived from 95 promoted sides (1994–2026) who averaged 1.0 pt/game and lost
49% of their first 5. Their cards show a `PROV` flag and **low** confidence, edges on their
games can never be "actionable" or "watch", and their FPL projections are discounted 30%.

### 5. Record predictions before kickoff, then grade them
`record_ledger()` writes every fixture's prediction **and the closing line** to `ledger.csv`
before kickoff (locked on first sighting, never overwritten), then grades each against the
actual result once it finishes. This is the always-on backtest of live predictions vs outcomes
and the market — the running evidence log behind the Backtest page.

---
Changing any of these is a deliberate edit here + in the code it points to — not a side effect.

### 6. Kalshi forward test (added 2026-08-07)
`kalshi.py` pulls the public `KXEPLGAME` series (no key). It is a **comparison layer only** —
it never feeds the model's probabilities or the sportsbook edge calc.

`record_ledger()` now logs three forecasts per fixture and freezes them at kickoff:
- `pred_*`  the model's FIRST locked prediction (made far ahead)
- `close_*` the model's LAST pre-kickoff prediction
- `kal_*`   Kalshi's LAST pre-kickoff mid (its closing price; refreshed each build until the
            game starts, because Kalshi only lists EPL ~a week ahead)
- `line_*`  the sportsbook's last pre-kickoff implied probabilities

On settlement each is scored with Brier + RPS and `closer` records whether the model or Kalshi
was nearer the truth. The running scorecard + settled-market board appear in the dashboard's
**Backtest -> Forward test** section.

### 7. Prop pricing + prop forward test (added 2026-08-07)
`kalshi.price_props(M)` derives BTTS, Over N.5 totals and win-by->N.5 spreads from the **same
Dixon-Coles score matrix** that prices the 1X2 — BTTS = P(both >=1), totals = P(sum > line),
spreads = P(margin > line). No extra model, no extra data.

`kalshi.fetch_props()` reads the public prop series `KXEPLBTTS` / `KXEPLTOTAL` / `KXEPLSPREAD`.
A quote is marked **thin** (and can never produce an entry) when volume < 50 or the bid/ask
spread exceeds 8c — which is every prop quote at the moment, since they open as placeholders.

`record_props_ledger()` locks the model price for **every** market on every fixture and Kalshi's
last pre-kickoff quote where one exists, then settles each from the final score with a binary
Brier. Results appear under **Backtest -> Forward test -> Props**.

Not covered: corners and goalscorer props. Kalshi lists neither for EPL today, and the FPL API
has no corner counts (only who takes them). Corner data does exist in
`~/premier_league_history/results.csv` (9,880 matches back to 2000-01) if a corners model is
ever wanted — it would be a separate Poisson fit, not free from the goals model.


### Fix log — 2026-08-20 (season-eve check)
**Ledger keys made stable.** Keys were `time|home|away`; when TV picks moved October kickoff
times, 25 fixtures got a SECOND ledger row and the orphaned originals could never settle
(ledger 380 -> 405, props 3040 -> 3240). Keys are now `home|away` (an ordered pair occurs
exactly once per PL season) with `kickoff` stored as a mutable column. Existing files were
migrated: duplicates merged, keeping the EARLIEST locked prediction and the newest closing
values. Verified stable across repeated builds.

### Fix log — 2026-08-24 (first graded matchweek)
**Forward-test snapshot now locks once per gameweek.** The FPL squad is regenerated on every
build (by design). The snapshot guard was `if key not in rows`, which ADDED each rebuild's new
picks to the same gameweek — GW1 grew to 26 players with 17 marked as the "XI" across 4 batches.
Now a gameweek is sealed once it has any snapshot. GW1 was reset to its first genuine locked
15 (snapshot 2026-07-29, 11 in XI); backup at /tmp/fpl_forward.bak.csv.

### 8. Corners model (added 2026-08-24) — a DIFFERENT method, on purpose
`prem_corners.py` does NOT reuse the Dixon-Coles architecture. I built it that way first and
it failed validation: team-corner MAE 2.314 vs 2.207 for a plain rolling average, and total
corners came out WORSE than naive. Corner rates track current tactics/personnel, which an
8-year window with a 3-year half-life follows far too slowly. Higher ridge made it worse still
(swept 8/50/200/800/3000), so the MLE was abandoned.

Shipped method: rolling mean of each club's last 30 matches (window swept 8/12/20/30/50/80),
expected corners = (team's corners-for + opponent's corners-against) / 2, Poisson for the
P(>= N) curves Kalshi quotes.

Validated walk-forward, leak-free, on 9,170 matches:
  TEAM  corners  MAE 2.207 vs 2.391 naive (+7.7%), correlation 0.375  -> real signal
  TOTAL corners  MAE 2.838 vs 2.849 naive (+0.4%), correlation 0.123  -> essentially none
Total-corner rows carry `total_weak: True` and are never treated as an edge. This season's
corner counts are folded in each build from ESPN match stats.

### 9. FPL chip advisor (added 2026-08-24)
`fpl_chips.py` values each chip in EXTRA POINTS from the same per-matchweek projections that
build the squad, against a threshold for what the chip is worth in a week that justifies it:
  Bench Boost    = sum(bench projections)              bar 16.0
  Triple Captain = captain's projection (one more x)   bar 9.0
  Wildcard       = optimal XI - squad held, minus the ~1 free transfer you get anyway   bar 18.0
  Free Hit       = same one-week gap, only pays in a blank/double/injury week            bar 18.0
Wildcard and Free Hit need a previous gameweek's locked squad, so they are "unknown" in MW1.

### Fix log — 2026-08-24 (fantasy grading)
Two forward-test bugs found via a user-reported score mismatch (they scored 52, the ledger
said 39): (1) the snapshot was taken during the LOCKED preview window (2026-07-29) instead of
at reveal, so it graded a month-old preview squad; (2) the captain was not doubled. Both fixed:
snapshots now only happen once the team is revealed, a `cap` column is stored, and totals count
the captain 2x on both projection and actual.

### Fix log — 2026-08-25 (leak I introduced, then closed)
Clearing GW1's bad fantasy snapshot let it RE-lock on the 2026-08-24 build, by which point 9 of
GW1's 10 games had finished and Rule 1 had already refit the model on those results. The squad
was therefore picked with hindsight and its 58.0 is not evidence of anything.

Two changes: (1) `forward_test()` now refuses to snapshot a gameweek whose first ball has been
kicked (`before_ko` guard, on top of the existing lock-once and reveal guards); (2) any week
whose snapshot post-dates its first kickoff is flagged `tainted` and the UI labels it
"not valid" with a banner excluding it from claims about model skill.
GW2 (locked 2026-08-26, first kickoff 2026-08-28) is the first clean fantasy forward test.

### 10. Current club comes from FPL, every build (added 2026-08-25)
The 2026/27 squad list was scraped from premierleague.com in July. Transfers kept happening,
so it went stale: 15 players were at the wrong club, including Morgan Rogers still shown at
Aston Villa after moving to Chelsea. That is not cosmetic - the scorer model splits each club's
expected goals by its players' historical goal shares, so a transferred player was still
contributing his share to his OLD club and inflating its scorer probabilities.

`refresh_clubs()` now re-derives current club from the live FPL API on every build and rewrites
it into squad_2026_27_linked.csv, player_rankings_2026_27.csv and top50_players.csv. It is
idempotent: a second run reports zero changes. Transfers self-correct from now on.

Bug found while doing this: FPL names the promoted clubs "Coventry City" / "Hull City" /
"Ipswich Town" but FPL_TEAMS only had the short forms, so the alias lookup fell through and
those leaked in as three EXTRA clubs (23 teams instead of 20), splitting their squads. Added
the full names to feeds.FPL_TEAMS.

### Rule 3 RETIRED — 2026-08-25 (validated harmful)
The xG-proxy blend was never validated when it was added. It has now been swept out-of-sample
over 2018-26, fitting on (1-w)*goals + w*(SoT * league conversion) and always GRADING against
real goals:
    w=0.00 goals only   RPS 0.2092   <- best
    w=0.45 (shipped)    RPS 0.2102
    w=1.00 proxy only   RPS 0.2141
Monotonically worse. SoT x conversion discards shot quality, so it is a noisier training target
than the goals it replaced. The blend is now OFF. Real per-shot xG (FPL publishes it per player
per gameweek from 2026-27) is worth re-testing once a season or two has accumulated.

### Finding — the model adds no incremental information on 1X2 (2026-08-25)
Blending model and closing line out-of-sample over 3,040 priced matches:
    100% model  RPS 0.2092
     50/50      RPS 0.1994
    100% market RPS 0.1953   <- optimal
Monotonic: every step toward the market improves RPS, log loss and accuracy. There is no
weight at which the model improves on the closing line, so for match outcomes it carries no
signal the market has not already priced. Consequences: do not trade 1X2 against a sharp book;
the model's value has to come from markets the book prices loosely (thin props) or from
domains with NO market at all (FPL team selection, which is where it demonstrably helps).

### Rule 4 REPAIRED — 2026-08-25 (evidence shrinkage for thin-data clubs, "K")

**The bug.** Rule 4 only ever applied to clubs *missing* from the fitted model. Rule 1 refits
after every matchday, so a promoted club stopped being missing the moment it played once —
and from then on carried a full-strength rating fit on a single game, with every Rule 4
guardrail silently switched off.

Live example that exposed it: Hull have **0** matches in the 8-year window. They beat Man
United 2-0 in MW1 and came out rated **4th in the league with its best defence** (defense
+0.673 vs Arsenal +0.504), `provisional: False`, and an unguarded 76%-EV "edge" on their
own MW2 fixture. Coventry, on one loss to Arsenal, sat 19th. All three markets had that game
at Coventry 51.8 / draw 26.8 / Hull 21.4; the model had Hull the favourite.

**The fix.** `apply_cold_start()` now shrinks toward the COLD_* prior by how much evidence a
club actually has, instead of discarding the prior on first contact:

    w = n_eff / (n_eff + COLD_K)      rating = w*fitted + (1-w)*cold_prior
    COLD_K = 15.0    PROVISIONAL_N = 40.0

`n_eff` is the model's own time-weighted match count, so established clubs sit in the
hundreds, w ~ 1, and nothing about them changes. Clubs under PROVISIONAL_N stay in the
`cold` set, which keeps Rule 4's flags, confidence penalty and edge-downgrade switched on
until they have roughly a season of weighted evidence.

**How K was chosen — and the honest verdict.** `sweep_coldstart.py` walk-forwards 14 seasons
(2012-13..2025-26) refitting before EVERY matchday, which is what Rule 1 does live and what
validate.py's season-level harness could not test. One fit per matchday serves all K, since
shrinkage is applied post-fit. Null (K=0 = shipped behaviour) included.

    ALL FIXTURES (n=5320)     K=0 0.1992 | K=15 0.1992 | K=25 0.1993 | K=inf 0.2321
    THIN ONLY   (n= 163)      K=0 0.1923 | K=15 0.1891 | K=25 0.1890 (+1.7%)

The thin-fixture gain is **NOT statistically significant** — paired t = 0.83 at K=25 (and
1.10, 0.98, 0.70 at K=10/15/40). Stratified, the gain sits almost entirely in n_eff 15-40;
in the 5-15 band where Hull actually sits (live n_eff = 5.96) it is flat.

So this ships as a **guardrail, not an accuracy improvement**, and must not be described as
one. It is justified because it costs nothing overall (RPS identical to 4dp), it restores
Rule 4's protections to the case they were written for, and it removes an output the model
had no business producing. Effect on the live pathology:

    Hull net rating  +0.578 (4th)  ->  -0.150 (18th),  provisional restored
    Coventry v Hull  25/36/39      ->  38/30/32        (market 52/27/21)
    the 76%-EV Hull "edge"         ->  graded low, confidence 50, tier low

Still open: even at K=40 the model has Coventry only ~38% where the market says 52%. The
shrinkage bounds the damage; it does not make the model right about promoted clubs.

### 11. Transfer reality — 2026-08-27 (the recommended squad must be REACHABLE)

`build_team()` drafts an optimal 15 from scratch every gameweek. That is correct in GW1 and
after a wildcard, and wrong every other week: you hold last week's squad and get **one free
transfer**, with each extra costing **-4 points**. A "best XI" you cannot reach is not advice,
and grading it in the forward test measures a team nobody could have owned.

`fpl_transfers.py` now plans from the squad actually held:
  * objective = points you would REALLY score (best legal XI with the **captain doubled**), so
    an upgrade that changes your captain is valued properly and a bench-only upgrade scores ~0
  * searches 1..3 transfers by beam search, subtracting the -4 hit per transfer beyond the free
    allowance, and reports the net at each count so "is a hit worth it" is visible, not implied
  * a held player who is injured is dropped by `project_gw` (zero expected minutes) but you STILL
    OWN HIM - he is carried at proj 0.0 from the full FPL list, which both keeps the squad at
    15 and correctly makes him first in line to sell
  * selling price is assumed = current price. Real FPL sells at purchase price plus half the
    profit, so a risen player nets slightly less. Affects affordability, not the ranking.

`forward_test()` now snapshots the **reachable** squad (held + recommended moves), not the
draft. The from-scratch draft is kept alongside as `team.draft` for reference. GW2's snapshot
had already locked the unreachable draft, so it was cleared and re-locked at 2026-08-27T19:47,
still ahead of the 2026-08-28 first kickoff (backup /tmp/fpl_forward.bak2.csv).

First run, GW1 -> GW2:
    0 transfers  +0.0 gross    0 hit   +0.0 net
    1 transfer   +3.9 gross    0 hit   +3.9 net   Gibbs-White (injured) -> Mbeumo   <- take
    2 transfers  +5.0 gross   -4 hit   +1.0 net
    3 transfers  +5.6 gross   -8 hit   -2.4 net
Reachable XI 44.9 vs the draft's 47.0 - that 2.1-point gap is the honest cost of holding one
free transfer, and it is now shown rather than hidden.

### 12. The held squad comes from the FPL API — 2026-08-27

Rule 11's planner started from the model's own GW1 snapshot. That snapshot was the **tainted**
one (locked 2026-08-24, after GW1 had finished), so it was never the user's real team — and
the planner duly recommended buying Mbeumo, a player the user already owned. Caught by the
user, not by a test.

`fpl_transfers.real_squad()` now reads the actual squad from the public FPL API using the
entry id in `fpl_config.json`:
  * `entry/{id}/event/{gw}/picks/` — picks for the LIVE gameweek are private until its
    deadline, so it walks back to the newest public gameweek and then applies
    `entry/{id}/transfers/` to carry forward anything already bought
  * `entry/{id}/history/` — real **bank** and squad value, replacing the assumed GBP100.0m
  * budget = sum(current prices of held players) + bank, keeping accounting consistent with
    the current-price basis used everywhere else (FPL's own `value` uses sell prices, which
    differ, so it is deliberately not used as the budget)
Falls back to the snapshot only when no entry id is configured, and says which source it used.

### 13. Availability is a probability, not a boolean — 2026-08-27

`fpl_players()` set `fit = (status == "a")`, and `project_gw` hard-filtered on it. A 75% doubt
was therefore valued at **0.00** — Gibbs-White was a 75% doubt whom FPL itself projected at
ep_next 2.1. That distorts the XI and makes a doubtful player look like free money to sell.

`_avail(p)` now returns `chance_of_playing_next_round / 100`, falling back to
`{a: 1.0, d: 0.75}` by status when FPL publishes no percentage. The likely XI is picked on
availability-weighted ep, and each projection is scaled by that probability.

Residual limitation, deliberately not fixed here: the likely XI is still a hard top-11 cut, so
a player outside it projects 0 no matter how available he is. Proper handling needs a rotation
/ minutes model rather than a cut-off. The GW2 recommendation was identical before and after
this change, so it is robust to the simplification.
**RESOLVED 2026-09-01 — see Rule 19.** The cut is gone.

### 14. Player projections now use this season's xG / xA / defensive contribution — 2026-08-27

`build_player_stats.py` compiles every FPL player's individual statistics to
`outputs/fpl_player_stats_2026_27.csv` (616 players x 50 cols) and
`outputs/fpl_player_gameweek_2026_27.csv` (per-player per-gameweek, from `event/{gw}/live`).

Source is `bootstrap-static`, NOT a scrape of the /statistics page: that page renders from the
same endpoint but shows one stat column at a time behind pagination. Verified field-by-field
against the live page - price and total points matched on 8/8 distinct players checked (the
9th apparent mismatch was two different players both named Palmer).

Three changes to `simulate_fpl.project_gw`:

1. **Goal share** was purely historical (`prem_scorer` shares over the fitting window), which
   cannot see a transfer, a new signing or a changed role - the same blind spot that mis-priced
   promoted clubs at team level. Now blended with each player's share of his projected XI's
   xG this season: `share = (1-w)*historical + w*current`.
2. **Assist share** was a flat `0.6 * goal_share`. Now driven by xA share on the same blend.
3. **Defensive contribution** was MISSING ENTIRELY. FPL awards 2 pts at a threshold of
   defensive actions (10 for DEF, 12 for MID/FWD); the formula modelled goals, assists and
   clean sheets but not this, understating defenders and holding midfielders.
   Added as `P(actions >= threshold) * 2`, Poisson on the per-90 rate.

All three are evidence-shrunk on minutes, `w = m/(m+900)` (~10 full matches for half weight),
because one gameweek of per-90 rates is meaningless - a player with 1 minute read as 270
defensive actions per 90. DC rates shrink toward the POSITIONAL MEDIAN, not toward zero;
shrinking to zero would permanently understate defenders, which is the bug being fixed. The
minutes bar for that median steps 180 -> 60 -> 1 so it does not silently return zero in August.

Effect after MW1 (w = 0.09, so this season barely counts yet, by design):
  * DC is near-flat at ~0.30-0.34 pts for every defender - it will only differentiate once
    minutes accumulate. Currently inert, deliberately.
  * The xG/xA blend does move players: White (xg90 0.18, xa90 0.21) displaced Calafiori
    (0.04, 0.10) in the XI despite identical fixture and near-identical DC.
  * XI projection 46.0 -> 47.8.
  * The GW2 transfer recommendation (Thiago -> Isak, +1.1 net) was UNCHANGED by this, by the
    availability fix, and by the real-squad fix - three independent changes, same answer.

Not yet done: these are correlational inputs, not a validated improvement. There is no
out-of-sample test that the blend beats historical shares alone, because 2026/27 has one
gameweek. Worth a walk-forward once a season of FPL expected-stats has accumulated.

### 15. Scheduled refresh — 2026-08-27 (the season rolls forward on its own)

A daily launchd agent (`com.samiakil.premrefresh`, 06:30 local) runs `refresh.sh`, which chains
`build_player_stats.py` -> `build_dashboard.py` -> `make_standalone.py`. It runs whether or not
Claude is open, which is the point: the desk going stale is what made the GW2 fantasy squad
still show as "locked" a full day after it had actually unlocked.

**Daily, not weekly.** Gameweeks do not land on a fixed weekday (midweek fixtures, TV
reschedules), FPL prices and injury news move nightly, and deadlines shift. A weekly job leaves
the desk up to six days stale. Every step is idempotent, so extra runs cost nothing:
stat snapshots de-dupe on date, the ledger locks once per fixture, the fantasy snapshot seals
once per gameweek and refuses to touch a gameweek whose first ball has been kicked.

**Roll-forward.** `fpl_player_history.csv` appends a dated snapshot per player per run (xG, xA,
xGI, xGC, price, ownership, form, minutes, points), so the season keeps its trajectory instead
of being overwritten by the latest totals. `fpl_player_gameweek_2026_27.csv` re-derives every
finished gameweek from `event/{gw}/live`, so it grows a gameweek at a time on its own.

**Three bugs the first unattended run exposed** — all of them invisible when running by hand:
1. `ProcessType Background` + `LowPriorityIO` throttled the job so hard it took ~2 hours for
   work that takes seconds, and starved a network read into "connection reset". Now `Standard`;
   the same run finishes in seconds.
2. `python x.py | grep ...` reports GREP's exit status, so `refresh.sh` logged `exit=0` while
   `build_dashboard.py` was crashing on a traceback. A scheduler that reports success while the
   build is broken is worse than no scheduler. Now uses `pipefail` + `pipestatus[1]`.
3. `(kp or {}).get("btts", {}).get("url", "")` crashed the whole build: the `{}` default does
   not apply when the key EXISTS and is null, which is what Kalshi returns for a prop series
   with no live market. Fixed here and at the one other instance of the pattern
   (`bt.get("meta", {})`). The per-gameweek fetch now retries 4x with backoff and exits
   non-zero if a gameweek is genuinely unfetchable, rather than silently dropping it.

To check on it: `tail ~/prem_predictor/dashboard/refresh.log`, or `launchctl list | grep
premrefresh` (second column is the last exit status; 0 is healthy).

### 16. Away-goal calibration — 2026-08-30 (a real, significant bias, found by score analysis)

The MLE fit systematically UNDER-predicts away goals. Walk-forward over 2,774 matches
(2018-19..2025-26): away rate **+0.103 goals light**, total goals **+0.114 light, t = +3.63**.
Per season it is positive in **8 of the last 10** (mean +0.099, sd 0.098), so it is a standing
bias, not one freak year. Home goals are well calibrated (+0.011).

`prem_dixon_coles.AWAY_CAL = 1.08`, set to cancel the measured bias rather than fitted to an
outcome metric. Swept effect:

    total-goals bias   +0.114 -> +0.017
    Over-2.5 pricing    51.3% ->  53.6%   (actual 55.1%)
    RPS                0.2052 -> 0.2051   (no cost)

**This matters for TOTALS and BTTS, not the moneyline.** 1X2 depends on the RATIO of the two
rates, so scaling one barely moves it - which is why the bias survived unnoticed while RPS
looked fine. It is also the likely explanation for the prop ledger being full of losing
"under"/"no" bets: an under-predicted goal rate makes every under look like value. Season to
date the props were 44/95 with P&L +$27.51 that collapses to -$16.33 once the top two winners
are removed, and the losing side was overwhelmingly unders.

Residual, deliberately not papered over: even at zero mean bias, Over-2.5 still prices ~1pt
under the actual 55.1%. That is over-dispersion - real scorelines have fatter tails than a
Poisson - and a multiplier cannot fix it. Inflating AWAY_CAL further would be fitting the
symptom. The proper fix is a negative-binomial count model.

### 17. Form vs quality ratings — 2026-08-30

`fpl_form.py` -> `outputs/fpl_form_ratings.csv`. Reports form and quality as SEPARATE numbers
because conflating them is what makes a raw form table recommend the player about to regress:

    QUALITY  established level, recency-weighted historical output per 90
    FORM     this season's underlying output per 90, z-scored within position
    HEAT     FORM - QUALITY, i.e. performance against the player's OWN baseline

Form is measured on UNDERLYING output (xGI/90), not points: points are lumpy - bonus, clean
sheets and hauls swamp a handful of games - while xGI accumulates from every chance. Points/90
is carried as a cross-check only. The blend is position-aware (GK/DEF lean on defensive work
and points, MID/FWD on xGI). `confidence = m/(m+900)`; `form_adj` is the shrunk value used for
ranking, `form_raw` is what the player actually did.

Bug fixed while building it: 16 surnames in `player_rankings_2026_27.csv` map to two different
players, usually an outfielder and a keeper at the same club. Keying a dict on the bare name
kept whichever row came last, so Cole Palmer was rated on Chelsea's GOALKEEPER Palmer's zero
xGI and scored -2.02 for quality. Now resolved on position first, then most minutes. Same class
of bug as the earlier Bruno Fernandes miss.

Also fixed: `build_player_stats.gameweeks()` only pulled FINISHED gameweeks, so a gameweek with
9 of 10 games played was invisible to every consumer until the last match ended - exactly when
form data is most wanted. The in-progress gameweek is now pulled and flagged `provisional`.

### 18. Team corners, correct score and team totals — 2026-08-30

We had been pricing **4 of the 30+ EPL series Kalshi lists**, and all four were markets the
1X2 evidence says we cannot beat. Enumerating `/series?category=Sports` turned up the rest.
Three are now priced and logged, all from models that already existed:

| Series | Market | Priced from |
|---|---|---|
| `KXEPLTCORNERS` | **Team** corners | `prem_corners` - the ONLY market where we have validated out-of-sample signal |
| `KXEPLSCORE` | Correct score | the Dixon-Coles matrix cell, which we were computing and discarding |
| `KXEPLTEAMTOTAL` | A team's goals | a marginal of that same matrix |

**Team corners is the one that matters.** Rule 8 validated the corners model at +7.7% MAE over
a naive baseline on TEAM corners (correlation 0.375) and +0.4% on TOTALS (correlation 0.123).
Kalshi lists both; only the team side is priced as a prop, and `KXEPLCORNERS` (totals) is
deliberately left alone. This is the first market we have entered where the model has
demonstrated signal against reality rather than against a price.

First live quote, Aston Villa v Arsenal:

    Arsenal 6+ corners   model 38.4%   YES ask 45c (EV -14.7%)   NO ask 57c (EV +8.1%)
    Villa 5+ corners     model 40.0%   YES ask 41c (EV  -2.4%)   NO ask 60c (EV  0.0%)

Not thin: 1,124 volume and a 2c spread on the Arsenal leg. Villa prices within a point of our
number; Arsenal is a 5.6-point disagreement.

Settlement uses the real corner counts already captured from ESPN into `corners.actual`.
`settle_prop` returns None when corners are not yet known, and the row is left ungraded for a
later build rather than being force-settled as a loss.

Correct score carries the (home, away) pair in the `line` column instead of a number, and
`settle_prop` special-cases it. Both `KXEPLSCORE` and `KXEPLTEAMTOTAL` currently have no open
markets; the parsers and pricing are in place for when they list.

Discipline unchanged: these are LOGGED, not traded. The prop ledger exists to build evidence
before anything is staked, and every previous "edge" in it collapsed once the top two winners
were removed.


### 19. Expected minutes replace the top-11 cut — 2026-09-01

**The bug this closes.** `project_gw` built each club's "likely XI" by sorting available players
on availability-weighted `ep_next` and keeping the top 1 GK + 10 outfield. Everyone below the
line projected **exactly 0.0** — not "less", zero — so a fit, in-form, coin-flip-to-start player
was valued identically to an injured one. That is a discontinuity at an arbitrary boundary, and
it is what benched Gibbs-White in weeks he returned points. Rule 13 named it and left it.

Measured on held-out seasons, the cut zeroes **~21,400 player-matches a season, of which 1,813
actually started and 4,188 played at all** (2025-26; 1,776 / 4,214 in 2024-25) — roughly 1.8 real
starters per team-match discarded. It was also silently deciding transfers: Thiago had started
both of this season's games and carried `ep_next` 1.0, so the cut dropped him and the planner
listed a player the user owns and who plays every week as "unavailable", worth 0.

**What replaced it.** `fpl_minutes.py`: two logistic models — P(start) and P(appear | !start) —
on recency-weighted prior minutes, price, and price rank inside the club roster, fit on
2023-24..2025-26 player-gameweeks (86,765 rows). Then the constraint the logistic does not know:
**exactly 11 players start**, so P(start) is scaled inside each club to sum to 1 (GK) and 10
(outfield), water-filled at the 1.0 cap. Availability multiplies BEFORE that scaling, so an
injured starter's minutes are redistributed to his own team-mates rather than vanishing.

Minutes then follow from four empirical constants (started: P(60+) 0.932, E[min] 82.8; sub:
P(60+) 0.013, E[min] 18.2; 4.11 sub appearances per team-match). Consumers get the whole state
distribution, because FPL pays 1 appearance point under 60 minutes and 2 at 60+, and a clean
sheet only counts for a player who reached 60 — thresholds that must be integrated over, not
evaluated at E[minutes].

**Validated walk-forward** (train on earlier seasons only; baseline credited with the empirical
P(60+|start) rather than the certainty of 90 minutes the shipped code actually assumed):

| test season | metric | top-11 cut | minutes model |
|---|---|---|---|
| 2024-25 | 3-state Brier | 0.4522 | **0.3198** |
| 2024-25 | 3-state log loss | 2.7718 | **0.5825** |
| 2024-25 | minutes RMSE | 29.15 | **23.71** |
| 2025-26 | 3-state Brier | 0.4140 | **0.2902** |
| 2025-26 | 3-state log loss | 2.5464 | **0.5388** |
| 2025-26 | minutes RMSE | 28.18 | **22.64** |
| both, GW<=6 | 3-state Brier | 0.5490 / 0.5299 | **0.3592 / 0.3376** |

The gain is largest in the opening weeks, which is where the season is. Minutes **MAE** goes the
other way (12.59 -> 15.11) and that is not a defect: minutes are bimodal, MAE is minimised by the
median, and everything downstream multiplies the expectation — so squared error and the proper
scoring rules are what bind.

`W_K` (history shrinkage, `w = n/(n+W_K)`) was **swept, and the first guess was wrong**. Set to
3.0 by analogy with the other shrinkage constants in this project, it predicted 0.67 for players
who had started both openers against an actual 0.84. Minutes persist far more strongly than
per-90 rates do. At W_K = 0.25: all-rounds Brier 0.0840 -> 0.0818, GW<=6 0.1077 -> 0.0986 (-8.5%),
and that cell lands at 0.83 against 0.84.

**Deliberately not included.** Last season's minutes (name-matched from
`player_season_totals.csv`) improved GW<=6 Brier by ~2%. It needs cross-source name matching,
which has produced three separate bugs here already (Bruno Fernandes, both Palmers, the two
Wilsons). Not worth 2%. `ep_next` is also excluded — history does not carry it, so its weight
cannot be validated; where FPL's ep disagrees with our minutes the disagreement is now visible
in the `p_start` / `xmin` columns instead of silently deciding the XI.

**Effect on the live GW3 decision:** held XI 58.5 -> 61.0, the "unavailable" list emptied
(Thiago 0.0 -> 4.5), the captain moved Isak -> Gibbs-White, and the recommended transfer changed
from **sell Mbeumo -> Gakpo (+0.9)** to **sell Núñez -> Iwobi**. The Mbeumo sale was a
consequence of the cut, not of the transfer objective.

Also added: `build_player_stats.py` now stores `status`, `chance_next` and `ep_next` in the
daily roll-forward snapshot. Without them a past gameweek cannot be re-projected from what was
known before kickoff, which is exactly what a head-to-head between two projection versions
needs. That test is not possible for GW1/GW2 — those inputs are already gone — and is possible
from GW3 on.

Refit: `python dashboard/fpl_minutes.py --fit` (writes `.state/fpl_minutes.json`).

### 20. The transfer objective looks past this week — 2026-09-01

`fpl_transfers.plan` optimised a **single gameweek**, which is how it came to recommend selling
the squad's best in-form player for +0.9: one awkward fixture is enough to tip a fractional
gain, and the fixture after it never entered the sum. The objective is now the decay-weighted
XI total (captain doubled, as before) over `HORIZON = 3` gameweeks, `DECAY = 0.65`.

`DECAY` is a judgement, not a swept constant, and is deliberately steep — you get another free
transfer every week, so a squad three weeks out is only loosely the squad you will own. Hits stay
a one-time -4, which is the correct economics: the cost lands once, the gain accrues over the
weeks you hold the player.

Each option reports the **raw gain in every gameweek** (`by_gw`) alongside the weighted score
(`gross_h`), and the panel shows them as columns, because the weighted number cannot be checked
otherwise. The first version of that panel showed a one-week gross next to a three-week net, so
the row did not add up left to right, and the verdict read "net +2.8 pts over 3 GW" — which is
wrong: 2.8 is the DECAYED score, and the undecayed three-week gain is about 4.0. Both are fixed;
the panel now spells out Score = GW3x1 + GW4x0.65 + GW5x0.42 and Net = Score - Hit - Form bar.

**Form is a bar, not a bonus.** `fpl_form.form_adj` (this season's underlying output per 90,
z-scored within position, shrunk by `w = m/(m+900)`) now sets a threshold a move must clear:

    bar = FORM_GUARD * max(0, form_adj(out) - form_adj(in)),  FORM_GUARD = 0.75

It is a threshold rather than something added to the projection because `project_gw` already
carries part of the same evidence (this season's xG/xA share, at the same shrinkage) and adding
it twice would double-count. Two things checked rather than assumed:

  * it uses `form_adj` (shrunk), **not** `heat`. At this point in the season `heat` is
    small-sample noise — the current top 8 by heat have confidence 0.05-0.17, i.e. 45-170
    minutes — and gating transfers on it is precisely the trap `fpl_form.py` warns about.
  * `FORM_GUARD` is **not** derived from the observed form-to-points relationship. That
    regression looks strong (pts90 = 3.6 x form_raw, r = 0.88) and is circular: pts90 is one of
    `form_raw`'s own inputs. It measures what a player has scored, not what he will score.

**Honest status: the form bar is currently near-inert, by construction.** Two gameweeks in,
confidence is ~0.13, so `form_adj` spans about +/-0.25 and the bar on any live move is 0.0-0.1
points. It scales up as minutes accumulate and is worth roughly a point by mid-season. The thing
that actually stopped the Mbeumo sale was Rule 19, not this.

Robustness: the GW3 recommendation (Núñez -> Iwobi, 1 transfer) is **unchanged** across
HORIZON 1/2/3, DECAY 0.5/0.65/0.8 and FORM_GUARD 0.0/0.75/2.0. Only the size of the gain moves
(+1.9 one-week, +3.5 over three).

**Still open, deliberately.** A banked free transfer has option value — FPL banks up to five —
and the planner still does not price it, so it will recommend a move worth slightly more than
zero. The honest fix needs a number this project cannot yet ground; the horizon at least means
that number is now a three-week gain rather than a one-week one.


### 21. Team match statistics now have a record — 2026-09-01

The player side of this desk has kept a per-gameweek record since Rule 15
(`fpl_player_gameweek_2026_27.csv`, plus a dated roll-forward of season totals). **The team side
had none.** Club ratings were recomputed in memory every build and overwritten, this season's
results were re-derived from the ESPN feed and discarded, and the 28-stat team line ESPN
publishes for every finished match was fetched, read for its two corner counts, and thrown away.
Nothing at team level had a trajectory: you could ask what the model thought of a player on a
given date and not what it thought of a club.

`team_stats.py` writes it down, and is now step 2 of `refresh.sh` (after the player build, whose
output it consumes):

| file | what |
|---|---|
| `outputs/team_match_2026_27.csv` | one row per team per match, full ESPN stat line + xG/xA |
| `outputs/team_match_history.csv` | the same core stats per team-match back to 2000-01 (19,760) |
| `outputs/team_rolling_2026_27.csv` | rolling form AS AT each match, prior matches only |
| `outputs/team_form_2026_27.csv` | each club's rolling form as of now, one row per club |

**Sources, and the gaps, because they are not one feed.** Shots, shots on target, corners, fouls
and cards come from ESPN this season and football-data back to 2000-01. Possession, passing,
crosses, long balls, tackles, interceptions, clearances, blocked shots, saves and **offsides**
are ESPN-only, so 2026-27 forward. Offsides in particular **cannot be backfilled**: football-data
never carried it and FPL's `offside` column is populated for three seasons (2016-19) at ~5% of
rows. Team **xG/xA** is summed from FPL's per-player per-gameweek expected goals, 2023-24 forward
(2022-23 is half-populated — season total 732 against ~1,100 in full seasons — the same cutoff
`fpl_minutes.py` uses for `starts`). **xGC is the opponent's xG in the same match**, so xG and
xGC are one scale by construction. **Free kicks won = the opponent's fouls committed**: no feed
publishes free kicks, this is the standard proxy, and it is labelled as one. Offsides are
deliberately not folded into it so the definition is identical in both eras.

**The join is validated against a number neither side was asked for.** FPL also publishes a
per-player `expected_goals_conceded` — the xG faced while that player was on the pitch — so a
club's xGC should equal its opponent's xG. It does on **18 of 20** matches this season, within
0.35; the two that miss are provider rounding on penalty/own-goal xG, and all 20 line up on club
identity. (Summing that per-player xGC would be badly wrong: a player who lasted 90 minutes
carries the WHOLE team's figure, so the sum is ~14x the truth. Aston Villa's GW1 read 40.25.)

**Two bugs caught in the first run, both from taking a shortcut:**
1. Stat blocks were matched to clubs by name, and ESPN's spellings are its own. "Brighton & Hove
   Albion" did not match "Brighton", so Brighton's entire stat line for one match came out blank
   while its opponent's filled in. Now: alias table, prefix match, then **block order** as the
   backstop — verified 20/20 that ESPN puts the home side first — and a printed warning whenever
   the fallback is used, so an unknown spelling is loud rather than silent.
2. The rolling window carried back across the season boundary with no age limit, so the form
   table filled up with Bolton and Portsmouth, and **Hull's "last 6" mixed 2026 matches with
   2016-17** — they were last in this division a decade ago. Now `MAX_AGE_DAYS = 400`, the table
   is restricted to this season's 20 clubs, and `l6_n` / `l20_n` report how many matches actually
   made the window (Hull and Coventry: 2; Arsenal: 6 and 20).

Rolling form is strictly leak-free — a match's own numbers never enter its own rolling columns —
and uses two windows mirroring the form/quality split in `fpl_form.py`: `l6` is current form,
`l20` is level. **Neither window is swept**, and should not be described as tuned: there is no
out-of-sample target these general team stats are fitted to. Where a window IS tuned against an
outcome, that is `prem_corners.py`, swept to 30.

**Found while doing this:** `outputs/team_rankings.csv`, `team_rankings_2026_27.csv` and
`team_strength_index.csv` were orphans — nothing called `build_rankings.py`,  `link_squads.py`
or `compute_strength.py`, so they were frozen at 2026-07-17/22 with `decay 8**(-age/4)` against
a shipped `DECAY_SPAN` of 3. **Fixed in Rule 22.**


### 22. The ranking exports are back in the build — 2026-09-01

Three scripts wrote files that nothing regenerated: `build_rankings.py`, `link_squads.py` and
`compute_strength.py` were in no build and in no schedule. Their outputs had been sitting at
17-22 July since before the season started. They are now steps 3-5 of `refresh.sh`, and the
order is a dependency chain rather than a preference:

    build_player_stats -> team_stats -> build_rankings -> link_squads -> compute_strength
                                     -> build_dashboard -> make_standalone

`build_rankings` fits the history and writes `player_rankings.csv` + `team_rankings.csv`;
`link_squads` reads BOTH and filters them to this season's squads; `compute_strength` reads
`player_rankings_2026_27.csv`; and `build_dashboard` re-derives every player's current club
(Rule 10) into those same files, so it has to run LAST of the four or `link_squads` silently
undoes the club corrections. Verified: the whole chain is byte-for-byte idempotent across two
consecutive runs.

**The decay discrepancy fixed itself.** `build_rankings` already read `dc.DECAY_BASE/DECAY_SPAN`
rather than hardcoding them — the CSV said `8**(-age/4)` only because it was written when
`DECAY_SPAN` was 4. Regenerating produces `8**(-age/3)`.

**Two real bugs had to be fixed first, or wiring these in would have shipped wrong numbers daily
instead of stale ones.**

1. **`compute_strength` was building this season's table out of last season's league.** It
   called `dc.get_model()` — the plain historical fit, no Rule 1 fold of this season's results,
   no Rule 4 cold-start shrinkage — over a hardcoded `season == "2025-26"` club filter. The
   output contained **Burnley, West Ham and Wolves, all relegated**, and was missing
   **Coventry, Hull and Ipswich, all promoted**. It now refits through
   `build_dashboard.live_model()`, over the clubs in this season's fixture list, and carries a
   `provisional` column. It agrees with `dashboard.json` on the net strength of all 20 clubs to
   4dp, where before it disagreed about which clubs were even in the division.

2. **`build_dashboard` was reading the stale file and throwing the result away.** Line 554 did
   `strength = read_csv(.../team_strength_index.csv)` and `strength` was never used again — the
   rendered table has always come from `build_strength_2627()`. A dead read of a wrong file is
   the kind of thing that looks load-bearing the moment someone tries to use it, so it is gone.

Rule 1's fold is now a function, `build_dashboard.fold_finished(events)`, with `live_model()`
wrapping it and `apply_cold_start` together. `compute_strength` calls that rather than keeping a
second copy of Rule 1 that can drift from the first — which is exactly how these files got out
of step in the first place.

**Known limitation, not fixed:** `team_rankings_2026_27.csv` contains **18 clubs, not 20**.
`link_squads` filters the fitted ranking to this season's squads, and Coventry and Hull have no
matches in the 8-year window, so they have nothing to filter down to. The file is honest about
what it is — a filtered historical fit — but it is not the current-season team table. That is
`team_strength_index.csv`, which now covers all 20 with cold-start priors and flags the two as
provisional. Use that one.

### 19. "Recent form" was not recent — 2026-09-03 (two bugs, user-spotted)

The card read `Brentford -1 vs Sunderland -3` after Brentford had opened W/D and Sunderland
L/W. Both numbers were wrong, for two separate reasons.

**(a) This season was not in it.** `current_form()` is goal difference over the last 6 matches
and reads the historical results file, which ends at the close of LAST season. Rule 1 folds
finished 26/27 games into the ratings but nothing was folding them into form, so every card
showed last season's form. Tottenham read **+2** after losing 0-3 and 0-2. Fixed by passing the
same folded frame the refit uses.

    Brentford -1 -> +3      Sunderland -3 -> +3      Tottenham +2 -> -4      Arsenal +6 -> +10

**(b) "Last 6" reached back a decade.** The deque had no time cutoff, so a promoted club's last
six *Premier League* matches were whenever it was last in the league:

    Coventry  - form built from matches played April-May 2001   (25 years old)
    Hull      - form built from matches played in 2017          (9 years old)

`FORM_MAX_AGE_DAYS = 550` (~18 months) now bounds it, and `with_counts=True` reports how many
matches actually contributed so a thin number can be labelled. Clubs with six genuine recent
games are untouched; only the stale cases move.

    Hull -9 -> +3 (2 games)      Coventry -9 -> -4 (2 games)

Cards now print "(2 games)" when fewer than six contributed. This was never cosmetic: the value
feeds the supremacy blend, which was fitted on contiguous in-season form and had no business
being handed a result from 2001.

### 20. FotMob team stats + an xG strength index — 2026-09-07

`team_stats.py` already records a per-match team line from ESPN, and keeps it. What ESPN does
not publish is the measure that separates a good performance from a lucky one: **expected
goals**. `team_stats.py` derives team xG second-hand by summing FPL's per-player xG (2023-24
forward only). FotMob publishes it directly, plus several things nothing else here carries:

    xG, xG open play, xG set play, xG non-penalty
    xGOT (post-shot xG)          shot quality AFTER it leaves the boot - separates finishing
                                 and goalkeeping from chance creation
    big chances / missed         FotMob's own high-quality-chance count
    shots inside / outside box   location, not just volume
    touches in opposition box    territory where it matters
    duels, dribbles, sprints, distance covered

`fotmob.py` pulls `api/data/matchDetails` per fixture (ids from `api/data/leagues?id=47`),
caches each match for 30 days since a finished stat line never changes, and writes
`outputs/team_fotmob_2026_27.csv` — one row per team per match, 42 columns.

**The index.** Attack and defence are fit by ridge least squares on xG:

    xg_for(i vs j, home) = attack_i - defence_j + home_adv

The Dixon-Coles structure, solved on expected goals instead of scored ones. Over three matches
a 4-0 and a 1-0 are both a win and say very different things; xG accumulates from every chance
and so says more per match played. RIDGE = 1.0 because 60 rows cannot support 41 confident
parameters. Home advantage comes out at **+0.71 xG**.

Bug caught by sanity-checking the output: `defence` is solved with a minus sign in the design
matrix, so a high value already means "concedes less". Negating it on the way out inverted the
whole table and put Arsenal - the best xGA in the league by a distance - **20th**.

**This is a measurement, not a change to the match model.** The Dixon-Coles predictor is
validated walk-forward on goals; swapping its training target to xG is a change that needs its
own out-of-sample test, not an assumption. The index sits alongside it.

Runs on every build (`refresh.sh`), and `finished_rounds()` auto-detects new matchweeks, so it
picks up MW4 with no edit.

### 21. Corner probabilities were Poisson; corners are not — 2026-09-07

The live 60-75% probability band was reading 65.7% predicted against 50.0% actual. Tested first
rather than fixed: **z = -1.59 on n=29, NOT significant**, and the breakdown showed totals (17 of
the 29) were well calibrated at 66.5% vs 64.7%. The band was not broken. Five corner props were.

Testing those where the sample is large found something real. Across **19,760 team-matches**
(2000-01 to 2025-26) team corners have **variance/mean = 1.665**, where a Poisson requires
exactly 1.0. Corners are over-dispersed, so pricing P(>= N) off a Poisson over-states the common
lines and under-states the tails:

        line   actual   Poisson    NegBinom
         3+     83.5%    90.4%       83.7%
         4+     70.9%    78.5%       71.0%
         5+     57.0%    62.5%       56.9%
         8+     22.2%    17.7%       21.9%
        10+      9.7%     4.8%        9.5%

    mean absolute calibration error   4.74pt  ->  0.24pt

The 4+/5+ lines are exactly where a 60-75% quote sits, which is why the failure surfaced in that
band. The band was the symptom; the distribution was the cause.

`_pois` is now a negative binomial with `DISPERSION = 1.665`, giving variance = mu * 1.665 via
r = mu/(dispersion - 1). Measured, not assumed - and stable across 26 years (1.576 / 1.652 /
1.671 / 1.696 / 1.751 by era), so it is a property of football rather than a fit to one period.

**`expected()` is untouched**, so Rule 8's validated +7.7% MAE over the naive baseline still
stands - only the distribution around the mean changed, not the mean.

### 22. Scheduler reliability — 2026-09-12 (one DNS blip was costing a whole day)

Reported symptom: fantasy squads not posting, Kalshi settlements not fetched, data going stale.
Audit of 14 scheduled runs found 2 not-ok and durations from 0 to 96 minutes.

**Root cause, 2026-09-11 06:35:** `socket.gaierror: nodename nor servname provided`. The Mac was
waking from sleep and DNS was not up. The first network step died, and because nothing retried,
the failure took out **four of eight steps** - build_player_stats, fotmob, compute_strength and
build_dashboard. No fantasy squad, no settlement, no dashboard. With a once-daily schedule,
nothing recovered for 24 hours.

Four fixes:

1. **Wait for the network.** `net_ready()` polls fotmob / FPL / ESPN on port 443, up to ~5
   minutes, before the chain starts. A machine waking at 06:30 no longer loses the day.
2. **Retry each step**, 3 attempts with backoff, re-checking connectivity between tries. Every
   script here is idempotent - predictions lock once per fixture, the fantasy snapshot seals
   once per gameweek, stat snapshots de-duplicate on date - so a retry costs nothing and
   rescues a transient error. Verified with a harness: a step that fails twice then succeeds is
   reported "recovered on attempt 3"; one that always fails still sets the failure flag.
3. **Every 3 hours, not daily.** `StartInterval 10800` also fires on wake when a slot was
   missed, which `StartCalendarInterval` does not do reliably. `RunAtLoad` is now true so a
   reboot does not leave the desk stale until the next slot. Caches (ESPN 180 min, FotMob 30
   days for finished matches) keep repeat runs cheap - a fully cached run takes 33 seconds.
4. **`refresh_status.json`** records per-step ok/failed and a finish timestamp, so a silent
   failure is visible instead of only surfacing as stale data days later.

### 23. Chip advice was valued against a squad you do not own — 2026-09-12

Found by tracing a stray worktree (`.claude/worktrees/gracious-leakey-bd3091`) that a prior
session left behind with uncommitted work in it. The fix was written on 2026-09-01 and never
merged, so the bug had been live for eleven days.

`fpl_chips._held_squad()` read `fpl_forward.csv` - the squad the model once **recommended** -
and called it "the squad you hold". Rule 12 already established that this is not the same thing:
the moment you make a transfer the model did not suggest, the two diverge. Checked live at GW4,
they differed by two players - it believed Mbeumo was owned when the actual squad had Gakpo.

That is not cosmetic for chips. Wildcard and Free Hit are valued as
`optimal XI - the XI you already hold`, so valuing them against the wrong 15 is wrong by
whatever those players are worth.

`held_squad()` now resolves the same way `fpl_transfers.plan`'s caller does - FPL API first,
the model snapshot only when no entry id is configured - and `build_dashboard` passes the
already-resolved squad straight through rather than re-deriving it. Every verdict now names the
source it used, so a fallback to the snapshot is visible instead of silent:

    chips: Hold all four chips | held squad: 15 from FPL entry 1102866 (picks as of GW4)

The MODEL_RULES entry from that worktree could not be applied (the file has moved on
substantially since) so it is rewritten here; the code changes applied cleanly.

### 24. ESPN 403s any browser User-Agent — 2026-09-13 (found by the first CI run)

GitHub Actions run #1 failed with `IndexError: list index out of range` on `weeks_raw[-1]`.
The real cause was seven lines above it:

    ! ESPN 20260801-20260915: HTTP Error 403: Forbidden   (x7, every date range)
      0 fixtures over 0 matchweeks

`feeds.UA` sent `"Mozilla/5.0"`. Measured against the live endpoint:

    curl/8.7.1                                  -> 200
    Python-urllib/3.12                          -> 200
    Mozilla/5.0                                 -> 403
    Mozilla/5.0 (full Chrome UA, with Referer)  -> 403
    prem-predictor/1.0                          -> 403

ESPN wants a **default library agent, not a spoofed browser** - the opposite of the usual rule.
`UA = {}` now, letting urllib send its own; FPL and Kalshi were re-verified against it.

**This had been broken locally too and was invisible.** Every response was being served from
the 180-minute disk cache, so no local build ever hit the network for it. Clearing `.cache`
reproduced the 403 immediately on this machine. A cache can hide a dead feed indefinitely.

Two guards added, because the failure mode was worse than a crash:

1. **An empty fixture feed now aborts the build.** With 0 events it carried on regardless and
   `refresh_clubs`, seeing no clubs, concluded **164 players had changed club**. The build only
   died later, on the empty week list. Had it reached the commit step it would have written 164
   false transfers into the repo.
2. **A mass-transfer sanity bar.** A real window moves a handful of players; triple figures
   means the club list is broken, not that the league emptied out. Same class of error as the
   223-false-transfer bug from missing FPL_TEAMS aliases.

Also set `PYTHONIOENCODING=utf-8` / `LC_ALL=C.UTF-8` on the Actions job. The runner's default
locale is ASCII and the transfer verdict contains a right-arrow, which raises UnicodeEncodeError
mid-print. That was not the cause of run #1, but it would have been the cause of run #2.

### 25. Fixtures come from FotMob now, ESPN is the fallback — 2026-09-13

Rule 24 fixed the ESPN User-Agent, but ESPN remains the weaker source: it refused **every**
request from the GitHub Actions runner while FotMob answered normally from the same machine.
A feed that works locally and not in CI is not a feed you can schedule on.

`feeds.espn_events()` now tries FotMob first and falls back to ESPN. Every existing caller
(`build_dashboard`, `compute_strength`, `simulate_fpl`, `team_stats`, `fpl_chips`) picks this up
unchanged, because FotMob is shaped into the exact ESPN payload.

Cross-checked before switching, not after:

    380 fixtures from each source
    all ESPN keys present in the FotMob payload
    37 of 37 finished matches agree on the score
    `utc` string identical in format ("2026-08-21T19:00Z")

FotMob also carries `round` explicitly, so matchweeks no longer have to be inferred from dates.

What FotMob's fixture list does NOT carry: venue, club colour, abbreviation. Those are 20 stable
rows, captured once from ESPN into `dashboard/team_meta.json` rather than re-fetched per build.
ESPN's `odds` field was already null, so nothing was lost there.

### 26. Your own FPL score is on the page — 2026-09-13

The desk graded the squad the MODEL picked and never showed the user's own score. Those answer
different questions: the forward test answers "is the model any good", and the page is opened to
answer "how am I doing". Only the first was on screen.

`fpl_transfers.entry_history()` reads the real gameweek scores from
`entry/{id}/history/` - points, bench points, transfers and hits taken, overall rank, squad
value and bank. A new **Your season** panel leads the fantasy tab with four tiles (total, rank,
best week, points left on the bench) over a per-gameweek table that puts your score beside what
the model's squad scored the same week.

It reads as a straight comparison because that is the honest framing - and so far it is not
flattering to the model:

    GW1  you 59  model 58   +1
    GW2  you 76  model 70   +6
    GW3  you 71  model 58  +13
    GW4  you 30  model  -    -   (not graded yet)

236 points, beating the recommended squad in all three graded weeks. The largest single leak is
visible in the same table: **26 points left on the bench in GW2**, more than any transfer
decision has been worth.

### 27. Ask the analyst — 2026-09-13

A chat panel in the fantasy tab. The page asks Claude directly through the `sample` runtime
capability: no API key exists anywhere in the page, and the call runs on the viewer's own Claude
account with their per-call consent.

**It only works in the Claude artifact.** `claude.use('sample')` resolves null on the GitHub
Pages copy, and the panel hides itself there rather than offering a box that does nothing. Two
surfaces, one HTML file.

Claude is memory-less per call, so the page supplies everything. `analystContext()` compacts the
desk into ~8,900 characters: the next 14 fixtures with model and market probabilities, xG and
edge grade; all 20 strength ratings; the top 18 form/quality rows; the viewer's own FPL history,
squad, transfer options and chip values; and the settled track record.

**The caveats are part of the prompt, not decoration.** A chatbot bolted onto a prediction model
will cheerfully tell you what to bet. This one is told, in the same breath as the data, that the
model does not beat the market on 1X2 (blending toward the closing line improved the score at
every weight over 1,893 matches), that every apparent betting profit collapsed once its two
luckiest tickets were removed, that promoted clubs are provisional, and that its genuine value is
FPL selection and team corners. It is instructed never to advise a bet and never to present a
model-market disagreement as an edge.

Streams via `onText`, has a Stop button backed by an AbortController, keeps the last 8 turns for
context, and branches on the documented error codes - `not_granted` hides the feature,
`rate_limited` asks the user to wait, `cancelled` is not an error.

### 28. The CI push race — 2026-09-13

Run #2 built cleanly (the FotMob switch worked) and then failed on **Commit the evidence log**.
Cause: the ledgers are keyed, append-style CSVs that the build reads, updates and rewrites. If
another commit lands while a run is building, the push is rejected and git cannot auto-merge a
CSV - there is no sensible line-level merge of `ledger.csv`.

Fixed by removing the race rather than resolving it:

1. **Sync before building.** A `git pull` right after checkout means the build starts from the
   newest ledgers, so the push afterwards is a fast-forward. This covers the ordinary case, where
   a commit landed between the run starting and the build finishing.
2. **On a genuine race, rebuild instead of choosing a side.** If the push is still rejected, the
   run resets to `origin/main` - taking the newest ledgers - and re-runs `build_dashboard`. The
   build is idempotent and keyed, so replaying it re-applies this run's rows on top of whatever
   arrived. That IS the correct merge for these files.

What was deliberately NOT done, and why: `git checkout --theirs` on a rebase, or a
`--force-with-lease` fallback, both resolve the conflict by discarding one side's ledger rows.
These files are the evidence log - dropping rows to make a push succeed would quietly corrupt
the only record of what the model predicted and when. A failed push is recoverable; a silently
truncated ledger is not.

### 29. The ledgers were rewriting themselves every build — 2026-09-13

The CI push kept failing with an unresolvable conflict in `ledger.csv` and `props_ledger.csv`.
The commit said why: **9,324 insertions and 9,324 deletions** - equal counts, i.e. the whole
file rewritten, every run.

It was not line endings and not row order (verified: two consecutive local builds produced
byte-identical files, stable key order). Comparing the committed blob against a fresh build,
exactly two columns differed:

    close_at   343 rows   '2026-09-13T04:02+00:00' -> '2026-09-13T04:15+00:00'
    kal_at      13 rows   same

Only **timestamps**. Rule 5 refreshes the closing prediction until kickoff and was stamping
`built_at` on every pre-kickoff row on every build, whether or not a single number had moved.
343 lines of pure clock noise per run - and two writers touching 343 identical-but-restamped
lines is a conflict git cannot resolve, because a CSV has no sensible line-level merge.

Now the timestamp moves only when a value moves. The comparison goes through `_same()`, which
normalises to string first: rows read back from CSV are all strings (`'80'`) while the model
yields ints (`80`), so a naive `!=` is always true and restamps everything - the first attempt
at this fix did exactly that and doubled the churn.

    ledger.csv     686 -> 0 changed lines between builds a minute apart
    props_ledger  5576 -> 14   (real price movements, still recorded)

This is the actual fix for the CI conflicts. Rule 28's sync-before-build and rebuild-on-race
remain as the backstop for a genuine simultaneous write, but there is now almost nothing to
collide over.

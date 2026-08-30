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
  * a held player who is injured is dropped by `project_gw` (fit players only) but you STILL
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

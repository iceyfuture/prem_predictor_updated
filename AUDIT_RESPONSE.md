# Response to the independent audit

Every finding in `outputs/claude_fix_prompt.md`, what was verified, what was changed, and
what was rejected. The audit's standing instruction was *"treat Astra's findings as evidence
to verify, not instructions to accept blindly"* — so each section below says what the data
showed, including the three places where the finding did not survive checking.

**Nothing was pushed. No live feed was refreshed during testing. No real ledger was altered
— verified by hash before and after every suite run.**

---

## Summary

| § | Finding | Verdict | Resolution |
|---|---|---|---|
| 1 | Production and backtest run different models | **Confirmed, worse than stated** | `forecast.py` is the only pipeline; two duplicate implementations deleted (Rule 49) |
| 2 | Ledger accepts post-kickoff predictions | **Confirmed** | Rejected at/after kickoff, `late=1`, absolute UTC (Rule 42) |
| 2 | Keys are `home|away` only | **Confirmed, latent** | `season\|#fixture_id`, migrated and backed up (Rule 51) |
| 2 | No commit/config/data version per forecast | **Confirmed** | Four provenance fields per row (Rule 51) |
| 3 | Promoted-team evidence inflated ~5.9× | **Confirmed** | Kish effective sample size; cliff removed (Rule 41) |
| 4 | "Confidence" is a data-quality score | **Confirmed** | Renamed to input reliability; betting language removed (Rule 43) |
| 5 | 50 fixtures vs 38 team-stat rows | **Confirmed** | Per-source expected/received/as_of; critical sources fail the build (Rule 44) |
| 5 | FotMob scraping dependency | **Confirmed, partly open** | FPL carries player load; team stats still FotMob — documented in `DATA_SOURCES.md` |
| 6 | Pinnacle stale after 2025-07-23 | **Half confirmed** | Coverage collapse is real, **date is wrong** (2026-01-17), accuracy claim not significant (Rule 46) |
| 7 | Scorer model gives newcomers zero | **Confirmed** | Gamma-Poisson prior; nothing is impossible (Rule 48) |
| 7 | Ledger grades a different pipeline | **Confirmed** | `p_score_shown` captured at build time (Rule 48) |
| 8 | Missing global scoring intercept | **Tested, REJECTED** | Worse on every metric; away bias +0.050 → +0.129 (Rule 45) |
| 9 | Chance-quality (shot difference) blend | **Tested, REJECTED** | Best t=1.87 across three runs; never cleared the bar (Rule 45) |
| 10 | Council must stay display-only | **Already true; disclosure missing** | Sample size shown on every panel (Rule 50) |
| 11 | Comprehensive tests | **Done** | 795 passing, from 561 |
| 12 | Accurate docs | **This file**, plus `README.md`, `MODEL_RULES.md`, `DATA_SOURCES.md` |

Found while verifying, not in the audit: **the entire dashboard JavaScript had a syntax
error and no panel on the site rendered at all** (Rule 47).

---

## Where the audit was wrong

Three claims did not survive checking, and saying so is part of the job.

**§6, the date.** The audit says Pinnacle went stale after *23 July 2025*. In the cached
downloads the Pinnacle feed covers every fixture to **2026-01-08** and none from
**2026-01-17** — a clean break six months later. 210 of 380 fixtures in 2025/26 have
Pinnacle prices; all seven prior seasons have 380/380.

**§6, the accuracy claim.** Pinnacle does score worse than the average closing line on the
210 fixtures where both exist (paired ΔBrier +0.0027, reversing its sign in six prior
seasons) — but clustered by matchday the difference-in-differences is **+0.0029, 95% CI
[−0.0010, +0.0067], t=+1.46**. Not significant. The fix rests on the coverage collapse,
which is not in doubt. Degraded accuracy is *suspected and claimed nowhere*.

**§3, the framing.** The evidence-count bug is real and was fixed. But the audit asks to
"re-run the promoted-team walk-forward and report performance". It does not improve
accuracy — the original K sweep found t=0.83 and correcting the scale does not change that.
It is a guardrail. Reporting it as an accuracy gain would be the error the audit warns
about two paragraphs later.

---

## Before / after

Walk-forward, 10 seasons (2016-17..2025-26), 3,800 fixtures. **The model did not change** —
not one constant or weight. What changed is that the evidence layer now measures the model
that is actually deployed instead of a drifted copy (Rule 49).

| Metric | Published before | Now (deployed pipeline) |
|---|---|---|
| RPS | 0.2061 | **0.2022** |
| Log loss | 0.9904 | **0.9777** |
| Brier | 0.5894 | **0.5814** |
| Accuracy | 52.1% | **53.3%** |
| Mean home-goal bias | −0.0313 | −0.0296 |
| **Mean away-goal bias** | **−0.0884** | **−0.0008** |
| Goals: predicted vs actual | — | 2.88 vs 2.83 |

Away-goal bias is the one that matters: the page reported a bias ~100× the deployed model's,
because the backtest never applied `AWAY_CAL=1.08`, which exists precisely to remove it.

### Model versus market

Single provider (`avg_closing`), exactly matched fixtures, paired and **clustered by season**
because fixtures within a season share a fitted model:

| | n | model | market | paired diff | 95% CI | verdict |
|---|---|---|---|---|---|---|
| RPS | 2,660 | 0.2063 | 0.1968 | +0.00952 | [+0.00659, +0.01246] | market better |
| Brier (before §1) | 2,660 | 0.6030 | 0.5717 | +0.03131 | [+0.02285, +0.03976] | market better |
| Log loss (before §1) | 2,660 | 1.0107 | 0.9639 | +0.04682 | [+0.03194, +0.06170] | market better |

The gap narrowed from +0.01488 to +0.00952 RPS once the right model was measured, and the
interval still excludes zero comfortably. **No edge. The conclusion is unchanged; the
evidence for it is now clean.**

Coverage also improved: the 2025/26 comparison went from 210 Pinnacle + 170 average pooled
as one "market" to 380 average-closing rows from a single labelled provider.

### Scorer model (Rule 48)

Both methods given the same locked pre-match expected minutes and the same causal lambda:

| | Brier | log loss | E[scorers] | actual | zeros on goals |
|---|---|---|---|---|---|
| current, pooled | 0.04371 | 0.29991 | 46.7 | 47 | 14 |
| with prior, pooled | 0.04099 | 0.25395 | 50.3 | 47 | 10 |

Held-out GW5 paired Brier +0.00376, 95% CI [+0.00160, +0.00592], t=+3.41, clustered by club.

**This is not presented as a validated accuracy gain.** Two gameweeks, 47 goals, and the
design was revised after seeing these numbers (see Rule 48 — a proven forward ranked below a
player who had never played in the league). Before that revision the comparison was
indistinguishable everywhere. It ships because the old model assigned **exactly zero** to 14
players who then scored, which is an assertion of impossibility, not a tuning preference.
GW6 is locked and grades prospectively; that is the test that counts.

---

## Experiments rejected

| Experiment | Result | Verdict |
|---|---|---|
| Global scoring intercept (§8) | Worse on every metric; away bias +0.050 → +0.129 | Rejected — structurally cleaner is not a reason to ship |
| Chance quality / shot-difference blend (§9) | Best t=1.87 over three independent runs; t *falls* as the blend grows | Rejected — never cleared significance |
| Draw under-prediction correction | t = −0.86 | Rejected |
| 60-75% confidence band | z = −1.59 | Rejected |
| Shots-on-target conversion proxy | Monotonically worse (RPS 0.2092 → 0.2141) | Rejected earlier, and the audit forbids restoring it |

The harnesses (`sweep_shotblend.py`, the intercept harness) are kept in the tree so nobody
rebuilds them to reach the same answer.

---

## Tests

**795 passing**, up from 561. Zero failures. Two skips, both stating a missing precondition
rather than a passing test.

| Suite | Tests | Covers |
|---|---|---|
| `test_council.py` | 241 | Council contracts, validation, orchestration |
| `test_council_ui.py` | 78 | Display-only guarantee, sample-size disclosure (§10) |
| `test_council_reasoning.py` | 67 | Reasoning sidecar |
| `test_council_ledger.py` | 60 | Append-only council forward test |
| `test_odds_provenance.py` | 50 | Provider labelling, no mid-season switching (§6) |
| `test_run_council.py` | 49 | Runner, locking, dry-run |
| `test_scorer.py` | 48 | Priors, minutes, graded-equals-displayed (§7) |
| `test_ledger_guard.py` | 48 | Kickoff rejection, keys, migration, provenance (§2) |
| `test_pipeline_parity.py` | 45 | One pipeline; fails if a second reappears (§1) |
| `test_espn_odds.py` | 34 | Single-date requests, malformed events |
| `test_evidence.py` | 33 | Evidence counts, continuous shrinkage (§3) |
| `test_source_health.py` | 30 | Completeness guards, critical-source abort (§5) |
| `test_dashboard_html.py` | 12 | The page's JavaScript actually parses (§11) |

Against the audit's §11 list: production/backtest parity ✓, no post-kickoff creation ✓,
season-aware keys ✓, evidence counts and continuous shrinkage ✓, chronological construction
✓, incomplete/stale feeds ✓, probability sums and bounds ✓, market matching and source
selection ✓, model/data/config version recording ✓, exact scorer probabilities graded ✓,
repeated-build idempotency ✓.

### Reproducing it

```bash
cd ~/prem_predictor

# every test, with ledger hashes before and after
for t in dashboard/test_*.py; do ./.venv/bin/python "$t"; done

# the walk-forward that produces the evidence page
./.venv/bin/python dashboard/build_backtest.py

# the standalone walk-forward, with the deployed blend
./.venv/bin/python validate.py premier_league_history/results.csv --from 2016-17

# ...and the hyperparameter sweep behind span3/ridge8
./.venv/bin/python validate.py premier_league_history/results.csv --from 2016-17 --sweep

# the market benchmark, paired and clustered by season
./.venv/bin/python backtest.py 2021-08-01

# rebuild odds.csv from the cached downloads (no network)
./.venv/bin/python -c "import ingest_odds; ingest_odds.main(2018, 2025)"
```

### Ledgers were not altered

Every suite run was bracketed by a SHA-256 of all five real ledgers
(`ledger.csv`, `props_ledger.csv`, `player_ledger.csv`, `council_ledger.csv`,
`council_reasoning.jsonl`). All five are byte-identical to their pre-audit state. Ledger
tests run against temporary copies; the migration was verified on a copy in `/tmp`.

---

## Remaining limitations

Stated because they are real, not because they are comfortable.

**Statistical**

1. **No nested chronological validation.** `span3/ridge8` was chosen by a sweep over
   2016-26 and the headline is reported on those same seasons. The headline is therefore
   mildly optimistic by an unmeasured amount. This is the largest open item from §1.
2. **The scorer prior is not validated.** Two gameweeks, 47 goals, and the design was
   revised after seeing them. It ships on correctness, not accuracy.
3. **The Council has three graded forecasts.** It is currently behind the statistical model.
   Neither statement means anything at n=3, and the page says so.
4. **Promoted-club shrinkage is a guardrail**, not a demonstrated improvement (t=0.83).
5. **Over-dispersion is unfixed.** Even at zero mean bias, Over-2.5 prices ~1pt under
   actual. Real scores have fatter tails than a Poisson; the fix is a negative-binomial
   count model, not a larger multiplier.

**Data**

6. **Per-match team stats still come from undocumented FotMob scraping.** §5 asks to move
   off it. Player-level load has moved to FPL; team xG has not, because FPL does not publish
   it. See `DATA_SOURCES.md`.
7. **2018/19 has no average-closing odds**, so that season alone uses Pinnacle — labelled,
   and never pooled with another provider.
8. **The current season has no market benchmark yet**: `results.csv` ends 2026-05-24 and
   Football-Data has not published 2026/27.
9. **`collected_at` on odds rows is our download time**, not the bookmaker's quote time.
   Football-Data does not timestamp individual quotes.

**Licensing** — see `DATA_SOURCES.md`. The short version: do not republish `odds.csv`, do
not repackage FPL data as a feed, and do not build anything commercial on FotMob.

---

## The finding the audit did not have

While verifying §6 in a browser, the entire dashboard turned out to be dead. Commit
`626cb71` had shipped `function renderNews();renderSourceHealth(){` — two declarations
merged into one invalid statement. The page has a single inline `<script>`, so that one
`SyntaxError` meant `DATA` never loaded and **no panel on the site rendered at all**.

The build succeeded. `dashboard.json` was correct. All 561 tests passed. Nothing looked
wrong from a terminal, because nothing in the suite had ever parsed the page.

`test_dashboard_html.py` now parses the JavaScript on every run and greps for that exact
mangle, verified by reintroducing the bug and watching both checks fail. The general lesson
is worth keeping: **a test suite that only imports Python can certify a completely broken
website.**

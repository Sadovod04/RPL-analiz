# RPL-analiz — report

## What it does

Given a Russian academy footballer's career up to a cutoff age (default 19), the
project estimates the probability that they will **ever reach ≥ 200 minutes in the
Russian Premier League**. It is a **candidate-ranking** tool with calibrated
probabilities — a shortlist of prospects, not a verdict on any one player.

Three views of the same question (SPEC §3):

| target | definition | used by |
|---|---|---|
| **binary** | `1` ≥ 200 RPL min ever · `0` below and age ≥ 26 · else *censored* | CatBoost, logreg |
| **ordinal** | none / lower leagues / RPL | auxiliary signal |
| **survival** | age at RPL debut, right-censored | Cox PH — P(breakthrough by any age) |

## Data

Planned core sources were **Transfermarkt** (career stats) and **youfl.ru** (the
official youth league). What actually works from a non-RU network:

| source | status | role |
|---|---|---|
| **`tmapi.transfermarkt.technology`** | ✅ open JSON, no WAF | **primary** — per-game minutes/goals/assists/age by competition (incl. `RUJL` youth league), master data, market value |
| **www.transfermarkt.com** (`kader`) | ⚠️ AWS WAF → headless Chromium | historical squad pages, for player-id discovery only |
| **ru.wikipedia** (ЮФЛ season articles) | ✅ | youth-league participants (academy universe) + standings |
| **youfl.ru** | ❌ TLS-rejected off RU IPs | deferred to M1b |
| РФС / Sofascore / FBref | ❌ 403 / low ROI | adapter skeletons only |

Entity resolution merges a player across sources by birth-year block + fuzzy name
match (Cyrillic transliterated), union-find → stable `player_id`.

**Volume:** the full `run_ingest` crawl (`--academy-seasons 2013-2026 --fast
--build`) discovers ~4000 player ids from ЮФЛ/academy `kader` pages across all
divisions and seasons, then pulls each via tmapi → **3982 players** in
`data/processed/features.parquet` (git-ignored). A 140-player `features_demo.parquet`
(`scripts/demo_dataset.py`, current RPL squads, no Playwright) is the quick
pipeline check. The crawl is resumable (checkpoint + skip-already-ingested).

## Method (the part that makes it defensible)

- **Strict time cutoff** — every season row goes through one tested function
  (`features/time_cutoff.py`) before it becomes a feature; nothing observed at/after
  the cutoff age can enter.
- **Temporal split** — train on cohorts resolved before `split.test_cohort_from`,
  test after. Imitates real use.
- **GroupKFold by `player_id`** for tuning — a player never straddles folds.
- **Leakage guard** (`eval/leakage_check.py`) — a test that fails if a post-outcome
  column (`target`, `rpl_minutes_*`, `current_age`, `reached_pro`, …) reaches the
  matrix. Run before every `fit`.
- **Time-aware `academy_conversion_rate`** — P(breakthrough) for an academy is
  computed from *earlier* cohorts only.
- **Imbalance** — PR-AUC / Recall@Top-K / Brier, class weights; never accuracy.
- **Censored players** are dropped for the binary model, kept for survival.

## Models

| model | file | notes |
|---|---|---|
| naive scout | `models/baseline.py` | rank by market value at cutoff — the bar to beat |
| logistic regression | `models/baseline.py` | simple youth features + position, class-weighted |
| **CatBoost** | `models/gbm.py` | native categoricals (`academy_club`, `position`), Optuna over GroupKFold, SHAP importance + interactions, MLflow |
| Cox PH | `models/survival.py` | P(breakthrough by age); RSF optional via `--extra survival` |

## Results

Full academy crawl: **3982 players**, 521 academies, birth years ~1995–2010.
Resolved outcomes — RPL: 542 made it / 1001 did not (2439 still open); pro
(RPL/FNL/FNL-2): 1591 / 493 (1898 open). Temporal split: train born < 1999,
test born 1999–2003.

**`pro_target` (reached RPL / FNL / FNL-2), test n=770, base rate 0.81**

| model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| **CatBoost** | **0.968** | 0.879 | 0.129 |
| logistic regression | 0.959 | 0.853 | 0.151 |
| naive scout (market value) | 0.801 | 0.468 | — |

CatBoost CV PR-AUC (GroupKFold by player): 0.86–0.92.

**`target` (RPL, ≥200 min), base rate ~0.2** — logreg PR-AUC **0.687** vs naive
scout 0.300; ROC-AUC 0.833.

The model clears the market-value baseline on both targets. **Top SHAP features:**
`best_level_pre_cutoff` (highest level reached young), `position_detail`,
`played_youth_league`, `height_cm`, `youth_ga_per90` / `ga_per90_U19`,
`youth_seasons`, `minutes_U19`. i.e. *how high a level they reached young, how much
they played, and how productive they were*.

### Phase A — trajectory & cohort features

Eight features added off the already-crawled DB (no re-scrape,
`scripts/build_features.py`), 24 → 32 model inputs:

- `birth_quarter`, `rel_age_frac` — relative age within the Jan-1 age group. The
  academy intake shows a textbook relative-age effect (Q1 37.6% of resolved
  players vs Q4 14.7%); among those who then reach the RPL it flattens
  (33 / 27 / 23 / 17), i.e. a late-born player who survives selection is a
  slightly *better* bet — weak signal, right sign.
- `min_/mean_age_gap_vs_peers` — season age minus the mean age of that
  league-season. Negative = playing above their age group. Enters SHAP top-8.
- `matches_share_min/_mean` — matches played vs the fullest season in that league
  (availability proxy).
- `minutes_dropoff_max`, `had_minutes_collapse` — biggest season-over-season fall
  in minutes. Intended as an injury proxy; in practice it correlates *positively*
  with success (pro +0.31, rpl +0.27) — it reads as "was pushed up a level early"
  more than "got hurt". Kept, with that caveat.
- `cohort_year` — era control (legionnaire-limit / 2022 changes). Dominates SHAP
  on the RPL target (≈0.86); in the temporal split test cohorts sit outside the
  train range so it can only extrapolate a trend, but whether it reads "less time
  elapsed" legitimately vs games the split is the first thing for the Phase 3
  honest-eval to probe.

Circularity check: all eight correlate weakly with `market_value_at_cutoff_eur`
(max |0.32|) — not a repackaged scout signal.

**Before → after (temporal split, CatBoost, 12 Optuna trials):**

| target | metric | before (24f) | after (32f) |
|---|---|---|---|
| **`target`** (RPL ≥200 min, base 30%) | PR-AUC | 0.688 | **0.750** |
| | ROC-AUC | 0.839 | 0.867 |
| | Brier | 0.161 | 0.135 |
| | CV PR-AUC (GroupKFold) | 0.67 | 0.80 |
| **`pro_target`** (base 86%) | PR-AUC | 0.972 | 0.977 |
| | ROC-AUC | 0.852 | 0.875 |
| | Brier | 0.136 | 0.127 |

Real gain on the hard RPL target; `pro_target` was already near-saturated. The
logreg baseline is left on the original feature list — the new inputs hurt it
(raw `cohort_year` scale, collinear gaps), and its job is to be a stable bar.

### Phase B — "recognition" (ru.wikipedia)

`ingest/sources/wikipedia_players.py` + `scripts/ingest_wikipedia.py`: for each
resolved player, find a ru.wikipedia footballer article and extract signals that
only count **before** the cutoff age:

- `wiki_article_pre_cutoff` — an article was first created before the player
  turned 19. "Has an article now" is post-hoc and never becomes a feature; the
  raw `article_created_age` / `wiki_title` / `honours_years` are blocked by
  `eval/leakage_check.py`.
- `wiki_youth_national_team` — a Russia youth-NT category at **U19 or below**
  (молодёжная / U21 is excluded — that cap can happen at 20-21).
- `wiki_youth_honours` — honours in the "Достижения" section dated at/before 18.
- `pre_cutoff_recognition_score = 3·article + 2·youth_NT + 1·honours`,
  `recognition_count`, `any_recognition` — the umbrella aggregate B3/B5 will
  extend later.

Name matching: our `name_home_country` carries a patronymic (Transfermarkt,
sometimes wrong — e.g. our "Пиняев Сергей *Андреевич*" vs the article "Пиняев,
Сергей *Максимович*"), so search/match is on **surname + given name only**,
disambiguated by birth year (intro text or "Родившиеся в YYYY году" category) and
the "Футболисты…" category.

Status: adapter built and unit-tested (`tests/test_wikipedia_players.py`);
feature wiring + leakage guards + tests in place. The 3982-player crawl runs at
`--rate 1.3` (ru.wikipedia 429s the dev IP under bursts); resumable via
`scripts/ingest_wikipedia.py` (skips done players). Metrics land here after the
crawl + retrain.

### Phase C — costlier trajectory features

Added off the DB (no new scrape), 32 → 42 model inputs:

- `first_senior_age` / `played_senior_pre_cutoff` — age at the first *pre-cutoff*
  season in a senior pro league (RPL / FNL / FNL-2). ~32% of players reach men's
  football before the cutoff.
- `min_per_appearance` — pre-cutoff minutes / appearances (starter-vs-sub proxy).
- `starter_share` — fraction of pre-cutoff seasons averaging >60'/match.

**No measurable gain** (CatBoost, 12 trials): RPL `target` PR-AUC 0.750 → 0.744
(within noise), `pro_target` 0.977 → 0.978. The "reached men's football young"
signal is already carried by `best_level_pre_cutoff` and the age-gap features.
Kept — cheap, interpretable, no harm — but not pulling weight yet.

Still open in Phase C: academy-to-academy transfers at 15–16 (needs `academy_club`
cleanup — it is `formerClubsNote` free text), `marktwertverlauf` for a historical
market-value series.

### Phase 3 — honest evaluation

`scripts/run_honest_eval.py` (RPL `target`, temporal split, CatBoost default
params, features = Phase A + C; re-run after the Phase B crawl):

**Recall@Top-20, per test cohort year** — the pooled number is misleading:

| cohort | n | positives | model | scout |
|---|---|---|---|---|
| 1999 | 173 | 27 | **0.63** | 0.07 |
| 2000 | 181 | 26 | 0.15 | 0.12 |
| 2001–2003 | 77 | 77 | — | — |

2001–2003 have *no negatives* (every resolved player is a success — the
`test_cohort_to` cap is not tight enough), so recall there is meaningless. Only
**1999–2000** are real test years: the model crushes the market-value scout on
1999 (0.63 vs 0.07) and barely beats it on 2000 (0.15 vs 0.12). n = 2 usable
cohorts — this *is* the "tiny, noisy" limitation, made visible.

**Calibration:** ECE 0.068, Brier 0.141 — roughly diagonal, some wobble at the
extremes. Acceptable for the sample size.

**Bootstrap 90% CI (test n = 431, 130 positives):**

| | mean | 90% CI |
|---|---|---|
| PR-AUC model | 0.70 | [0.63, 0.78] |
| PR-AUC scout | 0.30 | [0.26, 0.34] |
| ROC-AUC model | 0.85 | [0.82, 0.88] |

Wide, but the model and scout CIs do not overlap — the lift is real.

**`cohort_year` probe:** refit without it → PR-AUC 0.70 → 0.66, ROC 0.85 → 0.82,
but Recall@20 0.115 → 0.138 (it *hurts* top-K). Permutation importance ≈ **0.00**
— shuffling `cohort_year` does nothing to PR-AUC even though it is SHAP #1, i.e.
it is a proxy the model reconstructs from correlated features, not an independent
lever, and it is **not** gaming the temporal split. Verdict: keep it (small
honest PR-AUC/ROC lift), but it is not the star SHAP implies.

Caveats: the pro-target test window skews positive (most academy kids who reach the
1999–2003 cohort got at least FNL-2 minutes); `academy_club` is still free-text
from Transfermarkt's `formerClubsNote`, so that feature is weak.

## Honest limitations (SPEC §14)

- **Tiny, noisy target.** Even with all Russian academies, breakthroughs are a few
  hundred players historically. Injuries, form, and coach decisions are not in the
  numbers. This is ranking a stream, not calling one career.
- **Sub-U15 data barely exists** — features at ages 11–14 will be sparse; real
  signal starts ~14–16.
- **`academy_club` from `formerClubsNote`** is free text — for the conversion-rate
  feature to work, the real run must key academy off the discovery club, not the note.
- **Market value at cutoff age** is mostly missing (tmapi master gives only
  recent points); a full time series needs the `marktwertverlauf` endpoint.
- **youfl.ru** unreachable here — youth-league coverage currently leans on tmapi's
  `RUJL` rows and Wikipedia standings.

## Reproduce

```bash
uv sync && uv run playwright install chromium
docker compose up -d                       # Postgres for raw data
uv run pytest                              # ~77 pass, 1 skip (RSF extra)

# full academy crawl (~1 h, resumable) -> data/processed/features.parquet
uv run python -m ingest.run_ingest --academy-seasons 2013-2026 --fast --build

# rebuild features from the existing DB only (no re-scrape) — for feature iteration
uv run python scripts/build_features.py

# recognition signal (Phase B): ru.wikipedia bios -> wiki_recognition table
uv run python scripts/ingest_wikipedia.py --rate 1.3      # resumable; raise --rate on 429

# models on whatever parquet is present (real if built, else demo)
uv run python scripts/run_baseline.py --target pro_target
uv run python scripts/run_gbm.py --target pro_target --trials 25
uv run python scripts/run_honest_eval.py --target target  # Phase 3: per-year recall, calibration, CIs
uv run python scripts/run_survival.py

# bilingual dashboard
uv run --extra app streamlit run app/streamlit_dashboard.py

# quick pipeline check without Playwright (~4 min):
uv run python scripts/demo_dataset.py 140
```

## Next

Development plan is phased:

- **Phase A — trajectory & cohort features.** ✅ done.
- **Phase B — "recognition".** ✅ code + ru.wikipedia adapter; ⏳ 3982-player crawl
  running. TM talent tags dropped (tmapi has no such field). Later sources: RPL
  best-young-player of the month, ЮФЛ team-of-the-round (youthleague.ru is
  geo-blocked here; VK needs a token), RFS youth call-ups (`ingest/sources/rfs.py`
  skeleton).
- **Phase C — costlier features.** ✅ `first_senior_age`, `played_senior_pre_cutoff`,
  `min_per_appearance`, `starter_share`. ⏳ academy-to-academy transfers
  (`academy_club` cleanup), `marktwertverlauf` historical market value.
- **Phase 3 — honest eval.** ✅ `scripts/run_honest_eval.py` — per-year
  Recall@Top-20, calibration/ECE, bootstrap CIs, `cohort_year` probe. Finding:
  only 2 test cohorts (1999–2000) have a real candidate pool; `test_cohort_to`
  needs tightening; `cohort_year` is a proxy, not a leak.
- Re-run the eval + `run_gbm` after the Phase B crawl; fill Phase B metrics.
- Full `run_ingest` with `kader` discovery → real training set with negatives.

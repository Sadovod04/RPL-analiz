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

**Volume in this repo:** a 140-player convenience sample (`features_demo.parquet`,
current RPL squads via tmapi) — enough to exercise the whole pipeline end to end.
It is **heavily skewed to `target = 1`** (current top-flight players); it is **not**
a training set. A real dataset needs `run_ingest` with `kader` discovery over
resolved 1990–2004 cohorts, which supplies the negatives.

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

## Results (pipeline check only)

Real metrics are **pending a full dataset**. On the skewed demo sample every model
scores PR-AUC ≈ 1.0 (there are almost no negatives) — not informative. What *is*
visible and points the right way:

- **CatBoost SHAP** top features: `youth_minutes_trend`, `minutes_U19`,
  `minutes_U17`, `youth_minutes_total`, `best_level_pre_cutoff` — i.e. *how much
  and how fast the youth role grew*, and *how high a level they reached young*.
- **Cox** concordance ≈ 0.80 (inflated by the sample); P(breakthrough by age) rises
  monotonically 21 → 23 → 25 as it should.

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
uv run pytest                              # 70 pass, 1 skip (RSF extra)

# demo pipeline (no Playwright, ~4 min):
uv run python scripts/demo_dataset.py 140
uv run python scripts/run_baseline.py
uv run python scripts/run_gbm.py
uv run python scripts/run_survival.py
uv run --extra app streamlit run app/streamlit_dashboard.py

# real dataset:
uv run python -m ingest.run_ingest --sources wikipedia transfermarkt \
    --seasons 2010-2022 --limit 400
```

## Next

- Full `run_ingest` with `kader` discovery → real training set with negatives.
- `marktwertverlauf` endpoint for historical market value.
- Wire M1b sources (РФС youth caps) when run from an unblocked network.
- Recall@Top-20 against a properly defined per-year candidate pool.

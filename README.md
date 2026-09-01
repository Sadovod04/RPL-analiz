# RPL-analiz

**RU.** Модель, которая по ранней карьерной статистике футболиста из академии
России (11–21 год) оценивает вероятность когда-либо закрепиться в РПЛ. Это
**ранжирование потока кандидатов** с калибровкой вероятностей и оценкой
неопределённости — «топ-N перспективных», а не вердикт по одному игроку. Проект
портфельный. Полное ТЗ — [`SPEC.md`](SPEC.md).

**EN.** A model that, from a Russian academy footballer's early career stats
(ages 11–21), estimates the probability of ever establishing themselves in the
Russian Premier League (RPL). It is a **candidate-ranking** model with calibrated
probabilities and honest uncertainty — a top-N shortlist, not a verdict on one
player. Portfolio project. Full spec: [`SPEC.md`](SPEC.md).

---

## Быстрый старт / Quick start

```bash
# 1. окружение (Python 3.12, pinned) / environment
uv sync
uv run playwright install chromium        # for TM squad (kader) discovery

# 2. Postgres (сырьё / raw data)
docker compose up -d
cp .env.example .env

# 3. тесты / tests   (storage tests need Postgres up; else skipped)
uv run pytest

# 4. линт / lint
uv run ruff check .

# 5. сбор данных / ingest  (M1a)
uv run python -m ingest.run_ingest --sources wikipedia transfermarkt \
    --seasons 2015-2024 --limit 50
```

> **Docker в своём терминале:** после первого запуска нужен `newgrp docker` или
> перелогин (группа `docker` добавлена, но сессия старее).
> **youfl.ru** недоступен из не-RU сети — ЮФЛ-контекст берётся из Wikipedia +
> Transfermarkt `RUJL`; см. `SPEC.md` §5.

`make` targets: `install`, `db-up`, `db-down`, `test`, `lint`, `fmt`.

## Структура / Layout

| Path | Что / What |
|---|---|
| `config/settings.toml` | пороги таргета (200 мин / 26 лет), когорты (1990–2004), cutoff-возраст (11), пути |
| `ingest/` | скрейперы (Transfermarkt, ЮФЛ, …), rate limiter, pydantic-схемы, resolver, запись в Postgres |
| `features/` | `time_cutoff.py` (единый анти-leakage фильтр), `build_features.py`, `labels.py` |
| `models/` | `baseline.py` (логрег + наивный скаут), `gbm.py` (CatBoost), `survival.py` (Cox/RSF) |
| `eval/` | `metrics.py` (PR-AUC, Brier, Recall@TopK), `leakage_check.py`, `shap_analysis.py` |
| `notebooks/` | `01_eda.ipynb`, `02_model_report.ipynb` |
| `app/` | Streamlit-дашборд (M6) |

## Вехи / Milestones

`M0` скаффолд · `M1a` ядро сбора (TM + ЮФЛ + resolver) · `M1b` обогащение ·
`M2` фичи + метки + EDA · `M3` baseline · `M4` CatBoost + SHAP ·
`M5` survival · `M6` отчёт + дашборд. **v1 = M0→M4.**

## Данные и этика / Data & ethics

Сырой скрейпленный датасет **не** коммитится и **не** публикуется (`.gitignore`:
`data/`). В репозитории — только код сбора и агрегированные фичи. Скрейпинг
идёт с rate limiting и backoff. Не production-сервис. См. `SPEC.md` §5.

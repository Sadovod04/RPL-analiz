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

# 5. сбор данных / ingest
uv run python -m ingest.run_ingest --fast --build          # академии: Transfermarkt + Wikipedia
uv run python scripts/refresh_youth.py                     # регионы: ФФ СПб + Клубная лига Москвы

# 6. дашборд / dashboard
uv run --extra app streamlit run app/streamlit_dashboard.py
```

> **Docker в своём терминале:** после первого запуска нужен `newgrp docker` или
> перелогин (группа `docker` добавлена, но сессия старее).
> **youfl.ru** недоступен из не-RU сети — ЮФЛ-контекст берётся из Wikipedia +
> Transfermarkt `RUJL`; см. `SPEC.md` §5.

`make` targets: `install`, `db-up`, `db-down`, `test`, `lint`, `fmt`.

## Источники данных / Data sources

| Источник | Кто | Как |
|---|---|---|
| **Transfermarkt** (`tmapi`) | игроки академий, годы рождения ≤ 2010, с карьерой | открытый JSON API, `ingest.run_ingest` |
| **Wikipedia + TM `RUJL`** | контекст ЮФЛ | best-effort, см. `SPEC.md` §5 |
| **ФФ СПб** (`stat.ffspb.org`, платформа «Наградион») | дети СПб 2012–2015 г.р. | `scripts/ingest_ffspb.py` |
| **Клубная лига Москвы** (`mosff.ru`) | дети Москвы 2012–2014 г.р. | `scripts/ingest_mosff.py` (один JSON-эндпоинт) |

Академии оцениваются обученной моделью (CatBoost). У детей региональных
федераций карьеры ещё нет — им считается **прозрачная эвристика 0–100**
(голы за игру, объём игр, сила клуба), проранжированная внутри каждого источника.
Обе группы попадают в один датасет и одни и те же таблицы дашборда; колонка
«Метод» и фильтр «Источник» их разделяют.

Другие регионы добавляются одним адаптером — merge в дашборд автоматический
(`app.youth_features` подхватывает каждый `data/processed/<source>_players.parquet`).

## Дашборд / Dashboard

`uv run --extra app streamlit run app/streamlit_dashboard.py` — двуязычный
(RU/EN), 4 вкладки:

- **Перспективные** — общий ранжированный список (академии + дети регионов)
- **Уже в РПЛ / проф.** — игроки с решённой судьбой + их fitted-оценка
- **Сравнение** — 2–6 игроков бок о бок, метрики полосками
- **Юные (регионы)** — дети ФФ СПб + Москвы, фильтр по региону / году, прогноз
  уровня (РПЛ / ФНЛ / ФНЛ-2 / низкий), мини-сравнение

## Структура / Layout

| Path | Что / What |
|---|---|
| `config/settings.toml` | пороги таргета (200 мин / 26 лет), когорты (1990–2004), cutoff-возраст (11), пути |
| `ingest/` | скрейперы (`sources/`: transfermarkt, wikipedia, ffspb, mosff), rate limiter, pydantic-схемы, resolver, запись в Postgres |
| `features/` | `time_cutoff.py` (единый анти-leakage фильтр), `build_features.py` (в т.ч. траекторные/когортные фичи: relative age, «молод для лиги», провалы минут, `cohort_year`), `labels.py` |
| `models/` | `baseline.py` (логрег + наивный скаут), `gbm.py` (CatBoost), `survival.py` (Cox/RSF) |
| `eval/` | `metrics.py` (PR-AUC, Brier, Recall@TopK), `leakage_check.py`, `shap_analysis.py` |
| `app/` | Streamlit-дашборд, `ranking.py` (логика), `youth_features.py` (merge региональных пулов), `i18n.py` |
| `scripts/` | `ingest_ffspb.py`, `ingest_mosff.py`, `refresh_youth.py`, `build_features.py` (rebuild parquet from DB, no re-scrape), `run_*.py`, `demo_dataset.py` |
| `notebooks/` | `01_eda.ipynb`, `02_model_report.ipynb` |

## Вехи / Milestones

`M0` скаффолд · `M1a` ядро сбора (TM + ЮФЛ + resolver) · `M1b` обогащение ·
`M2` фичи + метки + EDA · `M3` baseline · `M4` CatBoost + SHAP ·
`M5` survival · `M6` отчёт + дашборд · `M7` региональные детские пулы (ФФ СПб,
Москва) в общем датасете. **v1 = M0→M4.**

## Данные и этика / Data & ethics

Сырой скрейпленный датасет **не** коммитится и **не** публикуется
(`.gitignore`: `data/`). Это касается и детских пулов: `ffspb_players.parquet` /
`mosff_players.parquet` содержат ФИО, отчества и даты рождения несовершеннолетних
из открытой статистики региональных федераций — они остаются **только локально**,
собираются скриптами при клоне репозитория. В git — только код сбора и, при
необходимости, агрегированные фичи взрослых игроков. Скрейпинг идёт с rate
limiting и backoff. Не production-сервис. См. `SPEC.md` §5.

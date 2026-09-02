"""Phase B / recognition: ru.wikipedia footballer bios for every resolved player.

    uv run python scripts/ingest_wikipedia.py [--limit N] [--refresh]

Idempotent and resumable: players already in ``wiki_recognition`` are skipped
unless ``--refresh``. Writes the raw match to ``raw_document`` (source=wikipedia,
doc_type=player_bio) and the typed row to ``wiki_recognition``. Rebuild features
afterwards with ``scripts/build_features.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import text

from ingest.db import get_engine
from ingest.rate_limiter import RateLimiter
from ingest.sources.wikipedia_players import WikiPlayerBios
from ingest.storage import store_raw, store_wiki_recognition
from ingest.tables import init_db


def _todo(engine, limit: int | None, refresh: bool) -> pd.DataFrame:
    done = set()
    if not refresh:
        with engine.connect() as c:
            done = {r[0] for r in c.execute(text("select player_id from wiki_recognition"))}
    with engine.connect() as c:
        players = pd.read_sql(
            text(
                "select player_id, canonical_name, name_home_country, birth_date from player"
            ),
            c,
        )
    todo = players[~players["player_id"].isin(done)]
    return todo.head(limit) if limit else todo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--refresh", action="store_true", help="re-check players already done")
    ap.add_argument(
        "--rate",
        type=float,
        default=1.3,
        help="min seconds between MediaWiki API calls (raise if you hit HTTP 429)",
    )
    args = ap.parse_args(argv)

    engine = get_engine()
    init_db(engine)
    todo = _todo(engine, args.limit, args.refresh)
    print(f"wikipedia bios: {len(todo)} player(s) to check  (rate={args.rate}s)")

    bios = WikiPlayerBios(rate_limiter=RateLimiter(min_interval=args.rate, jitter=args.rate / 2))
    hits = errors = 0
    for i, r in enumerate(todo.itertuples(index=False), 1):
        bd = r.birth_date if pd.notna(r.birth_date) else None
        by = bd.year if bd else None
        try:
            bio = bios.lookup(r.player_id, r.name_home_country, r.canonical_name, bd)
        except Exception as exc:  # noqa: BLE001 — one bad player must not kill an unattended run
            errors += 1
            print(f"  ! {r.canonical_name}: {type(exc).__name__}: {exc}  (will retry next run)")
            continue
        store_raw(
            engine,
            source="wikipedia",
            doc_type="player_bio",
            source_id=r.player_id,
            payload={
                "query_name": r.name_home_country or r.canonical_name,
                "birth_year": by,
                "wiki_title": bio.wiki_title,
                "match_score": bio.match_score,
                "article_created": bio.article_created.isoformat() if bio.article_created else None,
                "article_created_age": bio.article_created_age,
                "youth_honours_count": bio.youth_honours_count,
                "nt_youth_levels": bio.nt_youth_levels,
                "honours_years": bio.honours_years,
            },
            collection_season="static",
            url=(
                f"https://ru.wikipedia.org/wiki/{bio.wiki_title.replace(' ', '_')}"
                if bio.wiki_title
                else None
            ),
        )
        store_wiki_recognition(
            engine,
            [
                {
                    "player_id": bio.player_id,
                    "wiki_title": bio.wiki_title,
                    "match_score": bio.match_score,
                    "article_created": bio.article_created,
                    "article_created_age": bio.article_created_age,
                    "youth_honours_count": bio.youth_honours_count,
                    "nt_youth_levels": bio.nt_youth_levels,
                    "honours_years": bio.honours_years,
                }
            ],
        )
        if bio.wiki_title:
            hits += 1
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}  matched: {hits}  errors: {errors}")

    print(
        f"done. {hits}/{len(todo)} matched to a ru.wikipedia article; "
        f"{errors} errors (retry next run)"
    )


if __name__ == "__main__":
    main()

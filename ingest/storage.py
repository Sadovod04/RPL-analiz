"""Persistence helpers over :mod:`ingest.tables`.

Upserts are idempotent so a re-run of ``run_ingest`` refreshes rather than
duplicates. Postgres path uses ``ON CONFLICT``; the generic path (SQLite tests)
falls back to delete+insert.
"""

from __future__ import annotations

import json
from datetime import date

from sqlalchemy import delete, insert
from sqlalchemy.engine import Engine

from ingest import tables
from ingest.schemas import MarketValuePoint, Player, SeasonStats


def _pg_upsert(engine: Engine, table, rows: list[dict], index_elements: list[str]) -> None:
    if not rows:
        return
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = pg_insert(table).values(rows)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in table.columns
        if c.name not in index_elements and c.name != "id"
    }
    stmt = stmt.on_conflict_do_update(index_elements=index_elements, set_=update_cols)
    with engine.begin() as conn:
        conn.execute(stmt)


def _generic_upsert(engine: Engine, table, rows: list[dict], key_cols: list[str]) -> None:
    if not rows:
        return
    with engine.begin() as conn:
        for row in rows:
            conn.execute(delete(table).where(*[table.c[k] == row[k] for k in key_cols]))
        conn.execute(insert(table), rows)


def upsert(engine: Engine, table, rows: list[dict], key_cols: list[str]) -> None:
    if engine.dialect.name == "postgresql":
        _pg_upsert(engine, table, rows, key_cols)
    else:
        _generic_upsert(engine, table, rows, key_cols)


# --- raw landing zone --------------------------------------------------
def store_raw(
    engine: Engine,
    *,
    source: str,
    doc_type: str,
    source_id: str,
    payload,
    collection_season: str,
    url: str | None = None,
) -> None:
    row = {
        "source": source,
        "doc_type": doc_type,
        "source_id": str(source_id),
        "url": url,
        "payload": payload
        if isinstance(payload, (dict, list))
        else json.loads(json.dumps(payload)),
        "collection_season": collection_season,
    }
    upsert(
        engine, tables.raw_document, [row], ["source", "doc_type", "source_id", "collection_season"]
    )


# --- typed tables ----------------------------------------------------
def store_player(engine: Engine, player_id: str, p: Player) -> None:
    row = {
        "player_id": player_id,
        "canonical_name": p.full_name,
        "name_home_country": p.name_home_country,
        "birth_date": p.birth_date,
        "position": p.position.value if p.position else None,
        "position_detail": p.position_detail,
        "nationality": p.nationality,
        "is_foreigner": p.is_foreigner,
        "height_cm": p.height_cm,
        "academy_club": p.academy_club or (p.youth_clubs[0] if p.youth_clubs else None),
    }
    upsert(engine, tables.player, [row], ["player_id"])


def link_source(
    engine: Engine, player_id: str, source: str, source_id: str, match_score: float | None = None
) -> None:
    upsert(
        engine,
        tables.player_source_xref,
        [
            {
                "player_id": player_id,
                "source": source,
                "source_id": str(source_id),
                "match_score": match_score,
            }
        ],
        ["source", "source_id"],
    )


def store_seasons(engine: Engine, player_id: str, seasons: list[SeasonStats]) -> None:
    rows = [
        {
            "player_id": player_id,
            "source": s.source,
            "season": s.season or "",
            "club": s.club,
            "league": s.league,
            "age_at_season": s.age_at_season,
            "minutes": s.minutes,
            "matches": s.matches,
            "goals": s.goals,
            "assists": s.assists,
            "is_rpl": s.is_rpl,
        }
        for s in seasons
    ]
    upsert(engine, tables.season_stats, rows, ["player_id", "source", "season", "league", "club"])


def store_market_values(engine: Engine, player_id: str, points: list[MarketValuePoint]) -> None:
    rows = [
        {
            "player_id": player_id,
            "source": mv.source,
            "date": mv.date or date(1900, 1, 1),
            "value_eur": mv.value_eur,
            "club": mv.club,
            "age": mv.age,
        }
        for mv in points
        if mv.date is not None
    ]
    upsert(engine, tables.market_value, rows, ["player_id", "source", "date"])


def store_wiki_standings(engine: Engine, season: str, standings: list[dict], division: str = "1"):
    rows = [
        {
            "season": season,
            "division": division,
            "club": s["club"],
            "played": s.get("played"),
            "wins": s.get("wins"),
            "draws": s.get("draws"),
            "losses": s.get("losses"),
            "goals_for": s.get("goals_for"),
            "goals_against": s.get("goals_against"),
            "points": s.get("points"),
        }
        for s in standings
    ]
    upsert(engine, tables.wiki_standings, rows, ["season", "division", "club"])

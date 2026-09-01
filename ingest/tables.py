"""Raw + resolved schema (SQLAlchemy Core).

Kept deliberately small — a few thousand players. ``raw_document`` is the
append/replace landing zone (one row per fetched page/payload, JSONB); typed
tables below are populated from it so features can be rebuilt without re-scraping
(SPEC §6). ``init_db`` creates everything; no Alembic for a project this size.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

# JSONB on Postgres, plain JSON elsewhere (e.g. SQLite in unit tests)
_JSON = JSON().with_variant(JSONB(), "postgresql")

raw_document = Table(
    "raw_document",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("source", String(32), nullable=False),
    Column(
        "doc_type", String(32), nullable=False
    ),  # profile | performance | market_value | wiki_season
    Column("source_id", String(64), nullable=False),  # TM player id, wiki title, ...
    Column("url", String(512)),
    Column("payload", _JSON, nullable=False),
    Column("collection_season", String(16), nullable=False),  # partition key, e.g. "2024-25"
    Column("fetched_at", DateTime(timezone=True), server_default=func.now()),
    UniqueConstraint("source", "doc_type", "source_id", "collection_season", name="uq_raw_doc"),
)

player = Table(
    "player",
    metadata,
    Column("player_id", String(40), primary_key=True),  # stable hash, see resolve.py
    Column("canonical_name", String(200), nullable=False),
    Column("name_home_country", String(200)),
    Column("birth_date", Date),
    Column("position", String(16)),
    Column("nationality", String(64)),
    Column("is_foreigner", Boolean),
    Column("height_cm", Integer),
    Column("academy_club", String(120)),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

player_source_xref = Table(
    "player_source_xref",
    metadata,
    Column("source", String(32), primary_key=True),
    Column("source_id", String(64), primary_key=True),
    Column("player_id", String(40), nullable=False, index=True),
    Column("match_score", Float),
)

season_stats = Table(
    "season_stats",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", String(40), index=True),
    Column("source", String(32), nullable=False),
    Column("season", String(16), nullable=False),
    Column("club", String(120)),
    Column("league", String(120)),
    Column("minutes", Integer),
    Column("matches", Integer),
    Column("goals", Integer),
    Column("assists", Integer),
    Column("is_rpl", Boolean, server_default="false"),
    UniqueConstraint("player_id", "source", "season", "league", "club", name="uq_season_row"),
)

market_value = Table(
    "market_value",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", String(40), index=True),
    Column("source", String(32), nullable=False),
    Column("date", Date),
    Column("value_eur", Float),
    Column("club", String(120)),
    Column("age", Float),
    UniqueConstraint("player_id", "source", "date", name="uq_mv_point"),
)

wiki_standings = Table(
    "wiki_standings",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("season", String(16), nullable=False),
    Column("division", String(16), server_default="1"),
    Column("club", String(120), nullable=False),
    Column("played", Integer),
    Column("wins", Integer),
    Column("draws", Integer),
    Column("losses", Integer),
    Column("goals_for", Integer),
    Column("goals_against", Integer),
    Column("points", Integer),
    UniqueConstraint("season", "division", "club", name="uq_wiki_row"),
)


def init_db(engine) -> None:
    metadata.create_all(engine)


def drop_all(engine) -> None:
    metadata.drop_all(engine)

"""Storage round-trip against the live docker-compose Postgres (skipped if down)."""

from datetime import date

import pytest
from sqlalchemy import select

from ingest import storage, tables
from ingest.schemas import MarketValuePoint, Player, Position, SeasonStats


@pytest.fixture
def clean_db(db_engine):
    tables.drop_all(db_engine)
    tables.init_db(db_engine)
    yield db_engine
    tables.drop_all(db_engine)


def test_init_creates_all_tables(clean_db):
    from sqlalchemy import inspect

    names = set(inspect(clean_db).get_table_names())
    assert {
        "raw_document",
        "player",
        "player_source_xref",
        "season_stats",
        "market_value",
        "wiki_standings",
    } <= names


def test_raw_upsert_is_idempotent(clean_db):
    for _ in range(2):
        storage.store_raw(
            clean_db,
            source="transfermarkt",
            doc_type="profile",
            source_id="362570",
            payload={"a": 1},
            collection_season="2025-26",
        )
    with clean_db.connect() as c:
        rows = c.execute(select(tables.raw_document)).fetchall()
    assert len(rows) == 1


def test_player_and_children_round_trip(clean_db):
    p = Player(
        source="transfermarkt",
        source_id="362570",
        full_name="Andrey Mostovoy",
        birth_date=date(1997, 11, 5),
        position=Position.W,
        height_cm=180,
    )
    storage.store_player(clean_db, "pid123", p)
    storage.link_source(clean_db, "pid123", "transfermarkt", "362570", 100.0)
    storage.store_seasons(
        clean_db,
        "pid123",
        [
            SeasonStats(
                source="transfermarkt",
                source_player_id="362570",
                season="19/20",
                league="Premier Liga",
                minutes=900,
                matches=20,
                is_rpl=True,
            ),
        ],
    )
    storage.store_market_values(
        clean_db,
        "pid123",
        [
            MarketValuePoint(
                source="transfermarkt",
                source_player_id="362570",
                date=date(2020, 1, 1),
                value_eur=1_000_000,
            ),
        ],
    )
    with clean_db.connect() as c:
        assert c.execute(select(tables.player.c.canonical_name)).scalar() == "Andrey Mostovoy"
        assert c.execute(select(tables.season_stats.c.minutes)).scalar() == 900
        assert c.execute(select(tables.market_value.c.value_eur)).scalar() == 1_000_000
        assert c.execute(select(tables.player_source_xref.c.player_id)).scalar() == "pid123"


def test_season_upsert_updates_not_duplicates(clean_db):
    def row(minutes):
        return [
            SeasonStats(
                source="transfermarkt",
                source_player_id="1",
                season="19/20",
                league="Premier Liga",
                club="964",
                minutes=minutes,
                is_rpl=True,
            )
        ]

    storage.store_seasons(clean_db, "pid1", row(100))
    storage.store_seasons(clean_db, "pid1", row(950))
    with clean_db.connect() as c:
        vals = c.execute(select(tables.season_stats.c.minutes)).fetchall()
    assert vals == [(950,)]

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def load_fixture(*parts: str):
    p = FIXTURES.joinpath(*parts)
    text = p.read_text(encoding="utf-8")
    return json.loads(text) if p.suffix == ".json" else text


@pytest.fixture
def fx():
    return load_fixture


@pytest.fixture(scope="session")
def db_engine():
    """Isolated ``rpl_test`` database on the docker-compose Postgres.

    NEVER the main ``rpl`` DB — storage tests drop_all on teardown, which would
    wipe a running ``run_ingest`` crawl. Created on first use, left in place.
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError

    from ingest.db import database_url

    admin_url = database_url().rsplit("/", 1)[0] + "/postgres"
    test_url = database_url().rsplit("/", 1)[0] + "/rpl_test"
    try:
        admin = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
        with admin.connect() as c:
            exists = c.execute(
                text("select 1 from pg_database where datname = 'rpl_test'")
            ).scalar()
            if not exists:
                c.execute(text("create database rpl_test"))
        admin.dispose()
    except OperationalError:
        pytest.skip("Postgres not reachable (run: docker compose up -d)")
    return create_engine(test_url, future=True)

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
    """Live Postgres from docker-compose; skip the test if it isn't up."""
    from sqlalchemy.exc import OperationalError

    from ingest.db import get_engine

    engine = get_engine()
    try:
        with engine.connect():
            pass
    except OperationalError:
        pytest.skip("Postgres not reachable (run: docker compose up -d)")
    return engine

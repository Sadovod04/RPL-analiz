"""Postgres connection helper. Env vars win over ``config/settings.toml``."""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine

from settings import load_settings


def database_url() -> str:
    db = load_settings()["db"]
    user = os.getenv("PGUSER", db["user"])
    pw = os.getenv("PGPASSWORD", db["password"])
    host = os.getenv("PGHOST", db["host"])
    port = os.getenv("PGPORT", db["port"])
    name = os.getenv("PGDATABASE", db["name"])
    return f"postgresql+psycopg://{user}:{pw}@{host}:{port}/{name}"


def get_engine(**kwargs) -> Engine:
    return create_engine(database_url(), future=True, **kwargs)

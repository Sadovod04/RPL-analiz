"""Phase C: Transfermarkt market-value *history* for every player.

    uv run python scripts/ingest_market_value.py [--limit N] [--rate S]

tmapi's ``/player/{id}/market-value-history`` returns dated value points from the
player's teens onward (the site's "Marktwertverlauf"). The base crawl only stored
recent points, so ``market_value_at_cutoff_eur`` is NaN for almost everyone;
this fills the pre-cutoff points. Idempotent per (player, source, date). Rebuild
features afterwards with ``scripts/build_features.py``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import text

from ingest.db import get_engine
from ingest.fetcher import TmApiClient
from ingest.rate_limiter import RateLimiter
from ingest.storage import store_raw, upsert
from ingest.tables import market_value


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _points(payload: dict, player_id: str) -> list[dict]:
    # tmapi sometimes repeats a "determined" date (placeholder 01-01); the unique
    # key is (player_id, source, date), so keep the last point per date.
    by_date: dict[date, dict] = {}
    for h in payload.get("data", {}).get("history", []) or []:
        mv = h.get("marketValue") or {}
        d = _parse_date(mv.get("determined"))
        if d is None:
            continue
        by_date[d] = {
            "player_id": player_id,
            "source": "transfermarkt",
            "date": d,
            "value_eur": float(mv.get("value") or 0.0),
            "club": str(h.get("clubId")) if h.get("clubId") else None,
            "age": float(h["age"]) if h.get("age") is not None else None,
        }
    return list(by_date.values())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rate", type=float, default=0.25)
    ap.add_argument("--refresh", action="store_true", help="re-fetch players already done")
    args = ap.parse_args(argv)

    engine = get_engine()
    with engine.connect() as c:
        xref = pd.read_sql(
            text(
                "select x.player_id, x.source_id from player_source_xref x "
                "where x.source = 'transfermarkt'"
            ),
            c,
        )
        if not args.refresh:
            done = {
                r[0]
                for r in c.execute(
                    text("select distinct player_id from market_value where age < 19")
                )
            }
            xref = xref[~xref["player_id"].isin(done)]
    if args.limit:
        xref = xref.head(args.limit)
    print(f"market-value history: {len(xref)} player(s) to fetch  (rate={args.rate}s)")

    tm = TmApiClient(rate_limiter=RateLimiter(min_interval=args.rate, jitter=args.rate))
    pts_total = errs = 0
    for i, r in enumerate(xref.itertuples(index=False), 1):
        try:
            payload = tm.market_value_history(r.source_id)
        except Exception as exc:  # noqa: BLE001 — one bad id must not kill the run
            errs += 1
            print(f"  ! {r.source_id}: {type(exc).__name__}")
            continue
        rows = _points(payload, r.player_id)
        if rows:
            upsert(engine, market_value, rows, ["player_id", "source", "date"])
            pts_total += len(rows)
        store_raw(
            engine,
            source="transfermarkt",
            doc_type="market_value_history",
            source_id=r.source_id,
            payload=payload.get("data", {}),
            collection_season="static",
        )
        if i % 200 == 0:
            print(f"  {i}/{len(xref)}  points: {pts_total}  errors: {errs}")

    print(f"done. {pts_total} value points written; {errs} errors")


if __name__ == "__main__":
    main()

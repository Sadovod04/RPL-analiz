"""Ingest orchestration: sources -> Postgres (raw + resolved).

    uv run python -m ingest.run_ingest --sources wikipedia transfermarkt \
        --seasons 2015-2023 --academies 964 23176 --limit 50

    uv run python -m ingest.run_ingest --capture      # save test fixtures, no DB

Flow:
  1. wikipedia  -> ЮФЛ standings per season  -> seeds the academy universe
  2. transfermarkt (headless Chromium) -> per academy club, squad pages per
     season -> player ids -> profile + performance + market value
  3. resolve.resolve_players() clusters records across sources -> player table

Transfermarkt needs ``uv run playwright install chromium`` once, and (from a
non-RU network) works where youfl.ru does not.
"""

from __future__ import annotations

import argparse
from datetime import date

from settings import load_settings

CORE_SOURCES = ("wikipedia", "transfermarkt")
ENRICHMENT_SOURCES = ("rfs", "regional_federations", "sofascore", "fbref")


def collection_season(today: date | None = None) -> str:
    d = today or date.today()
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def parse_seasons(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(CORE_SOURCES),
        choices=CORE_SOURCES + ENRICHMENT_SOURCES,
    )
    p.add_argument("--seasons", default="2015-2024", help="year range 'A-B' or CSV of start years")
    p.add_argument(
        "--academies",
        nargs="*",
        default=None,
        help="TM club ids; default = config [academies].extra + wiki participants",
    )
    p.add_argument("--limit", type=int, default=None, help="max players per academy (debug)")
    p.add_argument(
        "--capture", action="store_true", help="save fixtures to tests/fixtures, skip DB"
    )
    p.add_argument("--dry-run", action="store_true", help="plan only, no fetch / no DB")
    return p.parse_args(argv)


def _run_wikipedia(engine, years: list[int]) -> set[str]:
    from ingest.sources.wikipedia import WikipediaYouthLeague
    from ingest.storage import store_raw, store_wiki_standings

    src = WikipediaYouthLeague()
    universe: set[str] = set()
    for y in years:
        season = src.fetch_season(y)
        if not season:
            continue
        universe.update(season["participants"])
        if engine is not None:
            store_raw(
                engine,
                source="wikipedia",
                doc_type="wiki_season",
                source_id=season["season"],
                payload=season,
                collection_season=collection_season(),
            )
            store_wiki_standings(engine, season["season"], season["standings"])
    return universe


def _discover_player_ids(club_ids: list[str], years: list[int], limit: int | None) -> list[str]:
    """Historical squad (kader) pages are WAF-guarded -> use the browser here."""
    from ingest.fetcher import BrowserFetcher
    from ingest.sources.transfermarkt import kader_url, parse_kader_html

    found: list[str] = []
    with BrowserFetcher() as bf:
        for club_id in club_ids:
            for y in years:
                try:
                    html = bf.get(kader_url(club_id, y)).html
                except Exception:  # noqa: BLE001
                    continue
                for pid in parse_kader_html(html):
                    if pid not in found:
                        found.append(pid)
                if limit and len(found) >= limit:
                    return found[:limit]
    return found[:limit] if limit else found


def _run_transfermarkt(engine, player_ids: list[str]) -> list[dict]:
    from ingest.fetcher import TmApiClient
    from ingest.sources.transfermarkt import TransfermarktSource
    from ingest.storage import store_raw

    collected: list[dict] = []
    with TmApiClient() as api:
        src = TransfermarktSource(api)
        for pid in player_ids:
            try:
                rec = src.fetch_player(pid)
            except Exception as exc:  # noqa: BLE001 - skip a bad id, keep going
                print(f"  tm player {pid} failed: {exc}")
                continue
            player = rec["player"]
            collected.append(
                {
                    "source": "transfermarkt",
                    "source_id": player.source_id,
                    "full_name": player.full_name,
                    "name_home_country": player.name_home_country,
                    "birth_date": player.birth_date,
                    "_payload": rec,
                }
            )
            if engine is not None:
                store_raw(
                    engine,
                    source="transfermarkt",
                    doc_type="profile",
                    source_id=player.source_id,
                    payload=player.model_dump(mode="json"),
                    collection_season=collection_season(),
                    url=player.profile_url,
                )
    return collected


def _resolve_and_store(engine, records: list[dict]) -> pd.DataFrame:  # noqa: F821
    from ingest.resolve import resolve_players
    from ingest.storage import link_source, store_market_values, store_player, store_seasons

    resolved = resolve_players([{k: v for k, v in r.items() if k != "_payload"} for r in records])
    if engine is None:
        return resolved
    by_key = {(r["source"], str(r["source_id"])): r.get("_payload") for r in records}
    for row in resolved.itertuples():
        payload = by_key.get((row.source, str(row.source_id)))
        if payload is None:
            continue
        store_player(engine, row.player_id, payload["player"])
        link_source(engine, row.player_id, row.source, row.source_id, row.match_score)
        store_seasons(engine, row.player_id, payload.get("seasons", []))
        store_market_values(engine, row.player_id, payload.get("market_values", []))
    return resolved


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_settings()
    years = parse_seasons(args.seasons)

    if args.dry_run:
        print(
            f"[dry-run] sources={args.sources} seasons={years[0]}..{years[-1]} "
            f"academies={args.academies or 'auto'} limit={args.limit}"
        )
        return

    if args.capture:
        from ingest.fetcher import capture_fixtures

        with capture_fixtures("tests/fixtures/transfermarkt") as grab:
            grab("profile_sample.html", "https://www.transfermarkt.com/-/profil/spieler/362570")
        print("fixtures written to tests/fixtures/")
        return

    from ingest.db import get_engine
    from ingest.tables import init_db

    engine = get_engine()
    init_db(engine)

    universe: set[str] = set()
    if "wikipedia" in args.sources:
        universe = _run_wikipedia(engine, years)
        print(f"wikipedia: {len(universe)} academy names from ЮФЛ participants")

    records: list[dict] = []
    if "transfermarkt" in args.sources:
        from ingest.fetcher import TmApiClient
        from ingest.sources.transfermarkt import TransfermarktSource

        club_ids = list(args.academies or cfg["academies"].get("extra", []))
        if not club_ids:
            with TmApiClient() as api:
                club_ids = sorted(TransfermarktSource(api).academy_universe(years))
            print(f"transfermarkt: academy universe = {len(club_ids)} youth clubs from ЮФЛ tables")
        player_ids = _discover_player_ids(club_ids, years, args.limit)
        print(f"transfermarkt: discovered {len(player_ids)} player ids")
        records = _run_transfermarkt(engine, player_ids)
        print(f"transfermarkt: {len(records)} player records")

    if records:
        resolved = _resolve_and_store(engine, records)
        print(
            f"resolved: {resolved['player_id'].nunique()} unique players "
            f"from {len(records)} records"
        )


if __name__ == "__main__":
    main()

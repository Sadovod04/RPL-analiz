"""Ingest orchestration: sources -> Postgres (raw + resolved) -> features parquet.

    # full academy-wide crawl (long — hours), resumable, then build features:
    uv run python -m ingest.run_ingest --academy-seasons 2013-2026 --build

    # bounded test run:
    uv run python -m ingest.run_ingest --academies 964 232 --academy-seasons 2018-2020 --limit 60

    uv run python -m ingest.run_ingest --dry-run

Flow:
  1. wikipedia -> ЮФЛ standings per season (context) + academy names.
  2. academy universe = ЮФЛ (RUJL) youth-club ids per season (+ config extras +
     --academies). Historical `kader` pages (WAF) -> player ids via headless
     Chromium. Discovered ids are checkpointed to data/raw/discovered_ids.txt and
     players already in raw_document are skipped, so the crawl resumes.
  3. tmapi (no WAF) -> per player: master + per-game performance + market value.
  4. resolve -> player table; --build then writes data/processed/features.parquet.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from settings import ROOT, load_settings

CORE_SOURCES = ("wikipedia", "transfermarkt")
ENRICHMENT_SOURCES = ("rfs", "regional_federations", "sofascore", "fbref")
CHECKPOINT = ROOT / "data" / "raw" / "discovered_ids.txt"


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
    p.add_argument("--seasons", default="2015-2024", help="wikipedia ЮФЛ season range")
    p.add_argument(
        "--academy-seasons",
        default="2013-2026",
        help="season range for kader discovery of youth squads",
    )
    p.add_argument(
        "--academies",
        nargs="*",
        default=None,
        help="extra TM club ids (senior or youth) to crawl on top of the ЮФЛ universe",
    )
    p.add_argument("--limit", type=int, default=None, help="cap total players (debug)")
    p.add_argument(
        "--build",
        action="store_true",
        help="after ingest, write data/processed/features.parquet from the DB",
    )
    p.add_argument("--fresh", action="store_true", help="ignore the discovery checkpoint")
    p.add_argument(
        "--redo-discovery",
        action="store_true",
        help="re-crawl kader pages even if the checkpoint is already populated",
    )
    p.add_argument(
        "--fast", action="store_true", help="lighter tmapi rate limit (0.5s) for bulk collection"
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


# --- wikipedia -----------------------------------------------------
def _run_wikipedia(engine, years: list[int]) -> set[str]:
    from ingest.sources.wikipedia import WikipediaYouthLeague
    from ingest.storage import store_raw, store_wiki_standings

    src = WikipediaYouthLeague()
    names: set[str] = set()
    for y in years:
        season = src.fetch_season(y)
        if not season:
            continue
        names.update(season["participants"])
        store_raw(
            engine,
            source="wikipedia",
            doc_type="wiki_season",
            source_id=season["season"],
            payload=season,
            collection_season=collection_season(),
        )
        store_wiki_standings(engine, season["season"], season["standings"])
    return names


# --- discovery ---------------------------------------------------
def _academy_universe(years: list[int], extra: list[str]) -> list[str]:
    from ingest.fetcher import TmApiClient
    from ingest.sources.transfermarkt import TransfermarktSource

    clubs = set(extra)
    with TmApiClient() as api:
        for comp in ("RUJL", "RUJL2", "RUJL3"):
            clubs |= TransfermarktSource(api).academy_universe(years, competition_id=comp)
    return sorted(clubs)


def _load_checkpoint(fresh: bool) -> list[str]:
    if fresh or not CHECKPOINT.exists():
        return []
    return [line.strip() for line in CHECKPOINT.read_text().splitlines() if line.strip()]


def _save_checkpoint(ids: list[str]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(ids))


def _discover_player_ids(
    club_ids: list[str], years: list[int], *, fresh: bool, redo: bool, limit: int | None
) -> list[str]:
    from ingest.fetcher import BrowserFetcher
    from ingest.sources.transfermarkt import kader_url, parse_kader_html

    found = list(dict.fromkeys(_load_checkpoint(fresh)))
    if found and not redo:
        print(
            f"discovery: using checkpoint as-is ({len(found)} ids); "
            f"pass --redo-discovery to re-crawl"
        )
        return found[:limit] if limit else found
    seen = set(found)
    grid = [(c, y) for c in club_ids for y in years]
    print(f"discovery: {len(grid)} club-seasons, {len(found)} ids already checkpointed")
    with BrowserFetcher() as bf:
        for i, (club_id, y) in enumerate(grid, 1):
            try:
                ids = parse_kader_html(bf.get(kader_url(club_id, y)).html)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(grid)}] club {club_id} {y}: {exc}")
                continue
            new = [p for p in ids if p not in seen]
            seen.update(new)
            found.extend(new)
            if new:
                _save_checkpoint(found)
            if i % 20 == 0:
                print(f"  [{i}/{len(grid)}] +{len(new)} (total {len(found)})")
            if limit and len(found) >= limit:
                break
    return found[:limit] if limit else found


# --- tmapi fetch + store ---------------------------------------
def _already_ingested(engine) -> set[str]:
    q = "select source_id from raw_document where source='transfermarkt' and doc_type='profile'"
    return set(pd.read_sql(q, engine)["source_id"].astype(str))


def _run_transfermarkt(engine, player_ids: list[str], *, fast: bool = False) -> list[dict]:
    from ingest.fetcher import TmApiClient
    from ingest.rate_limiter import RateLimiter
    from ingest.sources.transfermarkt import TransfermarktSource
    from ingest.storage import store_raw

    done = _already_ingested(engine)
    todo = [p for p in player_ids if p not in done]
    print(f"tmapi: {len(todo)} players to fetch ({len(player_ids) - len(todo)} already ingested)")

    limiter = RateLimiter(min_interval=0.5, jitter=0.3) if fast else None
    collected: list[dict] = []
    with TmApiClient(rate_limiter=limiter) as api:
        src = TransfermarktSource(api)
        for i, pid in enumerate(todo, 1):
            try:
                rec = src.fetch_player(pid)
            except Exception as exc:  # noqa: BLE001
                print(f"  tm {pid}: {exc}")
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
            store_raw(
                engine,
                source="transfermarkt",
                doc_type="profile",
                source_id=player.source_id,
                payload=player.model_dump(mode="json"),
                collection_season=collection_season(),
                url=player.profile_url,
            )
            if i % 50 == 0:
                print(f"  [{i}/{len(todo)}] fetched")
    return collected


def _resolve_and_store(engine, records: list[dict]) -> pd.DataFrame:
    from ingest.resolve import resolve_players
    from ingest.storage import link_source, store_market_values, store_player, store_seasons

    resolved = resolve_players([{k: v for k, v in r.items() if k != "_payload"} for r in records])
    by_key = {(r["source"], str(r["source_id"])): r["_payload"] for r in records}
    for row in resolved.itertuples():
        payload = by_key.get((row.source, str(row.source_id)))
        if payload is None:
            continue
        store_player(engine, row.player_id, payload["player"])
        link_source(engine, row.player_id, row.source, row.source_id, row.match_score)
        store_seasons(engine, row.player_id, payload.get("seasons", []))
        store_market_values(engine, row.player_id, payload.get("market_values", []))
    return resolved


def _build_features(engine) -> None:
    from features.build_features import build_feature_matrix, feature_columns, from_db

    players, seasons, mvs = from_db(engine)
    m = build_feature_matrix(players, seasons, mvs)
    out = ROOT / load_settings()["paths"]["data_processed"] / "features.parquet"
    m.to_parquet(out, index=False)
    print(f"features: wrote {out}  shape={m.shape}  features={len(feature_columns(m))}")
    print(f"  target: {m['target'].value_counts(dropna=False).to_dict()}")
    if "pro_target" in m:
        print(f"  pro_target: {m['pro_target'].value_counts(dropna=False).to_dict()}")


# --- main ------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_settings()
    wiki_years = parse_seasons(args.seasons)
    academy_years = parse_seasons(args.academy_seasons)

    if args.dry_run:
        print(
            f"[dry-run] wiki={wiki_years[0]}..{wiki_years[-1]} "
            f"academy_seasons={academy_years[0]}..{academy_years[-1]} "
            f"extra_academies={args.academies or '[]'} limit={args.limit} build={args.build}"
        )
        return

    from ingest.db import get_engine
    from ingest.tables import init_db

    engine = get_engine()
    init_db(engine)

    if "wikipedia" in args.sources:
        print(f"wikipedia: {len(_run_wikipedia(engine, wiki_years))} ЮФЛ participant names")

    if "transfermarkt" in args.sources:
        have_checkpoint = CHECKPOINT.exists() and CHECKPOINT.read_text().strip() and not args.fresh
        if have_checkpoint and not args.redo_discovery:
            universe = []  # skip the RUJL universe fetch too
        else:
            extra = list(args.academies or cfg["academies"].get("extra", []))
            universe = _academy_universe(academy_years, extra)
            print(f"academy universe: {len(universe)} clubs")
        player_ids = _discover_player_ids(
            universe, academy_years, fresh=args.fresh, redo=args.redo_discovery, limit=args.limit
        )
        print(f"discovered {len(player_ids)} player ids")
        records = _run_transfermarkt(engine, player_ids, fast=args.fast)
        if records:
            resolved = _resolve_and_store(engine, records)
            print(f"resolved {resolved['player_id'].nunique()} players from {len(records)} records")

    if args.build:
        _build_features(engine)


if __name__ == "__main__":
    main()

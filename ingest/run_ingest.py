"""Ingest entrypoint: scrape sources -> raw tables in Postgres.

    uv run python -m ingest.run_ingest --sources transfermarkt youfl

Status: skeleton — wired up in M1a.
"""

from __future__ import annotations

import argparse

from settings import load_settings

CORE_SOURCES = ("transfermarkt", "youfl")
ENRICHMENT_SOURCES = ("rfs", "regional_federations", "sofascore", "fbref")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(CORE_SOURCES),
        choices=CORE_SOURCES + ENRICHMENT_SOURCES,
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _ = load_settings()
    raise NotImplementedError(f"M1a — would ingest: {args.sources}")


if __name__ == "__main__":
    main()

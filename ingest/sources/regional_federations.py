"""Regional federations (МРО) adapter (M1b) — ENRICHMENT, low priority.

Youth championships for U11–U14 age groups not covered by ЮФЛ. Highly
unstructured, gaps expected. Used to plug specific holes only.

Status: skeleton — implemented in M1b.
"""

from __future__ import annotations

from collections.abc import Iterator

from ingest.sources.base import Source


class RegionalFederations(Source):
    name = "regional_federations"

    def iter_academy_players(self, academy_ref: str) -> Iterator[str]:
        raise NotImplementedError("M1b")

    def fetch_player(self, player_ref: str) -> dict:
        raise NotImplementedError("M1b")

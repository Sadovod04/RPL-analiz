"""FBref adapter (M1b) — ENRICHMENT.

Advanced metrics (xG/xA, progressive passes/carries) for senior leagues, used to
enrich the adult-career side of a player's record.

Status: skeleton — implemented in M1b.
"""

from __future__ import annotations

from collections.abc import Iterator

from ingest.sources.base import Source


class Fbref(Source):
    name = "fbref"

    def iter_academy_players(self, academy_ref: str) -> Iterator[str]:
        raise NotImplementedError("M1b")

    def fetch_player(self, player_ref: str) -> dict:
        raise NotImplementedError("M1b")

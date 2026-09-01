"""Sofascore adapter (M1b) — ENRICHMENT.

Undocumented JSON API: player match ratings, match detail. Handle with care —
strict rate limiting, respect ToS, do not redistribute raw payloads.

Status: skeleton — implemented in M1b.
"""

from __future__ import annotations

from collections.abc import Iterator

from ingest.sources.base import Source


class Sofascore(Source):
    name = "sofascore"

    def iter_academy_players(self, academy_ref: str) -> Iterator[str]:
        raise NotImplementedError("M1b")

    def fetch_player(self, player_ref: str) -> dict:
        raise NotImplementedError("M1b")

"""Transfermarkt adapter (M1a) — CORE source.

Career-by-season history: minutes, goals/assists, transfers, market value (with
as-of date), DOB, position, height. Stable HTML, no official API. Rate-limited.

Status: skeleton — implemented in M1a.
"""

from __future__ import annotations

from collections.abc import Iterator

from ingest.sources.base import Source


class Transfermarkt(Source):
    name = "transfermarkt"

    def iter_academy_players(self, academy_ref: str) -> Iterator[str]:
        raise NotImplementedError("M1a")

    def fetch_player(self, player_ref: str) -> dict:
        raise NotImplementedError("M1a")

"""РФС adapter (rfs.ru) (M1b) — ENRICHMENT.

Youth national-team call-ups (U16–U19), СШ/СШОР championships.

Status: skeleton — implemented in M1b.
"""

from __future__ import annotations

from collections.abc import Iterator

from ingest.sources.base import Source


class Rfs(Source):
    name = "rfs"

    def iter_academy_players(self, academy_ref: str) -> Iterator[str]:
        raise NotImplementedError("M1b")

    def fetch_player(self, player_ref: str) -> dict:
        raise NotImplementedError("M1b")

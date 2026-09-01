"""ЮФЛ adapter (youfl.ru) (M1a) — CORE source.

Russian national youth league: ЮФЛ-1 (U17/U18), ЮФЛ-2 (U16), ЮФЛ-3 (U15).
Rosters, minutes, goals per round. Also seeds the academy universe
(``config.academies.seed_source = "youfl"``).

Status: skeleton — implemented in M1a.
"""

from __future__ import annotations

from collections.abc import Iterator

from ingest.sources.base import Source


class Youfl(Source):
    name = "youfl"

    def iter_academies(self) -> Iterator[str]:
        """Yield club refs that field a team in any ЮФЛ division."""
        raise NotImplementedError("M1a")

    def iter_academy_players(self, academy_ref: str) -> Iterator[str]:
        raise NotImplementedError("M1a")

    def fetch_player(self, player_ref: str) -> dict:
        raise NotImplementedError("M1a")

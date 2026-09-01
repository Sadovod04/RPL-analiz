"""РФС adapter (rfs.ru) (M1b) — ENRICHMENT.

Adds: youth national-team call-ups (U16–U19) — a coarse "on the federation's
radar" signal (SPEC §7). rfs.ru is reachable from CI (unlike youfl.ru).

Approach: rfs.ru publishes squad announcements per age team as news / roster
pages. Fetch with :class:`ingest.fetcher.HttpFetcher`, parse the player list
(name + club + DOB), emit :class:`ingest.schemas.NationalTeamCap` rows keyed by
age team + call-up date. Cross-source linking is done later by ``resolve``.

Not on the critical path (v1 = M0→M4); implement when the U-team feature is added.
"""

from __future__ import annotations

from ingest.fetcher import HttpFetcher
from ingest.schemas import NationalTeamCap

RFS_BASE = "https://www.rfs.ru"
YOUTH_TEAMS = ("U-21", "U-19", "U-18", "U-17", "U-16")


class RfsYouthCallups:
    name = "rfs"

    def __init__(self, fetcher: HttpFetcher | None = None):
        self._fetcher = fetcher or HttpFetcher()

    def iter_callups(self, team: str, season_year: int) -> list[NationalTeamCap]:
        raise NotImplementedError(
            "M1b: fetch rfs.ru youth squad pages, parse name/club/DOB -> NationalTeamCap"
        )

"""Московская федерация футбола — mosff.ru (youngest players Transfermarkt lacks).

Plain httpx, one JSON endpoint, no browser / no WAF:

    POST https://mosff.ru/api/tournament-stats/players
    body {"tournamentId": <int>, "minGames": 0, "page": <1..>}
    -> {"success": true, "data": {"count": <total>, "stats": [ {playerUrl, title,
        teamTitle, games, minutes, goalsSum, penalties, yellowCards, redCards, ...} ]}}

100 rows per page; page through until ``count`` is covered. Each *tournament* is a
single birth-year cohort ("Клубная лига 2013 г.р." -> every player is 2013-born),
so the birth year comes from the tournament, not the player page (mosff player
pages carry no DOB).
"""

from __future__ import annotations

import re

from ingest.fetcher import HttpFetcher

MOSFF_BASE = "https://mosff.ru"
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
_PLAYERS_API = "/api/tournament-stats/players"

# Клубная лига child tournaments -> birth year (from mosff.ru/tournament/1161 tabs)
CLUB_LEAGUE_TOURNAMENTS = {
    "2012": 1169,
    "2013": 1168,
    "2014": 1167,
    "2015": 1165,
    "2016": 1163,
}


def split_title(title: str) -> tuple[str, str | None]:
    """ "Фамилия Имя Отчество" -> ("Имя Фамилия", "Отчество").

    Matches the ФФ СПб convention (``full_name`` = given + family, patronymic apart).
    """
    parts = re.sub(r"\s+", " ", (title or "").strip()).split(" ")
    if len(parts) >= 3:
        family, given, patronymic = parts[0], parts[1], " ".join(parts[2:])
        return f"{given} {family}", patronymic
    if len(parts) == 2:
        family, given = parts
        return f"{given} {family}", None
    return title or "", None


def parse_player_row(row: dict, birth_year: int, tournament_id: int) -> dict:
    full_name, patronymic = split_title(row.get("title", ""))
    m = re.search(r"/player/(\d+)", row.get("playerUrl", "") or "")
    return {
        "mosff_id": m.group(1) if m else None,
        "full_name": full_name,
        "patronymic": patronymic,
        "birth_date": None,  # not exposed by mosff
        "birth_year": birth_year,
        "teams": row.get("teamTitle") or "",
        "tournament_id": tournament_id,
        "n_tournaments": 1,
        "games": int(row.get("games") or 0),
        "goals": int(row.get("goalsSum") or 0),
        "penalties": int(row.get("penalties") or 0),
        "minutes": int(row.get("minutes") or 0),
        "yellows": int(row.get("yellowCards") or 0),
        "reds": int(row.get("redCards") or 0),
        "hat_tricks": int(row.get("hatTricks") or 0),
        "source": "mosff",
    }


class MosffSource:
    name = "mosff"

    def __init__(self, fetcher: HttpFetcher | None = None):
        self._f = fetcher or HttpFetcher(user_agent=_UA)

    def _post(self, body: dict) -> dict:
        self._f.rate_limiter.wait()
        r = self._f._client.post(
            f"{MOSFF_BASE}{_PLAYERS_API}",
            json=body,
            headers={"accept": "application/json", "content-type": "application/json"},
        )
        r.raise_for_status()
        return r.json()

    def tournament_players(self, tournament_id: int, birth_year: int) -> list[dict]:
        """All players of one tournament (every page), one dict per player."""
        out: list[dict] = []
        page = 1
        while True:
            j = self._post({"tournamentId": tournament_id, "minGames": 0, "page": page})
            data = j.get("data") or {}
            batch = data.get("stats") or []
            out.extend(parse_player_row(r, birth_year, tournament_id) for r in batch)
            total = int(data.get("count") or 0)
            if len(batch) < 100 or len(out) >= total or page > 20:
                break
            page += 1
        return out

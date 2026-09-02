"""Wikipedia adapter (M1a) — CORE source, stand-in for youfl.ru.

youfl.ru (official ЮФЛ site) is geo-blocked from CI, so youth-league *context*
comes from ru.wikipedia season articles instead: the participant list seeds the
academy universe, and the standings table gives per-club season strength.

This is club-season granularity, not player minutes — player-level data comes
from Transfermarkt. Fetched through the MediaWiki API (stable, no bot wall).
"""

from __future__ import annotations

import re
from urllib.parse import quote

from bs4 import BeautifulSoup

from ingest.fetcher import HttpFetcher

WIKI_API = "https://ru.wikipedia.org/w/api.php"
# MediaWiki asks for a descriptive UA (tool; contact) or it throttles hard
WIKI_UA = "RPL-analiz/0.1 (research, non-commercial; contact: local)"

SEASON_TITLES = (
    "Юношеская футбольная лига {a}/{b}",
    "Юношеская футбольная лига — 1 {a}/{b}",
    "Юношеская футбольная лига-1 {a}/{b}",
)

# single-letter Cyrillic column codes of a standings table
_STANDINGS_COLS = {"и", "в", "н", "п", "о"}


def season_titles(start_year: int) -> list[str]:
    a, b = start_year, start_year + 1
    return [t.format(a=a, b=b) for t in SEASON_TITLES]


def _headers(table) -> list[str]:
    first = table.find("tr")
    return [c.get_text(" ", strip=True) for c in first.select("th")] if first else []


def _is_standings(headers: list[str]) -> bool:
    return {h.strip().lower() for h in headers} >= _STANDINGS_COLS


def _rows(table) -> list[dict[str, str]]:
    headers = _headers(table)
    out = []
    for tr in table.select("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        vals = [c.get_text(" ", strip=True) for c in cells]
        out.append(dict(zip(headers, vals, strict=False)) if headers else {"_": vals})
    return out


def _pick(row: dict, *keys: str) -> str | None:
    lower = {k.strip().lower(): v for k, v in row.items()}
    for k in keys:
        if k.strip().lower() in lower:
            return lower[k.strip().lower()]
    return None


def _to_int(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"-?\d+", text.replace("−", "-").replace("+", ""))
    return int(m.group()) if m else None


def _norm_club(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", name or "")).strip(" .*")


def parse_season_html(html: str, season: str) -> dict:
    """-> {season, participants: [club], standings: [{club, played, points, ...}]}."""
    soup = BeautifulSoup(html, "lxml")
    participants: list[str] = []
    standings: list[dict] = []

    for table in soup.select("table.wikitable"):
        headers = _headers(table)
        rows = _rows(table)
        if _is_standings(headers):
            for r in rows:
                club = _pick(r, "Команда", "Клуб")
                if not club:
                    continue
                standings.append(
                    {
                        "club": _norm_club(club),
                        "played": _to_int(_pick(r, "И")),
                        "wins": _to_int(_pick(r, "В")),
                        "draws": _to_int(_pick(r, "Н")),
                        "losses": _to_int(_pick(r, "П")),
                        "goals_for": _to_int(_pick(r, "МЗ")),
                        "goals_against": _to_int(_pick(r, "МП")),
                        "points": _to_int(_pick(r, "О")),
                    }
                )
        elif not participants and (
            {"команды", "команда", "клуб"} & {h.strip().lower() for h in headers}
        ):
            for r in rows:
                club = _pick(r, "Команды", "Команда", "Клуб") or next(iter(r.values()), None)
                if club and not str(club).isdigit():
                    participants.append(_norm_club(club))

    if not participants and standings:
        participants = [s["club"] for s in standings]

    return {"season": season, "participants": participants, "standings": standings}


class WikipediaYouthLeague:
    name = "wikipedia"

    def __init__(self, fetcher: HttpFetcher | None = None):
        self._fetcher = fetcher or HttpFetcher(user_agent=WIKI_UA)

    def fetch_season_html(self, title: str) -> str | None:
        url = (
            f"{WIKI_API}?action=parse&page={quote(title)}"
            "&prop=text&formatversion=2&format=json&redirects=1"
        )
        try:
            data = self._fetcher.get_json(url)
        except Exception:  # noqa: BLE001 - wikipedia is best-effort context, never fatal
            return None
        if "error" in data:
            return None
        return data["parse"]["text"]

    def fetch_season(self, start_year: int) -> dict | None:
        for title in season_titles(start_year):
            html = self.fetch_season_html(title)
            if html:
                return parse_season_html(html, season=f"{start_year}/{start_year + 1}")
        return None

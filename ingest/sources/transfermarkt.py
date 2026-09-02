"""Transfermarkt adapter (M1a) — CORE source.

Primary path is Transfermarkt's own JSON API (``tmapi.transfermarkt.technology``),
which — unlike ``www.transfermarkt.com`` — has no AWS WAF, so plain rate-limited
HTTP works (see :class:`ingest.fetcher.TmApiClient`):

  * ``/players?ids[]=`` — master data: DOB, position, height, foot, Cyrillic
    passport name, ``formerClubsNote`` (youth clubs), market value.
  * ``/player/{id}/performance-game`` — every career game with competition, season,
    minutes, goals, assists, age. Aggregated here to (season, competition) rows.
    Crucially this includes ``RUJL`` (Russian youth league) appearances, partly
    recovering what youfl.ru would have given.
  * ``/competition/RUJL/table?seasonId=`` — youth-league club ids per season →
    academy universe.

The browser (:class:`ingest.fetcher.BrowserFetcher`) is only needed to read
historical ``kader`` (squad-by-season) pages for roster discovery.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from ingest.schemas import MarketValuePoint, Player, Position, SeasonStats

TM_BASE = "https://www.transfermarkt.com"

RUSSIA_NATIONALITY_ID = 141
RPL_COMPETITION_IDS = {"RU1"}

# tmapi competitionId -> readable name (only the ones we reason about)
COMPETITION_NAMES = {
    "RU1": "Premier Liga",
    "RU2": "Pervaya Liga",
    "RU3": "Vtoraya Liga",
    "RUP": "Russian Cup",
    "RUSS": "Russian Super Cup",
    "RUJL": "Russian Youth League",
    "RPLW": "Premier Liga (relegation)",
    "CL": "UEFA Champions League",
    "EL": "UEFA Europa League",
}

_POSITION_MAP = {
    "goalkeeper": Position.GK,
    "centre-back": Position.CB,
    "left-back": Position.FB,
    "right-back": Position.FB,
    "defender": Position.CB,
    "defensive midfield": Position.CM,
    "central midfield": Position.CM,
    "attacking midfield": Position.CM,
    "left midfield": Position.CM,
    "right midfield": Position.CM,
    "midfield": Position.CM,
    "left winger": Position.W,
    "right winger": Position.W,
    "second striker": Position.ST,
    "centre-forward": Position.ST,
    "forward": Position.ST,
}
_POSITION_GROUP_MAP = {
    "GOALKEEPER": Position.GK,
    "DEFENDER": Position.CB,
    "MIDFIELD": Position.CM,
    "MIDFIELDER": Position.CM,
    "FORWARD": Position.ST,
}


# --- URL builders ----------------------------------------------------------
def profile_url(source_id: str, slug: str = "-") -> str:
    return f"{TM_BASE}/{slug}/profil/spieler/{source_id}"


def kader_url(club_id: str, season_year: int, slug: str = "-") -> str:
    return f"{TM_BASE}/{slug}/kader/verein/{club_id}/saison_id/{season_year}/plus/1"


# --- small value parsers -------------------------------------------------
def parse_money(text: str | None) -> float | None:
    if not text:
        return None
    t = str(text).strip().replace("\xa0", " ").lower()
    m = re.search(r"([\d.,]+)\s*(m|k|bn)?", t)
    if not m or not m.group(1):
        return None
    num = float(m.group(1).replace(",", ""))
    return {"bn": 1e9, "m": 1e6, "k": 1e3, None: 1.0}[m.group(2)] * num


def parse_tm_date(text: str | None) -> date | None:
    if not text:
        return None
    t = str(text).strip().split("(")[0].strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


def map_position(text: str | None) -> Position:
    if not text:
        return Position.UNKNOWN
    t = text.strip().lower().split(" - ")[-1].strip()
    return _POSITION_MAP.get(t, Position.UNKNOWN)


def parse_former_clubs(note: str | None) -> list[str]:
    """'ZSKA Moskau (2002-2012), Lokomotiv Moskau (03.2012-31.12.2014)' -> [names]."""
    if not note:
        return []
    pairs = re.findall(r"([^,()]+?)\s*\(([^)]*)\)", note)
    if pairs:
        return [name.strip(" ,") for name, _ in pairs if name.strip(" ,")]
    return [c.strip() for c in note.split(",") if c.strip()]


# --- tmapi JSON parsers ------------------------------------------------
def parse_player_master(payload: dict, source_id: str | None = None) -> Player:
    data = payload.get("data") or []
    if not data:
        raise ValueError("tmapi players payload has no data")
    d = data[0]
    sid = str(source_id or d.get("id"))
    attrs = d.get("attributes") or {}
    nat = (d.get("nationalityDetails") or {}).get("nationalities") or {}
    nat_id = nat.get("nationalityId")
    height_m = attrs.get("height")
    pos_detail = (attrs.get("position") or {}).get("name")
    pos = map_position(pos_detail)
    if pos is Position.UNKNOWN:
        pos = _POSITION_GROUP_MAP.get((attrs.get("positionGroup") or "").upper(), Position.UNKNOWN)
    mv = ((d.get("marketValueDetails") or {}).get("current") or {}).get("value")

    return Player(
        source="transfermarkt",
        source_id=sid,
        full_name=d.get("name") or f"tm:{sid}",
        name_home_country=(d.get("nationalityDetails") or {}).get("passportName") or None,
        birth_date=parse_tm_date((d.get("lifeDates") or {}).get("dateOfBirth")),
        position=pos,
        position_detail=pos_detail,
        foot=(attrs.get("preferredFoot") or {}).get("name"),
        height_cm=round(height_m * 100)
        if isinstance(height_m, (int, float)) and height_m
        else None,
        nationality=str(nat_id) if nat_id else None,
        is_foreigner=(nat_id is not None and nat_id != RUSSIA_NATIONALITY_ID) or None,
        place_of_birth=(d.get("birthPlaceDetails") or {}).get("placeOfBirth") or None,
        youth_clubs=parse_former_clubs(attrs.get("formerClubsNote")),
        current_club=next(
            (
                str(a.get("clubId"))
                for a in d.get("clubAssignments") or []
                if a.get("type") == "current"
            ),
            None,
        ),
        market_value_eur=float(mv) if mv is not None else None,
        profile_url=TM_BASE + d.get("relativeUrl", profile_url(sid)[len(TM_BASE) :]),
    )


def _g(d: dict, *path, default=None):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return default
    return d


def parse_performance(payload: dict, source_id: str) -> list[SeasonStats]:
    """Aggregate per-game rows to one :class:`SeasonStats` per (season, competition)."""
    games = _g(payload, "data", "performance", default=[]) or []
    buckets: dict[tuple[str, str], dict] = {}
    for game in games:
        gi = game.get("gameInformation") or {}
        st = game.get("statistics") or {}
        comp_id = gi.get("competitionId") or "?"
        season = _g(gi, "season", "display") or str(gi.get("seasonId") or "")
        key = (season, comp_id)
        b = buckets.setdefault(
            key,
            {"minutes": 0, "matches": 0, "goals": 0, "assists": 0, "ages": [], "club": None},
        )
        played = _g(st, "generalStatistics", "participationState") == "played"
        mins = _g(st, "playingTimeStatistics", "playedMinutes")
        if mins:
            b["minutes"] += int(mins)
        if played:
            b["matches"] += 1
        gs = st.get("goalStatistics") or {}
        # "...Official" is null for cups -> fall back to the unofficial total
        b["goals"] += gs.get("goalsScoredTotal") or gs.get("goalsScoredTotalOfficial") or 0
        b["assists"] += gs.get("assists") or gs.get("assistsOfficial") or 0
        age = _g(st, "generalStatistics", "age")
        if isinstance(age, (int, float)):
            b["ages"].append(age)
        if b["club"] is None:
            b["club"] = _g(game, "clubsInformation", "club", "clubId")

    out: list[SeasonStats] = []
    for (season, comp_id), b in sorted(buckets.items()):
        out.append(
            SeasonStats(
                source="transfermarkt",
                source_player_id=str(source_id),
                season=season,
                age_at_season=min(b["ages"]) if b["ages"] else None,
                club=b["club"],
                league=COMPETITION_NAMES.get(comp_id, comp_id),
                minutes=b["minutes"] or None,
                matches=b["matches"] or None,
                goals=b["goals"] or None,
                assists=b["assists"] or None,
                is_rpl=comp_id in RPL_COMPETITION_IDS,
            )
        )
    return out


def parse_market_value_history(payload: dict, source_id: str) -> list[MarketValuePoint]:
    """From the master payload's ``marketValueDetails`` (current/previous/highest).

    A full time series would need the ``marktwertverlauf`` chart endpoint; the
    master payload gives the few points that matter for a cutoff-age snapshot.
    """
    data = payload.get("data") or []
    if not data:
        return []
    mvd = data[0].get("marketValueDetails") or {}
    out: list[MarketValuePoint] = []
    for kind in ("current", "previous", "highest"):
        pt = mvd.get(kind)
        if not pt:
            continue
        out.append(
            MarketValuePoint(
                source="transfermarkt",
                source_player_id=str(source_id),
                date=parse_tm_date(pt.get("determined")),
                value_eur=float(pt["value"]) if pt.get("value") is not None else None,
            )
        )
    # dedupe by date
    seen, uniq = set(), []
    for p in out:
        if p.date not in seen:
            seen.add(p.date)
            uniq.append(p)
    return uniq


def parse_competition_table(payload: dict) -> list[str]:
    """Youth-league table -> club ids (academy universe seed for a season)."""
    tables = _g(payload, "data", "tables", default=[]) or []
    ids: list[str] = []
    for t in tables:
        for c in t.get("clubs", []):
            if c.get("clubId"):
                ids.append(str(c["clubId"]))
    return ids


# --- HTML: roster discovery only -------------------------------------
_SPIELER_RE = re.compile(r"/profil/spieler/(\d+)")


def parse_kader_html(html: str) -> list[str]:
    """Extract unique player ids from a www.transfermarkt.com squad (kader) page."""
    soup = BeautifulSoup(html, "lxml")
    ids: list[str] = []
    for a in soup.select("a[href*='/profil/spieler/']"):
        m = _SPIELER_RE.search(a.get("href", ""))
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


# --- Source -----------------------------------------------------------
class TransfermarktSource:
    """tmapi-backed. ``fetcher`` is a :class:`ingest.fetcher.TmApiClient`."""

    name = "transfermarkt"

    def __init__(self, api):
        self.api = api

    def fetch_player(self, player_id: str) -> dict:
        master = self.api.players(player_id)
        perf = self.api.performance_game(player_id)
        return {
            "player": parse_player_master(master, player_id),
            "seasons": parse_performance(perf, player_id),
            "market_values": parse_market_value_history(master, player_id),
        }

    def academy_universe(self, season_years: list[int], competition_id: str = "RUJL") -> set[str]:
        clubs: set[str] = set()
        for y in season_years:
            try:
                clubs.update(parse_competition_table(self.api.competition_table(competition_id, y)))
            except Exception:  # noqa: BLE001 - a missing season shouldn't abort the run
                continue
        return clubs

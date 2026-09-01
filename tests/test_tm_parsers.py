"""Transfermarkt parsers against real tmapi fixtures (Andrey Mostovoy, id 362570)."""

from datetime import date

from ingest.schemas import Position
from ingest.sources.transfermarkt import (
    parse_competition_table,
    parse_former_clubs,
    parse_kader_html,
    parse_market_value_history,
    parse_money,
    parse_performance,
    parse_player_master,
    parse_tm_date,
)


def test_value_parsers():
    assert parse_money("€3.50m") == 3_500_000
    assert parse_money("€600k") == 600_000
    assert parse_money("-") is None
    assert parse_tm_date("1997-11-05") == date(1997, 11, 5)
    assert parse_tm_date("05/11/1997 (28)") == date(1997, 11, 5)


def test_parse_former_clubs():
    note = (
        "ZSKA Moskau (2002-2012), SDYuSShOR-94 Rublevo Moskau (02.2012 - 03.2012), "
        "Lokomotiv Moskau (03.2012-31.12.2014)"
    )
    assert parse_former_clubs(note) == [
        "ZSKA Moskau",
        "SDYuSShOR-94 Rublevo Moskau",
        "Lokomotiv Moskau",
    ]
    assert parse_former_clubs("Spartak Moscow, CSKA") == ["Spartak Moscow", "CSKA"]
    assert parse_former_clubs(None) == []


def test_parse_player_master(fx):
    p = parse_player_master(fx("transfermarkt", "tmapi_players_362570.json"))
    assert p.source == "transfermarkt"
    assert p.source_id == "362570"
    assert p.full_name == "Andrey Mostovoy"
    assert p.name_home_country == "Мостовой Андрей Андреевич"
    assert p.birth_date == date(1997, 11, 5)
    assert p.position == Position.W  # Left Winger
    assert p.foot == "right"
    assert p.height_cm == 180
    assert p.is_foreigner is None  # Russian (nationalityId 141)
    assert p.place_of_birth == "Omsk"
    assert "Lokomotiv Moskau" in p.youth_clubs
    assert p.market_value_eur == 3_500_000
    assert p.profile_url.endswith("/andrey-mostovoy/profil/spieler/362570")


def test_parse_performance_aggregates_by_season_competition(fx):
    rows = parse_performance(fx("transfermarkt", "tmapi_performance_362570.json"), "362570")
    assert rows, "expected aggregated season rows"
    # one row per (season, competition)
    keys = [(r.season, r.league) for r in rows]
    assert len(keys) == len(set(keys))
    # RU1 rows flagged as RPL, youth league not
    rpl_rows = [r for r in rows if r.is_rpl]
    assert rpl_rows and all(r.league == "Premier Liga" for r in rpl_rows)
    youth = [r for r in rows if r.league == "Russian Youth League"]
    assert youth and not any(r.is_rpl for r in youth)
    # aggregates are sane
    for r in rows:
        assert r.minutes is None or r.minutes >= 0
        assert r.matches is None or r.matches >= 1


def test_parse_market_value_history(fx):
    pts = parse_market_value_history(fx("transfermarkt", "tmapi_players_362570.json"), "362570")
    assert pts
    assert all(p.source == "transfermarkt" for p in pts)
    assert any(p.value_eur == 3_500_000 for p in pts)
    assert len({p.date for p in pts}) == len(pts)  # deduped by date


def test_parse_competition_table_returns_club_ids(fx):
    ids = parse_competition_table(fx("transfermarkt", "tmapi_RUJL_table_2022.json"))
    assert len(ids) == 20
    assert all(cid.isdigit() for cid in ids)


def test_parse_kader_html():
    html = """
    <table class="items"><tbody>
      <tr><td><a href="/andrey-mostovoy/profil/spieler/362570">Mostovoy</a></td></tr>
      <tr><td><a href="/x/leistungsdaten/spieler/362570">dup</a>
              <a href="/y/profil/spieler/99999">Y</a></td></tr>
    </tbody></table>"""
    assert parse_kader_html(html) == ["362570", "99999"]

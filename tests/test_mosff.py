"""mosff.ru (Московская федерация футбола) adapter — pure parsers."""

from __future__ import annotations

from ingest.sources.mosff import CLUB_LEAGUE_TOURNAMENTS, parse_player_row, split_title


def test_split_title():
    assert split_title("Московский Даниил Иванович") == ("Даниил Московский", "Иванович")
    assert split_title("Иванов Пётр") == ("Пётр Иванов", None)
    assert split_title("  Ким  Александр  Сергеевич  ") == ("Александр Ким", "Сергеевич")
    assert split_title("Пеле") == ("Пеле", None)


def test_parse_player_row():
    row = {
        "num": 2,
        "playerUrl": "/player/25716",
        "title": "Марусов Мирон Сергеевич",
        "teamTitle": "Локомотив 2013 г.р.",
        "games": 18,
        "minutes": 1056,
        "goalsSum": 13,
        "penalties": 5,
        "yellowCards": 1,
        "redCards": 0,
        "hatTricks": 1,
    }
    p = parse_player_row(row, birth_year=2013, tournament_id=1168)
    assert p["mosff_id"] == "25716"
    assert p["full_name"] == "Мирон Марусов" and p["patronymic"] == "Сергеевич"
    assert p["birth_year"] == 2013 and p["birth_date"] is None
    assert p["games"] == 18 and p["goals"] == 13 and p["minutes"] == 1056
    assert p["source"] == "mosff" and p["tournament_id"] == 1168


def test_parse_player_row_missing_fields():
    p = parse_player_row({"title": "Нет Данных", "playerUrl": ""}, 2012, 1169)
    assert p["mosff_id"] is None
    assert p["games"] == 0 and p["goals"] == 0
    assert p["birth_year"] == 2012


def test_club_league_map():
    assert CLUB_LEAGUE_TOURNAMENTS["2012"] == 1169
    assert CLUB_LEAGUE_TOURNAMENTS["2013"] == 1168

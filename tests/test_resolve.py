from datetime import date

from ingest.resolve import build_crosswalk, normalize_name, resolve_players, translit


def test_normalize_name():
    assert normalize_name("Мостовой, Андрей") == normalize_name("Andrey Mostovoy")
    # comma form is reordered to "first last"; plain form keeps token order
    assert normalize_name("Мостовой, Андрей") == "andrey mostovoy"
    assert normalize_name("  Головин  Александр ") == "golovin aleksandr"
    assert sorted(normalize_name("Golovin Aleksandr").split()) == ["aleksandr", "golovin"]
    assert translit("Головин") == "golovin"


def test_resolve_clusters_across_sources():
    records = [
        {
            "source": "transfermarkt",
            "source_id": "362570",
            "full_name": "Andrey Mostovoy",
            "name_home_country": "Мостовой Андрей Андреевич",
            "birth_date": date(1997, 11, 5),
        },
        {
            "source": "wikipedia",
            "source_id": "Мостовой,_Андрей_Андреевич",
            "full_name": "Мостовой, Андрей Андреевич",
            "birth_date": date(1997, 11, 5),
        },
        {
            "source": "transfermarkt",
            "source_id": "999",
            "full_name": "Ivan Petrov",
            "birth_date": date(2003, 1, 1),
        },
    ]
    resolved = resolve_players(records)
    by_src = {r.source_id: r.player_id for r in resolved.itertuples()}
    assert by_src["362570"] == by_src["Мостовой,_Андрей_Андреевич"]
    assert by_src["999"] != by_src["362570"]
    assert resolved["player_id"].nunique() == 2


def test_same_name_different_birth_year_not_merged():
    records = [
        {
            "source": "a",
            "source_id": "1",
            "full_name": "Sergey Ivanov",
            "birth_date": date(1999, 5, 1),
        },
        {
            "source": "b",
            "source_id": "2",
            "full_name": "Sergey Ivanov",
            "birth_date": date(2005, 5, 1),
        },
    ]
    resolved = resolve_players(records)
    assert resolved["player_id"].nunique() == 2


def test_crosswalk_shape():
    records = [
        {"source": "tm", "source_id": "1", "full_name": "A B", "birth_date": date(2000, 1, 1)},
    ]
    xw = build_crosswalk(resolve_players(records))
    assert list(xw.columns) == ["source", "source_id", "player_id", "match_score"]


def test_empty_input():
    resolved = resolve_players([])
    assert len(resolved) == 0

"""Phase B: ru.wikipedia bio adapter — pure parsing (no network)."""

from datetime import date

from ingest.sources.wikipedia_players import (
    WikiBio,
    WikiPlayerBios,
    _honour_years,
    _nt_youth_levels,
    _parse_ts,
    _search_name,
)


def test_search_name_drops_patronymic():
    assert _search_name("Пиняев Сергей Андреевич", "") == "Пиняев Сергей"
    assert _search_name("Абышов Руслан Ибрагим оглы", "") == "Абышов Руслан"
    assert _search_name("Мостовой Андрей", "") == "Мостовой Андрей"
    # no home name -> romanised fallback, "Last, First" reordered
    assert _search_name(None, "Batrakov, Aleksey") == "Aleksey Batrakov"
    assert _search_name("", "Aleksey Batrakov Andreevich") == "Aleksey Batrakov"


def test_nt_youth_levels_from_categories():
    cats = [
        "Категория:Игроки сборной России по футболу (до 17 лет)",
        "Категория:Игроки сборной России по футболу (до 21 года)",
        "Категория:Игроки молодёжной сборной России по футболу",
        "Категория:Футболисты России",
        "Категория:Игроки сборной Бразилии по футболу",  # other country -> ignored
    ]
    assert _nt_youth_levels(cats) == [17, 21]
    assert _nt_youth_levels(["Категория:Футболисты России"]) == []


def test_honour_years_only_from_bullet_lines_under_the_heading():
    wt = (
        "Some intro with a stray 1990.\n\n"
        "== Достижения ==\n"
        "* Чемпион России: [[2023]], 2024\n"
        "* Победитель первенства ЮФЛ: 2019\n\n"
        "== Личная жизнь ==\n"
        "* Родился в 1888 году\n"
    )
    assert _honour_years(wt) == [2019, 2023, 2024]
    assert _honour_years("no honours section here, year 2020") == []


def test_parse_ts():
    assert _parse_ts("2019-03-17T23:30:58Z") == date(2019, 3, 17)
    assert _parse_ts(None) is None
    assert _parse_ts("garbage") is None


def test_not_found_sentinel():
    b = WikiBio.not_found("pid1")
    assert b.player_id == "pid1"
    assert b.wiki_title is None
    assert b.nt_youth_levels == [] and b.honours_years == []
    assert b.article_created_age is None


class _FakeFetcher:
    """Serves canned MediaWiki API JSON keyed by the 'action'/'list' of the call."""

    def __init__(self, pages):
        self._pages = pages
        self.calls = []

    def get(self, url, params=None, **kw):
        self.calls.append(params)
        p = params or {}

        class _R:
            @staticmethod
            def json():
                if p.get("list") == "search":
                    return {"query": {"search": [{"title": "Пиняев, Сергей Максимович"}]}}
                if p.get("action") == "parse":
                    return {"parse": {"wikitext": "== Достижения ==\n* Кубок ЮФЛ: 2019\n"}}
                return {"query": {"pages": _FakeFetcher_pages}}

        return _R()


def test_lookup_happy_path_with_fake_fetcher():
    global _FakeFetcher_pages
    _FakeFetcher_pages = [
        {
            "title": "Пиняев, Сергей Максимович",
            "extract": "Сергей Максимович Пиняев (род. 2 ноября 2004) — российский футболист.",
            "categories": [
                {"title": "Категория:Футболисты России"},
                {"title": "Категория:Игроки сборной России по футболу (до 19 лет)"},
            ],
            "revisions": [{"timestamp": "2021-08-01T10:00:00Z"}],
        }
    ]
    bios = WikiPlayerBios(fetcher=_FakeFetcher(_FakeFetcher_pages))
    bio = bios.lookup("pid1", "Пиняев Сергей Андреевич", "Sergey Pinyaev", date(2004, 11, 2))
    assert bio.wiki_title == "Пиняев, Сергей Максимович"
    assert bio.article_created == date(2021, 8, 1)
    assert round(bio.article_created_age, 1) == 16.7
    assert bio.nt_youth_levels == [19]
    assert bio.youth_honours_count == 1  # 2019 <= 2004 + 18

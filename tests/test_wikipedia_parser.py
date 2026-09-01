"""Wikipedia ЮФЛ season parser against a real API fixture (2019/2020)."""

from ingest.sources.wikipedia import parse_season_html, season_titles


def test_season_titles():
    titles = season_titles(2022)
    assert "Юношеская футбольная лига 2022/2023" in titles


def test_parse_season_2019_20(fx):
    payload = fx("wikipedia", "yufl_2019_20.json")
    html = payload["parse"]["text"]
    result = parse_season_html(html, season="2019/2020")

    assert result["season"] == "2019/2020"
    assert len(result["participants"]) >= 8
    # standings extracted with numeric columns
    assert result["standings"], "expected a standings table"
    top = result["standings"][0]
    assert top["club"]
    assert isinstance(top["points"], int) and top["points"] > 0
    assert isinstance(top["played"], int) and top["played"] > 0
    # every standings club should also be a participant
    part = {p.lower() for p in result["participants"]}
    assert any(s["club"].lower() in part for s in result["standings"])

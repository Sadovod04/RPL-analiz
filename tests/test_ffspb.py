"""ФФ СПб (Наградион) adapter — parser + block-id extraction on real fixtures."""

from ingest.sources.ffspb import Nagradion, _snake, parse_matches


def test_snake_component_name():
    assert _snake("MatchFeed") == "match_feed"
    assert _snake("TopPlayersBlock") == "top_players_block"
    assert _snake("Header") == "header"


def test_parse_matches(fx):
    payload = fx("ffspb", "match_feed_40530.json")
    rows = parse_matches(payload)
    assert len(rows) > 100
    m = rows[0]
    assert {"match_id", "home", "guest", "goals", "url", "date", "finished"} <= m.keys()
    assert m["url"].startswith("https://stat.ffspb.org/tournament40530/match/")
    assert isinstance(m["match_id"], int)
    assert all(r["finished"] in (True, False) for r in rows)


def test_block_ids_extraction():
    html = (
        '<div data-block-id="16215" data-component-name="MatchFeed"></div>'
        '<div data-component-name="TournamentTable" data-block-id="400545"></div>'
        '<div data-block-id="400547" data-component-name="TournamentTable"></div>'
    )
    blocks = Nagradion().block_ids(html)
    assert blocks["MatchFeed"] == ["16215"]
    assert blocks["TournamentTable"] == ["400545", "400547"]

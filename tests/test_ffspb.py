"""ФФ СПб (Наградион) adapter — parsers on real fixtures (tournament 40530, U14 СПб 2025)."""

from datetime import date

from ingest.sources.ffspb import (
    _snake,
    parse_calendar,
    parse_lineup,
    parse_matches,
    parse_player_profile,
    render_props,
)


def test_snake_component_name():
    assert _snake("MatchFeed") == "match_feed"
    assert _snake("GameProtocolBlock") == "game_protocol_block"


def test_parse_matches(fx):
    rows = parse_matches(fx("ffspb", "match_feed_40530.json"))
    assert len(rows) > 100
    m = rows[0]
    assert {"match_id", "home", "guest", "goals", "url", "finished"} <= m.keys()
    assert m["url"].startswith("https://stat.ffspb.org/tournament40530/match/")
    assert isinstance(m["match_id"], int)


def test_parse_lineup(fx):
    gp = fx("ffspb", "game_protocol_3499308.json")
    players = parse_lineup(gp)
    assert 20 <= len(players) <= 40  # two teams, starters + subs
    p = players[0]
    assert {"ffspb_id", "full_name", "number", "side", "started", "tournament_id"} <= p.keys()
    assert p["tournament_id"] == "40530"
    assert p["ffspb_id"].isdigit()
    assert {"home", "away"} == {x["side"] for x in players}
    assert any(x["started"] for x in players) and any(not x["started"] for x in players)


def test_parse_player_profile(fx):
    html = fx("ffspb", "player_765150_frag.html")
    prof = parse_player_profile(html)
    assert prof["full_name"] == "Камиль Абдулмаликов"
    assert prof["birth_date"] == date(2012, 11, 23)
    assert "13" in (prof["age_text"] or "")


def test_parse_player_stats_tournaments(fx):
    # PlayerStats props embedded — reuse render_props path via a wrapper page
    import json

    ps = fx("ffspb", "player_stats_765150.json")
    html = f"<script>renderComponent(\"x-1\", 'PlayerStats', {json.dumps(ps)});</script>"
    prof = parse_player_profile("<div class='person-info'></div>" + html)
    trns = prof["tournaments"]
    assert trns and all("tournament_id" in x and "season" in x for x in trns)
    assert any("до 1" in (x["tournament_name"] or "") for x in trns)


def test_parse_calendar():
    html = """
    <a href="/tournament44327">Первенство. Мальчики до 14 лет</a>
    <a href="/tournament44325">Первенство. Мальчики до 15 лет</a>
    <a href="/tournament41244">МФЛ. Мальчики 2011-2012</a>
    <a href="/tournament99999">Чемпионат России. Премьер-лига</a>
    <a href="/tournament44346">Кубок. Юноши до 16 лет</a>
    """
    rows = parse_calendar(html, max_age=15)
    ids = {r["tournament_id"] for r in rows}
    assert "44327" in ids and "44325" in ids and "41244" in ids
    assert "99999" not in ids  # not a youth tournament
    assert "44346" not in ids  # age 16 > max_age 15


def test_render_props_brace_matching():
    html = 'foo renderComponent("uid", \'X\', {"a": 1, "b": {"c": [1,2]}, "d": "}"}); bar'
    assert render_props(html, "X") == {"a": 1, "b": {"c": [1, 2]}, "d": "}"}

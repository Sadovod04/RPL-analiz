from ingest.run_ingest import collection_season, parse_args, parse_seasons


def test_parse_seasons_range_and_csv():
    assert parse_seasons("2015-2018") == [2015, 2016, 2017, 2018]
    assert parse_seasons("2019,2021") == [2019, 2021]


def test_collection_season_boundary():
    import datetime

    assert collection_season(datetime.date(2026, 8, 1)) == "2026-27"
    assert collection_season(datetime.date(2026, 3, 1)) == "2025-26"


def test_parse_args_defaults():
    ns = parse_args([])
    assert ns.sources == ["wikipedia", "transfermarkt"]
    assert ns.seasons == "2015-2024"


def test_dry_run(capsys):
    from ingest.run_ingest import main

    main(["--dry-run", "--seasons", "2018-2020"])
    assert "dry-run" in capsys.readouterr().out

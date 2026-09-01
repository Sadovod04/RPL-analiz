from settings import load_settings


def test_settings_load_and_shape():
    cfg = load_settings()
    assert cfg["target"]["rpl_minutes_threshold"] == 200
    assert cfg["target"]["settled_age"] == 26
    assert cfg["cohorts"]["birth_year_min"] == 1990
    assert cfg["cohorts"]["birth_year_max"] == 2004
    assert cfg["cohorts"]["scoring_birth_year_min"] == 2005
    assert cfg["features"]["cutoff_age_start"] == 11
    assert cfg["features"]["age_buckets"] == ["U13", "U15", "U17", "U19", "U21"]

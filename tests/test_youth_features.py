"""youth_features — mapping regional youth pools onto the main feature schema."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from app.youth_features import combined_frame, youth_feature_rows, youth_frame
from features.build_features import feature_columns
from features.labels import CENSORED


def _ffspb_parquet(tmp_path):
    d = pd.DataFrame(
        {
            "ffspb_id": ["1", "2", "3"],
            "full_name": ["Иван Тест", "Иван Тест", "Петр Гол"],
            "patronymic": ["Иванович", "Иванович", "Петрович"],
            "birth_date": [dt.date(2013, 5, 1), dt.date(2013, 5, 1), dt.date(2012, 1, 2)],
            "age_text": ["12", "12", "13"],
            "n_tournaments": [2, 2, 1],
            "teams": ["СШОР Зенит-2", "Академия Зенит", "СШ Кировец"],
            "games": [18, 18, 20],
            "goals": [5, 5, 1],
            "yellows": [0, 1, 2],
            "reds": [0, 0, 0],
            "minutes": [1200, 1200, 1500],
            "source": ["ffspb"] * 3,
        }
    )
    p = tmp_path / "ffspb_players.parquet"
    d.to_parquet(p)
    return p


def _mosff_parquet(tmp_path):
    d = pd.DataFrame(
        {
            "mosff_id": ["10", "11", "12"],
            "full_name": ["Мирон Марусов", "Даниил Московский", "Азамат Сафин"],
            "patronymic": ["Сергеевич", "Иванович", None],
            "birth_date": [None, None, None],
            "birth_year": [2013, 2013, 2012],
            "teams": ["Локомотив 2013 г.р.", "Динамо 2013 г.р.", "Родина 2012 г.р."],
            "tournament_id": [1168, 1168, 1169],
            "n_tournaments": [1, 1, 1],
            "games": [18, 13, 15],
            "goals": [13, 15, 11],
            "penalties": [5, 0, 3],
            "minutes": [1056, 708, 892],
            "yellows": [1, 2, 0],
            "reds": [0, 1, 0],
            "hat_tricks": [1, 2, 0],
            "source": ["mosff"] * 3,
        }
    )
    p = tmp_path / "mosff_players.parquet"
    d.to_parquet(p)
    return p


def test_youth_frame_dedupes_and_scores(tmp_path):
    y = youth_frame([_ffspb_parquet(tmp_path)])
    assert len(y) == 2  # the two "Иван Тест" rows collapse
    ivan = y[y["full_name"] == "Иван Тест"].iloc[0]
    assert ivan["games"] == 18 and ivan["goals"] == 5  # max, not double-counted
    assert {"pers_score", "proj_level", "source"} <= set(y.columns)
    assert y["proj_level"].str.startswith("yl_").all()


def test_youth_frame_multi_source(tmp_path):
    y = youth_frame([_ffspb_parquet(tmp_path), _mosff_parquet(tmp_path)])
    assert set(y["source"].str.split(";").str[0]) == {"ffspb", "mosff"}
    assert len(y) == 5  # 2 spb + 3 moscow, no cross-region name clash here
    # pers_score is ranked within each source
    assert y[y["source"] == "mosff"]["pers_score"].between(0, 100).all()


def test_youth_feature_rows_match_schema(tmp_path):
    template = pd.read_parquet("data/processed/features.parquet")
    rows = youth_feature_rows([_ffspb_parquet(tmp_path), _mosff_parquet(tmp_path)], template)
    assert set(template.columns) <= set(rows.columns)
    for c in ("target", "pro_target", "ordinal_target"):
        if c in rows:
            assert (rows[c] == CENSORED).all()
    assert set(rows["source"]) <= {"ffspb", "mosff"}
    assert rows["player_id"].str.startswith("youth_").all()
    assert rows["youth_ga_per90"].notna().all()


def test_combined_frame_concat(tmp_path):
    fp = "data/processed/features.parquet"
    base = pd.read_parquet(fp)
    merged = combined_frame(fp, [_ffspb_parquet(tmp_path), _mosff_parquet(tmp_path)])
    assert len(merged) == len(base) + 5
    assert set(merged["source"]) == {"tm", "ffspb", "mosff"}
    assert not ({"source", "pers_score", "proj_level"} & set(feature_columns(merged)))
    kids = merged[merged["source"] != "tm"]
    assert (kids["pro_target"] == CENSORED).all()


def test_combined_frame_dir_autodiscovers(tmp_path):
    _ffspb_parquet(tmp_path)
    _mosff_parquet(tmp_path)
    # features.parquet must sit in the same dir for the dir form
    pd.read_parquet("data/processed/features.parquet").to_parquet(tmp_path / "features.parquet")
    merged = combined_frame(tmp_path / "features.parquet", tmp_path)
    assert set(merged["source"]) == {"tm", "ffspb", "mosff"}


def test_combined_frame_without_youth():
    base = pd.read_parquet("data/processed/features.parquet")
    merged = combined_frame("data/processed/features.parquet", None)
    assert len(merged) == len(base)
    assert (merged["source"] == "tm").all()

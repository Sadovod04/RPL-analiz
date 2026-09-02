"""youth_features — mapping ФФ СПб kids onto the main feature schema + combining."""

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
            "n_tournaments": [1, 1, 2],
            "teams": ["СШОР Зенит-2", "Академия Зенит", "СШ Кировец"],
            "games": [10, 8, 20],
            "goals": [2, 3, 1],
            "yellows": [0, 1, 2],
            "reds": [0, 0, 0],
            "minutes": [700, 500, 1500],
            "source": ["ffspb"] * 3,
        }
    )
    p = tmp_path / "ffspb_players.parquet"
    d.to_parquet(p)
    return p


def test_youth_frame_dedupes_and_scores(tmp_path):
    y = youth_frame(_ffspb_parquet(tmp_path))
    assert len(y) == 2  # the two "Иван Тест" rows collapse
    ivan = y[y["full_name"] == "Иван Тест"].iloc[0]
    assert ivan["games"] == 18 and ivan["goals"] == 5
    assert {"pers_score", "proj_level"} <= set(y.columns)
    assert y["proj_level"].str.startswith("yl_").all()


def test_youth_feature_rows_match_schema(tmp_path):
    template = pd.read_parquet("data/processed/features.parquet")
    rows = youth_feature_rows(_ffspb_parquet(tmp_path), template)
    # every template column present
    assert set(template.columns) <= set(rows.columns)
    # labels censored -> never train
    for c in ("target", "pro_target", "ordinal_target"):
        if c in rows:
            assert (rows[c] == CENSORED).all()
    assert (rows["source"] == "ffspb").all()
    assert rows["player_id"].str.startswith("ffspb_").all()
    # the youth stats we do have are carried
    assert (rows["youth_goals_total"] > 0).any()
    assert rows["youth_ga_per90"].notna().all()


def test_combined_frame_concat(tmp_path):
    fp = "data/processed/features.parquet"
    base = pd.read_parquet(fp)
    merged = combined_frame(fp, _ffspb_parquet(tmp_path))
    assert len(merged) == len(base) + 2
    assert set(merged["source"]) == {"tm", "ffspb"}
    # provenance / heuristic columns are NOT model features
    assert not ({"source", "pers_score", "proj_level"} & set(feature_columns(merged)))
    # the kids never land in the resolved (trainable) set
    kids = merged[merged["source"] == "ffspb"]
    assert (kids["pro_target"] == CENSORED).all()


def test_combined_frame_without_ffspb():
    base = pd.read_parquet("data/processed/features.parquet")
    merged = combined_frame("data/processed/features.parquet", None)
    assert len(merged) == len(base)
    assert (merged["source"] == "tm").all()

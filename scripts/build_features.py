"""Rebuild data/processed/features.parquet from the Postgres DB — no re-scrape.

    uv run python scripts/build_features.py

Same step as ``ingest.run_ingest --build`` but standalone, for iterating on the
feature code against an already-crawled DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.build_features import build_feature_matrix, feature_columns, from_db
from ingest.db import get_engine
from settings import load_settings

_NEW_COLS = (
    "cohort_year",
    "birth_quarter",
    "rel_age_frac",
    "min_age_gap_vs_peers",
    "mean_age_gap_vs_peers",
    "matches_share_min",
    "matches_share_mean",
    "minutes_dropoff_max",
    "had_minutes_collapse",
    "wiki_article_pre_cutoff",
    "wiki_youth_national_team",
    "wiki_youth_honours",
    "recognition_count",
    "pre_cutoff_recognition_score",
    "any_recognition",
)


def main() -> None:
    players, seasons, mvs, wiki = from_db(get_engine())
    m = build_feature_matrix(players, seasons, mvs, wiki=wiki)
    out = Path(load_settings()["paths"]["data_processed"]) / "features.parquet"
    m.to_parquet(out, index=False)
    print(f"wrote {out}  shape={m.shape}  features={len(feature_columns(m))}")
    print(f"  target:     {m['target'].value_counts(dropna=False).to_dict()}")
    if "pro_target" in m:
        print(f"  pro_target: {m['pro_target'].value_counts(dropna=False).to_dict()}")
    present = [c for c in _NEW_COLS if c in m.columns]
    print(f"  phase-A cols ({len(present)}/{len(_NEW_COLS)}): {present}")
    print(m[present].describe().T.to_string())


if __name__ == "__main__":
    main()

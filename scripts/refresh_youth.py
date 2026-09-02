"""Refresh every regional youth pool in one go, then print the combined summary.

    uv run python scripts/refresh_youth.py                # mosff (fast) + summary
    uv run python scripts/refresh_youth.py --with-ffspb   # also re-crawl ФФ СПб (slow)

mosff.ru is a single fast JSON endpoint; ФФ СПб is a multi-step crawl, so it is
opt-in here. The dashboard picks up the new parquets automatically.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.youth_features import youth_frame, youth_paths  # noqa: E402


def _run(*args: str) -> None:
    print(f"\n$ {' '.join(args)}")
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-ffspb", action="store_true", help="also re-crawl stat.ffspb.org (slow)")
    ap.add_argument("--mosff-years", nargs="*", default=["2012", "2013"])
    args = ap.parse_args(argv)

    _run("scripts/ingest_mosff.py", "--years", *args.mosff_years)
    if args.with_ffspb:
        _run("scripts/ingest_ffspb.py", "--max-age", "15")

    paths = youth_paths(ROOT / "data" / "processed")
    y = youth_frame(paths)
    print("\n=== combined youth pool ===")
    print("sources :", y["source"].str.split(";").str[0].value_counts().to_dict())
    print("by year :", y["birth_year"].value_counts().sort_index().to_dict())
    print("proj lvl:", y["proj_level"].value_counts().to_dict())
    cols = ["full_name", "birth_year", "source", "teams", "goals", "pers_score"]
    top = y.nlargest(10, "pers_score")[cols]
    with pd.option_context("display.max_colwidth", 40):
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()

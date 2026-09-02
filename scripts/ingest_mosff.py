"""Collect Moscow youth players from mosff.ru (Клубная лига) into their own parquet.

    uv run python scripts/ingest_mosff.py                       # 2012 + 2013 by default
    uv run python scripts/ingest_mosff.py --years 2012 2013 2014
    uv run python scripts/ingest_mosff.py --tournaments 1168:2013 1169:2012

Writes ``data/processed/mosff_players.parquet`` — one row per player per tournament
(games / goals / minutes / cards). Merge into the dashboard happens automatically:
``app.youth_features`` picks up every ``data/processed/<source>_players.parquet``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ingest.rate_limiter import RateLimiter  # noqa: E402
from ingest.sources.mosff import CLUB_LEAGUE_TOURNAMENTS, MosffSource  # noqa: E402

OUT = Path("data/processed/mosff_players.parquet")


def _pairs(args) -> list[tuple[int, int]]:
    if args.tournaments:
        out = []
        for tok in args.tournaments:
            tid, _, yr = tok.partition(":")
            out.append((int(tid), int(yr)))
        return out
    return [(CLUB_LEAGUE_TOURNAMENTS[y], int(y)) for y in args.years]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--years", nargs="*", default=["2012", "2013"], choices=sorted(CLUB_LEAGUE_TOURNAMENTS)
    )
    ap.add_argument("--tournaments", nargs="*", help="explicit <tournamentId>:<birthYear> tokens")
    args = ap.parse_args(argv)

    src = MosffSource()
    src._f.rate_limiter = RateLimiter(min_interval=0.4, jitter=0.3)

    rows: list[dict] = []
    for tid, yr in _pairs(args):
        players = src.tournament_players(tid, yr)
        print(f"tournament {tid} ({yr} г.р.): {len(players)} players")
        rows.extend(players)

    df = pd.DataFrame(rows)
    if len(df):
        # a kid who plays up appears in two tournaments -> keep both rows; the
        # youth_frame de-dupe (by name + patronymic + birth_year) folds them.
        df = df.drop_duplicates(subset=["mosff_id", "tournament_id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  shape={df.shape}")
    if len(df):
        print("by birth year:", df["birth_year"].value_counts().sort_index().to_dict())
        print("with >=1 goal:", int((df["goals"] > 0).sum()))
        print(
            df.nlargest(5, "goals")[
                ["full_name", "birth_year", "teams", "games", "goals"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f}s")

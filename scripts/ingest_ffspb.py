"""Collect youth players from stat.ffspb.org (Наградион) into their own parquet.

    uv run python scripts/ingest_ffspb.py --max-age 15 --seasons-back 4
    uv run python scripts/ingest_ffspb.py --tournaments 40530 37576 44327

Chain: discover youth tournaments -> matches -> lineups -> unique players ->
player profile (name + birth date + tournament history). Writes
``data/processed/ffspb_players.parquet``. Resumable via a checkpoint of visited
player ids. Merge into the main dataset is a separate step (ingest.resolve).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ingest.rate_limiter import RateLimiter  # noqa: E402
from ingest.sources.ffspb import FfspbSource, Nagradion  # noqa: E402

OUT = Path("data/processed/ffspb_players.parquet")
CKPT = Path("data/raw/ffspb_players_checkpoint.parquet")


def _collect_player_ids(
    src: FfspbSource, tids: list[str], limit: int | None
) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], dict] = {}
    for i, tid in enumerate(tids, 1):
        try:
            matches = [m for m in src.tournament_matches(tid) if m["finished"]]
        except Exception as exc:  # noqa: BLE001
            print(f"  tournament {tid}: {exc}")
            continue
        print(f"[{i}/{len(tids)}] tournament {tid}: {len(matches)} finished matches")
        for m in matches:
            try:
                for p in src.match_lineup(tid, m["match_id"]):
                    key = (p["tournament_id"], p["ffspb_id"])
                    seen.setdefault(key, {"full_name": p["full_name"]})
            except Exception as exc:  # noqa: BLE001
                print(f"    match {m['match_id']}: {exc}")
            if limit and len(seen) >= limit:
                return list(seen)
    return list(seen)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tournaments", nargs="*", help="explicit tournament ids (skips discovery)")
    ap.add_argument("--max-age", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None, help="cap unique players (debug)")
    args = ap.parse_args(argv)

    src = FfspbSource(Nagradion(fetcher=None))
    src.api._f.rate_limiter = RateLimiter(min_interval=0.3, jitter=0.2)

    if args.tournaments:
        tids = list(args.tournaments)
    else:
        disc = src.discover_youth_tournaments(max_age=args.max_age)
        tids = [d["tournament_id"] for d in disc]
        print(f"discovered {len(tids)} youth tournaments (<= {args.max_age} y.o.)")

    done: set[str] = set()
    rows: list[dict] = []
    if CKPT.exists():
        prev = pd.read_parquet(CKPT)
        rows = prev.to_dict("records")
        done = set(prev["ffspb_id"].astype(str))
        print(f"resume: {len(done)} players already collected")

    pairs = [(t, p) for t, p in _collect_player_ids(src, tids, args.limit) if p not in done]
    print(f"{len(pairs)} new players to fetch")

    for n, (tid, pid) in enumerate(pairs, 1):
        try:
            prof = src.player_profile(tid, pid, with_stats=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  player {pid}: {exc}")
            continue
        car = prof.get("career", {})
        rows.append(
            {
                "ffspb_id": pid,
                "full_name": prof["full_name"],
                "patronymic": prof.get("patronymic"),
                "birth_date": prof["birth_date"],
                "age_text": prof["age_text"],
                "n_tournaments": len(prof["tournaments"]),
                "teams": ";".join(sorted({x["team"] for x in prof["tournaments"] if x["team"]})),
                "games": car.get("games", 0),
                "goals": car.get("goals", 0),
                "yellows": car.get("yellows", 0),
                "reds": car.get("reds", 0),
                "minutes": car.get("minutes", 0),
                "source": "ffspb",
            }
        )
        if n % 100 == 0:
            pd.DataFrame(rows).to_parquet(CKPT, index=False)
            print(f"  [{n}/{len(pairs)}] checkpointed")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    df.to_parquet(CKPT, index=False)
    print(f"wrote {OUT}  shape={df.shape}")
    if len(df):
        by = pd.to_datetime(df["birth_date"], errors="coerce").dt.year
        print("birth years:", by.value_counts().sort_index().tail(8).to_dict())


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f}s")

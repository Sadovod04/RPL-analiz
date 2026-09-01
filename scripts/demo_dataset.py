"""Build a small REAL feature set via tmapi only (no Playwright) for the EDA notebook.

Convenience sample: current squads of RPL clubs + their reserve/youth sides. This
is target-skewed (few true negatives) — it demonstrates the M2 pipeline on real
data; the actual training set comes from `run_ingest` kader discovery over
resolved cohorts. Writes data/processed/features_demo.parquet.
"""

from __future__ import annotations

import sys

import pandas as pd

from features.build_features import build_feature_matrix, feature_columns
from ingest.fetcher import TmApiClient
from ingest.rate_limiter import RateLimiter
from ingest.sources.transfermarkt import TransfermarktSource

RPL_CLUBS = ["16704", "232", "964", "2410", "121", "2698", "2696", "1083", "932", "1124"]


def main(max_players: int = 200) -> None:
    # one-off demo pull -> lighter throttle than the default ingest cadence
    with TmApiClient(rate_limiter=RateLimiter(min_interval=0.5, jitter=0.3)) as api:
        src = TransfermarktSource(api)

        pids: list[str] = []
        for cid in RPL_CLUBS:
            try:
                sq = api.club_squad(cid)["data"]
                pids += [str(p["playerId"]) for p in sq.get("squad", [])]
            except Exception as e:  # noqa: BLE001
                print(f"squad {cid}: {e}")
        pids = list(dict.fromkeys(pids))[:max_players]
        print(f"{len(pids)} unique player ids")

        players, seasons, mvs = [], [], []
        for i, pid in enumerate(pids, 1):
            try:
                rec = src.fetch_player(pid)
            except Exception as e:  # noqa: BLE001
                print(f"  {pid}: {e}")
                continue
            p = rec["player"]
            by = p.birth_date.year if p.birth_date else None
            players.append(
                {
                    "player_id": pid,
                    "canonical_name": p.full_name,
                    "birth_year": by,
                    "position": p.position.value,
                    "height_cm": p.height_cm,
                    "is_foreigner": p.is_foreigner,
                    "academy_club": (p.youth_clubs[0] if p.youth_clubs else None),
                }
            )
            for s in rec["seasons"]:
                seasons.append(
                    {
                        "player_id": pid,
                        "season": s.season,
                        "league": s.league,
                        "club": s.club,
                        "minutes": s.minutes,
                        "matches": s.matches,
                        "goals": s.goals,
                        "assists": s.assists,
                        "is_rpl": s.is_rpl,
                        "age_at_season": s.age_at_season,
                    }
                )
            for m in rec["market_values"]:
                mvs.append({"player_id": pid, "date": m.date, "value_eur": m.value_eur})
            if i % 25 == 0:
                print(f"  {i}/{len(pids)}")

    players_df = pd.DataFrame(players)
    seasons_df = pd.DataFrame(seasons)
    mvs_df = pd.DataFrame(mvs)
    print(f"players={len(players_df)} seasons={len(seasons_df)} mv={len(mvs_df)}")

    m = build_feature_matrix(players_df, seasons_df, mvs_df, cutoff_age=19)
    out = "data/processed/features_demo.parquet"
    m.to_parquet(out, index=False)
    print(f"wrote {out}  shape={m.shape}  features={len(feature_columns(m))}")
    print(m["target"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)

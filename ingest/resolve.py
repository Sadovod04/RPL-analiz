"""Entity resolution: merge records for the same physical player across sources.

Blocking key: birth year. Within a block, fuzzy-match normalized names
(order-insensitive, Cyrillic transliterated to Latin) and cluster with union-find.
Each cluster gets a stable ``player_id`` = sha1 of its canonical (name, birth_date).

Pure/deterministic — unit-tested with synthetic multi-source records; no I/O.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz

_CYR2LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

MATCH_THRESHOLD = 87.0  # rapidfuzz token_sort_ratio


def translit(text: str) -> str:
    return "".join(_CYR2LAT.get(ch, ch) for ch in text.lower())


def normalize_name(name: str) -> str:
    """Lowercase, de-accent, 'Last, First' -> 'first last', translit Cyrillic, strip punct."""
    if not name:
        return ""
    s = name.strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    s = translit(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _sorted_key(norm: str) -> str:
    return " ".join(sorted(norm.split()))


def _birth_year(rec: dict) -> int | None:
    bd = rec.get("birth_date")
    if bd is None:
        return None
    if hasattr(bd, "year"):
        return bd.year
    m = re.search(r"\b(19|20)\d{2}\b", str(bd))
    return int(m.group()) if m else None


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class ResolveConfig:
    threshold: float = MATCH_THRESHOLD


def _player_id(canonical_name: str, birth_year: int | None) -> str:
    raw = f"{_sorted_key(normalize_name(canonical_name))}|{birth_year or '?'}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def resolve_players(records: list[dict], cfg: ResolveConfig | None = None) -> pd.DataFrame:
    """``records`` need: source, source_id, full_name; optional name_home_country, birth_date.

    Returns a DataFrame with all input columns plus ``norm_name``, ``birth_year``,
    ``player_id``, ``match_score``.
    """
    cfg = cfg or ResolveConfig()
    df = pd.DataFrame(records).copy()
    if df.empty:
        return df.assign(player_id=[], match_score=[])

    df["birth_year"] = df.apply(_birth_year, axis=1)
    df["norm_name"] = df["full_name"].fillna("").map(normalize_name)
    df["norm_home"] = (
        df.get("name_home_country", pd.Series([None] * len(df))).fillna("").map(normalize_name)
    )

    uf = _UnionFind(len(df))
    scores = [100.0] * len(df)
    idx = list(df.index)

    for block, sub in df.groupby(df["birth_year"].fillna(-1)):
        members = list(sub.index)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                s = max(
                    fuzz.token_sort_ratio(df.at[a, "norm_name"], df.at[b, "norm_name"]),
                    fuzz.token_sort_ratio(df.at[a, "norm_name"], df.at[b, "norm_home"]),
                    fuzz.token_sort_ratio(df.at[a, "norm_home"], df.at[b, "norm_name"]),
                )
                # unknown birth year -> require a stricter match
                needed = cfg.threshold + (6 if block == -1 else 0)
                if s >= needed:
                    uf.union(idx.index(a), idx.index(b))
                    scores[idx.index(a)] = min(scores[idx.index(a)], s)
                    scores[idx.index(b)] = min(scores[idx.index(b)], s)

    cluster_root = {i: uf.find(i) for i in range(len(df))}
    # canonical name per cluster = longest full_name (most complete)
    root_to_rows: dict[int, list[int]] = {}
    for pos, root in cluster_root.items():
        root_to_rows.setdefault(root, []).append(pos)

    pid_by_pos: dict[int, str] = {}
    for positions in root_to_rows.values():
        names = [df.iloc[p]["full_name"] or "" for p in positions]
        years = [df.iloc[p]["birth_year"] for p in positions if pd.notna(df.iloc[p]["birth_year"])]
        canonical = max(names, key=len)
        by = int(years[0]) if years else None
        pid = _player_id(canonical, by)
        for p in positions:
            pid_by_pos[p] = pid

    df = df.reset_index(drop=True)
    df["player_id"] = [pid_by_pos[i] for i in range(len(df))]
    df["match_score"] = scores
    return df.drop(columns=["norm_home"])


def build_crosswalk(resolved: pd.DataFrame) -> pd.DataFrame:
    return resolved[["source", "source_id", "player_id", "match_score"]].drop_duplicates()

"""Layer 0 -- scoring and per-case ranking.

score()      : one row -> scalar, given theta + registry (+ masked groups)
score_frame(): adds a 'score' column to a wide-table DataFrame
rank_case()  : collapse to variant/gene/row level, stable-sort by score desc
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from ._util import variant_key
from .contributors import ADDITIVE, GATE, GENE_PRIOR, Contributor


def default_theta(registry: list[Contributor]) -> dict[str, float]:
    """All-ones theta -> reproduces the README raw_pathogenic_score exactly."""
    return {c.key(): 1.0 for c in registry}


def score(row: dict, theta: dict, registry: list[Contributor],
          masked_groups: frozenset = frozenset()) -> float:
    additive = 0.0
    gate = 1.0
    for c in registry:
        if c.group is not None and c.group in masked_groups:
            continue
        w = theta.get(c.key(), 1.0)
        if c.kind in (ADDITIVE, GENE_PRIOR):
            additive += w * c.calibrate(row)
        elif c.kind == GATE:
            gate *= c.calibrate(row)  # B-class factor in [0,1]; not used in v1
    return gate * additive


def score_frame(df: pd.DataFrame, theta: dict, registry: list[Contributor],
                masked_groups: frozenset = frozenset()) -> pd.DataFrame:
    out = df.copy()
    recs = out.to_dict("records")
    out["score"] = [score(r, theta, registry, masked_groups) for r in recs]
    return out


def _collapse(scored: pd.DataFrame, level: str) -> pd.DataFrame:
    """Collapse transcript rows to the ranking unit, keeping max score.

    'variant' -> one entry per (chrom,pos,ref,alt)  (default ranking unit)
    'gene'    -> one entry per gene_symbol
    'row'     -> no collapse (rank raw transcript rows)
    Returns a frame with columns: _unit, score, gene_symbol, _vkey, _order.
    """
    s = scored.reset_index(drop=True).copy()
    s["_order"] = range(len(s))  # preserves original (VEP) order for tie-break
    s["_vkey"] = [variant_key(r.chrom, r.pos, r.ref, r.alt)
                  for r in s.itertuples(index=False)]
    if level == "row":
        s["_unit"] = s["_order"]
        return s
    by = "_vkey" if level == "variant" else "gene_symbol"
    # max score per unit; keep the earliest original order among that unit's rows
    idx = s.groupby(by, sort=False)["score"].idxmax()
    out = s.loc[idx].copy()
    out["_unit"] = out[by]
    return out


def rank_case(scored: pd.DataFrame, level: str = "variant") -> pd.DataFrame:
    """Return units sorted by score desc, ties broken by original order.

    Adds a 1-based 'rank' column. Stable mergesort preserves VEP order on ties,
    matching the README's pathogenic_rank behaviour.
    """
    collapsed = _collapse(scored, level)
    ranked = collapsed.sort_values(
        ["score", "_order"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def truth_rank(ranked: pd.DataFrame, truth_units: Iterable, level: str) -> Optional[int]:
    """Best (smallest) 1-based rank achieved by any truth unit; None if absent."""
    truth = set(truth_units)
    hits = ranked[ranked["_unit"].isin(truth)]["rank"]
    return int(hits.min()) if len(hits) else None

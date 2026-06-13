"""Layer 0 -- scoring, feature extraction, and per-case ranking.

KEY CHANGE vs v1: calibration ("how strong is this evidence", expensive string
parsing) is split from combination ("apply theta", cheap linear algebra).

  extract_features(df, registry) -> FeatureBundle   # parse ONCE, persist
  apply_theta(bundle, theta, masked) -> scores       # vectorized, run N times
  rank_units(bundle, scores, level)  -> ranked units # collapse + stable sort

This is what makes the optimizer (Layer 3) and agent (Layer 4) tractable at
9 GB scale: the 9 GB table becomes a small (n_rows x n_contributors) matrix that
the inner loop hits thousands of times without ever re-parsing.

theta is a dict of NON-NEGATIVE magnitude multipliers (one per contributor).
Direction lives in calibrate(); theta only scales. theta == all 1.0 still
reproduces the README raw_pathogenic_score exactly (the regression anchor).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ._util import to_int, variant_key
from .contributors import ADDITIVE, GATE, GENE_PRIOR, Contributor

_IDENTITY = ("chrom", "pos", "ref", "alt")
_INDEX_EXTRA = ("gene_symbol", "tx_rank_within_variant", "tx_selected_reason")


# --------------------------------------------------------------------------
# theta helpers (the optimizer/agent interface)
# --------------------------------------------------------------------------
def theta_keys(registry: list[Contributor]) -> list[str]:
    return [c.key() for c in registry]


def default_theta(registry: list[Contributor]) -> dict[str, float]:
    """All-ones theta -> reproduces the README raw_pathogenic_score exactly."""
    return {c.key(): 1.0 for c in registry}


def default_bounds(registry: list[Contributor]) -> dict[str, tuple[float, float]]:
    """Non-negative magnitude bounds. >=0 preserves evidence sign & auditability.

    The optimizer/agent search inside these; widen per-key as needed but keeping
    low >= 0 is what guarantees a learned weight is still interpretable as
    'we trust this evidence Nx'.
    """
    return {c.key(): (0.0, 5.0) for c in registry}


def theta_to_vector(theta: dict, keys: list[str]) -> np.ndarray:
    return np.array([float(theta.get(k, 1.0)) for k in keys], dtype=float)


def vector_to_theta(vec, keys: list[str]) -> dict:
    return {k: float(v) for k, v in zip(keys, vec)}


def needed_columns(registry: list[Contributor]) -> list[str]:
    """Exact column projection for loading: identity + index + every reads col.

    Makes Contributor.reads load-bearing (was documentation-only): at 9 GB you
    read only these columns, not the whole table.
    """
    cols = list(_IDENTITY) + list(_INDEX_EXTRA)
    for c in registry:
        cols.extend(c.reads)
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# --------------------------------------------------------------------------
# the precompute artifact
# --------------------------------------------------------------------------
@dataclass
class FeatureBundle:
    """Compact, persistable result of parsing one case's wide table.

    F: (n_rows, n_contributors) calibrated sub-scores -- the only thing the
       optimizer/agent loop ever touches.
    index: tiny frame (vkey, gene, original order, tx rank) for collapse/rank.
    """
    F: np.ndarray
    keys: list[str]
    kinds: list[str]
    groups: list[Optional[str]]
    index: pd.DataFrame

    def col(self, key: str) -> np.ndarray:
        return self.F[:, self.keys.index(key)]


def extract_features(df: pd.DataFrame, registry: list[Contributor]) -> FeatureBundle:
    """Run every calibrate() ONCE per row. This is the only expensive step;
    everything downstream is linear algebra on F."""
    recs = df.to_dict("records")          # at true scale: chunk + project columns
    n, m = len(recs), len(registry)
    F = np.zeros((n, m), dtype=float)
    for j, c in enumerate(registry):
        cal = c.calibrate
        F[:, j] = [cal(r) for r in recs]
    idx = pd.DataFrame({
        "_order": np.arange(n),
        "_vkey": [variant_key(r.get("chrom"), r.get("pos"), r.get("ref"), r.get("alt"))
                  for r in recs],
        "gene_symbol": [r.get("gene_symbol") for r in recs],
    })
    if "tx_rank_within_variant" in df.columns:
        idx["tx_rank"] = [to_int(r.get("tx_rank_within_variant")) for r in recs]
    return FeatureBundle(F=F, keys=theta_keys(registry),
                         kinds=[c.kind for c in registry],
                         groups=[c.group for c in registry], index=idx)


def apply_theta(bundle: FeatureBundle, theta: dict,
                masked_groups: frozenset = frozenset()) -> np.ndarray:
    """Vectorized score per row. additive cols: F @ theta; gate cols: product.
    Masking: additive masked -> 0 contribution; gate masked -> factor 1."""
    keys, kinds, groups = bundle.keys, bundle.kinds, bundle.groups
    tvec = theta_to_vector(theta, keys)
    is_add = np.array([k in (ADDITIVE, GENE_PRIOR) for k in kinds])
    is_gate = np.array([k == GATE for k in kinds])
    masked = np.array([g is not None and g in masked_groups for g in groups])

    add_theta = tvec.copy()
    add_theta[~is_add] = 0.0
    add_theta[masked & is_add] = 0.0
    additive = bundle.F @ add_theta

    if is_gate.any():
        G = bundle.F[:, is_gate].copy()
        G[:, masked[is_gate]] = 1.0          # masked gate is identity, not 0
        gate = np.prod(G, axis=1)
    else:
        gate = np.ones(bundle.F.shape[0])
    return gate * additive


# --------------------------------------------------------------------------
# ranking (collapse respects the transcript-selection layer when present)
# --------------------------------------------------------------------------
def _variant_reps(idx: pd.DataFrame, collapse: str) -> pd.DataFrame:
    """One representative row per variant.

    'auto'/'selected_transcript': if tx_rank present, prefer tx_rank==1, else
        fall back to the highest-scoring transcript within that variant.
        (Avoids 'transcript shopping': a variant cannot climb just because some
        non-selected transcript happens to score high.)
    'max': highest-scoring transcript per variant (v1 legacy behaviour).
    """
    have_tx = "tx_rank" in idx.columns and idx["tx_rank"].notna().any()
    if collapse in ("auto", "selected_transcript") and have_tx:
        # rank key: selected (tx_rank==1) first, then by tx_rank, then score
        idx = idx.copy()
        idx["_sel"] = (idx["tx_rank"] == 1).astype(int)
        idx = idx.sort_values(["_vkey", "_sel", "score"],
                              ascending=[True, False, False], kind="mergesort")
        return idx.groupby("_vkey", sort=False).head(1)
    # max-score collapse
    keep = idx.groupby("_vkey", sort=False)["score"].idxmax()
    return idx.loc[keep]


def rank_units(bundle: FeatureBundle, scores: np.ndarray, level: str = "variant",
               collapse: str = "auto") -> pd.DataFrame:
    """Collapse to ranking unit, stable-sort desc. Adds 1-based 'rank' and keeps
    '_rep_row' (the original row index that represents the unit) for explain."""
    idx = bundle.index.copy()
    idx["score"] = scores
    if level == "row":
        idx["_unit"] = idx["_order"]
        idx["_rep_row"] = idx["_order"]
    else:
        reps = _variant_reps(idx, collapse)        # one row per variant
        if level == "variant":
            reps = reps.copy()
            reps["_unit"] = reps["_vkey"]
        elif level == "gene":
            keep = reps.groupby("gene_symbol", sort=False)["score"].idxmax()
            reps = reps.loc[keep].copy()
            reps["_unit"] = reps["gene_symbol"]
        else:
            raise ValueError(level)
        reps["_rep_row"] = reps["_order"]
        idx = reps
    ranked = idx.sort_values(["score", "_order"], ascending=[False, True],
                             kind="mergesort").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def truth_rank(ranked: pd.DataFrame, truth_units: Iterable, level: str = "variant") -> Optional[int]:
    truth = set(truth_units)
    hits = ranked[ranked["_unit"].isin(truth)]["rank"]
    return int(hits.min()) if len(hits) else None


# --------------------------------------------------------------------------
# convenience single-row / single-frame API (kept for ad-hoc use & tests)
# --------------------------------------------------------------------------
def score(row: dict, theta: dict, registry: list[Contributor],
          masked_groups: frozenset = frozenset()) -> float:
    """One row -> scalar. Identical maths to apply_theta, kept for tests."""
    additive, gate = 0.0, 1.0
    for c in registry:
        if c.group is not None and c.group in masked_groups:
            continue
        w = theta.get(c.key(), 1.0)
        if c.kind in (ADDITIVE, GENE_PRIOR):
            additive += w * c.calibrate(row)
        elif c.kind == GATE:
            gate *= c.calibrate(row)
    return gate * additive


def score_frame(df: pd.DataFrame, theta: dict, registry: list[Contributor],
                masked_groups: frozenset = frozenset()) -> pd.DataFrame:
    """Adds a 'score' column. Now routed through extract+apply for consistency."""
    bundle = extract_features(df, registry)
    out = df.copy()
    out["score"] = apply_theta(bundle, theta, masked_groups)
    return out

"""Vectorized metric fast-path for Layer 3.

The collapse grouping (row->variant, variant->gene) and tie-break order do NOT
depend on theta, so they are precomputed ONCE per case. Per-theta we then only
need the TRUTH unit's rank, which is:

    rank = 1 + #units strictly better than the best truth unit
         where "better" = (score > s_t) or (score == s_t and order < order_t)

No pandas, no full sort -- microseconds per case. Used for recall@k / MRR;
the pandas rank_units() is kept for human-readable explain dumps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

_TOL = 1e-9


@dataclass
class FastIndex:
    v_codes: np.ndarray        # row -> variant-group id
    v_order: np.ndarray        # variant-group -> min original order (tie-break)
    v_rep: np.ndarray | None   # variant-group -> representative row (selected mode)
    v_truth: np.ndarray        # truth variant-group ids
    vg_gene: np.ndarray        # variant-group -> gene-group id
    g_codes: np.ndarray        # row -> gene-group id
    g_order: np.ndarray        # gene-group -> min original order
    g_truth: np.ndarray        # truth gene-group ids
    selected: bool             # True = tx-selected collapse, False = max collapse


def _seg_min(values, codes, n):
    out = np.full(n, np.inf)
    np.minimum.at(out, codes, values)
    return out


def _seg_max(values, codes, n):
    out = np.full(n, -np.inf)
    np.maximum.at(out, codes, values)
    return out


def build_fast_index(bundle, truth_variant, truth_gene, collapse="auto") -> FastIndex:
    idx = bundle.index
    order = idx["_order"].to_numpy()
    vkeys = list(idx["_vkey"])
    genes = list(idx["gene_symbol"])

    v_codes, v_uniques = pd.factorize(pd.Index(vkeys), sort=False)
    G_v = len(v_uniques)
    g_codes, g_uniques = pd.factorize(pd.Index(genes), sort=False)
    G_g = len(g_uniques)

    v_order = _seg_min(order, v_codes, G_v)
    g_order = _seg_min(order, g_codes, G_g)

    v_truth = np.array([i for i, u in enumerate(v_uniques) if u in truth_variant], dtype=int)
    g_truth = np.array([i for i, u in enumerate(g_uniques) if u in truth_gene], dtype=int)

    # variant-group -> gene-group (first row of each variant decides its gene)
    vg_gene = np.full(G_v, -1, dtype=int)
    for r in range(len(v_codes)):
        vg = v_codes[r]
        if vg_gene[vg] < 0:
            vg_gene[vg] = g_codes[r]

    selected = (collapse in ("auto", "selected_transcript")
                and "tx_rank" in idx.columns and idx["tx_rank"].notna().any())
    v_rep = None
    if selected:
        tx = idx["tx_rank"].to_numpy()
        tx = np.where(pd.isna(tx), np.inf, tx).astype(float)
        v_rep = np.full(G_v, -1, dtype=int)
        best = np.full(G_v, np.inf)
        for r in range(len(v_codes)):
            vg = v_codes[r]
            if tx[r] < best[vg]:
                best[vg] = tx[r]
                v_rep[vg] = r
    return FastIndex(v_codes, v_order, v_rep, v_truth, vg_gene,
                     g_codes, g_order, g_truth, selected)


def _rank_from(unit_score, unit_order, truth_ids):
    if len(truth_ids) == 0:
        return None
    s_t = unit_score[truth_ids].max()
    cand = truth_ids[np.isclose(unit_score[truth_ids], s_t, atol=_TOL)]
    o_t = unit_order[cand].min()
    better = (unit_score > s_t + _TOL) | (
        np.isclose(unit_score, s_t, atol=_TOL) & (unit_order < o_t))
    return int(better.sum()) + 1


def fast_ranks(fi: FastIndex, row_scores: np.ndarray) -> dict:
    G_v = len(fi.v_order)
    if fi.selected:
        v_score = row_scores[fi.v_rep]
    else:
        v_score = _seg_max(row_scores, fi.v_codes, G_v)
    g_score = _seg_max(v_score, fi.vg_gene, len(fi.g_order))
    return {
        "variant": _rank_from(v_score, fi.v_order, fi.v_truth),
        "gene": _rank_from(g_score, fi.g_order, fi.g_truth),
    }

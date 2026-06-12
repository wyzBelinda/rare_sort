"""Layer 1 -- the evaluation harness.

This is the single source of truth (the fitness function) that every later
layer -- numerical optimizer, agent -- optimizes against. Pure and
deterministic: same (cases, theta) -> same metrics.

Core ideas wired in:
  * supervision = per-CASE ranking of the known causal variant, not pointwise
    classification. Primary metrics: recall@k and MRR.
  * reported at BOTH variant level and gene level (rare-disease cares about
    "did the causal gene surface" as well as the exact variant).
  * leakage controls as toggles -> MASK_CONDITIONS gives the clinvar x
    phenotype 2x2. (phenotype mask is a no-op until a 'phenotype' contributor
    exists, but the plumbing is here so it costs nothing later.)
  * splits are BY CASE, never by variant.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

from ._util import variant_key
from .contributors import Contributor
from .scorer import rank_case, score_frame, truth_rank

DEFAULT_KS = (1, 5, 10, 20, 50)

# The clinvar x phenotype 2x2. Pick the condition that matches your deployment
# goal as the PRIMARY objective (e.g. 'both_masked' for novel discovery).
MASK_CONDITIONS = {
    "none": frozenset(),
    "clinvar_masked": frozenset({"clinvar"}),
    "phenotype_masked": frozenset({"phenotype"}),
    "both_masked": frozenset({"clinvar", "phenotype"}),
}


@dataclass
class Case:
    """One sample / one VCF's wide table, plus its ground truth."""
    case_id: str
    df: pd.DataFrame
    causal_variants: list[tuple]              # variant_key tuples
    causal_genes: Optional[set] = None        # derived from df if None

    def __post_init__(self):
        self.causal_variants = [
            v if isinstance(v, tuple) and len(v) == 4 else variant_key(*v)
            for v in self.causal_variants
        ]
        if self.causal_genes is None:
            self.causal_genes = self._derive_genes()

    def _derive_genes(self) -> set:
        truth = set(self.causal_variants)
        genes = set()
        for r in self.df.itertuples(index=False):
            if variant_key(r.chrom, r.pos, r.ref, r.alt) in truth:
                if not pd.isna(getattr(r, "gene_symbol", None)):
                    genes.add(r.gene_symbol)
        return genes

    def truth_units(self, level: str):
        if level == "gene":
            return self.causal_genes
        if level == "variant":
            return set(self.causal_variants)
        # row level: truth = any row index whose variant is causal
        truth = set(self.causal_variants)
        return {i for i, r in enumerate(self.df.itertuples(index=False))
                if variant_key(r.chrom, r.pos, r.ref, r.alt) in truth}


def _agg(ranks: list[Optional[int]], ks: Iterable[int]) -> dict:
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    out = {"n_cases": n, "n_found": len(found)}
    for k in ks:
        out[f"recall@{k}"] = sum(r <= k for r in found) / n if n else 0.0
    out["MRR"] = sum(1.0 / r for r in found) / n if n else 0.0
    out["median_rank"] = float(pd.Series(found).median()) if found else None
    out["mean_rank"] = float(pd.Series(found).mean()) if found else None
    return out


def evaluate(cases: list[Case], theta: dict, registry: list[Contributor],
             masked_groups: frozenset = frozenset(),
             levels=("variant", "gene"), ks: Iterable[int] = DEFAULT_KS) -> dict:
    """Score & rank every case, aggregate recall@k / MRR per level."""
    per_level_ranks = {lvl: [] for lvl in levels}
    per_case = []
    for case in cases:
        scored = score_frame(case.df, theta, registry, masked_groups)
        row = {"case_id": case.case_id}
        for lvl in levels:
            ranked = rank_case(scored, level=lvl)
            r = truth_rank(ranked, case.truth_units(lvl), lvl)
            per_level_ranks[lvl].append(r)
            row[f"{lvl}_rank"] = r
        per_case.append(row)
    return {
        "masked_groups": sorted(masked_groups),
        "levels": {lvl: _agg(per_level_ranks[lvl], ks) for lvl in levels},
        "per_case": pd.DataFrame(per_case),
    }


def evaluate_all_conditions(cases, theta, registry, **kw) -> dict:
    """Run the full clinvar x phenotype 2x2 in one call."""
    return {name: evaluate(cases, theta, registry, masked_groups=mg, **kw)
            for name, mg in MASK_CONDITIONS.items()}


def load_cases(manifest: pd.DataFrame, wide_table_dir: str = ".",
               na_values=("-",)) -> list[Case]:
    """Build Cases from a truth manifest + per-case wide-table CSVs.

    manifest columns (one row PER CAUSAL VARIANT; multiple rows per case ok):
        case_id, csv_file, chrom, pos, ref, alt
    csv_file is resolved relative to wide_table_dir. All columns are read as
    strings (dtype=str) so '-' stays '-' and AF scientific notation is exact.
    """
    import os
    cases = []
    for case_id, grp in manifest.groupby("case_id", sort=False):
        path = os.path.join(wide_table_dir, str(grp.iloc[0]["csv_file"]))
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=list(na_values))
        df = df.fillna("-")
        truth = [variant_key(r.chrom, r.pos, r.ref, r.alt)
                 for r in grp.itertuples(index=False)]
        cases.append(Case(str(case_id), df, causal_variants=truth))
    return cases


def case_cv_folds(cases: list[Case], n_folds: int = 5, seed: int = 0):
    """Yield (train_cases, test_cases) split BY CASE (never by variant)."""
    ids = [c.case_id for c in cases]
    rng = random.Random(seed)
    rng.shuffle(ids)
    by_id = {c.case_id: c for c in cases}
    folds = [ids[i::n_folds] for i in range(n_folds)]
    for f in range(n_folds):
        test_ids = set(folds[f])
        train = [by_id[i] for i in ids if i not in test_ids]
        test = [by_id[i] for i in folds[f]]
        yield train, test

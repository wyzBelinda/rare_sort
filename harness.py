"""Layer 1 -- evaluation harness, plus the Layer 3/4 hooks.

Single source of truth (fitness function). Pure & deterministic.

What changed for scale + optimizer/agent compatibility:
  * prepare(): extract features ONCE per case -> Prepared. evaluate() and the
    2x2 condition sweep reuse these bundles instead of re-parsing. This is the
    fast path the optimizer hammers: objective(theta) only does linear algebra.
  * explain_failures(): structured per-contributor breakdown of the causal unit
    vs the decoys outranking it -- the failure dump the agent reasons over.
  * bootstrap_ci(): case-resampled CI on any metric -- the small-N guardrail
    that stops the agent chasing noise.
  * describe_model(): a frozen, signed, JSON-able model artifact (registry
    signature + theta + metrics + rationale) -- provenance for Layer 4.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from ._util import variant_key
from .contributors import Contributor
from .fastrank import FastIndex, build_fast_index, fast_ranks
from .scorer import (FeatureBundle, apply_theta, extract_features,
                     needed_columns, rank_units, theta_keys, truth_rank)

DEFAULT_KS = (1, 5, 10, 20, 50)

MASK_CONDITIONS = {
    "none": frozenset(),
    "clinvar_masked": frozenset({"clinvar"}),
    "phenotype_masked": frozenset({"phenotype"}),
    "both_masked": frozenset({"clinvar", "phenotype"}),
}


@dataclass
class Case:
    case_id: str
    df: pd.DataFrame
    causal_variants: list[tuple]
    causal_genes: Optional[set] = None

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
                g = getattr(r, "gene_symbol", None)
                if not pd.isna(g):
                    genes.add(g)
        return genes


@dataclass
class Prepared:
    """A Case with its features already extracted (parse-once artifact)."""
    case_id: str
    bundle: FeatureBundle
    truth_variant: set
    truth_gene: set
    fast: Optional[FastIndex] = None        # vectorized metric index (Layer 3)

    def truth_units(self, level: str):
        if level == "gene":
            return self.truth_gene
        if level == "variant":
            return self.truth_variant
        idx = self.bundle.index
        return set(idx.loc[idx["_vkey"].isin(self.truth_variant), "_order"])


def prepare(cases: list[Case], registry: list[Contributor],
            collapse: str = "auto") -> list[Prepared]:
    out = []
    for c in cases:
        b = extract_features(c.df, registry)
        tv, tg = set(c.causal_variants), set(c.causal_genes)
        out.append(Prepared(c.case_id, b, tv, tg,
                            fast=build_fast_index(b, tv, tg, collapse)))
    return out


def _as_prepared(items, registry) -> list[Prepared]:
    if items and isinstance(items[0], Prepared):
        return items
    if registry is None:
        raise ValueError("registry required when passing raw Cases")
    return prepare(items, registry)


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


def evaluate(items, theta: dict, registry: list[Contributor] = None,
             masked_groups: frozenset = frozenset(),
             levels=("variant", "gene"), ks: Iterable[int] = DEFAULT_KS,
             collapse: str = "auto") -> dict:
    """Accepts Cases (auto-prepared) OR Prepared (fast path, no re-parse)."""
    prepared = _as_prepared(items, registry)
    per_level = {lvl: [] for lvl in levels}
    per_case = []
    fast_ok = collapse == "auto"
    for p in prepared:
        scores = apply_theta(p.bundle, theta, masked_groups)
        rec = {"case_id": p.case_id}
        fr = fast_ranks(p.fast, scores) if (fast_ok and p.fast is not None) else None
        for lvl in levels:
            if fr is not None and lvl in fr:
                r = fr[lvl]
            else:
                ranked = rank_units(p.bundle, scores, level=lvl, collapse=collapse)
                r = truth_rank(ranked, p.truth_units(lvl), lvl)
            per_level[lvl].append(r)
            rec[f"{lvl}_rank"] = r
        per_case.append(rec)
    return {
        "masked_groups": sorted(masked_groups),
        "levels": {lvl: _agg(per_level[lvl], ks) for lvl in levels},
        "per_case": pd.DataFrame(per_case),
    }


def evaluate_all_conditions(items, theta, registry=None, **kw) -> dict:
    """clinvar x phenotype 2x2 -- features extracted once, reused 4x."""
    prepared = _as_prepared(items, registry)
    return {name: evaluate(prepared, theta, masked_groups=mg, **kw)
            for name, mg in MASK_CONDITIONS.items()}


# --------------------------------------------------------------------------
# Layer 4 hook: structured failure dump for the agent to reason over
# --------------------------------------------------------------------------
def _contribs(bundle: FeatureBundle, row_idx: int, theta: dict,
              masked_groups: frozenset) -> dict:
    out = {}
    for j, key in enumerate(bundle.keys):
        g = bundle.groups[j]
        masked = g is not None and g in masked_groups
        w = 1.0 if masked else float(theta.get(key, 1.0))
        out[key] = round(w * float(bundle.F[row_idx, j]) * (0.0 if masked else 1.0), 3)
    return out


def explain_failures(items, theta, registry=None, k: int = 10,
                     level: str = "variant", masked_groups: frozenset = frozenset(),
                     collapse: str = "auto", max_blockers: int = 5) -> list[dict]:
    """For every case whose causal unit ranks worse than k, return the causal
    unit's per-contributor contributions and those of the units outranking it."""
    prepared = _as_prepared(items, registry)
    failures = []
    for p in prepared:
        scores = apply_theta(p.bundle, theta, masked_groups)
        ranked = rank_units(p.bundle, scores, level=level, collapse=collapse)
        truth = p.truth_units(level)
        hit = ranked[ranked["_unit"].isin(truth)]
        if len(hit) == 0:
            failures.append({"case_id": p.case_id, "causal_rank": None,
                             "reason": "causal unit absent from table"})
            continue
        crank = int(hit["rank"].min())
        if crank <= k:
            continue
        crow = int(hit.sort_values("rank").iloc[0]["_rep_row"])
        blockers = ranked[ranked["rank"] < crank].head(max_blockers)
        failures.append({
            "case_id": p.case_id,
            "causal_rank": crank,
            "causal_unit": str(hit.sort_values("rank").iloc[0]["_unit"]),
            "causal_score": round(float(hit["score"].max()), 3),
            "causal_contributions": _contribs(p.bundle, crow, theta, masked_groups),
            "blockers": [
                {"unit": str(b._unit), "score": round(float(b.score), 3),
                 "contributions": _contribs(p.bundle, int(b._rep_row), theta, masked_groups)}
                for b in blockers.itertuples(index=False)
            ],
        })
    return failures


# --------------------------------------------------------------------------
# Layer 3/4 guardrail: case-resampled bootstrap CI on any scalar metric
# --------------------------------------------------------------------------
def bootstrap_ci(items, theta, metric_fn: Callable[[dict], float],
                 registry=None, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05, **eval_kw) -> dict:
    prepared = _as_prepared(items, registry)
    point = metric_fn(evaluate(prepared, theta, **eval_kw))
    rng = random.Random(seed)
    n = len(prepared)
    boots = []
    for _ in range(n_boot):
        sample = [prepared[rng.randrange(n)] for _ in range(n)]
        boots.append(metric_fn(evaluate(sample, theta, **eval_kw)))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"point": float(point), "lo": lo, "hi": hi, "n_cases": n, "n_boot": n_boot}


# --------------------------------------------------------------------------
# Layer 4 hook: frozen, signed model artifact (provenance)
# --------------------------------------------------------------------------
def describe_model(registry: list[Contributor], theta: dict,
                   metrics: dict = None, data_snapshot: str = None,
                   rationale: str = None) -> dict:
    sig_src = [{"name": c.name, "kind": c.kind, "group": c.group,
                "reads": list(c.reads)} for c in registry]
    payload = {"registry": sig_src, "theta": {k: float(theta.get(k, 1.0))
                                              for k in theta_keys(registry)}}
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return {
        "version_signature": signature,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registry": sig_src,
        "theta": payload["theta"],
        "metrics": metrics or {},
        "data_snapshot": data_snapshot,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------
# loading (with column projection) + by-case CV
# --------------------------------------------------------------------------
def load_cases(manifest: pd.DataFrame, wide_table_dir: str = ".",
               registry: list[Contributor] = None, na_values=("-",)) -> list[Case]:
    """manifest: one row per causal variant (case_id, csv_file, chrom,pos,ref,alt).
    If registry is given, only its needed_columns() are read (column projection
    -- the difference between loading 9 GB and loading a few hundred MB)."""
    import os
    usecols = None
    if registry is not None:
        want = set(needed_columns(registry))
        usecols = lambda c: c in want  # noqa: E731
    cases = []
    for case_id, grp in manifest.groupby("case_id", sort=False):
        path = os.path.join(wide_table_dir, str(grp.iloc[0]["csv_file"]))
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         na_values=list(na_values), usecols=usecols)
        df = df.fillna("-")
        truth = [variant_key(r.chrom, r.pos, r.ref, r.alt)
                 for r in grp.itertuples(index=False)]
        cases.append(Case(str(case_id), df, causal_variants=truth))
    return cases


def case_cv_folds(cases, n_folds: int = 5, seed: int = 0):
    ids = [c.case_id for c in cases]
    rng = random.Random(seed)
    rng.shuffle(ids)
    by_id = {c.case_id: c for c in cases}
    folds = [ids[i::n_folds] for i in range(n_folds)]
    for f in range(n_folds):
        test_ids = set(folds[f])
        yield ([by_id[i] for i in ids if i not in test_ids],
               [by_id[i] for i in folds[f]])

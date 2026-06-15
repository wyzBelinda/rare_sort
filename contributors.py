"""Layer 0 -- the contributor registry.

A Contributor is one evidence source. It declares:
  - name        : stable id, also the default weight key in theta
  - group       : masking group ("clinvar", "phenotype", ...). None = unmaskable.
  - kind        : ADDITIVE | GATE | GENE_PRIOR  (v1 only uses ADDITIVE)
  - calibrate() : row(dict) -> raw sub-score, using FIXED domain bucket maps

The split is deliberate:
  * calibrate() holds the domain-driven bucket maps (the "calibration layer").
    These stay fixed; they encode "how strong is this evidence".
  * theta holds ONE multiplier per contributor (the "combination weights").
    These are what the optimizer/agent later tunes.

With theta == all 1.0, score() reproduces the README's raw_pathogenic_score
exactly. That is the v1 contract and the regression anchor.

Adding a new feature later == append one Contributor (+ an adapter that joins
its source table and emits the columns it reads). scorer.py never changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ._util import is_missing, split_terms, to_float, to_int, norm_token

ADDITIVE = "ADDITIVE"
GATE = "GATE"          # reserved for B-class quality/validity gates (not v1)
GENE_PRIOR = "GENE_PRIOR"  # reserved for joined gene-level priors (not v1)


@dataclass(frozen=True)
class Contributor:
    name: str
    calibrate: Callable[[dict], float]
    kind: str = ADDITIVE
    group: Optional[str] = None          # masking group; None = never masked
    weight_key: Optional[str] = None     # defaults to name
    reads: tuple[str, ...] = field(default_factory=tuple)  # documentation only

    def key(self) -> str:
        return self.weight_key or self.name


# --------------------------------------------------------------------------
# 1. clinvar_score
# --------------------------------------------------------------------------
_SIG_BASE = {
    "pathogenic": 40,
    "likely_pathogenic": 30,
    "pathogenic_likely_pathogenic": 35,
    "uncertain_significance": 3,
    "likely_benign": -25,
    "benign": -35,
    "benign_likely_benign": -30,
}
_BENIGN_ADJUST = {
    "benign": 0.95,
    "likely_benign": 0.90,
    "benign_likely_benign": 0.925,
}
_STAR_FACTOR = {4: 1.20, 3: 1.15, 2: 1.10, 1: 1.00, 0: 1.00}


def _sig_base(sig_norm: str) -> Optional[int]:
    if sig_norm in _SIG_BASE:
        return _SIG_BASE[sig_norm]
    # conflicting interpretations / classifications -> 0 (any release wording)
    if "conflicting" in sig_norm:
        return 0
    if "uncertain" in sig_norm:
        return 3
    return None


def _review_factor(review_norm: str) -> float:
    if "practice_guideline" in review_norm or "expert_panel" in review_norm:
        return 1.15
    if "multiple_submitters" in review_norm and "no_conflicts" in review_norm:
        return 1.10
    return 1.00


def _clinvar(row: dict) -> float:
    sig = norm_token(row.get("clinvar_significance"))
    base = _sig_base(sig)
    if base is None or base == 0:
        return 0.0
    star = to_int(row.get("clinvar_star_rating")) or 0
    star = max(0, min(4, star))
    star_factor = _STAR_FACTOR[star]
    review_factor = _review_factor(norm_token(row.get("clinvar_review_status")))
    benign_adjust = _BENIGN_ADJUST.get(sig, 1.0)
    return float(round(base * star_factor * review_factor * benign_adjust))


# --------------------------------------------------------------------------
# 2. consequence_score
# --------------------------------------------------------------------------
_CONS_BASE = {
    "transcript_ablation": 30,
    "splice_acceptor_variant": 28,
    "splice_donor_variant": 28,
    "stop_gained": 28,
    "frameshift_variant": 28,
    "stop_lost": 24,
    "start_lost": 22,
    "missense_variant": 12,
    "protein_altering_variant": 12,
    "inframe_insertion": 10,
    "inframe_deletion": 10,
    "splice_region_variant": 6,
    "synonymous_variant": 2,
    "intron_variant": 0,
    "upstream_gene_variant": -3,
    "downstream_gene_variant": -3,
    "intergenic_variant": -5,
}
_IMPACT_SCORE = {"HIGH": 5, "MODERATE": 3, "LOW": 0, "MODIFIER": -3}


def _consequence(row: dict) -> float:
    terms = split_terms(row.get("consequence"))
    mapped = [_CONS_BASE[t] for t in terms if t in _CONS_BASE]
    base = max(mapped) if mapped else 0
    impact = row.get("impact")
    impact_score = _IMPACT_SCORE.get(str(impact).strip().upper(), 0) if not is_missing(impact) else 0
    return float(base + impact_score)


# --------------------------------------------------------------------------
# 3. splice_lof_score
# --------------------------------------------------------------------------
# Consequence terms LOFTEE reasons about. Adjustable domain constant.
_LOF_CONSEQUENCES = {
    "transcript_ablation",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",
}


def _splice_lof(row: dict) -> float:
    s = 0.0
    ds = to_float(row.get("spliceAI_ds_max"))
    if ds is not None:
        if ds >= 0.8:
            s += 20
        elif ds >= 0.5:
            s += 15
        elif ds >= 0.2:
            s += 8
        elif ds >= 0.1:
            s += 3
        if ds >= 0.2 and not is_missing(row.get("spliceAI_type")):
            s += 2
    terms = set(split_terms(row.get("consequence")))
    if terms & _LOF_CONSEQUENCES:
        if not is_missing(row.get("loftee_lof_filter")):
            s += -8
        flag = norm_token(row.get("loftee_lof_flag")).upper()
        if flag == "HC":
            s += 15
        elif flag == "LC":
            s += 5
        elif flag:  # any other non-empty flag
            s += -3
    return s


# --------------------------------------------------------------------------
# 4. prediction_score  (REVEL + CADD; only for missense / protein_altering)
# --------------------------------------------------------------------------
def _prediction(row: dict) -> float:
    terms = set(split_terms(row.get("consequence")))
    if not (terms & {"missense_variant", "protein_altering_variant"}):
        return 0.0
    s = 0.0
    revel = to_float(row.get("revel_score"))
    if revel is not None:
        if revel >= 0.9:
            s += 15
        elif revel >= 0.75:
            s += 12
        elif revel >= 0.5:
            s += 8
        elif revel >= 0.25:
            s += 3
    cadd = to_float(row.get("cadd_phred"))
    if cadd is not None:
        if cadd >= 30:
            s += 12
        elif cadd >= 25:
            s += 9
        elif cadd >= 20:
            s += 6
        elif cadd >= 10:
            s += 2
    return s


# --------------------------------------------------------------------------
# 5. frequency_score
# --------------------------------------------------------------------------
def _frequency(row: dict) -> float:
    s = 0.0
    eas = to_float(row.get("gnomAD_eas_AF"))
    popmax = to_float(row.get("gnomAD_popmax_AF"))
    nhomalt = to_int(row.get("gnomAD_nhomalt"))

    if eas is not None:
        if eas < 0.0001:
            s += 10
        elif eas < 0.001:
            s += 8
        elif eas < 0.01:
            s += 3
        elif eas <= 0.05:
            s += -10
        else:
            s += -25
        # popmax escalation. README wording is ambiguous (two separate "if"s);
        # implemented as EXCLUSIVE so a common variant is not double-penalized.
        if popmax is not None:
            if popmax > 0.05:
                s += -10
            elif popmax > 0.01:
                s += -5
    elif popmax is not None:
        if popmax < 0.0001:
            s += 8
        elif popmax < 0.001:
            s += 6
        elif popmax < 0.01:
            s += 2
        elif popmax <= 0.05:
            s += -8
        else:
            s += -20

    if nhomalt is not None:
        if 1 <= nhomalt <= 5:
            s += -3
        elif 6 <= nhomalt <= 20:
            s += -8
        elif nhomalt > 20:
            s += -15

    # absent from gnomAD (both EAS and popmax missing) = extremely rare = +signal
    if eas is None and popmax is None:
        s += 5
    return s


# --------------------------------------------------------------------------
# 6. domain_score
# --------------------------------------------------------------------------
def _domain(row: dict) -> float:
    return 3.0 if not is_missing(row.get("protein_domains")) else 0.0


# --------------------------------------------------------------------------
# The v1 registry: faithful to README. theta == 1.0 reproduces raw score.
# --------------------------------------------------------------------------
def default_registry() -> list[Contributor]:
    return [
        Contributor("clinvar_score", _clinvar, group="clinvar",
                    reads=("clinvar_significance", "clinvar_star_rating",
                           "clinvar_review_status")),
        Contributor("consequence_score", _consequence,
                    reads=("consequence", "impact")),
        Contributor("splice_lof_score", _splice_lof,
                    reads=("spliceAI_ds_max", "spliceAI_type", "consequence",
                           "loftee_lof_flag", "loftee_lof_filter")),
        Contributor("prediction_score", _prediction,
                    reads=("consequence", "revel_score", "cadd_phred")),
        Contributor("frequency_score", _frequency,
                    reads=("gnomAD_eas_AF", "gnomAD_popmax_AF", "gnomAD_nhomalt")),
        Contributor("domain_score", _domain, reads=("protein_domains",)),
    ]

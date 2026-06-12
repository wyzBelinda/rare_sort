"""ranker -- deterministic Layer 0/1 core for variant prioritization.

Layer 0 (scoring/ranking) and Layer 1 (evaluation harness) only. No optimizer,
no LLM. theta == all 1.0 reproduces the README raw_pathogenic_score.
"""
from .contributors import Contributor, default_registry, ADDITIVE, GATE, GENE_PRIOR
from .scorer import default_theta, score, score_frame, rank_case, truth_rank
from .harness import (
    Case, evaluate, evaluate_all_conditions, case_cv_folds, load_cases,
    MASK_CONDITIONS, DEFAULT_KS,
)

__all__ = [
    "Contributor", "default_registry", "ADDITIVE", "GATE", "GENE_PRIOR",
    "default_theta", "score", "score_frame", "rank_case", "truth_rank",
    "Case", "evaluate", "evaluate_all_conditions", "case_cv_folds", "load_cases",
    "MASK_CONDITIONS", "DEFAULT_KS",
]

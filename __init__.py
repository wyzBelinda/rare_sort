"""ranker -- deterministic Layer 0/1 core for variant prioritization.

Layer 0 (scoring/ranking) + Layer 1 (evaluation harness), plus the structural
hooks Layer 3 (optimizer) and Layer 4 (agent) need. No optimizer, no LLM here.
theta == all 1.0 reproduces the README raw_pathogenic_score.
"""
from .contributors import Contributor, default_registry, ADDITIVE, GATE, GENE_PRIOR
from .scorer import (
    default_theta, score, score_frame, rank_units, truth_rank,
    FeatureBundle, extract_features, apply_theta,
    theta_keys, default_bounds, theta_to_vector, vector_to_theta, needed_columns,
)
from .harness import (
    Case, Prepared, prepare, evaluate, evaluate_all_conditions,
    explain_failures, bootstrap_ci, describe_model,
    case_cv_folds, load_cases, MASK_CONDITIONS, DEFAULT_KS,
)

__all__ = [
    "Contributor", "default_registry", "ADDITIVE", "GATE", "GENE_PRIOR",
    "default_theta", "score", "score_frame", "rank_units", "truth_rank",
    "FeatureBundle", "extract_features", "apply_theta",
    "theta_keys", "default_bounds", "theta_to_vector", "vector_to_theta", "needed_columns",
    "Case", "Prepared", "prepare", "evaluate", "evaluate_all_conditions",
    "explain_failures", "bootstrap_ci", "describe_model",
    "case_cv_folds", "load_cases", "MASK_CONDITIONS", "DEFAULT_KS",
]

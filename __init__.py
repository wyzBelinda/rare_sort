"""ranker -- deterministic Layer 0/1 core + Layer 3 optimizer + Layer 4 agent.

Layer 0 scoring/ranking, Layer 1 evaluation harness, Layer 3 numerical weight
optimization (optuna TPE/CMA-ES + by-case CV + margin/L2 objective), and Layer 4
the closed-loop agent (propose -> optimize -> held-out gate -> freeze).
theta == all 1.0 reproduces the README raw_pathogenic_score.
"""
from .contributors import Contributor, default_registry, ADDITIVE, GATE, GENE_PRIOR
from .scorer import (
    default_theta, score, score_frame, rank_units, truth_rank,
    FeatureBundle, extract_features, apply_theta,
    theta_keys, default_bounds, theta_to_vector, vector_to_theta, needed_columns,
)
from .fastrank import build_fast_index, fast_ranks, unit_scores, FastIndex
from .harness import (
    Case, Prepared, prepare, evaluate, evaluate_all_conditions,
    explain_failures, bootstrap_ci, describe_model,
    case_cv_folds, load_cases, MASK_CONDITIONS, DEFAULT_KS,
)
from .optimize import (
    optimize_theta, cv_optimize, fit_and_freeze, metric_getter, make_margin_objective,
)
from .agent import (
    Proposal, AgentState, Proposer, HeuristicProposer, run_agent_loop,
    apply_proposal, llm_proposer_payload,
)
from .llm_proposer import LLMProposer, create_llm_proposer

__all__ = [
    "Contributor", "default_registry", "ADDITIVE", "GATE", "GENE_PRIOR",
    "default_theta", "score", "score_frame", "rank_units", "truth_rank",
    "FeatureBundle", "extract_features", "apply_theta",
    "theta_keys", "default_bounds", "theta_to_vector", "vector_to_theta", "needed_columns",
    "build_fast_index", "fast_ranks", "unit_scores", "FastIndex",
    "Case", "Prepared", "prepare", "evaluate", "evaluate_all_conditions",
    "explain_failures", "bootstrap_ci", "describe_model",
    "case_cv_folds", "load_cases", "MASK_CONDITIONS", "DEFAULT_KS",
    "optimize_theta", "cv_optimize", "fit_and_freeze", "metric_getter", "make_margin_objective",
    "Proposal", "AgentState", "Proposer", "HeuristicProposer", "run_agent_loop",
    "apply_proposal", "llm_proposer_payload",
    "LLMProposer", "create_llm_proposer",
]

"""Layer 3 -- numerical weight optimization over the frozen functional form.

Two objective families:
  * ranking metric (MRR / recall@k via evaluate) -- what we ultimately care about,
    but SATURATES (flat once causal ranks #1) so it under-identifies theta.
  * margin objective (make_margin_objective) -- a listwise log-softmax that keeps
    separating the causal unit from decoys even after rank #1, plus an L2 pull
    toward a prior (the auditable README defaults). This identifies theta and
    keeps it near a sane, explainable point.

theta is a non-negative magnitude vector. Backends via optuna: 'tpe','cma','random'.
cv_optimize() trains per by-case fold and scores the HELD-OUT fold; OOF
improvement vs theta==1 is the only signal to trust.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import optuna

from .contributors import Contributor
from .fastrank import unit_scores
from .harness import (DEFAULT_KS, Prepared, _agg, bootstrap_ci, case_cv_folds,
                      describe_model, evaluate, prepare)
from .scorer import (apply_theta, default_bounds, default_theta, theta_keys,
                     theta_to_vector, vector_to_theta)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SAMPLERS = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "cma": lambda seed: optuna.samplers.CmaEsSampler(seed=seed),
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
}


def metric_getter(level: str = "variant", name: str = "MRR") -> Callable[[dict], float]:
    return lambda res: res["levels"][level][name]


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def make_margin_objective(registry: list[Contributor], *, level: str = "variant",
                          masked_groups=frozenset(), l2: float = 1e-2,
                          prior: Optional[dict] = None) -> Callable:
    """Objective callable f(prepared, theta_dict) -> float to MAXIMIZE.

    mean_case [ logsumexp(z_causal) - logsumexp(z_all) ] - l2 * || theta - prior ||^2

    where z = per-case standardized unit scores. Standardizing is the temperature
    that stops the log-softmax from collapsing to a hard-max (raw sub-scores are
    tens of points, so exp() would saturate exactly like MRR did). The log-softmax
    term keeps pulling the causal unit above the decoys in a GRADED way (no
    saturation -> theta direction is identified); l2 anchors toward `prior`."""
    keys = theta_keys(registry)
    pvec = theta_to_vector(prior or default_theta(registry), keys)

    def obj(prepared: list[Prepared], theta: dict) -> float:
        tvec = theta_to_vector(theta, keys)
        total, n = 0.0, 0
        for p in prepared:
            s = apply_theta(p.bundle, theta, masked_groups)
            scores, truth = unit_scores(p.fast, s)[level]
            if len(truth) == 0 or len(scores) < 2:
                continue
            z = (scores - scores.mean()) / (scores.std() + 1e-9)   # temperature
            total += _logsumexp(z[truth]) - _logsumexp(z)
            n += 1
        mean = total / n if n else 0.0
        return mean - l2 * float(np.sum((tvec - pvec) ** 2))

    return obj


def _prep(items, registry):
    return items if (items and isinstance(items[0], Prepared)) else prepare(items, registry)


def optimize_theta(train, registry: list[Contributor], *, metric_fn=None,
                   objective: Optional[Callable] = None, masked_groups=frozenset(),
                   n_trials: int = 200, backend: str = "tpe", seed: int = 0,
                   levels=("variant",), bounds: Optional[dict] = None):
    """Tune theta on `train`. If `objective` (f(prepared,theta)->float) is given it
    is maximized directly; otherwise metric_fn(evaluate(...)) is used. Features are
    extracted ONCE; every trial is vectorized. Returns (best_theta, best_value, study)."""
    metric_fn = metric_fn or metric_getter("variant", "MRR")
    prepared = _prep(train, registry)
    keys = theta_keys(registry)
    bnds = bounds or default_bounds(registry)

    def trial_fn(trial):
        vec = [trial.suggest_float(k, bnds[k][0], bnds[k][1]) for k in keys]
        theta = vector_to_theta(vec, keys)
        if objective is not None:
            return objective(prepared, theta)
        return metric_fn(evaluate(prepared, theta, masked_groups=masked_groups, levels=levels))

    study = optuna.create_study(direction="maximize", sampler=_SAMPLERS[backend](seed))
    study.optimize(trial_fn, n_trials=n_trials)
    return {k: study.best_params[k] for k in keys}, study.best_value, study


def cv_optimize(cases, registry, *, n_folds: int = 5, seed: int = 0,
                metric_fn=None, objective: Optional[Callable] = None,
                masked_groups=frozenset(), n_trials: int = 200, backend: str = "tpe",
                levels=("variant", "gene"), ks=DEFAULT_KS,
                bounds: Optional[dict] = None) -> dict:
    """By-case CV. Train each fold (with `objective` if given), score the held-out
    fold with the RANKING metric, aggregate OUT-OF-FOLD ranks for tuned vs theta==1."""
    metric_fn = metric_fn or metric_getter("variant", "MRR")
    base_theta = default_theta(registry)
    oof_tuned = {lvl: [] for lvl in levels}
    oof_base = {lvl: [] for lvl in levels}
    per_fold = []
    for fi, (train, test) in enumerate(case_cv_folds(cases, n_folds, seed)):
        best, train_val, _ = optimize_theta(
            train, registry, metric_fn=metric_fn, objective=objective,
            masked_groups=masked_groups, n_trials=n_trials, backend=backend,
            seed=seed + fi, levels=("variant",), bounds=bounds)
        res_t = evaluate(test, best, registry, masked_groups=masked_groups, levels=levels, ks=ks)
        res_b = evaluate(test, base_theta, registry, masked_groups=masked_groups, levels=levels, ks=ks)
        for lvl in levels:
            oof_tuned[lvl] += list(res_t["per_case"][f"{lvl}_rank"])
            oof_base[lvl] += list(res_b["per_case"][f"{lvl}_rank"])
        per_fold.append({"fold": fi, "n_train": len(train), "n_test": len(test),
                         "train_obj": train_val, "test_tuned": metric_fn(res_t),
                         "test_baseline": metric_fn(res_b), "theta": best})
    return {
        "oof_tuned": {lvl: _agg(oof_tuned[lvl], ks) for lvl in levels},
        "oof_baseline": {lvl: _agg(oof_base[lvl], ks) for lvl in levels},
        "per_fold": per_fold,
        "config": {"n_folds": n_folds, "backend": backend, "n_trials": n_trials,
                   "masked_groups": sorted(masked_groups)},
    }


def fit_and_freeze(cases, registry, *, n_trials: int = 300, backend: str = "tpe",
                   objective: Optional[Callable] = None, masked_groups=frozenset(),
                   metric_fn=None, seed: int = 0, cv_report: Optional[dict] = None,
                   n_boot: int = 500, bounds: Optional[dict] = None,
                   data_snapshot: str = None, rationale: str = None,
                   extra: Optional[dict] = None):
    """Final fit on ALL cases -> frozen, signed model artifact with in-sample value,
    bootstrap CI on the ranking metric, and (if given) the CV-OOF estimate + extra."""
    metric_fn = metric_fn or metric_getter("variant", "MRR")
    best, val, _ = optimize_theta(cases, registry, metric_fn=metric_fn,
                                  objective=objective, masked_groups=masked_groups,
                                  n_trials=n_trials, backend=backend, seed=seed,
                                  bounds=bounds)
    ci = bootstrap_ci(cases, best, metric_fn, registry=registry, n_boot=n_boot,
                      seed=seed, masked_groups=masked_groups, levels=("variant",))
    metrics = {"in_sample_objective": val, "bootstrap_ci_MRR": ci,
               "cv_oof": cv_report["oof_tuned"] if cv_report else None,
               "cv_oof_baseline": cv_report["oof_baseline"] if cv_report else None}
    model = describe_model(registry, best, metrics=metrics,
                           data_snapshot=data_snapshot, rationale=rationale)
    if extra:
        model["provenance"] = extra
    return best, model

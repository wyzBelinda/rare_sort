"""Layer 3 -- numerical weight optimization over the frozen functional form.

theta is a non-negative magnitude vector; the optimizer searches it against the
Layer 1 held-out ranking metric. Backends (all via optuna, uniform interface):
  'tpe'    -- Bayesian TPE
  'cma'    -- CMA-ES
  'random' -- random search (baseline for "is the optimizer doing anything")

cv_optimize() is the honest signal: for each by-case fold it tunes on train and
scores on the held-out test, then aggregates OUT-OF-FOLD ranks. In-sample
improvement is meaningless; OOF improvement vs theta==1 is the thing to trust.
"""
from __future__ import annotations

from typing import Callable, Optional

import optuna

from .contributors import Contributor
from .harness import (DEFAULT_KS, Prepared, _agg, bootstrap_ci, case_cv_folds,
                      describe_model, evaluate, prepare)
from .scorer import default_bounds, default_theta, theta_keys, vector_to_theta

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SAMPLERS = {
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "cma": lambda seed: optuna.samplers.CmaEsSampler(seed=seed),
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
}


def metric_getter(level: str = "variant", name: str = "MRR") -> Callable[[dict], float]:
    return lambda res: res["levels"][level][name]


def _prep(items, registry):
    if items and isinstance(items[0], Prepared):
        return items
    return prepare(items, registry)


def optimize_theta(train, registry: list[Contributor], *, metric_fn=None,
                   masked_groups=frozenset(), n_trials: int = 200,
                   backend: str = "tpe", seed: int = 0, levels=("variant",),
                   bounds: Optional[dict] = None):
    """Tune theta on `train` (Cases or Prepared). Features are extracted ONCE;
    every trial is a vectorized evaluate(). Returns (best_theta, best_value, study)."""
    metric_fn = metric_fn or metric_getter("variant", "MRR")
    prepared = _prep(train, registry)
    keys = theta_keys(registry)
    bnds = bounds or default_bounds(registry)

    def objective(trial):
        vec = [trial.suggest_float(k, bnds[k][0], bnds[k][1]) for k in keys]
        res = evaluate(prepared, vector_to_theta(vec, keys),
                       masked_groups=masked_groups, levels=levels)
        return metric_fn(res)

    study = optuna.create_study(direction="maximize", sampler=_SAMPLERS[backend](seed))
    study.optimize(objective, n_trials=n_trials)
    best = {k: study.best_params[k] for k in keys}
    return best, study.best_value, study


def cv_optimize(cases, registry, *, n_folds: int = 5, seed: int = 0,
                metric_fn=None, masked_groups=frozenset(), n_trials: int = 200,
                backend: str = "tpe", levels=("variant", "gene"),
                ks=DEFAULT_KS) -> dict:
    """By-case CV. Tune on each train fold, score the held-out test fold, then
    aggregate OUT-OF-FOLD ranks for both the tuned theta and the theta==1 baseline."""
    metric_fn = metric_fn or metric_getter("variant", "MRR")
    base_theta = default_theta(registry)
    oof_tuned = {lvl: [] for lvl in levels}
    oof_base = {lvl: [] for lvl in levels}
    per_fold = []
    for fi, (train, test) in enumerate(case_cv_folds(cases, n_folds, seed)):
        best, train_val, _ = optimize_theta(
            train, registry, metric_fn=metric_fn, masked_groups=masked_groups,
            n_trials=n_trials, backend=backend, seed=seed + fi, levels=("variant",))
        res_t = evaluate(test, best, registry, masked_groups=masked_groups,
                         levels=levels, ks=ks)
        res_b = evaluate(test, base_theta, registry, masked_groups=masked_groups,
                         levels=levels, ks=ks)
        for lvl in levels:
            oof_tuned[lvl] += list(res_t["per_case"][f"{lvl}_rank"])
            oof_base[lvl] += list(res_b["per_case"][f"{lvl}_rank"])
        per_fold.append({"fold": fi, "n_train": len(train), "n_test": len(test),
                         "train_metric": train_val,
                         "test_tuned": metric_fn(res_t),
                         "test_baseline": metric_fn(res_b),
                         "theta": best})
    return {
        "oof_tuned": {lvl: _agg(oof_tuned[lvl], ks) for lvl in levels},
        "oof_baseline": {lvl: _agg(oof_base[lvl], ks) for lvl in levels},
        "per_fold": per_fold,
        "config": {"n_folds": n_folds, "backend": backend, "n_trials": n_trials,
                   "masked_groups": sorted(masked_groups)},
    }


def fit_and_freeze(cases, registry, *, n_trials: int = 300, backend: str = "tpe",
                   masked_groups=frozenset(), metric_fn=None, seed: int = 0,
                   cv_report: Optional[dict] = None, n_boot: int = 500,
                   data_snapshot: str = None, rationale: str = None):
    """Final fit on ALL cases -> frozen, signed model artifact carrying the
    in-sample value, a bootstrap CI, and (if provided) the CV-OOF estimate."""
    metric_fn = metric_fn or metric_getter("variant", "MRR")
    best, val, _ = optimize_theta(cases, registry, metric_fn=metric_fn,
                                  masked_groups=masked_groups, n_trials=n_trials,
                                  backend=backend, seed=seed)
    ci = bootstrap_ci(cases, best, metric_fn, registry=registry, n_boot=n_boot,
                      seed=seed, masked_groups=masked_groups, levels=("variant",))
    metrics = {"in_sample": val, "bootstrap_ci": ci,
               "cv_oof": cv_report["oof_tuned"] if cv_report else None,
               "cv_oof_baseline": cv_report["oof_baseline"] if cv_report else None}
    model = describe_model(registry, best, metrics=metrics,
                           data_snapshot=data_snapshot, rationale=rationale)
    return best, model

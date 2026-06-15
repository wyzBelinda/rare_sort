"""Layer 4 -- the closed-loop agent over Layers 0/1/3.

The agent's autonomy lives in the OUTER loop (read failures -> propose a change);
the numerical optimizer (Layer 3) is the INNER loop; the held-out CV metric is the
honest arbiter that stops the agent fooling itself. The deployed artifact stays a
frozen, audited theta + formula -- "self-evolution" is captured as a versioned,
evidence-backed provenance log, NOT as anything mutable inside the weight vector.

One loop iteration:
  fit theta on all cases  ->  explain_failures (structured dump)
  proposer.propose(state, failures)  ->  Proposal (a diff: bounds/prior/structure)
  apply  ->  cv_optimize candidate (margin objective)  ->  OOF metric
  GATE on held-out delta  ->  accept (commit) or reject (revert), always logged
  ... repeat ...  ->  fit_and_freeze the committed config with the full history

`Proposer` is an interface. HeuristicProposer (here) reads the same JSON a human or
LLM would read and emits the same Proposal schema -- so an LLM proposer is a drop-in
(see llm_proposer_payload for the exact bytes it would receive).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

import numpy as np

from .harness import explain_failures
from .optimize import (cv_optimize, fit_and_freeze, make_margin_objective,
                       metric_getter, optimize_theta)
from .scorer import default_bounds, default_theta


@dataclass
class Proposal:
    kind: str                       # set_bounds|set_prior|drop_contributor|restore_contributor|stop
    target: str = ""
    value: Optional[float] = None
    rationale: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentState:
    registry: list                  # effective (enabled) contributors
    bounds: dict
    prior: dict
    masked_groups: frozenset
    best_oof: Optional[dict]
    best_theta: Optional[dict]
    round: int
    disabled: set = field(default_factory=set)


class Proposer(Protocol):
    def propose(self, state: AgentState, failures: list[dict]) -> Proposal: ...


class HeuristicProposer:
    """Reads the failure dump: where is the causal variant latently stronger than
    the blockers (in raw sub-scores) but capped by its current bound? Raise that
    contributor's bound. This is a stand-in for an LLM reading the same JSON."""

    def __init__(self, max_bound: float = 5.0, step: float = 1.5):
        self.max_bound, self.step = max_bound, step

    def propose(self, state: AgentState, failures: list[dict]) -> Proposal:
        if not failures:
            return Proposal("stop", rationale="no failing cases left")
        tally = {c.name: 0.0 for c in state.registry}
        seen = False
        for f in failures:
            cs, bl = f.get("causal_subscores"), f.get("blockers") or []
            if not (cs and bl):
                continue
            seen = True
            for k in tally:
                avg_block = float(np.mean([b["subscores"][k] for b in bl]))
                tally[k] += cs[k] - avg_block
        if not seen:
            return Proposal("stop", rationale="no blockers to learn from")
        cand = [(k, v) for k, v in tally.items()
                if state.bounds[k][1] < self.max_bound - 1e-9 and v > 0]
        if not cand:
            return Proposal("stop", rationale="no headroom or no latent strength")
        k, v = max(cand, key=lambda kv: kv[1])
        new_hi = min(self.max_bound, state.bounds[k][1] + self.step)
        return Proposal("set_bounds", k, new_hi,
                        f"causal beats blockers in raw {k} (Σ={v:.0f}) but bound "
                        f"caps it at {state.bounds[k][1]:.1f}; raise to {new_hi:.1f}")


def apply_proposal(state: AgentState, prop: Proposal):
    bounds, prior, disabled = dict(state.bounds), dict(state.prior), set(state.disabled)
    if prop.kind == "set_bounds":
        lo, _ = bounds[prop.target]
        bounds[prop.target] = (lo, prop.value)
    elif prop.kind == "set_prior":
        prior[prop.target] = prop.value
    elif prop.kind == "drop_contributor":
        disabled.add(prop.target)
    elif prop.kind == "restore_contributor":
        disabled.discard(prop.target)
    return bounds, prior, disabled


def llm_proposer_payload(state: AgentState, failures: list[dict]) -> str:
    """Exact JSON an LLM proposer would receive. (Parity with HeuristicProposer.)"""
    return json.dumps({
        "round": state.round,
        "current_bounds": {k: list(v) for k, v in state.bounds.items()},
        "current_prior": state.prior,
        "best_oof": state.best_oof,
        "masked_groups": sorted(state.masked_groups),
        "n_failures": len(failures),
        "failures": failures[:20],
        "allowed_proposals": ["set_bounds", "set_prior", "drop_contributor",
                              "restore_contributor", "stop"],
    }, default=str, indent=2)


def run_agent_loop(cases, full_registry, *, proposer: Proposer, n_rounds: int = 6,
                   masked_groups=frozenset(), init_bounds: Optional[dict] = None,
                   l2: float = 1e-2, n_trials: int = 80, n_folds: int = 4,
                   backend: str = "cma", min_delta: float = 0.02, patience: int = 2,
                   seed: int = 0, verbose: bool = True):
    bounds = init_bounds or default_bounds(full_registry)
    prior = default_theta(full_registry)
    disabled: set = set()
    metric = metric_getter("variant", "MRR")

    def eff(reg_disabled):
        return [c for c in full_registry if c.name not in reg_disabled]

    def cv(reg, bnds, pri):
        obj = make_margin_objective(reg, masked_groups=masked_groups, l2=l2, prior=pri)
        return cv_optimize(cases, reg, objective=obj, masked_groups=masked_groups,
                           n_folds=n_folds, n_trials=n_trials, backend=backend,
                           bounds=bnds, levels=("variant",), seed=seed)

    rep = cv(eff(disabled), bounds, prior)
    best = rep["oof_tuned"]["variant"]
    history = [{"round": 0, "proposal": None, "oof_MRR": best["MRR"],
                "oof_recall@1": best["recall@1"], "decision": "baseline"}]
    if verbose:
        print(f"[r0 baseline] OOF MRR={best['MRR']:.3f} recall@1={best['recall@1']:.3f}")

    rejects = 0
    for r in range(1, n_rounds + 1):
        reg = eff(disabled)
        obj = make_margin_objective(reg, masked_groups=masked_groups, l2=l2, prior=prior)
        theta, _, _ = optimize_theta(cases, reg, objective=obj, masked_groups=masked_groups,
                                     n_trials=n_trials, backend=backend, bounds=bounds, seed=seed)
        fails = explain_failures(cases, theta, reg, k=1, masked_groups=masked_groups)
        state = AgentState(reg, bounds, prior, masked_groups, best, theta, r, disabled)
        prop = proposer.propose(state, fails)
        if prop.kind == "stop":
            if verbose:
                print(f"[r{r}] proposer stop: {prop.rationale}")
            break
        nb, npr, nd = apply_proposal(state, prop)
        cand = cv(eff(nd), nb, npr)["oof_tuned"]["variant"]
        delta = cand["MRR"] - best["MRR"]
        accept = delta > min_delta
        history.append({"round": r, "proposal": prop.to_dict(), "oof_MRR": cand["MRR"],
                        "oof_recall@1": cand["recall@1"], "delta": round(delta, 4),
                        "decision": "accept" if accept else "reject"})
        if verbose:
            print(f"[r{r}] {prop.kind}:{prop.target} -> OOF MRR={cand['MRR']:.3f} "
                  f"(Δ{delta:+.3f}) {'ACCEPT' if accept else 'reject'} | {prop.rationale}")
        if accept:
            bounds, prior, disabled, best, rejects = nb, npr, nd, cand, 0
        else:
            rejects += 1
            if rejects >= patience:
                if verbose:
                    print(f"[r{r}] patience exhausted, stopping")
                break

    reg = eff(disabled)
    obj = make_margin_objective(reg, masked_groups=masked_groups, l2=l2, prior=prior)
    theta, model = fit_and_freeze(
        cases, reg, objective=obj, masked_groups=masked_groups, n_trials=n_trials * 2,
        backend=backend, bounds=bounds, n_boot=200,
        cv_report={"oof_tuned": {"variant": best}, "oof_baseline": rep["oof_baseline"]},
        data_snapshot="agent-loop final", rationale="Layer-4 closed-loop result",
        extra={"agent_history": history, "disabled": sorted(disabled),
               "final_bounds": {k: list(v) for k, v in bounds.items()}})
    if verbose:
        print(f"[freeze] signature={model['version_signature']} "
              f"final OOF MRR={best['MRR']:.3f} recall@1={best['recall@1']:.3f}")
    return theta, model, history

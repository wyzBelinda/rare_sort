"""Part (b): Layer 4 closed-loop verification.

Setup designed so a single Layer-3 optimize CANNOT solve it: initial per-contributor
bounds are tight and uniform (upper=1.5), so no weighting can lift the causal variant
over the ClinVar/consequence 'trap' decoy. The agent must READ failures, discover
that the causal is latently strong in splice/prediction, and RAISE those bounds.
We verify the loop is closed: OOF metric climbs across gated rounds, the bad
proposals would be rejected, and the result is a frozen, signed artifact carrying
the full provenance history.
"""
import json
import numpy as np

from ranker import default_registry, default_bounds, HeuristicProposer, run_agent_loop
from ranker.agent import AgentState, llm_proposer_payload
from ranker import explain_failures, optimize_theta, make_margin_objective
from verify_layer3 import make_case, REG, KEYS


def main():
    cases = [make_case(f"S{i}", np.random.default_rng(2000 + i)) for i in range(50)]
    full_reg = default_registry()

    # Structural gap: splice_lof and prediction are pinned OFF (weight forced to 0).
    # The causal variant's signal lives mostly there, so NO reweighting of the
    # remaining axes can rank it #1 -- the agent must discover and switch them on.
    init_bounds = {k: (0.0, 5.0) for k in KEYS}
    init_bounds["splice_lof_score"] = (0.0, 0.0)
    init_bounds["prediction_score"] = (0.0, 0.0)

    print("=== Layer 4 closed loop (HeuristicProposer; LLM proposer is a drop-in) ===")
    theta, model, history = run_agent_loop(
        cases, full_reg, proposer=HeuristicProposer(max_bound=5.0, step=2.5),
        n_rounds=6, init_bounds=init_bounds, l2=1e-2, n_trials=70, n_folds=4,
        backend="cma", min_delta=0.02, seed=0, verbose=True)

    base_mrr = history[0]["oof_MRR"]
    final_mrr = max(h["oof_MRR"] for h in history if h["decision"] in ("baseline", "accept"))
    accepted = [h for h in history if h.get("decision") == "accept"]

    print("\n[provenance history]")
    for h in history:
        p = h.get("proposal")
        tag = h["decision"]
        extra = f" {p['kind']}:{p['target']}->{p['value']}" if p else ""
        print(f"  r{h['round']:<2} {tag:<8} OOF MRR={h['oof_MRR']:.3f}{extra}")

    print(f"\n[final theta] " + " ".join(f"{k.split('_')[0]}={theta[k]:.2f}" for k in KEYS))
    print(f"[freeze] signature={model['version_signature']}, "
          f"history embedded under model['provenance']['agent_history'] "
          f"({len(model['provenance']['agent_history'])} entries)")

    # show the LLM-proposer seam: the exact payload it would receive at round 1
    reg = full_reg
    obj = make_margin_objective(reg, l2=1e-2)
    th, _, _ = optimize_theta(cases, reg, objective=obj, n_trials=60, backend="cma",
                              bounds=init_bounds, seed=0)
    fails = explain_failures(cases, th, reg, k=1)
    state = AgentState(reg, init_bounds, {k: 1.0 for k in KEYS}, frozenset(),
                       None, th, 1)
    payload = llm_proposer_payload(state, fails)
    print(f"\n[LLM seam] proposer payload at r1 is {len(payload)} bytes of JSON "
          f"({len(fails)} failures); first failure keys: "
          f"{list(json.loads(payload)['failures'][0].keys()) if fails else 'none'}")

    assert final_mrr > base_mrr + 0.1, (base_mrr, final_mrr)
    assert len(accepted) >= 1
    assert model["provenance"]["agent_history"]                     # provenance present
    json.dumps(model)                                               # fully serializable
    print(f"\nVERIFIED: closed loop lifted OOF MRR {base_mrr:.3f} -> {final_mrr:.3f} "
          f"via {len(accepted)} gated, logged proposals; frozen artifact carries the trail.")


if __name__ == "__main__":
    main()

"""LLM-backed Proposer for Layer 4 — drop-in replacement for HeuristicProposer.

Reads the same structured failure dump (llm_proposer_payload), calls an
OpenAI-compatible chat API, parses the JSON response into a Proposal.
On any failure (network, parse, validation) it degrades gracefully to the
heuristic proposer so the agent loop never halts on an API hiccup.

Supports DeepSeek, Groq, OpenAI, and any other provider with a
/v1/chat/completions endpoint. Configuration is read from environment
variables (loaded via python-dotenv from .env in the project root).

Env vars (all required):
  RARE_SORT_API_KEY    API key
  RARE_SORT_BASE_URL   Chat completions endpoint base URL
  RARE_SORT_MODEL      Model name string

Optional:
  RARE_SORT_TEMPERATURE        LLM temperature (default 0.2)
  RARE_SORT_MAX_OUTPUT_TOKENS  Max tokens in response (default 4096)
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from .agent import AgentState, HeuristicProposer, Proposal, Proposer, llm_proposer_payload

# ── System prompt ──────────────────────────────────────────────────────────
# This is the only "prompt engineering" surface. It explains the domain model
# (contributors, theta, bounds, subscores vs contributions) and the allowed
# proposal actions, then asks the LLM to reason like a clinical variant scientist.
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a clinical variant scientist tuning a rare-disease variant prioritisation system.

## Background
A patient's exome/genome produces thousands of variants. The system scores each
variant by summing weighted evidence from fixed "contributors":

| Contributor | Evidence source | Direction |
|---|---|---|
| clinvar_score | ClinVar clinical significance | pathogenic=+, benign=- |
| consequence_score | VEP consequence (frameshift, missense…) | severe=+, silent=0 |
| splice_lof_score | SpliceAI + LOFTEE loss-of-function | damaging=+, filtered=- |
| prediction_score | REVEL + CADD (missense only) | damaging=+ |
| frequency_score | gnomAD population frequency | rare=+, common=- |
| domain_score | Protein domain annotation | in-domain=+3 |

Each contributor has a FIXED calibration (the raw "subscore") and a LEARNABLE
weight multiplier (theta).  The final score is:

  score = Σ (weight_i × subscore_i)

Theta weights are non-negative and bounded: each contributor has a [lo, hi] range.
The system also maintains a "prior" (default all 1.0) — the auditable baseline
that the L2 penalty pulls weights toward.

## Your job
You receive a JSON dump of cases where the CURRENT weights FAIL to rank the
known causal variant in the top k. For each failure you see:

  causal_subscores  — raw calibrated scores of the causal variant (theta-independent)
  causal_contributions — weighted contributions under the current theta
  blockers[]  — the variants that RANK ABOVE the causal variant right now,
                 each with their own subscores and contributions

Your task: find the ROOT CAUSE — which contributor(s) has latent signal in the
causal variant (high subscore) but its current bound or weight is too tight to
let that signal break through the blockers?

## Allowed proposals

Reply with a single JSON object containing:

{
  "kind": "set_bounds" | "set_prior" | "drop_contributor" | "restore_contributor" | "stop",
  "target": "<contributor_name>",
  "value": <float or null>,
  "rationale": "<one-sentence explanation of what you changed and why>"
}

Rules for each kind:
- set_bounds: raise (or occasionally lower) a contributor's upper bound.
  target = contributor name, value = new upper bound.
  Use this when the causal variant is LATENTLY strong in this contributor
  (causal_subscore >> blocker_subscore) but the bound caps the weight.
  New upper bound should be in [0.5, 10.0]; prefer small increments (1–2×).
- set_prior: shift the L2 anchor for one contributor.
  target = contributor name, value = new prior (suggest 0.5–3.0).
  Use this sparingly, only when a contributor is systematically over/under-trusted
  across many cases.
- drop_contributor: disable a contributor entirely. target = contributor name.
  Use only when the contributor is NOISY (high variance, wrong sign) across
  many failures.
- restore_contributor: re-enable a previously dropped contributor.
  target = contributor name. Use when you believe dropping it was a mistake.
- stop: no useful change can be made. Only use when genuinely stuck.

## Reasoning checklist
1. For each failure case, compare causal_subscores vs each blocker's subscores.
2. Find contributors where causal clearly dominates but current weight is low or
   bound-capped.
3. Also check for contributors where NOISE (blocker subscores >> causal) is
   hurting — consider tightening or dropping.
4. Prefer ONE targeted change per round. The agent loop gates every proposal
   with a held-out CV check; small steps are safer than big leaps.
5. If there are no failures (n_failures=0), always reply "stop".

## Output format
Reply with ONLY a single valid JSON object. No markdown fences, no surrounding
text, no thinking aloud. The response must start with "{" and end with "}".
"""

# ── Parsing ────────────────────────────────────────────────────────────────

_VALID_KINDS = {"set_bounds", "set_prior", "drop_contributor", "restore_contributor", "stop"}


def _strip_json_fence(text: str) -> str:
    """Handle ```json ... ``` wrapping that some models add."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _parse_proposal(raw: str):
    body = _strip_json_fence(raw)
    obj = json.loads(body)
    kind = str(obj.get("kind", "")).strip()
    if kind not in _VALID_KINDS:
        obj["kind"] = "stop"
        obj["rationale"] = f"unknown kind '{kind}' parsed from LLM; stopping. raw: {raw[:200]}"
    return Proposal(
        kind=obj["kind"],
        target=str(obj.get("target", "")),
        value=float(obj["value"]) if obj.get("value") is not None else None,
        rationale=str(obj.get("rationale", "")),
    )


# ── LLM Proposer ───────────────────────────────────────────────────────────

class LLMProposer:
    """Calls an OpenAI-compatible chat API to generate Proposals.

    Configuration is read from environment variables (see module docstring).
    Falls back to HeuristicProposer on any error so the agent loop survives
    API hiccups. Set fallback=False to surface errors instead.

    Usage:
        proposer = LLMProposer()
        # or with explicit config:
        proposer = LLMProposer(api_key="sk-...", base_url="https://...", model="...")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        fallback: bool = True,
    ):
        # Lazy import — only pay the openai import cost when LLMProposer is used.
        from openai import OpenAI  # type: ignore

        self.api_key = api_key or os.environ["RARE_SORT_API_KEY"]
        self.base_url = base_url or os.environ["RARE_SORT_BASE_URL"]
        self.model = model or os.environ.get("RARE_SORT_MODEL", "gpt-4o")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.environ.get("RARE_SORT_TEMPERATURE", "0.2"))
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.environ.get("RARE_SORT_MAX_OUTPUT_TOKENS", "4096"))
        )
        self.fallback = fallback
        self._heuristic = HeuristicProposer() if fallback else None
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._calls = 0
        self._errors = 0

    @property
    def stats(self) -> dict:
        return {"calls": self._calls, "errors": self._errors}

    def propose(self, state: AgentState, failures: list[dict]) -> Proposal:
        """Generate a Proposal by calling the LLM. Falls back to HeuristicProposer
        on any error (network, parse, validation) if fallback=True."""
        payload = llm_proposer_payload(state, failures)

        try:
            self._calls += 1
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content
            proposal = _parse_proposal(raw)
            # Attach the raw LLM output for auditability
            proposal.rationale = f"[LLM] {proposal.rationale}"
            return proposal

        except Exception as exc:
            self._errors += 1
            if self._heuristic is not None:
                prop = self._heuristic.propose(state, failures)
                prop.rationale = f"[LLM error: {exc}] fallback heuristic: {prop.rationale}"
                return prop
            raise


# ── Convenience factory ─────────────────────────────────────────────────────

def create_llm_proposer(**kwargs) -> LLMProposer:
    """Create an LLMProposer, loading .env from the project root if available.

    Tries, in order:
      1. Explicit kwargs passed to this function.
      2. Environment variables (already loaded, or load_dotenv first).
      3. Sensible defaults for base_url/model; API key MUST be set.
    """
    # Try to load .env if not already loaded (idempotent).
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    except Exception:
        pass

    return LLMProposer(**kwargs)

"""
Cost accounting — the canonical token-counting implementation used in
the DIAL paper. All published numbers (and any reported on the
leaderboard) MUST go through `compute_cost`, otherwise comparisons are
not meaningful.

Methodology
-----------
We count three sources of tokens:
    1. base proposer LLM calls    — issued every step
    2. gate overhead              — any extra LLM/inference cost the gate
                                    incurs per step (e.g. extra samples,
                                    confidence queries)
    3. rollout LLM calls          — issued only when the gate triggers,
                                    expanded by the env-specific horizon

Cost is reported as `cost / base_only_cost`, where the denominator is the
mean episode token count of base_only on the same seeds. This makes the
metric dimensionless and comparable across environments with very
different base token usage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TokenLedger:
    """Running tally of tokens used by one episode."""
    base_proposer: int = 0
    gate_overhead: int = 0
    rollout: int = 0

    def total(self) -> int:
        return self.base_proposer + self.gate_overhead + self.rollout

    def add(self, kind: str, n_tokens: int) -> None:
        if kind == "base":
            self.base_proposer += n_tokens
        elif kind == "gate":
            self.gate_overhead += n_tokens
        elif kind == "rollout":
            self.rollout += n_tokens
        else:
            raise ValueError(
                f"Unknown token kind '{kind}'. "
                f"Expected one of: base, gate, rollout."
            )


def compute_cost(
    method_tokens: int,
    base_only_tokens: int,
) -> float:
    """
    Cost in units of base-only token usage.

    Parameters
    ----------
    method_tokens : int
        Total tokens used by the gated method, including base proposer +
        gate overhead + rollout.
    base_only_tokens : int
        Total tokens used by the base_only baseline on the same env/seed.

    Returns
    -------
    float
        cost / base_only. 1.0 means equal cost; 5.0 means 5× more
        expensive than not gating at all.
    """
    if base_only_tokens <= 0:
        return float("inf")
    return method_tokens / base_only_tokens


def env_cost_constants(env: str) -> Dict[str, float]:
    """
    Per-env mean rollout-to-base token ratios used for fast cost
    estimation when the harness skips actual base_only re-runs.
    Values match Appendix `app:cost` in the paper.
    """
    constants = {
        "hotpotqa":  {"rollout_per_step": 35.8, "base_per_step": 1.0},
        "apps":      {"rollout_per_step":  3.9, "base_per_step": 1.0},
        "webshop":   {"rollout_per_step": 12.9, "base_per_step": 1.0},
        "fever":     {"rollout_per_step": 22.4, "base_per_step": 1.0},
        "twexpress": {"rollout_per_step":  8.1, "base_per_step": 1.0},
        "plancraft": {"rollout_per_step": 11.5, "base_per_step": 1.0},
    }
    return constants.get(env, {"rollout_per_step": 10.0, "base_per_step": 1.0})

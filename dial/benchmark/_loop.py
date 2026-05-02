"""
Single-episode rollout loop used by the real-runner mode.

Kept separate from `harness.py` so that env adapters can be imported
lazily and `--stub` mode never touches them.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict

from dial.benchmark.cost import TokenLedger, env_cost_constants
from dial.benchmark.gate_interface import Decision, GateInterface

logger = logging.getLogger("DIAL.benchmark")


@dataclass
class EpisodeOutcome:
    success: bool
    n_steps: int
    n_rollouts: int
    base_only_tokens_estimate: int


def run_episode(
    env,
    gate: GateInterface,
    valuator,
    episode_idx: int,
    seed: int,
    rng: random.Random,
    ledger: TokenLedger,
) -> EpisodeOutcome:
    """
    Drive one episode: at each step, compute signals, ask gate, optionally
    invoke optimizer, and record tokens via `ledger`.

    The implementation is intentionally minimal — production paper runs
    use `experiments/run_dial.py`, which adds richer logging and supports
    multi-method dispatch. This loop is the canonical reference for what
    a third-party benchmark run looks like.
    """
    obs = env.reset(seed=seed + episode_idx)
    cost_const = env_cost_constants(env.env_name if hasattr(env, "env_name") else "")
    n_rollouts = 0
    n_steps = 0
    base_tokens = 0
    done = False

    while not done:
        n_steps += 1

        signals = _compute_signals(env, obs)
        ledger.add("base", _token_estimate_base(env, obs))
        base_tokens += _token_estimate_base(env, obs)

        decision = gate.should_rollout(_state_text(obs), signals)
        if not isinstance(decision, Decision):
            decision = Decision(trigger=bool(decision))

        if decision.trigger:
            n_rollouts += 1
            utility = valuator.estimate_utility(env)
            rollout_tokens = int(_token_estimate_base(env, obs) * cost_const["rollout_per_step"])
            ledger.add("rollout", rollout_tokens)
            gate.update(_state_text(obs), signals, float(utility))
            action = env.greedy_oracle_action() if utility > 0 else _base_action(env, obs)
        else:
            action = _base_action(env, obs)

        obs, reward, done, info = env.step(action)

    success = bool(info.get("success", reward > 0)) if isinstance(info, dict) else (reward > 0)
    return EpisodeOutcome(
        success=success,
        n_steps=n_steps,
        n_rollouts=n_rollouts,
        base_only_tokens_estimate=base_tokens,
    )


def _compute_signals(env, obs) -> Dict[str, float]:
    """
    Universal signal pool defined in Appendix `app:universal-features`.
    Env adapters may attach additional signals via `env.last_signals`.
    """
    sig: Dict[str, float] = {
        "token_entropy": float(getattr(env, "_last_token_entropy", 0.0)),
        "action_entropy": float(getattr(env, "_last_action_entropy", 0.0)),
        "step_count": float(getattr(env, "_step", 0)),
        "state_length": float(len(_state_text(obs))),
        "num_avail_actions": float(len(getattr(env, "_actions", []) or [])),
    }
    extra = getattr(env, "last_signals", None)
    if isinstance(extra, dict):
        for k, v in extra.items():
            try:
                sig[k] = float(v)
            except (TypeError, ValueError):
                continue
    return sig


def _state_text(obs: Any) -> str:
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict) and "text" in obs:
        return str(obs["text"])
    return str(obs)


def _token_estimate_base(env, obs) -> int:
    """
    Coarse estimate when the env adapter does not return exact token counts.
    Replaced by the real proposer's reported usage when available.
    """
    return max(50, len(_state_text(obs)) // 4)


def _base_action(env, obs):
    actions = getattr(env, "_actions", None)
    if actions:
        return actions[0]
    if hasattr(env, "default_action"):
        return env.default_action()
    raise RuntimeError(
        f"Env {env} exposes neither '_actions' nor 'default_action()'."
    )

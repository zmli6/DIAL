"""
Environment-agnostic value estimators.

These wrap the BaseEnv interface so the rest of DIAL (oracle, VoC, etc.)
can call ``compute_value(env, action)`` regardless of environment type.
The original MiniGrid-specific valuators in valuator.py are preserved
for backward compatibility.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .envs.base import BaseEnv

logger = logging.getLogger("DIAL")


# ──────────────────────────────────────────────────────────────────
# GENERIC FORWARD EVALUATOR
# ──────────────────────────────────────────────────────────────────

class GenericForwardValuator:
    """
    Forward (myopic) evaluator that delegates to BaseEnv.forward_value().

    Each environment adapter implements its own intentionally-flawed
    forward heuristic.
    """

    def __init__(self, shaping_weight: float = 1.0, **kwargs):
        self.shaping_weight = shaping_weight

    def compute_value(self, env: BaseEnv, action: int) -> float:
        return env.forward_value(action) * self.shaping_weight


# ──────────────────────────────────────────────────────────────────
# GENERIC RETROSPECTIVE EVALUATOR
# ──────────────────────────────────────────────────────────────────

class GenericRetrospectiveValuator:
    """
    Rollout-based evaluator that works with any BaseEnv.

    For each candidate action *a*, do N rollouts of horizon H:
      1. deepcopy env  →  execute *a*
      2. follow rollout policy for H-1 more steps
      3. accumulate discounted return
    Ṽ_R(s, a) = mean over N rollouts.
    """

    def __init__(
        self,
        horizon: int = 5,
        num_samples: int = 8,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        tie_break: str = "random",
        **kwargs,
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.gamma = gamma
        self.epsilon = epsilon
        self.tie_break = tie_break

    def compute_value(self, env: BaseEnv, action: int) -> float:
        returns = []
        for _ in range(self.num_samples):
            ret = self._single_rollout(env, action)
            returns.append(ret)
        return float(np.mean(returns))

    def compute_values_batch(self, env: BaseEnv, actions: List[int]) -> Dict[int, float]:
        """
        Compute V_R for multiple actions more efficiently.

        Optimizations over calling compute_value() in a loop:
          1. First-step caching: actions that terminate on step 1
             return the same reward for all N samples, so we only
             need 1 deepcopy instead of N.
          2. Save/restore state instead of deepcopy-from-original
             for each rollout — avoid copying unchanged data (like
             the full HotpotQA question bank).
        """
        results: Dict[int, float] = {}
        state_snapshot = env.get_state()

        for action in actions:
            # Probe first step with a single deepcopy
            probe = env.deepcopy()
            try:
                obs, reward, terminated, truncated, info = probe.step(action)
            except Exception:
                results[action] = 0.0
                continue
            finally:
                del probe

            first_reward = reward

            # If first step already terminates, all N samples are identical
            if terminated or truncated:
                results[action] = float(first_reward)
                continue

            # Otherwise, do N rollouts from post-first-step state
            returns = []
            for _ in range(self.num_samples):
                ret = self._single_rollout(env, action)
                returns.append(ret)
            results[action] = float(np.mean(returns))

        return results

    def compute_value_detailed(self, env: BaseEnv, action: int) -> Dict:
        returns = []
        for _ in range(self.num_samples):
            ret = self._single_rollout(env, action)
            returns.append(ret)
        return {
            "value": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "returns": [float(r) for r in returns],
        }

    def _single_rollout(self, env: BaseEnv, first_action: int) -> float:
        env_copy = env.deepcopy()

        try:
            total_return = 0.0
            discount = 1.0

            # Step 1: forced action
            obs, reward, terminated, truncated, info = env_copy.step(first_action)
            total_return += discount * reward
            discount *= self.gamma

            if terminated or truncated:
                return total_return

            # Steps 2..H: rollout policy
            for _ in range(1, self.horizon):
                a = env_copy.rollout_action(epsilon=self.epsilon)
                obs, reward, terminated, truncated, info = env_copy.step(a)
                total_return += discount * reward
                discount *= self.gamma

                if terminated or truncated:
                    break

            # Bootstrap at horizon end
            if not (terminated or truncated):
                # Use 0.0 as neutral bootstrap — forward_value() is
                # intentionally flawed (e.g. ignores lava in MiniGrid,
                # keyword heuristic in ALfWorld), so using it here would
                # inject noise and inflate variance in oracle values.
                pass

            return total_return
        finally:
            # Ensure clone is cleaned up promptly (important for envs
            # like ALfWorld where the clone shares the parent's _alf_env)
            del env_copy

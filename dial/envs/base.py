"""
Base environment adapter — defines the common interface all DIAL
environments must implement.

The DIAL framework needs these capabilities from every environment:
  1. reset / step  (standard RL loop)
  2. get_state / set_state  (snapshot & restore for rollouts)
  3. get_actions  (discrete action enumeration)
  4. state_description  (text representation for LLM proposer)
  5. success metric  (did the episode succeed?)
  6. forward_value / heuristic_value  (cheap myopic evaluator)
  7. classify_state  (stratified analysis in VoC)
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EnvState:
    """Serialisable snapshot of environment state."""
    data: Dict[str, Any] = field(default_factory=dict)
    env_type: str = ""


class BaseEnv(ABC):
    """
    Abstract base environment adapter.

    Subclasses wrap a concrete environment (MiniGrid, ALFWorld, …)
    and expose the unified interface that the rest of DIAL relies on.
    """

    # ── class-level metadata (override in subclass) ──────────────
    ENV_TYPE: str = "base"          # "minigrid", "alfworld", "webshop", "hotpotqa"
    ACTION_NAMES: Dict[int, str] = {}
    ACTION_DESCRIPTIONS: Dict[int, str] = {}

    def __init__(self, env_name: str, **kwargs):
        self.env_name = env_name
        self._env = None                    # the underlying native env
        self.max_steps: int = kwargs.get("max_steps", 200)

    @property
    def env_type(self) -> str:
        """Convenience instance-level accessor for ENV_TYPE."""
        return self.ENV_TYPE

    # ── lifecycle ─────────────────────────────────────────────────

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        """Reset env; return (obs, info)."""
        ...

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """Take one step; return (obs, reward, terminated, truncated, info)."""
        ...

    def close(self):
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()

    # ── snapshot / restore (required for rollouts) ────────────────

    @abstractmethod
    def get_state(self) -> EnvState:
        """Return a restorable snapshot of the current state."""
        ...

    @abstractmethod
    def set_state(self, state: EnvState):
        """Restore a previously captured snapshot."""
        ...

    def deepcopy(self) -> "BaseEnv":
        """Return an independent deep-copy of this adapter + its env."""
        return copy.deepcopy(self)

    # ── action space ──────────────────────────────────────────────

    @abstractmethod
    def get_actions(self) -> List[int]:
        """Return the full list of valid discrete action indices."""
        ...

    def get_action_name(self, action: int) -> str:
        return self.ACTION_NAMES.get(action, str(action))

    def get_action_description(self, action: int) -> str:
        return self.ACTION_DESCRIPTIONS.get(action, f"Action {action}")

    @property
    def num_actions(self) -> int:
        return len(self.get_actions())

    # ── observation helpers ───────────────────────────────────────

    @abstractmethod
    def get_text_description(self) -> str:
        """
        Return a natural-language description of the current state.
        Used by the LLM proposer to decide which actions to take.
        """
        ...

    def get_image(self):
        """Return an RGB image (PIL or ndarray) of the current state, or None."""
        return None

    # ── evaluation helpers ────────────────────────────────────────

    @abstractmethod
    def forward_value(self, action: int) -> float:
        """
        Cheap / myopic value estimate for *action* from current state.
        This is the "forward evaluator" — it may be intentionally
        flawed (e.g. ignores lava in MiniGrid).
        """
        ...

    @abstractmethod
    def is_success(self, reward: float, terminated: bool, info: Dict) -> bool:
        """Return True if the episode ended successfully."""
        ...

    @abstractmethod
    def classify_state(self) -> str:
        """
        Classify the current state into a category for stratified analysis.
        E.g. "safe", "near_lava", "path_diverges" for MiniGrid.
        """
        ...

    # ── state key (for oracle dict) ──────────────────────────────

    @abstractmethod
    def make_state_key(self) -> str:
        """Hashable string key for the current state."""
        ...

    # ── greedy oracle action ─────────────────────────────────────

    @abstractmethod
    def greedy_oracle_action(self) -> int:
        """
        Near-optimal greedy action for oracle value collection.
        Should use full environment knowledge (no intentional blindness).
        """
        ...

    # ── rollout policy action ────────────────────────────────────

    def rollout_action(self, epsilon: float = 0.3) -> int:
        """
        Action for rollout policy (ε-greedy around greedy_oracle_action).
        Default implementation; can be overridden per environment.
        """
        import numpy as np
        if np.random.random() < epsilon:
            return int(np.random.choice(self.get_actions()))
        return self.greedy_oracle_action()

    # ── convenience ──────────────────────────────────────────────

    @property
    def unwrapped(self):
        """Access the raw underlying env (for debugging)."""
        if self._env is not None and hasattr(self._env, "unwrapped"):
            return self._env.unwrapped
        return self._env

    def __repr__(self):
        return f"<{self.__class__.__name__} env={self.env_name}>"

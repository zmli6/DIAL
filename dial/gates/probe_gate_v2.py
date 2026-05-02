"""
Probe Gate v2 — Two strategies for threshold-calibrated probe gating.

Strategy 1 (Offline):
  Pre-compute threshold from offline B1 data, skip exploration entirely.
  Gate starts in exploitation immediately.

Strategy 2 (Adaptive RL):
  Warm-start threshold from offline data, then continuously adapt
  via epsilon-greedy exploration + gradient-like threshold updates.
  Balances exploration (random triggers to discover new info)
  and exploitation (use learned threshold).
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict

import numpy as np

from dial.gates._scg_base import SCGBase

logger = logging.getLogger("DIAL")


# ══════════════════════════════════════════════════════════════════
# Strategy 1: Offline Pre-computed Threshold
# ══════════════════════════════════════════════════════════════════

class OfflineProbeGate(SCGBase):
    """
    Probe gate with pre-computed threshold. No exploration phase.

    Parameters
    ----------
    probe : HiddenStateProbe
    threshold : float
        Pre-computed decision threshold (from offline F1/quantile analysis).
    device : str
    """

    VARIANT = "probe_offline"

    def __init__(self, probe, threshold: float, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self._probe = probe
        self._device = device
        self._threshold = threshold
        # Skip exploration entirely
        self.phase = "exploitation"
        self.min_cal_points = 0

    def _exploit_decision(self, consistency: float, **ctx) -> bool:
        hidden_state = ctx.get("hidden_state")
        if hidden_state is None:
            return True
        means, _ = self._probe.probe.predict(
            hidden_state.reshape(1, -1), self._device
        )
        return bool(means[0] > self._threshold)

    def _on_transition(self):
        pass

    def get_estimated_pattern(self) -> Dict[str, Any]:
        return {
            "strategy": "offline",
            "threshold": self._threshold,
        }


# ══════════════════════════════════════════════════════════════════
# Strategy 2: Adaptive RL-like Threshold
# ══════════════════════════════════════════════════════════════════

class AdaptiveProbeGate(SCGBase):
    """
    Probe gate with RL-like adaptive threshold.

    Starts with an offline warm-start threshold, then continuously
    adapts via epsilon-greedy exploration + gradient-like updates.

    The threshold moves:
      - DOWN after useful rollouts (trigger more)
      - UP after useless rollouts (trigger less)
    Step size is proportional to |pred_U - threshold|.

    Epsilon decays over time: always explore a bit but less as we learn.

    Parameters
    ----------
    probe : HiddenStateProbe
    init_threshold : float
        Warm-start threshold (from offline analysis).
    epsilon_init : float
        Initial exploration rate (P of random trigger).
    epsilon_decay : float
        Multiplicative decay per step (e.g., 0.998).
    epsilon_min : float
        Minimum exploration rate.
    lr : float
        Threshold learning rate.
    cost_penalty : float
        Lambda for cost-aware updates. Higher = more conservative.
    device : str
    """

    VARIANT = "probe_adaptive"

    def __init__(
        self,
        probe,
        init_threshold: float,
        epsilon_init: float = 0.15,
        epsilon_decay: float = 0.998,
        epsilon_min: float = 0.02,
        lr: float = 0.05,
        cost_penalty: float = 0.0,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._probe = probe
        self._device = device
        self._threshold = init_threshold
        self._init_threshold = init_threshold
        self._epsilon = epsilon_init
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._lr = lr
        self._cost_penalty = cost_penalty

        # Skip fixed exploration phase — we do continuous exploration
        self.phase = "exploitation"
        self.min_cal_points = 0

        # Tracking
        self._n_triggers = 0
        self._n_useful = 0
        self._threshold_history = [init_threshold]
        self._current_pred = None

    def _exploit_decision(self, consistency: float, **ctx) -> bool:
        hidden_state = ctx.get("hidden_state")
        if hidden_state is None:
            return True

        means, _ = self._probe.probe.predict(
            hidden_state.reshape(1, -1), self._device
        )
        pred_U = float(means[0])
        self._current_pred = pred_U

        # Epsilon-greedy: explore with probability epsilon
        if random.random() < self._epsilon:
            # Exploration: random trigger
            decision = random.random() < 0.5
        else:
            # Exploitation: use threshold
            decision = pred_U > self._threshold

        return decision

    def update(self, consistency: float, utility: float, **ctx):
        super().update(consistency, utility, **ctx)
        self._n_triggers += 1

        pred_U = self._current_pred
        if pred_U is None:
            return

        # Adaptive threshold update
        if utility > 0:
            # Useful rollout → this pred_U was worth triggering
            # → lower threshold toward pred_U (trigger more like this)
            self._n_useful += 1
            step = self._lr * max(self._threshold - pred_U, 0)
            self._threshold -= step
        else:
            # Useless rollout → shouldn't have triggered at this pred_U
            # → raise threshold above pred_U (trigger less like this)
            step = self._lr * max(pred_U - self._threshold + 0.1, 0)
            self._threshold += step

        # Cost-aware regularization: gently push threshold up
        if self._cost_penalty > 0:
            self._threshold += self._cost_penalty * self._lr * 0.01

        # Clamp threshold to reasonable range
        self._threshold = max(min(self._threshold, 2.0), -2.0)

        # Decay epsilon
        self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)

        self._threshold_history.append(self._threshold)

    def _on_transition(self):
        pass

    def get_estimated_pattern(self) -> Dict[str, Any]:
        return {
            "strategy": "adaptive_rl",
            "init_threshold": self._init_threshold,
            "current_threshold": self._threshold,
            "epsilon": self._epsilon,
            "n_triggers": self._n_triggers,
            "n_useful": self._n_useful,
            "useful_rate": self._n_useful / max(self._n_triggers, 1),
            "threshold_history_len": len(self._threshold_history),
        }

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "threshold": self._threshold,
            "epsilon": self._epsilon,
            "n_triggers": self._n_triggers,
            "n_useful": self._n_useful,
            "useful_rate": self._n_useful / max(self._n_triggers, 1),
        })
        return stats

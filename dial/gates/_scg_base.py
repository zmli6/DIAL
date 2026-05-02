"""
SCG (Self-Calibrating Gate) — Base class and common utilities.

The SCG family of gates decides at each step whether to spend computation
on a retrospective rollout.  All variants share a two-phase lifecycle:

  1. **Exploration** — rollout decisions are (partially) random so that
     (C, U) calibration data is collected.
  2. **Exploitation** — rollout decisions are driven by the learned
     C → P(rollout) mapping.

Subclasses: SCG_MLP (statistical), SCG_Prompt (LLM in-context).
"""
from __future__ import annotations

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("DIAL")


# ── calibration data point ───────────────────────────────────────

@dataclass
class CalibrationPoint:
    """One (consistency, utility) observation."""
    consistency: float
    utility: float
    state_type: str = ""
    episode: int = 0
    step: int = 0
    decision: str = ""          # "rollout" | "skip"
    gate_phase: str = ""        # "exploration" | "exploitation"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consistency": self.consistency,
            "utility": self.utility,
            "state_type": self.state_type,
            "episode": self.episode,
            "step": self.step,
            "decision": self.decision,
            "gate_phase": self.gate_phase,
            **self.extra,
        }


# ── base gate ────────────────────────────────────────────────────

class SCGBase(ABC):
    """
    Abstract Self-Calibrating Gate.

    Parameters
    ----------
    explore_rate : float
        Probability of triggering a rollout during exploration phase.
    min_cal_points : int
        Minimum calibration points before switching to exploitation.
    window_size : int
        Sliding-window size for the calibration buffer.
    utility_threshold : float
        U > this value counts as "rollout was useful" for labelling.
    """

    VARIANT: str = "base"       # override in subclass

    def __init__(
        self,
        explore_rate: float = 0.5,
        min_cal_points: int = 50,
        window_size: int = 500,
        utility_threshold: float = 0.05,
    ):
        self.explore_rate = explore_rate
        self.min_cal_points = min_cal_points
        self.window_size = window_size
        self.utility_threshold = utility_threshold

        self.buffer: List[CalibrationPoint] = []
        self.phase: str = "exploration"
        self._step_counter: int = 0
        self._decision_log: List[Dict[str, Any]] = []

    # ── public API (used by gated agent) ─────────────────────────

    def should_rollout(self, consistency: float, **ctx) -> bool:
        """
        Decide whether to perform a retrospective rollout.

        Parameters
        ----------
        consistency : float
            The forward–retrospective consistency score for the current
            state (e.g. avg |V_F − V_R| across actions).
        ctx : dict
            Optional context (state_type, episode, step, …).

        Returns
        -------
        bool
            True ↔ execute the rollout.
        """
        self._step_counter += 1

        if self.phase == "exploration":
            decision = random.random() < self.explore_rate
            # Check whether to transition
            if len(self.buffer) >= self.min_cal_points:
                self._on_transition()
                self.phase = "exploitation"
                logger.info(
                    f"[{self.VARIANT}] → exploitation "
                    f"(n={len(self.buffer)} calibration points)"
                )
        else:
            decision = self._exploit_decision(consistency, **ctx)

        self._decision_log.append({
            "step": self._step_counter,
            "consistency": consistency,
            "decision": "rollout" if decision else "skip",
            "phase": self.phase,
            "buffer_size": len(self.buffer),
            **{k: v for k, v in ctx.items()
               if isinstance(v, (int, float, str, bool))},
        })
        return decision

    def update(self, consistency: float, utility: float, **ctx):
        """
        Record a new calibration observation and update the gate.
        Called only when a rollout was actually performed.
        """
        # Store all serialisable ctx fields into extra so gates can use
        # multi-signal features (evidence_count, state_category, …)
        extra = {
            k: v for k, v in ctx.items()
            if isinstance(v, (int, float, str, bool, type(None)))
            and k not in ("state_type", "episode", "step")
        }
        pt = CalibrationPoint(
            consistency=consistency,
            utility=utility,
            state_type=ctx.get("state_type", ""),
            episode=ctx.get("episode", 0),
            step=ctx.get("step", 0),
            decision="rollout",
            gate_phase=self.phase,
            extra=extra,
        )
        self.buffer.append(pt)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        self._on_update(pt)

    # ── subclass hooks ───────────────────────────────────────────

    @abstractmethod
    def _exploit_decision(self, consistency: float, **ctx) -> bool:
        """Exploitation-phase decision.  Override in subclass."""
        ...

    def _on_transition(self):
        """Called once when switching from exploration → exploitation."""
        pass

    def _on_update(self, pt: CalibrationPoint):
        """Called after every update (can trigger retraining etc.)."""
        pass

    # ── diagnostics ──────────────────────────────────────────────

    def get_estimated_pattern(self) -> Dict[str, Any]:
        """
        Return the gate's current estimate of the C-U pattern.
        Useful for convergence plots.
        """
        if len(self.buffer) < 10:
            return {"direction": "unknown", "n": len(self.buffer)}

        cs = np.array([p.consistency for p in self.buffer])
        us = np.array([p.utility for p in self.buffer])
        from scipy.stats import pearsonr
        r, p = pearsonr(cs, us)
        if p > 0.1:
            direction = "null"
        elif r > 0:
            direction = "positive"
        else:
            direction = "negative"
        return {
            "direction": direction,
            "pearson_r": float(r),
            "pearson_p": float(p),
            "n": len(self.buffer),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics for logging."""
        n_rollout = sum(1 for d in self._decision_log if d["decision"] == "rollout")
        n_total = len(self._decision_log)
        return {
            "variant": self.VARIANT,
            "phase": self.phase,
            "buffer_size": len(self.buffer),
            "total_decisions": n_total,
            "rollout_count": n_rollout,
            "rollout_rate": n_rollout / max(n_total, 1),
            "estimated_pattern": self.get_estimated_pattern(),
        }

    # ── serialisation ────────────────────────────────────────────

    def save_logs(self, output_dir: str, prefix: str = ""):
        """Persist decision log and calibration buffer."""
        os.makedirs(output_dir, exist_ok=True)
        tag = f"{prefix}_" if prefix else ""

        log_path = os.path.join(output_dir, f"{tag}scg_{self.VARIANT}_decision_log.json")
        with open(log_path, "w") as f:
            json.dump(self._decision_log, f, indent=2)

        cal_path = os.path.join(output_dir, f"{tag}scg_{self.VARIANT}_calibration.json")
        with open(cal_path, "w") as f:
            json.dump([p.to_dict() for p in self.buffer], f, indent=2)

        stats_path = os.path.join(output_dir, f"{tag}scg_{self.VARIANT}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.get_stats(), f, indent=2)

        logger.info(f"[{self.VARIANT}] logs saved to {output_dir}")

    def load_buffer(self, filepath: str):
        """Load pre-collected calibration data (e.g. from Quick Probe)."""
        with open(filepath) as f:
            raw = json.load(f)
        for item in raw:
            if isinstance(item, dict):
                pt = CalibrationPoint(
                    consistency=item.get("consistency_avg", item.get("consistency", 0)),
                    utility=item.get("utility", 0),
                    state_type=item.get("state_type", ""),
                    episode=item.get("episode", 0),
                    step=item.get("step", 0),
                    decision="rollout",
                    gate_phase="preloaded",
                )
                self.buffer.append(pt)
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]
        logger.info(
            f"[{self.VARIANT}] loaded {len(self.buffer)} calibration points from {filepath}"
        )

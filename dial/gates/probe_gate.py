"""
Calibrated Probe Gate — 4 threshold calibration strategies for Hidden State Probe.

Strategies:
  A. Quantile-adaptive:  threshold = quantile matching positive rate
  B. F1-optimal:         threshold maximizing F1 score
  C. Cost-EV:            threshold maximizing E[utility] - λ·trigger_cost
  D. Online Bayesian:    EMA-adaptive threshold during exploitation

All strategies share:
  - Exploration phase: random 50% trigger, collect (pred_U, actual_U) pairs
  - Transition: calibrate threshold from exploration data
  - Exploitation: trigger when probe_predicted_U > calibrated_threshold
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

import numpy as np

from dial.gates._scg_base import SCGBase

logger = logging.getLogger("DIAL")


class CalibratedProbeGate(SCGBase):
    """
    Gate using pre-trained HiddenStateProbe with calibrated threshold.

    Parameters
    ----------
    probe : HiddenStateProbe
        Pre-trained probe for utility prediction.
    calibration_method : str
        One of 'A' (quantile), 'B' (F1), 'C' (cost_ev), 'D' (online).
    cost_ratio : float
        C_rollout / C_base for method C.  Higher = more conservative.
    online_lr : float
        Learning rate for method D threshold adaptation.
    device : str
        Torch device.
    """

    VARIANT = "calibrated_probe"

    def __init__(
        self,
        probe,
        calibration_method: str = "A",
        cost_ratio: float = 10.0,
        online_lr: float = 0.02,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._probe = probe
        self._device = device
        self._cal_method = calibration_method.upper()
        self._cost_ratio = cost_ratio
        self._online_lr = online_lr

        # Exploration data collection
        self._explore_preds = []     # pred_U for triggered steps
        self._explore_utils = []     # actual_U for triggered steps
        self._all_preds = []         # pred_U for ALL steps (for quantile)

        # Calibrated threshold (set at transition)
        self._calibrated_threshold = 0.05  # fallback
        self._current_pred = None

        self.VARIANT = f"probe_cal_{self._cal_method}"

    def _exploit_decision(self, consistency: float, **ctx) -> bool:
        """Required by SCGBase ABC. Actual logic is in should_rollout()."""
        hidden_state = ctx.get("hidden_state")
        pred_U = self._predict_utility(hidden_state)
        return pred_U > self._calibrated_threshold

    def _predict_utility(self, hidden_state) -> float:
        """Get probe's predicted utility for a single hidden state."""
        if hidden_state is None:
            return 1.0  # trigger if no hidden state
        means, _ = self._probe.probe.predict(
            hidden_state.reshape(1, -1), self._device
        )
        return float(means[0])

    def should_rollout(self, consistency: float, **ctx) -> bool:
        self._step_counter += 1
        hidden_state = ctx.get("hidden_state")
        pred_U = self._predict_utility(hidden_state)

        if self.phase == "exploration":
            # Record ALL predictions (for quantile calibration)
            self._all_preds.append(pred_U)
            self._current_pred = pred_U

            decision = random.random() < self.explore_rate

            # Check transition
            if len(self.buffer) >= self.min_cal_points:
                self._on_transition()
                self.phase = "exploitation"
                logger.info(
                    f"[{self.VARIANT}] → exploitation "
                    f"(threshold={self._calibrated_threshold:.4f}, "
                    f"n={len(self.buffer)} cal points)"
                )
        else:
            # Exploitation: use calibrated threshold
            decision = pred_U > self._calibrated_threshold
            self._current_pred = pred_U

        self._decision_log.append({
            "step": self._step_counter,
            "consistency": consistency,
            "pred_U": pred_U,
            "threshold": self._calibrated_threshold,
            "decision": "rollout" if decision else "skip",
            "phase": self.phase,
            "buffer_size": len(self.buffer),
        })
        return decision

    def update(self, consistency: float, utility: float, **ctx):
        super().update(consistency, utility, **ctx)

        # Record (pred, actual) for calibration
        if self._current_pred is not None:
            self._explore_preds.append(self._current_pred)
            self._explore_utils.append(utility)

        # Method D: online adaptation during exploitation
        if self.phase == "exploitation" and self._cal_method == "D":
            self._online_adapt(utility)

    # ── Calibration Methods ──────────────────────────────────

    def _on_transition(self):
        preds = np.array(self._explore_preds)
        utils = np.array(self._explore_utils)
        all_preds = np.array(self._all_preds)

        if len(preds) < 5:
            logger.warning(f"[{self.VARIANT}] Too few calibration points ({len(preds)})")
            return

        if self._cal_method == "A":
            self._calibrate_quantile(preds, utils, all_preds)
        elif self._cal_method == "B":
            self._calibrate_f1(preds, utils)
        elif self._cal_method == "C":
            self._calibrate_cost_ev(preds, utils)
        elif self._cal_method == "D":
            self._calibrate_online_init(preds, utils)
        else:
            raise ValueError(f"Unknown calibration method: {self._cal_method}")

    def _calibrate_quantile(self, preds, utils, all_preds):
        """A. Quantile-adaptive: match positive rate."""
        positive_rate = float((utils > 0).mean())
        if positive_rate <= 0:
            self._calibrated_threshold = float(np.max(all_preds)) + 0.01
        elif positive_rate >= 1:
            self._calibrated_threshold = float(np.min(all_preds)) - 0.01
        else:
            # threshold = percentile of ALL predictions such that
            # trigger_rate ≈ positive_rate
            self._calibrated_threshold = float(
                np.percentile(all_preds, 100 * (1 - positive_rate))
            )
        logger.info(
            f"[{self.VARIANT}] Method A: positive_rate={positive_rate:.3f}, "
            f"threshold={self._calibrated_threshold:.4f}"
        )

    def _calibrate_f1(self, preds, utils):
        """B. F1-optimal: sweep thresholds, maximize F1."""
        labels = (utils > 0).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            self._calibrated_threshold = float(np.median(preds))
            return

        best_f1, best_t = 0.0, float(np.median(preds))
        for t in np.linspace(preds.min() - 0.01, preds.max() + 0.01, 200):
            pred_pos = (preds > t).astype(int)
            tp = ((pred_pos == 1) & (labels == 1)).sum()
            fp = ((pred_pos == 1) & (labels == 0)).sum()
            fn = ((pred_pos == 0) & (labels == 1)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        self._calibrated_threshold = float(best_t)
        logger.info(
            f"[{self.VARIANT}] Method B: best_F1={best_f1:.3f}, "
            f"threshold={self._calibrated_threshold:.4f}"
        )

    def _calibrate_cost_ev(self, preds, utils):
        """C. Cost-EV: maximize E[utility|trigger] - λ·P(trigger)."""
        lam = self._cost_ratio  # penalty for triggering

        best_ev, best_t = -float("inf"), float(np.median(preds))
        for t in np.linspace(preds.min() - 0.01, preds.max() + 0.01, 200):
            triggered = preds > t
            trigger_rate = triggered.mean()
            if trigger_rate < 0.01:
                continue
            mean_util_if_triggered = utils[triggered].mean()
            # EV = expected utility gain - cost penalty
            ev = mean_util_if_triggered * trigger_rate - lam * trigger_rate
            if ev > best_ev:
                best_ev = ev
                best_t = t

        self._calibrated_threshold = float(best_t)
        logger.info(
            f"[{self.VARIANT}] Method C: best_EV={best_ev:.4f}, "
            f"cost_ratio={lam:.1f}, threshold={self._calibrated_threshold:.4f}"
        )

    def _calibrate_online_init(self, preds, utils):
        """D. Online Bayesian: initialize with F1, then adapt online."""
        # Start with F1-optimal as warm start
        self._calibrate_f1(preds, utils)
        self._online_threshold = self._calibrated_threshold
        self._online_n = 0
        logger.info(
            f"[{self.VARIANT}] Method D: init_threshold={self._calibrated_threshold:.4f}, "
            f"will adapt with lr={self._online_lr:.3f}"
        )

    def _online_adapt(self, utility: float):
        """D. Online adaptation: adjust threshold based on rollout outcome."""
        self._online_n += 1
        if utility > 0:
            # Useful rollout → slightly lower threshold (trigger more)
            self._calibrated_threshold *= (1 - self._online_lr)
        else:
            # Useless rollout → slightly raise threshold (trigger less)
            self._calibrated_threshold *= (1 + self._online_lr)
        # Clamp to reasonable range
        self._calibrated_threshold = max(
            min(self._calibrated_threshold, 2.0), -2.0
        )

    # ── Stats ────────────────────────────────────────────────

    def get_estimated_pattern(self) -> Dict[str, Any]:
        return {
            "calibration_method": self._cal_method,
            "calibrated_threshold": self._calibrated_threshold,
            "cost_ratio": self._cost_ratio,
            "n_explore_pairs": len(self._explore_preds),
        }

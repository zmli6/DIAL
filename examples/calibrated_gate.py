"""
Slightly less trivial example: fits a logistic regression on the explore
cache, then deploys it. Demonstrates how a calibration-based method
plugs into DIAL's harness without re-running exploration.

Requires scikit-learn::

    pip install scikit-learn

Run::

    python -m dial.benchmark \\
        --gate examples/calibrated_gate.py:CalibratedGate \\
        --explore-data-dir paper_results/explore_cache/ \\
        --stub
"""
from typing import Any, Dict, List, Optional

import numpy as np

from dial.benchmark import GateInterface, Decision


class CalibratedGate(GateInterface):
    """L2-logistic-regression gate fit on the supplied explore cache."""

    name = "calibrated_logistic"
    feature_keys = ("token_entropy", "action_entropy", "step_count",
                    "state_length", "num_avail_actions")

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None

    def setup(self, env_name, backbone, explore_data=None):
        if not explore_data:
            self.model = None
            return
        X = np.array([
            [rec["signals"].get(k, 0.0) for k in self.feature_keys]
            for rec in explore_data
        ])
        y = np.array([1 if rec.get("utility", 0) > 0 else 0 for rec in explore_data])
        if len(set(y)) < 2:
            self.model = None
            return
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=1.0, max_iter=500).fit(X, y)

    def should_rollout(self, state, signals):
        if self.model is None:
            return Decision(trigger=False, score=0.0, reason="no model")
        x = np.array([[signals.get(k, 0.0) for k in self.feature_keys]])
        p = float(self.model.predict_proba(x)[0, 1])
        return Decision(trigger=p > self.threshold, score=p)

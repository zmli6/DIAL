"""
Minimal example: a fixed-direction entropy gate.

This is the simplest non-trivial GateInterface: trigger when token
entropy exceeds a threshold. It deliberately matches the assumption
DIAL's paper falsifies, so on Type-I environments (FEVER, HotpotQA,
TWExpress) you should see SR drop below base_only.

Run::

    python -m dial.benchmark --gate examples/threshold_gate.py:ThresholdGate --stub
"""
from dial.benchmark import GateInterface, Decision


class ThresholdGate(GateInterface):
    name = "entropy_threshold"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def setup(self, env_name, backbone, explore_data=None):
        # If explore data is available, set threshold to its 60th percentile.
        if explore_data:
            entropies = sorted(
                rec["signals"].get("token_entropy", 0.0) for rec in explore_data
            )
            if entropies:
                self.threshold = entropies[int(0.6 * len(entropies))]

    def should_rollout(self, state, signals):
        score = signals.get("token_entropy", 0.0)
        return Decision(trigger=score > self.threshold, score=score)

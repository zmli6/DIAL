"""
GateInterface — the contract a third-party gate must implement to be
evaluable by `dial.benchmark.run_benchmark`.

Designed to be minimal: a method only needs `should_rollout`. All other
hooks (setup, update, finalize) are optional with sensible defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Decision:
    """Returned by `should_rollout`. `trigger=True` means invoke optimizer T."""
    trigger: bool
    score: float = 0.0          # optional confidence in [0, 1] for diagnostics
    reason: str = ""            # optional human-readable rationale


class GateInterface:
    """
    Implement this class to plug a custom adaptive gate into DIAL's harness.

    The harness will call methods in this order:
        1. `setup(env_name, backbone, explore_data)` — once per (env, seed)
        2. for each step:
             decision = `should_rollout(state, signals)`
             if rollout actually happened:
                 `update(state, signals, utility)`
        3. `finalize()` — once at the end (optional, return diagnostics)

    Signals
    -------
    `signals` is a dict containing all universal features computed by
    the harness, plus any env-specific features the env adapter exposes.
    Universal keys (always present):
        - 'token_entropy'      : float, entropy of next-token distribution
        - 'action_entropy'     : float, entropy over candidate actions
        - 'step_count'         : int,   current step in the episode
        - 'state_length'       : int,   token count of state text
        - 'num_avail_actions'  : int,   size of candidate-action set (or 0)

    Optional env-specific keys (presence varies):
        - 'evidence_count', 'has_output', 'inventory_size', ...

    See `docs/extending.md` for how to add new signals from a new env.
    """

    name: str = "custom_gate"   # override for nice column names in reports

    # ─── required ────────────────────────────────────────────────

    def should_rollout(self, state: str, signals: Dict[str, float]) -> Decision:
        """
        Decide whether to invoke the test-time optimizer at this step.
        Must be deterministic given (state, signals) for reproducibility.
        """
        raise NotImplementedError

    # ─── optional hooks ──────────────────────────────────────────

    def setup(
        self,
        env_name: str,
        backbone: str,
        explore_data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Called once before each (env, seed) run.

        `explore_data` is a list of pre-collected (signals, utility) records
        from DIAL's randomized exploration phase, supplied automatically by
        `--use-explore-cache`. Format::

            [{'signals': {'token_entropy': 0.7, ...}, 'utility': 1.0}, ...]

        Methods that need calibration data (Platt, isotonic, probe-based)
        can use this to skip running their own exploration phase, making
        comparisons fair and fast. Stateless methods can ignore it.
        """
        pass

    def update(
        self,
        state: str,
        signals: Dict[str, float],
        utility: float,
    ) -> None:
        """
        Called after a rollout was actually performed, exposing the
        observed utility for online-learning gates. No-op by default.
        """
        pass

    def finalize(self) -> Dict[str, Any]:
        """Return diagnostics included in the per-env report."""
        return {}

"""
Self-Evolving Gate V2 — Multi-Cycle + Feedback-Guided Evolution
================================================================

Phase 6.1 improvements:
  - Phase 2 (feedback): After Cycle 1, feed back LASSO results + failure cases
    to LLM for a second reflection → generate improved features
  - Phase 3 (multi-cycle): Support 2 or 3 cycle iterations

Extends SelfEvolvingGate with:
  - Cycle tracking (on_episode_end counts episodes)
  - Re-reflect at cycle boundaries
  - Feedback prompt with LASSO importance + SR + failure/success cases
  - Feature management: selective (keep LASSO-selected, replace unselected)
  - Data strategy: cumulative (use all past data)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from dial.gates._dial_base import SelfEvolvingGate

logger = logging.getLogger("DIAL")


class SelfEvolvingGateV2(SelfEvolvingGate):
    """
    Multi-cycle self-evolving gate with feedback.

    Parameters
    ----------
    num_cycles : int
        Number of reflect-exploit cycles (2 or 3).
    use_feedback : bool
        If True, include LASSO results + failure cases in re-reflect prompt.
    max_llm_features : int
        Features per LLM reflection (default 5).
    feature_filter : bool
        Pre-filter by correlation.
    feature_strategy : str
        "selective" — keep LASSO-selected + replace unselected with new LLM features.
    data_strategy : str
        "cumulative" — use all accumulated data for LASSO retraining.
    """

    def __init__(
        self,
        num_cycles: int = 2,
        use_feedback: bool = True,
        feature_strategy: str = "selective",
        data_strategy: str = "cumulative",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_cycles = num_cycles
        self._use_feedback = use_feedback
        self._feature_strategy = feature_strategy
        self._data_strategy = data_strategy

        # Cycle tracking
        self._current_cycle = 1
        self._episode_count = 0
        self._cycle_boundaries = self._compute_cycle_boundaries()

        # History for feedback
        self._prev_selected_features = []
        self._prev_lasso_coefs = {}
        self._prev_sr = 0.0
        self._prev_trigger_rate = 0.0
        self._exploit_states = []     # states during exploitation (for failure/success cases)
        self._exploit_actions = []
        self._exploit_utilities = []
        self._exploit_triggered = []  # whether we triggered rollout

    def _compute_cycle_boundaries(self):
        """Compute episode indices where re-reflection happens."""
        if self._num_cycles == 2:
            # Explore 50 → Exploit 50 → Re-reflect → Exploit 100
            return [100]  # re-reflect after episode 100
        elif self._num_cycles == 3:
            # Explore 50 → Exploit 30 → Re-reflect → Exploit 30 → Re-reflect → Exploit 40
            return [80, 110]
        return []

    def on_episode_end(self):
        """Called at end of each episode. Check if it's time to re-reflect."""
        self._episode_count += 1

        if self._current_cycle <= len(self._cycle_boundaries):
            boundary = self._cycle_boundaries[self._current_cycle - 1]
            if self._episode_count >= boundary:
                logger.info(
                    f"[{self.VARIANT}] Cycle {self._current_cycle} → {self._current_cycle + 1} "
                    f"at episode {self._episode_count}"
                )
                self._re_reflect_and_retrain()
                self._current_cycle += 1

    def should_rollout(self, consistency: float, **ctx) -> bool:
        """Override to track exploitation data for feedback."""
        result = super().should_rollout(consistency, **ctx)

        # Track exploitation data for feedback
        if self.phase == "exploitation":
            obs = ctx.get("_obs")
            state_text = str(obs)[:2000] if obs is not None else ""
            action_text = ctx.get("action_text", "")
            self._exploit_states.append(state_text)
            self._exploit_actions.append(action_text)
            self._exploit_triggered.append(result)

        return result

    def update(self, consistency: float, utility: float, **ctx):
        """Override to track exploitation utilities."""
        super().update(consistency, utility, **ctx)

        if self.phase == "exploitation":
            self._exploit_utilities.append(utility)

    def _re_reflect_and_retrain(self):
        """Re-reflect with feedback from previous cycle, then retrain LASSO."""

        # Save previous cycle's results for feedback
        if self._selected_features:
            self._prev_selected_features = list(self._selected_features)
            if self._model is not None and hasattr(self._model, 'coef_'):
                self._prev_lasso_coefs = dict(zip(
                    self._selected_features,
                    self._model.coef_[0].tolist()
                ))

        # Compute previous cycle stats
        if self._exploit_utilities:
            exploit_sr = np.mean([u > 0 for u in self._exploit_utilities])
            exploit_trigger_rate = np.mean(self._exploit_triggered) if self._exploit_triggered else 0
            self._prev_sr = exploit_sr
            self._prev_trigger_rate = exploit_trigger_rate

        # Build feedback prompt and re-reflect
        if self._use_feedback:
            self._self_reflect_with_feedback()
        else:
            self._self_reflect()  # standard reflection without feedback

        # Augment accumulated data with new LLM features
        if self._llm_feature_fn is not None:
            if self._feature_strategy == "selective":
                self._selective_augment()
            else:
                self._augment_explore_features_with_llm()

        # Retrain LASSO with accumulated data
        if self._data_strategy == "cumulative":
            # Use all exploration + exploitation data
            self._accumulate_exploit_data()

        self._on_transition()

        logger.info(
            f"[{self.VARIANT}] Re-trained after cycle {self._current_cycle}: "
            f"features={len(self._selected_features or [])}, "
            f"threshold={self._cmdp_threshold:.4f}"
        )

        # Reset exploitation tracking for next cycle
        self._exploit_states = []
        self._exploit_actions = []
        self._exploit_utilities = []
        self._exploit_triggered = []

    def _accumulate_exploit_data(self):
        """Add exploitation data to explore pool for LASSO retraining."""
        for i, (state, action, triggered) in enumerate(zip(
            self._exploit_states, self._exploit_actions, self._exploit_triggered
        )):
            if triggered and i < len(self._exploit_utilities):
                utility = self._exploit_utilities[i] if i < len(self._exploit_utilities) else 0
                # Build feature dict
                feats = {}
                for name in self._feature_names_all or []:
                    feats[name] = 0.0

                # Extract universal features (approximate from stored data)
                step_idx = len(self._explore_features)
                feats['step_count'] = float(step_idx % 20)
                feats['state_length'] = float(len(state.split()))

                # Extract LLM features
                if self._llm_feature_fn is not None:
                    llm_feats = self._extract_llm_features(state, step_idx, action)
                    feats.update(llm_feats)

                self._explore_features.append(feats)
                self._explore_utils.append(utility)
                self._explore_states.append(state)
                self._explore_actions.append(action)

    def _selective_augment(self):
        """Selective feature strategy: keep LASSO-selected, replace unselected."""
        if self._llm_feature_fn is None:
            return

        # Remove old unselected LLM features from explore data
        old_llm_keys = [k for k in (self._feature_names_all or []) if k.startswith('llm_')]
        selected_llm = [k for k in (self._prev_selected_features or []) if k.startswith('llm_')]
        to_remove = [k for k in old_llm_keys if k not in selected_llm]

        if to_remove:
            for feats in self._explore_features:
                for key in to_remove:
                    feats.pop(key, None)
            logger.info(f"[{self.VARIANT}] Selective: removed {len(to_remove)} unselected LLM features")

        # Add new LLM features
        self._augment_explore_features_with_llm()

    def _self_reflect_with_feedback(self):
        """Reflect with feedback from previous cycle."""
        logger.info(f"[{self.VARIANT}] Starting feedback-guided reflection...")

        U = np.array(self._explore_utils)
        positive_mask = U > 0
        positive_rate = positive_mask.mean()

        # Build feedback section
        feedback_text = self._build_feedback_text()

        # Build base prompt (same examples as before)
        pos_indices = np.where(positive_mask)[0][:3]
        neg_indices = np.where(~positive_mask)[0][:3]
        base_prompt = self._build_reflection_prompt(pos_indices, neg_indices, positive_rate, U)

        # Combine with feedback
        prompt = f"""{base_prompt}

## Feedback from Previous Cycle

{feedback_text}

Based on this feedback, generate a NEW and IMPROVED feature extractor.
Focus on addressing the failure cases — what patterns did the previous features miss?
Generate exactly {self._max_llm_features} features."""

        # Call LLM with retries (same logic as parent)
        code = None
        last_error = "unknown"
        for attempt in range(self._max_retries):
            try:
                if attempt == 0:
                    response = self._call_llm(prompt)
                else:
                    error_prompt = (
                        f"{prompt}\n\nPrevious attempt failed: {last_error}\n"
                        f"Please fix. Attempt {attempt+1}/{self._max_retries}."
                    )
                    response = self._call_llm(error_prompt)

                code = self._extract_code(response)
                if code is None:
                    last_error = "No code block found"
                    continue

                fn, feature_names = self._validate_code(code)
                if fn is not None:
                    self._llm_feature_fn = fn
                    self._llm_feature_code = code
                    self._llm_feature_names = feature_names
                    logger.info(
                        f"[{self.VARIANT}] Feedback reflection succeeded: "
                        f"{len(feature_names)} features: {feature_names}"
                    )
                    self._reflect_log = {
                        "backend": self._reflect_backend,
                        "cycle": self._current_cycle + 1,
                        "attempts": attempt + 1,
                        "features_discovered": feature_names,
                        "used_feedback": True,
                    }
                    return
                else:
                    last_error = "Validation failed"

            except Exception as e:
                last_error = str(e)

        logger.warning(f"[{self.VARIANT}] Feedback reflection failed after {self._max_retries} attempts")

    def _build_feedback_text(self) -> str:
        """Build feedback text from previous cycle results."""
        parts = []

        # b) Feature importance (LASSO coefficients)
        if self._prev_lasso_coefs:
            parts.append("### Features selected by LASSO (with importance):")
            sorted_feats = sorted(self._prev_lasso_coefs.items(), key=lambda x: -abs(x[1]))
            for name, coef in sorted_feats:
                parts.append(f"  - {name}: coefficient = {coef:.4f}")

        # c) SR and trigger rate
        parts.append(f"\n### Performance:")
        parts.append(f"  - Success Rate: {self._prev_sr:.1%}")
        parts.append(f"  - Trigger Rate: {self._prev_trigger_rate:.1%}")

        # d) Failure cases: triggered but utility was 0
        failures = []
        for i, (triggered, state, action) in enumerate(zip(
            self._exploit_triggered, self._exploit_states, self._exploit_actions
        )):
            if triggered and i < len(self._exploit_utilities) and self._exploit_utilities[i] <= 0:
                failures.append((state[:300], action))
            if len(failures) >= 3:
                break

        if failures:
            parts.append("\n### Failure Cases (triggered rollout but it was NOT useful):")
            for i, (state, action) in enumerate(failures):
                parts.append(f"\n  Case {i+1}:")
                parts.append(f"  State: {state}")
                parts.append(f"  Action: {action}")

        # e) Success cases: correctly didn't trigger
        successes = []
        for i, (triggered, state, action) in enumerate(zip(
            self._exploit_triggered, self._exploit_states, self._exploit_actions
        )):
            if not triggered:
                successes.append((state[:300], action))
            if len(successes) >= 3:
                break

        if successes:
            parts.append("\n### Success Cases (correctly skipped rollout):")
            for i, (state, action) in enumerate(successes):
                parts.append(f"\n  Case {i+1}:")
                parts.append(f"  State: {state}")
                parts.append(f"  Action: {action}")

        return "\n".join(parts)

    def get_estimated_pattern(self) -> Dict[str, Any]:
        pattern = super().get_estimated_pattern()
        pattern.update({
            "num_cycles": self._num_cycles,
            "current_cycle": self._current_cycle,
            "use_feedback": self._use_feedback,
            "feature_strategy": self._feature_strategy,
            "data_strategy": self._data_strategy,
            "episode_count": self._episode_count,
        })
        return pattern

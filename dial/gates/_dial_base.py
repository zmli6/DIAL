"""
Self-Evolving Adaptive Gating
==============================

The agent reflects on its own exploration experience to discover
what state features predict rollout utility, then evolves its
gating strategy accordingly.

Pipeline:
  1. Explore (50 ep): collect (state, action, utility) trajectories
  2. Self-Reflect (LLM): analyze experience → generate feature extractor code
  3. Evolve (LASSO): select from self-discovered features → calibrate threshold
  4. Exploit: use evolved gating strategy

Supports two LLM backends:
  - "local": same Qwen3-4B via vLLM (truly self-contained)
  - "openrouter": Claude-opus-4.6 via OpenRouter API (stronger reflection)
"""
from __future__ import annotations

import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from dial.gates.dial_universal import PrincipledSCGGate

logger = logging.getLogger("DIAL")

OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "anthropic/claude-opus-4.6"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class SelfEvolvingGate(PrincipledSCGGate):
    """
    Self-Evolving gate that uses LLM reflection to discover features.

    Extends PrincipledSCGGate by adding a self-reflection step before
    LASSO feature selection. The LLM analyzes exploration trajectories
    and generates a Python feature extractor function.

    Parameters
    ----------
    reflect_backend : str
        "local" (vLLM Qwen3-4B) or "openrouter" (Claude-opus-4.6)
    max_retries : int
        Max retries for LLM code generation validation.
    """

    VARIANT = "self_evolving"

    def __init__(
        self,
        reflect_backend: str = "local",
        max_retries: int = 10,
        max_llm_features: int = 15,
        feature_filter: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        max_llm_features : int
            Max features to ask LLM to generate (5 for "fewer" mode, 15 for default).
        feature_filter : bool
            If True, pre-filter LLM features by |correlation| > 0.05 before LASSO.
        """
        super().__init__(**kwargs)
        self._reflect_backend = reflect_backend
        self._max_retries = max_retries
        self._max_llm_features = max_llm_features
        self._feature_filter = feature_filter
        self._llm_feature_fn = None
        self._llm_feature_code = None
        self._llm_feature_names = []
        self._reflect_log = {}

        # Store raw exploration data for reflection
        self._explore_states = []      # state texts
        self._explore_actions = []     # action texts
        self._explore_steps = []       # step counts

        self.VARIANT = f"self_evolving_{reflect_backend}"

    def should_rollout(self, consistency: float, **ctx) -> bool:
        self._step_counter += 1

        if self.phase == "exploration":
            import random
            decision = random.random() < self.explore_rate

            # Store raw state/action for reflection
            obs = ctx.get("_obs")
            state_text = str(obs)[:2000] if obs is not None else ""
            action_text = ctx.get("action_text", "")
            step = ctx.get("step", 0)
            self._explore_states.append(state_text)
            self._explore_actions.append(action_text)
            self._explore_steps.append(step)

            # Collect hidden states for online PCA
            hidden_state = ctx.get("hidden_state")
            if self._online_pca and hidden_state is not None:
                self._online_pca_buffer.append(hidden_state.copy())

            # Build and store features (universal only at this stage)
            feats = self._build_feature_pool(ctx)
            if self._feature_names_all is None:
                self._feature_names_all = sorted(feats.keys())
            self._current_feats = feats

            if len(self.buffer) >= self.min_cal_points:
                # Self-reflect before transition
                self._self_reflect()

                # Rebuild feature pool with LLM features
                if self._llm_feature_fn is not None:
                    self._augment_explore_features_with_llm()

                self._on_transition()
                self.phase = "exploitation"
                logger.info(
                    f"[{self.VARIANT}] → exploitation "
                    f"(selected {len(self._selected_features or [])} features, "
                    f"threshold={self._cmdp_threshold:.4f}, "
                    f"llm_features={len(self._llm_feature_names)})"
                )
            return decision

        else:
            if self._model is None:
                if self._cmdp_threshold >= 0.9:
                    return False
                return True

            feats = self._build_feature_pool(ctx)

            # Add LLM features if available
            if self._llm_feature_fn is not None:
                obs = ctx.get("_obs")
                state_text = str(obs)[:2000] if obs is not None else ""
                action_text = ctx.get("action_text", "")
                step = ctx.get("step", 0)
                llm_feats = self._extract_llm_features(state_text, step, action_text)
                feats.update(llm_feats)

            x = np.array([feats.get(f, 0) for f in self._selected_features]).reshape(1, -1)
            x_scaled = self._scaler.transform(x)
            prob = self._model.predict_proba(x_scaled)[0, 1]
            decision = bool(prob > self._cmdp_threshold)

            self._current_feats = feats
            self._decision_log.append({
                "step": self._step_counter,
                "prob": float(prob),
                "threshold": self._cmdp_threshold,
                "decision": "rollout" if decision else "skip",
                "phase": self.phase,
            })
            return decision

    def _self_reflect(self):
        """Use LLM to analyze exploration experience and generate feature extractor."""
        logger.info(f"[{self.VARIANT}] Starting self-reflection ({self._reflect_backend})...")

        # Prepare reflection data
        U = np.array(self._explore_utils)
        positive_mask = U > 0
        negative_mask = U <= 0
        positive_rate = positive_mask.mean()

        # Select representative examples
        pos_indices = np.where(positive_mask)[0][:3]
        neg_indices = np.where(negative_mask)[0][:3]

        # Build prompt
        prompt = self._build_reflection_prompt(
            pos_indices, neg_indices, positive_rate, U
        )

        # Call LLM with retries
        code = None
        for attempt in range(self._max_retries):
            try:
                if attempt == 0:
                    response = self._call_llm(prompt)
                else:
                    error_prompt = (
                        f"{prompt}\n\n"
                        f"Previous attempt failed with error:\n{last_error}\n\n"
                        f"Please fix the code and try again. Attempt {attempt+1}/{self._max_retries}."
                    )
                    response = self._call_llm(error_prompt)

                # Extract code from response
                code = self._extract_code(response)
                if code is None:
                    last_error = "No Python code block found in response"
                    continue

                # Validate code
                fn, feature_names = self._validate_code(code)
                if fn is not None:
                    self._llm_feature_fn = fn
                    self._llm_feature_code = code
                    self._llm_feature_names = feature_names
                    logger.info(
                        f"[{self.VARIANT}] Self-reflection succeeded on attempt {attempt+1}: "
                        f"{len(feature_names)} features discovered: {feature_names}"
                    )
                    self._reflect_log = {
                        "backend": self._reflect_backend,
                        "attempts": attempt + 1,
                        "features_discovered": feature_names,
                        "code": code,
                    }
                    return
                else:
                    last_error = "Code validation failed"

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{self.VARIANT}] Reflection attempt {attempt+1} failed: {e}")

        logger.warning(
            f"[{self.VARIANT}] Self-reflection failed after {self._max_retries} attempts. "
            f"Proceeding with universal features only."
        )
        self._reflect_log = {
            "backend": self._reflect_backend,
            "attempts": self._max_retries,
            "features_discovered": [],
            "error": last_error if 'last_error' in dir() else "unknown",
        }

    def _build_reflection_prompt(self, pos_indices, neg_indices, positive_rate, U):
        """Build the self-reflection prompt with exploration data."""

        # Collect representative examples
        examples_text = ""

        if len(pos_indices) > 0:
            examples_text += "\n### Episodes where rollout WAS useful (utility > 0):\n"
            for i, idx in enumerate(pos_indices):
                state = self._explore_states[idx][:500] if idx < len(self._explore_states) else "N/A"
                action = self._explore_actions[idx] if idx < len(self._explore_actions) else "N/A"
                step = self._explore_steps[idx] if idx < len(self._explore_steps) else 0
                util = U[idx]
                examples_text += f"\nExample {i+1} (step {step}, utility={util:.2f}):\n"
                examples_text += f"State: {state}\n"
                examples_text += f"Action: {action}\n"

        if len(neg_indices) > 0:
            examples_text += "\n### Episodes where rollout was NOT useful (utility = 0):\n"
            for i, idx in enumerate(neg_indices[:3]):
                state = self._explore_states[idx][:500] if idx < len(self._explore_states) else "N/A"
                action = self._explore_actions[idx] if idx < len(self._explore_actions) else "N/A"
                step = self._explore_steps[idx] if idx < len(self._explore_steps) else 0
                util = U[idx]
                examples_text += f"\nExample {i+1} (step {step}, utility={util:.2f}):\n"
                examples_text += f"State: {state}\n"
                examples_text += f"Action: {action}\n"

        # Statistics
        mean_pos_util = float(U[U > 0].mean()) if (U > 0).sum() > 0 else 0.0
        mean_util = float(U.mean())
        step_min = min(self._explore_steps) if self._explore_steps else 0
        step_max = max(self._explore_steps) if self._explore_steps else 0
        stats_text = f"""
### Exploration Statistics:
- Total decision points: {len(U)}
- Rollout was useful: {positive_rate:.1%} of the time
- Mean utility when useful: {mean_pos_util:.3f}
- Mean utility overall: {mean_util:.3f}
- Step range: {step_min} to {step_max}
"""

        prompt = f"""You are an AI agent that has been exploring an interactive environment. During exploration, you sometimes performed "rollouts" (additional computation) to improve your decisions. Sometimes the rollout was useful (positive utility), sometimes it was not.

Your task: Analyze the patterns in your experience and write a Python function that extracts features from the state text that could predict whether a rollout would be useful.

{stats_text}

{examples_text}

## Task

Write a Python function `extract_features` that takes the state text, step count, and action text, and returns a dictionary of numeric features that might predict rollout utility.

```python
def extract_features(state_text: str, step_count: int, action_text: str) -> dict:
    \"\"\"Extract features that predict rollout utility.

    Args:
        state_text: Current state observation text
        step_count: Current step number in the episode
        action_text: The proposed action text

    Returns:
        dict mapping feature names to float values
    \"\"\"
    import re
    features = {{}}

    # Your analysis-based features here
    # Example: features['text_length'] = len(state_text)

    return features
```

Requirements:
- Return a dict with string keys and float values
- Use only Python standard library (re, math, len, etc.)
- Extract exactly {self._max_llm_features} features based on the patterns you observe
- Features should be numeric (counts, ratios, booleans as 0/1)
- Focus on features that DIFFER between useful and not-useful rollout cases
- Quality over quantity — each feature should capture a distinct, meaningful signal
- Do NOT use any external libraries"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for reflection."""
        if self._reflect_backend == "openrouter":
            return self._call_openrouter(prompt)
        else:
            return self._call_local(prompt)

    def _call_local(self, prompt: str) -> str:
        """Call local vLLM (Qwen3-4B)."""
        import httpx

        endpoint = os.environ.get("DIAL_VLLM_ENDPOINT", "http://localhost:8900/v1")

        response = httpx.post(
            f"{endpoint}/chat/completions",
            json={
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call_openrouter(self, prompt: str) -> str:
        """Call Claude-opus-4.6 via OpenRouter."""
        import httpx

        response = httpx.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code block from LLM response."""
        # Try to find code block
        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'def extract_features\(.*?\).*?return features',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1) if '```' in pattern else match.group(0)
                # Ensure it has the function definition
                if 'def extract_features' in code:
                    return code.strip()

        # Fallback: try to find function definition directly
        if 'def extract_features' in response:
            start = response.index('def extract_features')
            # Find the end (next function or end of text)
            lines = response[start:].split('\n')
            code_lines = []
            for line in lines:
                code_lines.append(line)
                if line.strip().startswith('return'):
                    break
            return '\n'.join(code_lines)

        return None

    def _validate_code(self, code: str):
        """Validate generated feature extractor code. Returns (fn, feature_names) or (None, [])."""
        try:
            # Compile
            namespace = {}
            exec(code, namespace)

            if 'extract_features' not in namespace:
                return None, []

            fn = namespace['extract_features']

            # Test with sample data
            test_states = self._explore_states[:3] if self._explore_states else ["test state"]
            test_results = []

            for state in test_states:
                result = fn(state, 1, "test action")
                if not isinstance(result, dict):
                    return None, []
                # Check all values are numeric
                for k, v in result.items():
                    if not isinstance(v, (int, float)):
                        return None, []
                test_results.append(result)

            if not test_results:
                return None, []

            feature_names = sorted(test_results[0].keys())
            if len(feature_names) == 0:
                return None, []

            return fn, feature_names

        except Exception as e:
            logger.debug(f"Code validation failed: {e}")
            return None, []

    def _extract_llm_features(self, state_text: str, step: int, action_text: str) -> Dict[str, float]:
        """Extract features using LLM-generated function."""
        if self._llm_feature_fn is None:
            return {}
        try:
            result = self._llm_feature_fn(state_text, step, action_text)
            # Prefix with 'llm_' to distinguish from universal features
            return {f"llm_{k}": float(v) for k, v in result.items()}
        except Exception:
            return {f"llm_{k}": 0.0 for k in self._llm_feature_names}

    def _augment_explore_features_with_llm(self):
        """Add LLM features to exploration data (retroactively)."""
        if self._llm_feature_fn is None:
            return

        for i, feats in enumerate(self._explore_features):
            state = self._explore_states[i] if i < len(self._explore_states) else ""
            action = self._explore_actions[i] if i < len(self._explore_actions) else ""
            step = self._explore_steps[i] if i < len(self._explore_steps) else 0
            llm_feats = self._extract_llm_features(state, step, action)
            feats.update(llm_feats)

        # Update feature names
        if self._explore_features:
            self._feature_names_all = sorted(self._explore_features[0].keys())

        # Feature filter: remove LLM features with low correlation to utility
        if self._feature_filter and len(self._explore_features) > 5:
            U = np.array(self._explore_utils)
            if len(U) == len(self._explore_features):
                filtered_out = []
                for fname in list(self._llm_feature_names):
                    llm_key = f"llm_{fname}"
                    vals = np.array([f.get(llm_key, 0) for f in self._explore_features])
                    if np.std(vals) < 1e-8:
                        # Constant feature → remove
                        filtered_out.append(llm_key)
                        continue
                    corr = abs(np.corrcoef(vals, U)[0, 1])
                    if np.isnan(corr) or corr < 0.05:
                        filtered_out.append(llm_key)

                if filtered_out:
                    for feats in self._explore_features:
                        for key in filtered_out:
                            feats.pop(key, None)
                    logger.info(
                        f"[{self.VARIANT}] Feature filter: removed {len(filtered_out)} "
                        f"low-correlation features: {filtered_out}"
                    )

        # Update feature names
        if self._explore_features:
            self._feature_names_all = sorted(self._explore_features[0].keys())

        n_llm_remaining = sum(1 for k in self._feature_names_all if k.startswith('llm_'))
        logger.info(
            f"[{self.VARIANT}] Augmented {len(self._explore_features)} exploration "
            f"samples with {n_llm_remaining} LLM features"
            f"{' (filtered from ' + str(len(self._llm_feature_names)) + ')' if self._feature_filter else ''}"
        )

    def get_estimated_pattern(self) -> Dict[str, Any]:
        pattern = super().get_estimated_pattern()
        pattern.update({
            "method": "self_evolving",
            "reflect_backend": self._reflect_backend,
            "llm_features": self._llm_feature_names,
            "llm_code": self._llm_feature_code,
            "reflect_log": self._reflect_log,
        })
        return pattern

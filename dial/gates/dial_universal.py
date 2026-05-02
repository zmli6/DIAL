"""
E3: Principled SCG — Auto Feature Selection + CMDP Threshold
=============================================================

Automatically builds a feature pool, selects optimal features via LASSO,
and uses CMDP-optimal threshold.

No manual feature engineering needed. Threshold is theoretically motivated.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from dial.gates._scg_base import SCGBase

logger = logging.getLogger("DIAL")


class PrincipledSCGGate(SCGBase):
    """
    Principled SCG with auto feature selection.

    During exploration: collect (auto_features, utility) pairs.
    At transition: LASSO selects best features, set CMDP threshold.
    During exploitation: selected features → LR → trigger if P > threshold.

    Parameters
    ----------
    lambda_cost : float
        C_rollout / C_base ratio for CMDP threshold.
    max_features : int
        Max features to select via LASSO.
    pca_model : optional
        Pre-fitted PCA for hidden state features.
    """

    VARIANT = "principled_scg"

    def __init__(
        self,
        lambda_cost: float = 10.0,
        max_features: int = 10,
        pca_model=None,
        threshold_mode: str = "adaptive_lambda",
        explore_mode: str = "random",
        regularizer: str = "l1",
        **kwargs,
    ):
        """
        threshold_mode:
            "adaptive_lambda" — 自适应 λ sweep (default, 方案 B)
            "fbeta"           — F-beta score, β from positive_rate (完全 online)
        explore_mode:
            "random"     — random explore_rate gate (default, original behavior)
            "optimistic" — always trigger until enough data per state group,
                           then switch to random for that group
        regularizer:
            "l1"         — LASSO (DIAL default; sparse + shrinkage)
            "l2"         — Ridge (shrinkage only; keeps all pre-filtered feats)
            "none"       — unregularized logistic on full pre-filtered pool
            "elastic"    — Elastic Net (l1_ratio=0.5)
            "mi_topk"    — hard MI top-k selection, then unregularized LR
        """
        super().__init__(**kwargs)
        self._lambda_cost = lambda_cost
        self._max_features = max_features
        self._pca = pca_model
        self._threshold_mode = threshold_mode
        self._explore_mode = explore_mode
        self._regularizer = regularizer

        # Model (set at transition)
        self._model = None
        self._scaler = None
        self._selected_features = None
        self._cmdp_threshold = 0.5

        # Exploration data
        self._explore_features = []
        self._explore_utils = []
        self._feature_names_all = None

        # Optimistic exploration state
        self._optimistic_visit_counts = {}  # (step_bucket, state_cat) → count
        self._optimistic_min_visits = 3     # min visits before allowing non-trigger

        # Online PCA support
        self._online_pca = False
        self._online_pca_buffer = []

    def _build_feature_pool(self, ctx: Dict) -> Dict[str, float]:
        """Build candidate feature pool from context."""
        features = {}
        signals = ctx.get("signals", ctx)

        # === Type U: Universal Features ===
        features['step_count'] = float(signals.get('step_count', 0) or 0)
        features['token_entropy'] = float(signals.get('token_entropy', 0) or 0)
        features['evidence_count'] = float(signals.get('evidence_count', 0) or 0)
        features['num_available_actions'] = float(signals.get('num_available_actions', 0) or 0)

        is_finish = signals.get('is_finish_proposed', False)
        features['is_finish'] = 1.0 if is_finish else 0.0

        # Derived features
        max_steps = float(signals.get('max_steps', 10) or 10)
        features['step_ratio'] = features['step_count'] / max(max_steps, 1)
        features['entropy_sq'] = features['token_entropy'] ** 2
        features['step_x_entropy'] = features['step_count'] * features['token_entropy']

        # === Type H: Hidden State PCA Features ===
        hidden_state = ctx.get("hidden_state")
        if hidden_state is not None and self._pca is not None:
            pca_feats = self._pca.transform(hidden_state.reshape(1, -1))[0][:20]
            for i, v in enumerate(pca_feats):
                features[f'h_pca_{i}'] = float(v)
        else:
            for i in range(20):
                features[f'h_pca_{i}'] = 0.0

        # === Type E: Auto-extracted from state text ===
        obs = ctx.get('_obs')
        if obs is not None:
            state_text = str(obs) if not isinstance(obs, str) else obs
            features['state_length'] = float(len(state_text.split()))
            features['num_numbers'] = float(len(re.findall(r'\d+', state_text)))
            features['has_error'] = float(bool(re.search(r'error|fail|invalid', state_text, re.I)))
        else:
            features['state_length'] = 0.0
            features['num_numbers'] = 0.0
            features['has_error'] = 0.0

        return features

    def should_rollout(self, consistency: float, **ctx) -> bool:
        self._step_counter += 1

        if self.phase == "exploration":
            import random
            if self._explore_mode == "optimistic":
                # Optimistic: always trigger until enough visits per state group
                step = ctx.get("step_count", ctx.get("step", 0))
                state_cat = ctx.get("state_category", "unknown")
                key = (min(step, 5), state_cat)  # bucket steps 0-5+
                visits = self._optimistic_visit_counts.get(key, 0)
                self._optimistic_visit_counts[key] = visits + 1
                if visits < self._optimistic_min_visits:
                    decision = True  # not enough data → trigger
                else:
                    decision = random.random() < self.explore_rate
            else:
                decision = random.random() < self.explore_rate

            # Collect hidden states for online PCA
            hidden_state = ctx.get("hidden_state")
            if self._online_pca and hidden_state is not None:
                self._online_pca_buffer.append(hidden_state.copy())

            # Build and store features
            feats = self._build_feature_pool(ctx)
            if self._feature_names_all is None:
                self._feature_names_all = sorted(feats.keys())
            self._current_feats = feats

            if len(self.buffer) >= self.min_cal_points:
                # Fit online PCA before transition if enough data
                if self._online_pca and len(self._online_pca_buffer) >= 20 and self._pca is None:
                    self._fit_online_pca()
                    # Rebuild features with PCA now available
                    for i, stored_feats in enumerate(self._explore_features):
                        if i < len(self._online_pca_buffer):
                            h = self._online_pca_buffer[i]
                            pca_feats = self._pca.transform(h.reshape(1, -1))[0][:20]
                            for j, v in enumerate(pca_feats):
                                stored_feats[f'h_pca_{j}'] = float(v)
                self._on_transition()
                self.phase = "exploitation"
                logger.info(
                    f"[{self.VARIANT}] → exploitation "
                    f"(selected {len(self._selected_features or [])} features, "
                    f"threshold={self._cmdp_threshold:.4f})"
                )
            return decision

        else:
            if self._model is None:
                # If threshold is high (fallback for low pos_rate), don't trigger
                if self._cmdp_threshold >= 0.9:
                    return False
                return True

            feats = self._build_feature_pool(ctx)
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

    def update(self, consistency: float, utility: float, **ctx):
        super().update(consistency, utility, **ctx)

        if hasattr(self, '_current_feats'):
            self._explore_features.append(self._current_feats)
            self._explore_utils.append(utility)

    def _exploit_decision(self, consistency: float, **ctx) -> bool:
        return True  # Logic in should_rollout

    def _on_transition(self):
        """LASSO feature selection + CMDP threshold."""
        if len(self._explore_features) < 10:
            return

        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import mutual_info_classif

        # Build feature matrix
        names = self._feature_names_all or sorted(self._explore_features[0].keys())
        X = np.array([[f.get(n, 0) for n in names] for f in self._explore_features])
        U = np.array(self._explore_utils)
        y = (U > 0).astype(int)

        if y.sum() == 0 or y.sum() == len(y):
            positive_rate = y.mean()
            # Low positive rate → rollout rarely useful → almost never trigger
            if positive_rate < 0.02:
                self._cmdp_threshold = 0.95
                logger.info(
                    f"[{self.VARIANT}] fallback: single class, "
                    f"pos_rate={positive_rate:.3f} < 2% → threshold=0.95 (almost never trigger)"
                )
            else:
                self._cmdp_threshold = 0.5
                logger.warning(f"[{self.VARIANT}] Single class, using default threshold=0.5")
            self._lambda_adaptive = None
            return

        # Stage 1: MI-based pre-filtering
        scaler_pre = StandardScaler()
        X_scaled = scaler_pre.fit_transform(X)

        # Replace NaN/inf
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

        try:
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        except Exception:
            mi_scores = np.zeros(len(names))

        # Keep top features by MI
        top_k = min(30, len(names))
        top_idx = np.argsort(mi_scores)[::-1][:top_k]
        pre_filtered_names = [names[i] for i in top_idx]
        X_filtered = X_scaled[:, top_idx]

        # Stage 2: regularized selection (regularizer-ablation switch)
        reg = getattr(self, "_regularizer", "l1")
        cv_folds = min(5, max(2, int(y.sum())))
        from sklearn.linear_model import LogisticRegression
        try:
            if reg == "l1":
                fit = LogisticRegressionCV(
                    penalty='l1', solver='saga', cv=cv_folds,
                    max_iter=2000, class_weight='balanced', random_state=42,
                )
                fit.fit(X_filtered, y)
                nonzero = np.abs(fit.coef_[0]) > 1e-6
                if nonzero.sum() == 0:
                    nonzero[:self._max_features] = True
                selected_idx = np.where(nonzero)[0][:self._max_features]
            elif reg == "l2":
                fit = LogisticRegressionCV(
                    penalty='l2', solver='lbfgs', cv=cv_folds,
                    max_iter=2000, class_weight='balanced', random_state=42,
                )
                fit.fit(X_filtered, y)
                # Ridge keeps all pre-filtered features
                selected_idx = np.arange(X_filtered.shape[1])
            elif reg == "none":
                fit = LogisticRegression(
                    penalty=None, solver='lbfgs', max_iter=2000,
                    class_weight='balanced', random_state=42,
                )
                fit.fit(X_filtered, y)
                selected_idx = np.arange(X_filtered.shape[1])
            elif reg == "elastic":
                fit = LogisticRegressionCV(
                    penalty='elasticnet', solver='saga', cv=cv_folds,
                    l1_ratios=[0.5], max_iter=2000,
                    class_weight='balanced', random_state=42,
                )
                fit.fit(X_filtered, y)
                nonzero = np.abs(fit.coef_[0]) > 1e-6
                if nonzero.sum() == 0:
                    nonzero[:self._max_features] = True
                selected_idx = np.where(nonzero)[0][:self._max_features]
            elif reg == "mi_topk":
                # Top-3 by MI (already sorted from Stage 1: top_idx)
                k = min(3, X_filtered.shape[1])
                selected_idx = np.arange(k)
            else:
                raise ValueError(f"Unknown regularizer: {reg}")
        except Exception as e:
            logger.warning(f"[{self.VARIANT}] regularizer={reg} fit failed: {e}; fallback to first {self._max_features} feats")
            selected_idx = np.arange(min(self._max_features, len(pre_filtered_names)))

        self._selected_features = [pre_filtered_names[i] for i in selected_idx]

        # Re-fit final model on selected features
        X_sel = np.array([[f.get(n, 0) for n in self._selected_features]
                          for f in self._explore_features])
        self._scaler = StandardScaler()
        X_sel_scaled = self._scaler.fit_transform(X_sel)
        X_sel_scaled = np.nan_to_num(X_sel_scaled, nan=0, posinf=0, neginf=0)

        if reg == "none" or reg == "mi_topk":
            self._model = LogisticRegression(
                penalty=None, solver='lbfgs', max_iter=1000,
                class_weight='balanced', random_state=42,
            )
        else:
            self._model = LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42,
            )
        self._model.fit(X_sel_scaled, y)

        probs = self._model.predict_proba(X_sel_scaled)[:, 1]
        positive_rate = y.mean()

        if self._threshold_mode == "fbeta":
            self._tune_threshold_fbeta(probs, y, positive_rate)
        else:
            self._tune_threshold_adaptive_lambda(probs, y, U, positive_rate)

        logger.info(
            f"[{self.VARIANT}] Selected {len(self._selected_features)} features: "
            f"{self._selected_features}, threshold={self._cmdp_threshold:.4f}"
        )

    def _tune_threshold_adaptive_lambda(self, probs, y, U, positive_rate):
        """Adaptive λ threshold tuning (方案 B): λ = gain / (cost - 1)."""
        mean_pos_util = float(U[U > 0].mean()) if (U > 0).sum() > 0 else 0
        always_sr_gain = positive_rate * mean_pos_util
        always_cost = 1.0 + self._lambda_cost

        if always_cost > 1.01 and always_sr_gain > 0:
            lambda_adaptive = always_sr_gain / (always_cost - 1.0)
        elif always_sr_gain <= 0:
            lambda_adaptive = -0.01
        else:
            lambda_adaptive = 0.05

        best_adj, best_t = -float('inf'), 0.5
        for t in np.linspace(0.05, 0.95, 50):
            triggered = probs > t
            trigger_rate = triggered.mean()
            if trigger_rate < 0.01 or trigger_rate > 0.99:
                continue
            mean_util_triggered = U[triggered].mean() if triggered.sum() > 0 else 0
            adj = mean_util_triggered * trigger_rate - lambda_adaptive * trigger_rate
            if adj > best_adj:
                best_adj = adj
                best_t = t
        self._cmdp_threshold = best_t
        self._lambda_adaptive = lambda_adaptive

        logger.info(
            f"[{self.VARIANT}] adaptive_lambda: λ={lambda_adaptive:.4f}, "
            f"pos_rate={positive_rate:.3f}, threshold={self._cmdp_threshold:.4f}"
        )

    def _tune_threshold_fbeta(self, probs, y, positive_rate):
        """F-beta threshold tuning: β from positive_rate, fully online."""
        # β = sqrt(positive_rate / (1 - positive_rate))
        # Low positive_rate → low β → precision-focused → fewer triggers
        pos_rate_clipped = np.clip(positive_rate, 0.005, 0.995)
        beta = np.sqrt(pos_rate_clipped / (1 - pos_rate_clipped))

        # Fallback: if positive_rate < 1%, almost never trigger
        if positive_rate < 0.02:
            self._cmdp_threshold = 0.95
            self._lambda_adaptive = None
            self._fbeta = beta
            logger.info(
                f"[{self.VARIANT}] fbeta: pos_rate={positive_rate:.3f} < 2% "
                f"→ fallback threshold=0.95 (almost never trigger)"
            )
            return

        best_fbeta, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 50):
            pred_pos = (probs > t).astype(int)
            tp = ((pred_pos == 1) & (y == 1)).sum()
            fp = ((pred_pos == 1) & (y == 0)).sum()
            fn = ((pred_pos == 0) & (y == 1)).sum()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)

            if precision + recall < 1e-8:
                continue

            fb = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            if fb > best_fbeta:
                best_fbeta = fb
                best_t = t

        self._cmdp_threshold = best_t
        self._lambda_adaptive = None
        self._fbeta = beta

        trigger_rate = (probs > best_t).mean()
        logger.info(
            f"[{self.VARIANT}] fbeta: β={beta:.3f} (pos_rate={positive_rate:.3f}), "
            f"best_F{beta:.2f}={best_fbeta:.3f}, threshold={best_t:.4f}, "
            f"trigger_rate={trigger_rate:.1%}"
        )

    def _fit_online_pca(self):
        """Fit PCA from exploration hidden states (online mode)."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        H = np.array(self._online_pca_buffer)
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        n_comp = min(20, H.shape[0] - 1, H.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(H_scaled)

        original_transform = pca.transform
        def scaled_transform(X_new):
            return original_transform(scaler.transform(X_new))
        pca.transform = scaled_transform

        self._pca = pca
        logger.info(
            f"[{self.VARIANT}] Online PCA fitted: n_samples={len(H)}, "
            f"n_components={n_comp}, explained_var={pca.explained_variance_ratio_.sum():.3f}"
        )

    def get_estimated_pattern(self) -> Dict[str, Any]:
        return {
            "method": "principled_scg",
            "selected_features": self._selected_features,
            "cmdp_threshold": self._cmdp_threshold,
            "lambda_adaptive": getattr(self, '_lambda_adaptive', None),
            "n_explore": len(self._explore_features),
            "lambda_cost": self._lambda_cost,
        }

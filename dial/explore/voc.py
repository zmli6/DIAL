"""
Environment-agnostic VoC (Value of Computation) estimator.

Works with any BaseEnv + GenericForwardValuator + GenericRetrospectiveValuator.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind

from .envs.base import BaseEnv
from .generic_valuator import GenericForwardValuator, GenericRetrospectiveValuator

logger = logging.getLogger("DIAL")


class GenericVoCEstimator:
    """
    Estimates VoC signal for any environment type.

    consistency = |V_F(s,a) - Ṽ_R(s,a)|
    utility     = max(Ṽ_R(s,a')) − Ṽ_R(s, a_proposed)
    """

    def __init__(
        self,
        forward: GenericForwardValuator,
        retro: GenericRetrospectiveValuator,
        rollout_prob: float = 0.5,
    ):
        self.forward = forward
        self.retro = retro
        self.rollout_prob = rollout_prob

    def compute_consistency(self, env: BaseEnv, action: int) -> float:
        v_f = self.forward.compute_value(env, action)
        v_r = self.retro.compute_value(env, action)
        return abs(v_f - v_r)

    def compute_consistency_all(self, env: BaseEnv) -> Dict[int, Dict]:
        results = {}
        for a in env.get_actions():
            v_f = self.forward.compute_value(env, a)
            v_r = self.retro.compute_value(env, a)
            results[a] = {
                "v_forward": float(v_f),
                "v_retro": float(v_r),
                "consistency": abs(v_f - v_r),
            }
        return results

    def compute_rollout_utility(
        self, env: BaseEnv, proposed_action: int, top_k: int = 3,
    ) -> Dict[str, Any]:
        actions = env.get_actions()[:top_k]
        action_values = {a: self.retro.compute_value(env, a) for a in actions}

        best_action = max(action_values, key=action_values.get)
        best_val = action_values[best_action]
        proposed_val = action_values.get(
            proposed_action, self.retro.compute_value(env, proposed_action)
        )
        utility = best_val - proposed_val

        return {
            "utility": float(utility),
            "best_action": best_action,
            "proposed_action": proposed_action,
            "decision_changed": best_action != proposed_action,
            "action_values": {int(k): float(v) for k, v in action_values.items()},
        }

    def sample_voc(
        self,
        env: BaseEnv,
        num_episodes: int = 100,
        seed_start: int = 42,
        sample_every: int = 3,
        verbose: bool = True,
    ) -> List[Dict]:
        from tqdm import tqdm

        data_points = []
        iterator = range(num_episodes)
        if verbose:
            iterator = tqdm(iterator, desc=f"VoC sampling [{env.ENV_TYPE}]")

        for ep_idx in iterator:
            obs, info = env.reset(seed=seed_start + ep_idx)
            step_count = 0
            terminated = truncated = False

            while not (terminated or truncated):
                if step_count % sample_every == 0:
                    consistency_data = self.compute_consistency_all(env)

                    if not consistency_data:
                        action = env.greedy_oracle_action()
                        obs, reward, terminated, truncated, info = env.step(action)
                        step_count += 1
                        continue

                    proposed = max(
                        consistency_data,
                        key=lambda a: consistency_data[a]["v_forward"],
                    )
                    avg_c = np.mean([d["consistency"] for d in consistency_data.values()])
                    max_c = max(d["consistency"] for d in consistency_data.values())

                    if np.random.random() < self.rollout_prob:
                        util_data = self.compute_rollout_utility(env, proposed)
                        data_points.append({
                            "episode": ep_idx,
                            "step": step_count,
                            "consistency_avg": float(avg_c),
                            "consistency_max": float(max_c),
                            "consistency_proposed": float(
                                consistency_data[proposed]["consistency"]
                            ),
                            "utility": util_data["utility"],
                            "decision_changed": util_data["decision_changed"],
                            "proposed_action": proposed,
                            "best_action": util_data["best_action"],
                            "state_type": env.classify_state(),
                            "env_type": env.ENV_TYPE,
                        })

                action = env.greedy_oracle_action()
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1

        logger.info(f"VoC [{env.ENV_TYPE}]: {len(data_points)} data points")
        return data_points

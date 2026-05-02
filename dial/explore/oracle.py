"""
Environment-agnostic Oracle Value Collector and Controlled Value Generator.

Works with any BaseEnv implementation.  Delegates environment-specific
logic (greedy oracle action, state key, success detection) to the
environment adapter.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from .envs.base import BaseEnv
from .generic_valuator import GenericRetrospectiveValuator
from .utils import NumpyEncoder

logger = logging.getLogger("DIAL")


class GenericOracleCollector:
    """
    Collects oracle (ground truth) values for (state, action) pairs
    using multi-rollout estimation.  Works with any BaseEnv.
    """

    def __init__(
        self,
        num_rollouts: int = 20,
        rollout_horizon: int = 10,
        epsilon: float = 0.3,
        gamma: float = 0.99,
        sample_every: int = 3,
    ):
        self.num_rollouts = num_rollouts
        self.sample_every = sample_every

        self._retro = GenericRetrospectiveValuator(
            horizon=rollout_horizon,
            num_samples=num_rollouts,
            gamma=gamma,
            epsilon=epsilon,
        )

    def collect(
        self,
        env: BaseEnv,
        num_episodes: int = 100,
        seed_start: int = 42,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        from tqdm import tqdm

        oracle_values: Dict[str, Dict] = {}
        episode_summaries = []
        total_pairs = 0
        total_successes = 0

        iterator = range(num_episodes)
        if verbose:
            iterator = tqdm(iterator, desc=f"Collecting oracle [{env.ENV_TYPE}]")

        for ep_idx in iterator:
            seed = seed_start + ep_idx
            obs, info = env.reset(seed=seed)
            ep_reward = 0.0
            ep_steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                if ep_steps % self.sample_every == 0:
                    state_key = env.make_state_key()

                    if state_key not in oracle_values:
                        oracle_values[state_key] = {}

                    actions = env.get_actions()
                    for action in actions:
                        if action not in oracle_values[state_key]:
                            detailed = self._retro.compute_value_detailed(env, action)
                            oracle_values[state_key][action] = {
                                "mean": detailed["value"],
                                "std": detailed["std"],
                                "returns": detailed["returns"],
                            }
                            total_pairs += 1

                # Advance with greedy oracle action
                action = env.greedy_oracle_action()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps += 1

            success = env.is_success(reward, terminated, info)
            if success:
                total_successes += 1

            episode_summaries.append({
                "episode": ep_idx, "seed": seed,
                "reward": float(ep_reward), "steps": ep_steps,
                "success": success,
            })

        success_rate = total_successes / max(num_episodes, 1)
        all_stds = [
            v["std"]
            for sv in oracle_values.values()
            for v in sv.values()
        ]
        avg_std = float(np.mean(all_stds)) if all_stds else 0.0

        quality = {
            "success_rate": success_rate,
            "avg_std": avg_std,
            "num_states": len(oracle_values),
            "num_pairs": total_pairs,
            "num_episodes": num_episodes,
            "env_type": env.ENV_TYPE,
        }

        logger.info(
            f"Oracle [{env.ENV_TYPE}]: {len(oracle_values)} states, "
            f"{total_pairs} pairs, SR={success_rate:.2%}"
        )

        return {"values": oracle_values, "episodes": episode_summaries, "quality": quality}

    def validate_oracle(
        self,
        env: BaseEnv,
        num_episodes: int = 100,
        seed_start: int = 1000,
    ) -> Dict[str, Any]:
        successes = 0
        total_reward = 0.0

        for ep_idx in range(num_episodes):
            obs, info = env.reset(seed=seed_start + ep_idx)
            ep_reward = 0.0
            steps = 0
            terminated = truncated = False

            while not (terminated or truncated) and steps < env.max_steps:
                action = env.greedy_oracle_action()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1

            if env.is_success(reward, terminated, info):
                successes += 1
            total_reward += ep_reward

        sr = successes / max(num_episodes, 1)
        return {
            "success_rate": sr,
            "avg_reward": total_reward / max(num_episodes, 1),
            "num_episodes": num_episodes,
            "pass": sr > 0.85,
        }

    @staticmethod
    def save(data: Dict, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Oracle saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> Dict:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Oracle loaded from {filepath}")
        return data

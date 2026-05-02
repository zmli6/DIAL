#!/usr/bin/env python3
"""
Phase 6.1 — Counterfactual Paired Evaluation: Phase A Sanity Check
===================================================================

Validates the core assumptions of the counterfactual Δ(s) approach:
  (a) Δ(s) variance: is the signal stable across repeated pairs?
  (b) Δ(s) vs step: does early-step Δ dominate?
  (c) Proxy vs full: does one-step proxy correlate with full-branch Δ?

For each visited state in an always-trigger episode:
  1. Snapshot state s
  2. TRIGGER branch: run rollout → choose best action → continue to episode end → R_trigger
  3. Rollback to snapshot
  4. NO-TRIGGER branch: use LLM proposed action (no rollout) → continue to episode end → R_no_trigger
  5. Δ(s) = R_trigger - R_no_trigger

Optionally repeat each pair N_REPEATS times for variance estimation.

Usage:
    python experiments/p6_counterfactual_sanity.py \\
        --config configs/phase6_fever.yaml \\
        --env fever \\
        --episodes 50 \\
        --repeats 3 \\
        --max-paired-steps 2
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dial.envs import make_env
from dial.inference.proposer import ActionProposer
from dial.utils.legacy import NumpyEncoder

from experiments.p5_new_env_experiments import (
    compute_rollout_utility_new,
    extract_signals_new_env,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DIAL")


def run_branch(env, proposer, rollout_proposer, rollout_cfg, use_rollout: bool,
               max_steps: int = 10) -> Dict[str, Any]:
    """
    Run a branch (trigger or no-trigger) from the current env state
    until episode ends. Returns outcome dict.

    If use_rollout=True: at each step, do rollout and pick best action.
    If use_rollout=False: at each step, use LLM proposed action directly.
    """
    total_reward = 0.0
    steps = 0
    success = False

    while steps < max_steps:
        if env._done:
            break

        obs = env._obs_text

        # Propose action
        try:
            result = proposer.choose_action_with_logprobs(env, obs)
        except Exception:
            try:
                action = proposer.choose_action(env, obs)
                result = {"action": action, "token_logprobs": [], "text": ""}
            except Exception:
                break

        proposed_action = result["action"]

        if use_rollout:
            # Trigger rollout and pick best action
            try:
                env_type = getattr(env, "ENV_TYPE", "fever")
                rollout_result = compute_rollout_utility_new(
                    env, env_type, rollout_proposer, rollout_cfg, proposed_action,
                )
                if rollout_result["utility"] > 0:
                    chosen_action = rollout_result["best_action"]
                else:
                    chosen_action = proposed_action
            except Exception:
                chosen_action = proposed_action
        else:
            chosen_action = proposed_action

        obs, reward, terminated, truncated, info = env.step(chosen_action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            success = env.is_success(reward, terminated, info)
            break

    return {
        "total_reward": float(total_reward),
        "steps": steps,
        "success": success,
    }


def run_counterfactual_episode(
    env, proposer, rollout_proposer, rollout_cfg,
    episode_idx: int, seed: int,
    max_paired_steps: int = 2,
    n_repeats: int = 1,
) -> Dict[str, Any]:
    """
    Run one episode with always-trigger, collecting counterfactual pairs
    at each of the first max_paired_steps steps.
    """
    obs, info = env.reset(seed=seed)

    pairs = []
    step = 0
    episode_max_steps = env.max_steps

    while not env._done and step < episode_max_steps:
        # Snapshot current state
        state_snap = env.get_state()

        # Get proposed action (for no-trigger branch)
        try:
            result = proposer.choose_action_with_logprobs(env, obs)
        except Exception:
            try:
                action = proposer.choose_action(env, obs)
                result = {"action": action, "token_logprobs": [], "text": ""}
            except Exception:
                break

        proposed_action = result["action"]
        signals = extract_signals_new_env(env, obs, result)

        if step < max_paired_steps:
            # === Collect counterfactual pair ===
            trigger_results = []
            no_trigger_results = []

            for rep in range(n_repeats):
                # --- TRIGGER branch ---
                env.set_state(state_snap)
                env_trigger = env.deepcopy()
                trigger_out = run_branch(
                    env_trigger, proposer, rollout_proposer, rollout_cfg,
                    use_rollout=True, max_steps=episode_max_steps - step,
                )
                trigger_results.append(trigger_out)
                del env_trigger

                # --- NO-TRIGGER branch ---
                env.set_state(state_snap)
                env_no_trigger = env.deepcopy()
                no_trigger_out = run_branch(
                    env_no_trigger, proposer, rollout_proposer, rollout_cfg,
                    use_rollout=False, max_steps=episode_max_steps - step,
                )
                no_trigger_results.append(no_trigger_out)
                del env_no_trigger

            # Compute Δ(s) statistics
            trigger_successes = [r["success"] for r in trigger_results]
            no_trigger_successes = [r["success"] for r in no_trigger_results]
            trigger_rewards = [r["total_reward"] for r in trigger_results]
            no_trigger_rewards = [r["total_reward"] for r in no_trigger_results]

            delta_success = np.mean(trigger_successes) - np.mean(no_trigger_successes)
            delta_reward = np.mean(trigger_rewards) - np.mean(no_trigger_rewards)

            pair = {
                "episode": episode_idx,
                "step": step,
                "seed": seed,
                "state_category": signals.get("state_category", ""),
                "step_count": signals.get("step_count", step),
                "evidence_count": signals.get("evidence_count", 0),
                "token_entropy": signals.get("token_entropy", 0.0),
                # Trigger branch stats
                "trigger_success_rate": float(np.mean(trigger_successes)),
                "trigger_reward_mean": float(np.mean(trigger_rewards)),
                "trigger_steps_mean": float(np.mean([r["steps"] for r in trigger_results])),
                # No-trigger branch stats
                "no_trigger_success_rate": float(np.mean(no_trigger_successes)),
                "no_trigger_reward_mean": float(np.mean(no_trigger_rewards)),
                "no_trigger_steps_mean": float(np.mean([r["steps"] for r in no_trigger_results])),
                # Delta
                "delta_success": float(delta_success),
                "delta_reward": float(delta_reward),
                # Variance (if repeats > 1)
                "trigger_success_std": float(np.std(trigger_successes)) if n_repeats > 1 else None,
                "no_trigger_success_std": float(np.std(no_trigger_successes)) if n_repeats > 1 else None,
                "n_repeats": n_repeats,
            }
            pairs.append(pair)

        # === Continue main episode with always-trigger ===
        env.set_state(state_snap)

        # Do rollout and step (always-trigger main path)
        try:
            env_type = getattr(env, "ENV_TYPE", "fever")
            rollout_result = compute_rollout_utility_new(
                env, env_type, rollout_proposer, rollout_cfg, proposed_action,
            )
            if rollout_result["utility"] > 0:
                chosen = rollout_result["best_action"]
            else:
                chosen = proposed_action
        except Exception:
            chosen = proposed_action

        obs, reward, terminated, truncated, info = env.step(chosen)
        step += 1

        if terminated or truncated:
            break

    return {
        "episode": episode_idx,
        "seed": seed,
        "pairs": pairs,
        "total_steps": step,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase A: Counterfactual Δ(s) Sanity Check"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeats per state for variance estimation")
    parser.add_argument("--max-paired-steps", type=int, default=2,
                        help="Only collect pairs for first K steps")
    parser.add_argument("--seed-start", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_name = args.env
    env_section = cfg.get(env_name, {})
    env_cfg = env_section.get("environment", {"name": env_name, "type": env_name})
    proposer_cfg = env_section.get("proposer", {})
    rollout_cfg = env_section.get("rollout", {})

    output_dir = f"results/phase6/counterfactual/{env_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Counterfactual Δ(s) Sanity Check — Phase A                 ║")
    print(f"║  Env: {env_name}  |  Episodes: {args.episodes}  |  Repeats: {args.repeats}")
    print(f"║  Max paired steps: {args.max_paired_steps}  |  Seed start: {args.seed_start}")
    print(f"║  Start: {datetime.now().isoformat()}")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    # Create environment
    env = make_env(env_cfg)
    logger.info(f"Environment: {env}")

    # Create proposers
    llm_cfg = proposer_cfg.get("llm_config", {})
    base_proposer = ActionProposer(mode="llm_api", llm_config=llm_cfg)
    rollout_proposer = ActionProposer(mode="llm_api", llm_config=llm_cfg)

    # Run episodes
    all_pairs = []
    from tqdm import tqdm

    for ep in tqdm(range(args.episodes), desc=f"counterfactual [{env_name}]"):
        seed = args.seed_start + ep
        result = run_counterfactual_episode(
            env, base_proposer, rollout_proposer, rollout_cfg,
            episode_idx=ep, seed=seed,
            max_paired_steps=args.max_paired_steps,
            n_repeats=args.repeats,
        )
        all_pairs.extend(result["pairs"])

    # === Analysis ===
    print(f"\n{'='*60}")
    print(f"Results: {len(all_pairs)} counterfactual pairs from {args.episodes} episodes")
    print(f"{'='*60}\n")

    if not all_pairs:
        print("No pairs collected!")
        return

    # (a) Δ(s) distribution
    deltas = [p["delta_success"] for p in all_pairs]
    print(f"(a) Δ(s) distribution:")
    print(f"    mean  = {np.mean(deltas):.3f}")
    print(f"    std   = {np.std(deltas):.3f}")
    print(f"    min   = {np.min(deltas):.3f}")
    print(f"    max   = {np.max(deltas):.3f}")
    print(f"    >0    = {sum(1 for d in deltas if d > 0)}/{len(deltas)} ({sum(1 for d in deltas if d > 0)/len(deltas)*100:.1f}%)")
    print(f"    =0    = {sum(1 for d in deltas if d == 0)}/{len(deltas)}")
    print(f"    <0    = {sum(1 for d in deltas if d < 0)}/{len(deltas)}")

    # (a') Variance across repeats
    if args.repeats > 1:
        trigger_stds = [p["trigger_success_std"] for p in all_pairs if p["trigger_success_std"] is not None]
        no_trigger_stds = [p["no_trigger_success_std"] for p in all_pairs if p["no_trigger_success_std"] is not None]
        print(f"\n    Repeat variance (across {args.repeats} repeats per state):")
        print(f"    trigger success std:    mean={np.mean(trigger_stds):.3f}")
        print(f"    no-trigger success std: mean={np.mean(no_trigger_stds):.3f}")

    # (b) Δ(s) by step
    print(f"\n(b) Δ(s) by step:")
    step_groups = {}
    for p in all_pairs:
        s = p["step"]
        step_groups.setdefault(s, []).append(p["delta_success"])
    for s in sorted(step_groups):
        vals = step_groups[s]
        print(f"    step {s}: mean Δ = {np.mean(vals):+.3f}  (n={len(vals)}, >0: {sum(1 for v in vals if v > 0)})")

    # (c) Δ by state_category
    print(f"\n(c) Δ(s) by state_category:")
    cat_groups = {}
    for p in all_pairs:
        c = p["state_category"]
        cat_groups.setdefault(c, []).append(p["delta_success"])
    for c in sorted(cat_groups):
        vals = cat_groups[c]
        print(f"    {c:20s}: mean Δ = {np.mean(vals):+.3f}  (n={len(vals)})")

    # (d) Trigger vs No-trigger success rates
    print(f"\n(d) Branch success rates:")
    trigger_sr = np.mean([p["trigger_success_rate"] for p in all_pairs])
    no_trigger_sr = np.mean([p["no_trigger_success_rate"] for p in all_pairs])
    print(f"    trigger:    {trigger_sr:.3f}")
    print(f"    no-trigger: {no_trigger_sr:.3f}")
    print(f"    gap:        {trigger_sr - no_trigger_sr:+.3f}")

    # Save results
    output = {
        "env_name": env_name,
        "n_episodes": args.episodes,
        "n_pairs": len(all_pairs),
        "n_repeats": args.repeats,
        "max_paired_steps": args.max_paired_steps,
        "seed_start": args.seed_start,
        "pairs": all_pairs,
        "summary": {
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "delta_positive_rate": float(sum(1 for d in deltas if d > 0) / len(deltas)),
            "trigger_sr": float(trigger_sr),
            "no_trigger_sr": float(no_trigger_sr),
            "by_step": {
                str(s): {"mean": float(np.mean(v)), "n": len(v)}
                for s, v in step_groups.items()
            },
        },
        "timestamp": datetime.now().isoformat(),
    }

    out_file = os.path.join(output_dir, "phase_a_sanity_check.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()

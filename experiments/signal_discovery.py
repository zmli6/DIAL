#!/usr/bin/env python3
"""
Phase 1: Signal Discovery — HotpotQA + MBPP
=============================================================================

Core question (from README):
  哪些信号能预测 rollout utility？方向/形状是否在两个环境中一致？

Design:
  For each environment (HotpotQA, MBPP):
    1. Clean base agent (LLM proposer, temperature=0)
    2. At each step, collect signals (σ1-σ7) and compute rollout utility U
    3. Save per-step data: (episode, step, signals..., utility)
  Then run phase1_analysis.py for the multi-indicator analysis.

Signals collected:
  σ4  step_count       — episode step number
  σ1  token_entropy    — LLM output token entropy (from logprobs)
  σ5  state_category   — env.classify_state()
  σ6  evidence_count   — continuous evidence measure
  σ7  action_type      — type of proposed action
  σ_test pass_rate     — MBPP-specific: current test pass rate

Changes from Phase 0:
  - N=5 rollouts (was 3)
  - sample_every=1 (was 2)
  - Collect additional signals (entropy, evidence_count, action_type)
  - Run on both HotpotQA AND MBPP

Usage:
    # Full experiment (both environments):
    python experiments/phase1_signal_discovery.py --config configs/phase1_signal_discovery.yaml

    # HotpotQA only:
    python experiments/phase1_signal_discovery.py --config configs/phase1_signal_discovery.yaml --env hotpotqa

    # MBPP only:
    python experiments/phase1_signal_discovery.py --config configs/phase1_signal_discovery.yaml --env mbpp

    # Quick test:
    python experiments/phase1_signal_discovery.py --config configs/phase1_signal_discovery.yaml --episodes 10
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dial.envs import make_env
from dial.inference.proposer import ActionProposer
from dial.utils.legacy import setup_logger, NumpyEncoder


# ══════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ══════════════════════════════════════════════════════════════════

def compute_token_entropy(token_logprobs: List[Dict]) -> float:
    """
    Compute average token-level entropy from logprobs.

    For each token position, entropy = -Σ p_i * log(p_i) over top-K alternatives.
    Returns the mean entropy across all tokens.
    """
    if not token_logprobs:
        return 0.0

    entropies = []
    for tok_info in token_logprobs:
        top_lps = tok_info.get("top_logprobs", [])
        if not top_lps:
            continue
        # Convert logprobs to probs
        lps = [t["logprob"] for t in top_lps]
        # Numerically stable softmax over top-K
        max_lp = max(lps)
        probs = [math.exp(lp - max_lp) for lp in lps]
        total = sum(probs)
        probs = [p / total for p in probs]
        # Shannon entropy
        h = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        entropies.append(h)

    return float(np.mean(entropies)) if entropies else 0.0


def extract_hotpotqa_signals(env, obs, proposer_result: Dict) -> Dict[str, Any]:
    """Extract Phase 1 signals for HotpotQA environment."""
    # σ4: step count
    step_count = env._step_count

    # σ1: token entropy (from proposer logprobs)
    token_entropy = compute_token_entropy(
        proposer_result.get("token_logprobs", [])
    )

    # σ5: state category
    state_category = env.classify_state()

    # σ6: evidence count (continuous — number of retrieved paragraphs)
    evidence_count = len(env._context)

    # σ7: action type (search / lookup / finish)
    proposed_text = proposer_result.get("action_text", "")
    match = re.match(r'(search|lookup|finish)\[', proposed_text)
    action_type = match.group(1) if match else "unknown"

    # Additional: is this a "finish shortcut" scenario?
    # (agent chose non-finish, but rollout found finish is better)
    is_finish_proposed = action_type == "finish"

    return {
        "step_count": step_count,
        "token_entropy": float(token_entropy),
        "state_category": state_category,
        "evidence_count": evidence_count,
        "action_type": action_type,
        "is_finish_proposed": is_finish_proposed,
        # MBPP-specific signals (N/A for HotpotQA)
        "test_pass_rate": None,
        "error_type": None,
        "code_change_type": None,
    }


def extract_mbpp_signals(env, obs, proposer_result: Dict) -> Dict[str, Any]:
    """Extract Phase 1 signals for MBPP environment."""
    # σ4: step count
    step_count = env._step_count

    # σ1: token entropy
    token_entropy = compute_token_entropy(
        proposer_result.get("token_logprobs", [])
    )

    # σ5: state category
    state_category = env.classify_state()

    # σ6: evidence count → for MBPP, use test pass rate as the
    #     continuous progress indicator
    test_pass_rate = env.get_test_pass_rate()

    # σ7: action type → for MBPP, classify code change magnitude
    action_idx = proposer_result.get("action", 0)
    if action_idx < len(env._action_texts):
        action_code = env._action_texts[action_idx]
        code_change_type = env.get_code_change_magnitude(action_code)
    else:
        code_change_type = "unknown"

    error_type = env.get_error_type()

    return {
        "step_count": step_count,
        "token_entropy": float(token_entropy),
        "state_category": state_category,
        "evidence_count": test_pass_rate,  # continuous progress signal
        "action_type": code_change_type,   # analogous to σ7
        "is_finish_proposed": False,
        # MBPP-specific
        "test_pass_rate": test_pass_rate,
        "error_type": error_type,
        "code_change_type": code_change_type,
    }


# ══════════════════════════════════════════════════════════════════
# HOTPOTQA LLM ROLLOUT UTILITY (Phase 1: no more oracle)
# ══════════════════════════════════════════════════════════════════

def compute_hotpotqa_rollout_utility(
    env,
    rollout_proposer: ActionProposer,
    base_proposer: ActionProposer,
    rollout_cfg: Dict,
    proposed_action: int = None,
) -> Dict[str, Any]:
    """
    Compute HotpotQA utility per README Phase 1 definition using
    per-action evaluation (matching Exp A methodology).

    For each available action a at the current state:
        V(a) = mean of N LLM rollout chains (temp=0.7), each H-1 steps
               after forcing action a as the first step.

    Utility U = V(best_action) - V(proposed_action)

    This matches Exp A's approach: force each action, then LLM rollout
    for continuation. Phase 0 used oracle; Phase 1 uses LLM (temp=0.7).

    Args:
        env: Current environment state (not modified)
        rollout_proposer: LLM proposer at temperature=0.7
        base_proposer: LLM proposer at temperature=0 (only used if
                       proposed_action not provided)
        rollout_cfg: dict with num_chains, horizon, top_k_actions
        proposed_action: The base agent's chosen action (from Step A).
                        If None, will call base_proposer to determine it.

    Returns:
        dict with utility, action_values, best_action, proposed_value, etc.
    """
    N = rollout_cfg.get("num_chains", 5)
    H = rollout_cfg.get("horizon", 3)
    top_k = rollout_cfg.get("top_k_actions", 5)

    def _single_rollout(env_snapshot, first_action: int, proposer_to_use) -> float:
        """
        Force first_action, then run H-1 steps of LLM rollout.
        Returns the terminal F1 (or 0 if no finish reached).
        """
        env_copy = env_snapshot.deepcopy()
        try:
            total_return = 0.0

            # Step 1: forced action
            obs, reward, terminated, truncated, info = env_copy.step(first_action)
            total_return += reward

            if terminated or truncated:
                return total_return

            # Steps 2..H: LLM rollout policy (temperature=0.7)
            for _ in range(1, H):
                try:
                    obs_text = env_copy.get_text_description()
                    a = proposer_to_use.choose_action(env_copy, obs_text)
                    a = a if isinstance(a, int) else a
                except Exception:
                    acts = env_copy.get_actions()
                    a = acts[0] if acts else 0

                obs, reward, terminated, truncated, info = env_copy.step(a)
                total_return += reward

                if terminated or truncated:
                    break

            return total_return
        finally:
            del env_copy

    def _compute_action_value(env_snapshot, action: int) -> float:
        """Compute V(action) = mean of N rollouts starting with action."""
        returns = []
        for _ in range(N):
            ret = _single_rollout(env_snapshot, action, rollout_proposer)
            returns.append(ret)
        return float(np.mean(returns))

    # ── Get available actions (top-K) ──
    available_actions = env.get_actions()
    if len(available_actions) > top_k:
        available_actions = available_actions[:top_k]

    # ── Evaluate each available action ──
    action_values = {}
    llm_calls = 0
    for a in available_actions:
        v = _compute_action_value(env, a)
        action_values[a] = v
        # Each rollout does up to (H-1) LLM calls, N rollouts per action
        llm_calls += N * (H - 1)

    # ── Find best action ──
    if action_values:
        best_action = max(action_values, key=action_values.get)
        best_value = action_values[best_action]
    else:
        best_action = available_actions[0] if available_actions else 0
        best_value = 0.0

    # ── Proposed action value ──
    # Use the proposed_action from the caller (Step A) if provided
    if proposed_action is None:
        # Fallback: call base proposer (this shouldn't happen normally)
        base_copy = env.deepcopy()
        try:
            obs_text = base_copy.get_text_description()
            result = base_proposer.choose_action(base_copy, obs_text)
            proposed_action = result if isinstance(result, int) else result
        except Exception:
            acts = base_copy.get_actions()
            proposed_action = acts[0] if acts else 0
        finally:
            del base_copy

    proposed_value = action_values.get(proposed_action, 0.0)
    # If proposed action wasn't in top-K, evaluate it too
    if proposed_action not in action_values:
        proposed_value = _compute_action_value(env, proposed_action)
        action_values[proposed_action] = proposed_value
        llm_calls += N * (H - 1)

    utility = best_value - proposed_value

    # ── Finish shortcut detection ──
    is_finish_best = False
    best_action_name = env.get_action_name(best_action) if best_action is not None else ""
    if best_action_name.startswith("finish["):
        is_finish_best = True

    return {
        "utility": float(utility),
        "base_f1": float(proposed_value),   # V(proposed_action)
        "best_f1": float(best_value),       # V(best_action)
        "best_action": int(best_action),
        "best_action_text": best_action_name,
        "proposed_action": int(proposed_action),
        "action_values": {int(k): float(v) for k, v in action_values.items()},
        "num_actions_evaluated": len(action_values),
        "llm_calls": llm_calls,
        "is_finish_best": is_finish_best,
    }


# ══════════════════════════════════════════════════════════════════
# MBPP ROLLOUT UTILITY (per README: K code variants, not multi-step)
# ══════════════════════════════════════════════════════════════════

def compute_mbpp_rollout_utility(
    env,
    proposer: ActionProposer,
    rollout_cfg: Dict,
    proposed_action: int,
) -> Dict[str, Any]:
    """
    Compute MBPP utility per README definition:

        Rollout = 生成 K 个代码变体（temperature=0.7），各自执行单元测试
        Utility U = max(K 个变体的 test pass rate) - base 代码的 test pass rate

    This is fundamentally different from HotpotQA's multi-step rollout.
    It's a single-step "generate K alternatives and evaluate" procedure.

    Returns:
        dict with utility, variant_pass_rates, best_variant_idx, etc.
    """
    from dial.envs.mbpp_env import _safe_exec

    K = rollout_cfg.get("num_variants", 5)
    rollout_temp = rollout_cfg.get("temperature", 0.7)

    # ── Base code pass rate (the proposed action) ──
    if proposed_action < len(env._action_texts):
        base_code = env._action_texts[proposed_action]
    else:
        base_code = env._current_code or "pass"

    base_result = _safe_exec(base_code, env._test_code)
    base_pass_rate = base_result["pass_rate"]

    # ── Generate K code variants via LLM (temperature > 0) ──
    # Build the prompt from current env state
    obs_text = env._build_observation(initial=(env._step_count == 0))
    prompt = (
        "You are an expert Python programmer. Generate a solution for the following task.\n\n"
        f"{obs_text}\n\n"
        "Respond with ONLY the Python code, no explanation.\n"
    )

    messages = [{"role": "user", "content": prompt}]
    variant_pass_rates = []
    variant_codes = []
    llm_calls = 0

    for k in range(K):
        try:
            resp = proposer._client.chat.completions.create(
                model=proposer.model_name,
                messages=messages,
                temperature=rollout_temp,
                max_tokens=proposer.llm_config.get("max_tokens", 512),
            )
            code_text = resp.choices[0].message.content or ""
            # Extract code from markdown blocks if present
            import re as _re
            code_match = _re.search(r'```python\n(.*?)```', code_text, _re.DOTALL)
            if code_match:
                code_text = code_match.group(1).strip()
            else:
                # Try bare code block
                code_match = _re.search(r'```\n(.*?)```', code_text, _re.DOTALL)
                if code_match:
                    code_text = code_match.group(1).strip()

            variant_codes.append(code_text)
            result = _safe_exec(code_text, env._test_code)
            variant_pass_rates.append(result["pass_rate"])
            llm_calls += 1
        except Exception as e:
            logger.warning(f"MBPP variant generation {k} failed: {e}")
            variant_codes.append("")
            variant_pass_rates.append(0.0)

    # ── Utility = max(variant pass rates) - base pass rate ──
    if variant_pass_rates:
        best_variant_rate = max(variant_pass_rates)
        best_variant_idx = variant_pass_rates.index(best_variant_rate)
    else:
        best_variant_rate = base_pass_rate
        best_variant_idx = -1

    utility = best_variant_rate - base_pass_rate

    return {
        "utility": float(utility),
        "base_pass_rate": float(base_pass_rate),
        "best_variant_rate": float(best_variant_rate),
        "best_variant_idx": best_variant_idx,
        "variant_pass_rates": [float(r) for r in variant_pass_rates],
        "num_variants": len(variant_pass_rates),
        "llm_calls": llm_calls,
    }


# ══════════════════════════════════════════════════════════════════
# SINGLE-ENVIRONMENT EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run_signal_collection(
    env_name: str,
    env_cfg: Dict,
    proposer_cfg: Dict,
    retro_cfg: Dict,
    phase1_cfg: Dict,
    rollout_cfg: Optional[Dict] = None,
    num_episodes: int = 200,
    seed_start: int = 42,
    output_dir: str = "results/phase1_signal_discovery",
    verbose: bool = False,
    seed_list: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run Phase 1 signal collection for a single environment.

    For each episode, at each step:
      1. LLM proposes action (with logprobs for entropy)
      2. Extract signals (σ1-σ7)
      3. Compute rollout utility U
      4. Record (signals, U) data point

    Returns dict with all data points and summary.

    If seed_list is provided, runs only those seeds (ignoring num_episodes/seed_start).
    This is used for MBPP-Hard mode: only run on problems the base agent fails.
    """
    from tqdm import tqdm

    sample_every = phase1_cfg.get("sample_every", 1)
    top_k = phase1_cfg.get("top_k_actions", 5)

    # ── Create environment ──
    env = make_env(env_cfg)
    is_mbpp = env.ENV_TYPE == "mbpp"

    # ── Create LLM proposer (base agent: temperature=0) ──
    llm_config = dict(proposer_cfg.get("llm_config", {}))
    llm_config["temperature"] = 0.0  # greedy base agent
    proposer = ActionProposer(
        mode=proposer_cfg.get("mode", "llm_api"),
        llm_config=llm_config,
    )

    # ── Create rollout proposer (temperature=0.7) ──
    # README: "Phase 1+ 用 LLM rollout (temperature=0.7)"
    # Both HotpotQA and MBPP use LLM self as rollout policy.
    env_rollout_cfg = rollout_cfg or {}
    rollout_temp = env_rollout_cfg.get("temperature", 0.7)
    rollout_proposer_config = dict(proposer_cfg.get("llm_config", {}))
    rollout_proposer_config["temperature"] = rollout_temp
    rollout_proposer = ActionProposer(
        mode=proposer_cfg.get("mode", "llm_api"),
        llm_config=rollout_proposer_config,
    )

    print(f"\n  Environment: {env_name}")
    print(f"  LLM model:   {llm_config.get('model_name', 'unknown')}")
    print(f"  Endpoint:    {llm_config.get('endpoint', 'unknown')}")
    print(f"  Rollout:     LLM self (temperature={rollout_temp})")
    if is_mbpp:
        print(f"  MBPP rollout: K={env_rollout_cfg.get('num_variants', 5)} variants")
    else:
        N = env_rollout_cfg.get("num_chains", retro_cfg.get("num_samples", 5))
        H = env_rollout_cfg.get("horizon", retro_cfg.get("horizon", 3))
        K = env_rollout_cfg.get("top_k_actions", 5)
        print(f"  HotpotQA rollout: {K} actions × N={N} chains × H={H} steps")
    print(f"  Top K actions: {top_k}")
    print(f"  Sample every:  {sample_every}")
    print(f"  Episodes:      {num_episodes}")
    print()

    # ── Collect data ──
    all_data_points = []
    episode_summaries = []
    total_steps = 0
    t_start = time.time()
    llm_calls = 0
    llm_errors = 0

    # Support seed_list for MBPP-Hard mode
    if seed_list is not None:
        seeds_to_run = seed_list
        num_episodes = len(seeds_to_run)
        print(f"  🎯 Hard-problems mode: {num_episodes} specific seeds")
    else:
        seeds_to_run = [seed_start + i for i in range(num_episodes)]

    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc=f"Phase 1: {env_name}")

    for ep_idx in iterator:
        seed = seeds_to_run[ep_idx]
        obs, info = env.reset(seed=seed)
        step_count = 0
        terminated = truncated = False
        ep_reward = 0.0
        ep_utilities = []

        while not (terminated or truncated):
            # Step A: LLM proposes action (with logprobs for entropy)
            proposer_result = {"action": 0, "token_logprobs": [], "text": ""}
            try:
                result = proposer.choose_action_with_logprobs(env, obs)
                proposed_action = result["action"]
                proposer_result = result
                proposer_result["action_text"] = env.get_action_name(proposed_action)
                llm_calls += 1
            except Exception as e:
                logger.warning(f"LLM error ep={ep_idx} step={step_count}: {e}")
                llm_errors += 1
                actions = env.get_actions()
                proposed_action = actions[0] if actions else 0
                proposer_result["action"] = proposed_action
                proposer_result["action_text"] = env.get_action_name(proposed_action)

            # Step B: Extract signals
            if is_mbpp:
                signals = extract_mbpp_signals(env, obs, proposer_result)
            else:
                signals = extract_hotpotqa_signals(env, obs, proposer_result)

            # Step C: Compute rollout utility (if sampling this step)
            if step_count % sample_every == 0:

                if is_mbpp:
                    # ── MBPP: generate K code variants (temperature=0.7) ──
                    # README: U = max(K variants' pass rate) - base pass rate
                    mbpp_result = compute_mbpp_rollout_utility(
                        env, rollout_proposer, env_rollout_cfg, proposed_action,
                    )
                    utility = mbpp_result["utility"]
                    best_value = mbpp_result["best_variant_rate"]
                    proposed_value = mbpp_result["base_pass_rate"]
                    best_action = proposed_action  # N/A for MBPP variant rollout
                    best_action_text = f"variant_{mbpp_result['best_variant_idx']}"
                    is_finish_best = False
                    finish_shortcut = False
                    num_actions_evaluated = mbpp_result["num_variants"]
                    llm_calls += mbpp_result["llm_calls"]

                else:
                    # ── HotpotQA: per-action LLM rollout (Phase 1) ──
                    # For each available action a:
                    #   V(a) = mean of N rollouts forcing a as first step,
                    #          then LLM (temp=0.7) for H-1 remaining steps
                    # U = V(best) - V(proposed)
                    # This matches Exp A methodology with LLM rollout.
                    hotpot_result = compute_hotpotqa_rollout_utility(
                        env, rollout_proposer, proposer, env_rollout_cfg,
                        proposed_action=proposed_action,
                    )
                    utility = hotpot_result["utility"]
                    best_value = hotpot_result["best_f1"]
                    proposed_value = hotpot_result["base_f1"]
                    best_action = hotpot_result["best_action"]
                    best_action_text = hotpot_result["best_action_text"]
                    is_finish_best = hotpot_result["is_finish_best"]
                    finish_shortcut = (
                        not signals.get("is_finish_proposed", False)
                        and is_finish_best
                    )
                    num_actions_evaluated = hotpot_result["num_actions_evaluated"]
                    llm_calls += hotpot_result["llm_calls"]

                data_point = {
                    # Identifiers
                    "environment": env_name,
                    "episode": ep_idx,
                    "step": step_count,
                    "seed": seed,
                    # Proposed action info
                    "proposed_action": proposed_action,
                    "proposed_action_text": env.get_action_name(proposed_action),
                    "proposed_value": float(proposed_value),
                    # Best action info
                    "best_action": best_action,
                    "best_action_text": best_action_text,
                    "best_value": float(best_value),
                    # Utility
                    "utility": float(utility),
                    "decision_changed": best_action != proposed_action,
                    # Signals (σ1-σ7)
                    **signals,
                    # Finish shortcut (HotpotQA-specific)
                    "is_finish_best": is_finish_best,
                    "finish_shortcut": finish_shortcut,
                    # Meta
                    "num_actions_evaluated": num_actions_evaluated,
                }
                all_data_points.append(data_point)
                ep_utilities.append(utility)

            # Step D: Execute proposed action
            obs, reward, terminated, truncated, info = env.step(proposed_action)
            ep_reward += reward
            step_count += 1
            total_steps += 1

        episode_summaries.append({
            "environment": env_name,
            "episode": ep_idx,
            "seed": seed,
            "reward": float(ep_reward),
            "steps": step_count,
            "success": env.is_success(reward, terminated, info),
            "mean_utility": float(np.mean(ep_utilities)) if ep_utilities else 0.0,
            "max_utility": float(np.max(ep_utilities)) if ep_utilities else 0.0,
            "positive_utility_ratio": float(
                np.mean([u > 0 for u in ep_utilities])
            ) if ep_utilities else 0.0,
        })

        if (ep_idx + 1) % 20 == 0:
            elapsed = time.time() - t_start
            sr = np.mean([e["success"] for e in episode_summaries])
            if not verbose:
                print(f"  [{env_name}] Episode {ep_idx+1}/{num_episodes} | "
                      f"SR: {sr:.2%} | Data pts: {len(all_data_points)} | "
                      f"Time: {elapsed:.0f}s | LLM calls: {llm_calls}")

            # Incremental checkpoint — protects against job timeout
            try:
                ckpt_path = os.path.join(output_dir, "phase1_signal_data_checkpoint.json")
                with open(ckpt_path, "w") as _f:
                    json.dump(all_data_points, _f, cls=NumpyEncoder)
            except Exception:
                pass  # non-critical

    elapsed = time.time() - t_start
    env.close()

    # ── Summary statistics ──
    utilities = [dp["utility"] for dp in all_data_points]
    u = np.array(utilities) if utilities else np.array([0.0])

    summary = {
        "environment": env_name,
        "num_episodes": num_episodes,
        "num_data_points": len(all_data_points),
        "total_time_sec": round(elapsed, 1),
        "llm_calls": llm_calls,
        "llm_errors": llm_errors,
        # Core utility metrics
        "utility_mean": float(np.mean(u)),
        "utility_std": float(np.std(u)),
        "utility_median": float(np.median(u)),
        "utility_positive_ratio": float(np.mean(u > 0)),
        "decision_changed_ratio": float(
            np.mean([dp["decision_changed"] for dp in all_data_points])
        ) if all_data_points else 0.0,
        # Base agent
        "base_agent_sr": float(np.mean([e["success"] for e in episode_summaries])),
        # Signal summary
        "signal_means": {},
    }

    # Per-signal summary
    for sig in ["step_count", "token_entropy", "evidence_count", "test_pass_rate"]:
        vals = [dp[sig] for dp in all_data_points if dp.get(sig) is not None]
        if vals:
            summary["signal_means"][sig] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    # Per state_category breakdown
    cats = set(dp["state_category"] for dp in all_data_points)
    summary["per_state_category"] = {}
    for cat in cats:
        cat_utils = [dp["utility"] for dp in all_data_points
                     if dp["state_category"] == cat]
        if cat_utils:
            cu = np.array(cat_utils)
            summary["per_state_category"][cat] = {
                "count": len(cat_utils),
                "mean_utility": float(np.mean(cu)),
                "std_utility": float(np.std(cu)),
                "positive_ratio": float(np.mean(cu > 0)),
            }

    # Per step breakdown
    steps = sorted(set(dp["step"] for dp in all_data_points))
    summary["per_step"] = {}
    for s in steps:
        s_utils = [dp["utility"] for dp in all_data_points if dp["step"] == s]
        if s_utils:
            su = np.array(s_utils)
            summary["per_step"][s] = {
                "count": len(s_utils),
                "mean_utility": float(np.mean(su)),
                "std_utility": float(np.std(su)),
                "positive_ratio": float(np.mean(su > 0)),
            }

    return {
        "data_points": all_data_points,
        "episode_summaries": episode_summaries,
        "summary": summary,
    }


# ══════════════════════════════════════════════════════════════════
# FINISH SHORTCUT SUB-ANALYSIS (HotpotQA only)
# ══════════════════════════════════════════════════════════════════

def finish_shortcut_analysis(data_points: List[Dict]) -> Dict:
    """
    Separate HotpotQA utility into finish-type and strategy-type.

    finish_U:   rollout best action is finish[...] but agent chose non-finish
    strategy_U: rollout best action is search/lookup (non-trivial strategy)
    """
    finish_type = [dp for dp in data_points if dp.get("finish_shortcut", False)]
    strategy_type = [dp for dp in data_points
                     if not dp.get("is_finish_best", False) and dp["decision_changed"]]
    same_type = [dp for dp in data_points if not dp["decision_changed"]]

    results = {}
    for label, subset in [("finish_shortcut", finish_type),
                          ("strategy_change", strategy_type),
                          ("no_change", same_type)]:
        if subset:
            utils = np.array([dp["utility"] for dp in subset])
            results[label] = {
                "count": len(subset),
                "fraction": len(subset) / len(data_points),
                "mean_utility": float(np.mean(utils)),
                "std_utility": float(np.std(utils)),
                "positive_ratio": float(np.mean(utils > 0)),
            }
        else:
            results[label] = {
                "count": 0, "fraction": 0.0,
                "mean_utility": 0.0, "std_utility": 0.0, "positive_ratio": 0.0,
            }

    return results


# ══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_collection_report(
    results_by_env: Dict[str, Dict],
    output_dir: str,
):
    """Generate a data collection summary report (before full analysis)."""
    lines = [
        "# Phase 1: Signal Discovery — Data Collection Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]

    for env_name, res in results_by_env.items():
        s = res["summary"]
        lines.extend([
            f"## {env_name.upper()}",
            "",
            "### Utility Distribution",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Episodes | {s['num_episodes']} |",
            f"| Data points | {s['num_data_points']} |",
            f"| **Utility std** | **{s['utility_std']:.4f}** |",
            f"| Utility mean | {s['utility_mean']:.4f} |",
            f"| Utility median | {s['utility_median']:.4f} |",
            f"| **Positive ratio** | **{s['utility_positive_ratio']:.2%}** |",
            f"| Decision changed | {s['decision_changed_ratio']:.2%} |",
            f"| Base agent SR | {s['base_agent_sr']:.2%} |",
            f"| Time | {s['total_time_sec']:.0f}s |",
            f"| LLM calls | {s['llm_calls']} |",
            "",
            "### Per-Step Breakdown",
            "",
            "| Step | N | Mean U | Std U | U>0% |",
            "|------|---|--------|-------|------|",
        ])
        for step, info in sorted(s.get("per_step", {}).items()):
            lines.append(
                f"| {step} | {info['count']} | {info['mean_utility']:.3f} | "
                f"{info['std_utility']:.3f} | {info['positive_ratio']:.0%} |"
            )

        lines.extend([
            "",
            "### Per State-Category Breakdown",
            "",
            "| Category | N | Mean U | U>0% |",
            "|----------|---|--------|------|",
        ])
        for cat, info in sorted(s.get("per_state_category", {}).items()):
            lines.append(
                f"| {cat} | {info['count']} | {info['mean_utility']:.3f} | "
                f"{info['positive_ratio']:.0%} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "phase1_collection_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Collection report: {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Signal Discovery — Data Collection"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Config YAML file")
    parser.add_argument("--env", type=str, default=None,
                        choices=["hotpotqa", "mbpp", "both"],
                        help="Which environment(s) to run (default: both)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override num_episodes per environment")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed_start")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Override vLLM endpoint (e.g. http://localhost:8001/v1)")
    parser.add_argument("--hard-problems", type=str, default=None,
                        help="Path to hard_problem_ids.json (MBPP-Hard mode)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    phase1_cfg = cfg.get("phase1", {})
    num_episodes = args.episodes or phase1_cfg.get("num_episodes", 200)
    seed_start = args.seed or phase1_cfg.get("seed_start", 42)
    output_dir = args.output_dir or cfg.get("output", {}).get(
        "results_dir", "results/phase1_signal_discovery"
    )
    os.makedirs(output_dir, exist_ok=True)

    global logger
    logger = setup_logger(
        "DIAL",
        level="DEBUG" if args.verbose else "INFO",
        log_to_file=True,
        log_file="phase1_signal_discovery.log",
        results_dir=output_dir,
    )

    envs_to_run = []
    if args.env == "hotpotqa":
        envs_to_run = ["hotpotqa"]
    elif args.env == "mbpp":
        envs_to_run = ["mbpp"]
    else:
        envs_to_run = ["hotpotqa", "mbpp"]

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Phase 1: Signal Discovery                                      ║")
    print("║  Core Q: Which signals predict utility? Direction differences?   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Environments: {', '.join(envs_to_run)}")
    print(f"  Episodes:     {num_episodes} per env")
    print(f"  Seed start:   {seed_start}")
    print(f"  Output:       {output_dir}")
    print()

    # Load hard-problems seed list if provided (MBPP-Hard mode)
    hard_problem_seeds = None
    if args.hard_problems:
        with open(args.hard_problems) as f:
            hard_info = json.load(f)
        hard_problem_seeds = hard_info.get("hard_seeds", [])
        print(f"  🎯 MBPP-Hard mode: {len(hard_problem_seeds)} hard problems loaded")
        print(f"     from: {args.hard_problems}")
        print()

    results_by_env = {}

    for env_name in envs_to_run:
        env_section = cfg.get(env_name, {})
        env_cfg = env_section.get("environment", {})
        proposer_cfg = env_section.get("proposer", {})
        retro_cfg = env_section.get("retrospective", {})
        rollout_cfg = env_section.get("rollout", {})

        # Allow endpoint override via CLI or env var (for parallel jobs on different ports)
        endpoint_override = args.endpoint or os.environ.get("DIAL_VLLM_ENDPOINT")
        if endpoint_override:
            proposer_cfg = dict(proposer_cfg)  # copy to avoid mutating config
            llm_cfg = dict(proposer_cfg.get("llm_config", {}))
            llm_cfg["endpoint"] = endpoint_override
            proposer_cfg["llm_config"] = llm_cfg

        # Determine seed list for this environment
        env_seed_list = None
        if hard_problem_seeds is not None and env_name == "mbpp":
            env_seed_list = hard_problem_seeds

        env_output_dir = os.path.join(output_dir, env_name)
        os.makedirs(env_output_dir, exist_ok=True)

        ep_display = len(env_seed_list) if env_seed_list else num_episodes
        print("═" * 65)
        print(f"  DATA COLLECTION: {env_name.upper()}")
        if env_seed_list:
            print(f"  Hard problems: {len(env_seed_list)}  |  Seeds: [{env_seed_list[0]}..{env_seed_list[-1]}]")
        else:
            print(f"  Episodes: {num_episodes}  |  Seed: {seed_start}")
        print("═" * 65)

        results = run_signal_collection(
            env_name=env_name,
            env_cfg=env_cfg,
            proposer_cfg=proposer_cfg,
            retro_cfg=retro_cfg,
            phase1_cfg=phase1_cfg,
            rollout_cfg=rollout_cfg,
            num_episodes=num_episodes,
            seed_start=seed_start,
            output_dir=env_output_dir,
            verbose=args.verbose,
            seed_list=env_seed_list,
        )
        results_by_env[env_name] = results

        # Save raw data
        json_path = os.path.join(env_output_dir, "phase1_signal_data.json")
        with open(json_path, "w") as f:
            json.dump(results["data_points"], f, indent=2, cls=NumpyEncoder)
        print(f"\n  Raw data: {json_path}")

        # Save CSV
        if results["data_points"]:
            csv_path = os.path.join(env_output_dir, "phase1_signal_data.csv")
            # Flatten for CSV (skip nested dicts)
            csv_keys = [k for k in results["data_points"][0].keys()
                        if not isinstance(results["data_points"][0][k], (dict, list))]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_keys)
                writer.writeheader()
                for dp in results["data_points"]:
                    row = {k: dp.get(k) for k in csv_keys}
                    writer.writerow(row)
            print(f"  CSV data: {csv_path}")

        # Save episode summaries
        ep_path = os.path.join(env_output_dir, "phase1_episode_summaries.json")
        with open(ep_path, "w") as f:
            json.dump(results["episode_summaries"], f, indent=2, cls=NumpyEncoder)

        # Save summary
        summary_path = os.path.join(env_output_dir, "phase1_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results["summary"], f, indent=2, cls=NumpyEncoder)

        # Print summary
        s = results["summary"]
        print()
        print(f"  ╔═══════════════════════════════════════════╗")
        print(f"  ║  {env_name.upper()} SIGNAL COLLECTION SUMMARY     ║")
        print(f"  ╠═══════════════════════════════════════════╣")
        print(f"  ║  Data points:      {s['num_data_points']:<22d}║")
        print(f"  ║  Utility mean:     {s['utility_mean']:<22.4f}║")
        print(f"  ║  Utility std:      {s['utility_std']:<22.4f}║")
        print(f"  ║  Positive ratio:   {s['utility_positive_ratio']:<22.2%}║")
        print(f"  ║  Base agent SR:    {s['base_agent_sr']:<22.2%}║")
        print(f"  ╚═══════════════════════════════════════════╝")

        # HotpotQA finish shortcut analysis
        if env_name == "hotpotqa" and phase1_cfg.get("finish_shortcut_analysis"):
            print("\n  Finish Shortcut Analysis:")
            fs = finish_shortcut_analysis(results["data_points"])
            fs_path = os.path.join(env_output_dir, "phase1_finish_shortcut.json")
            with open(fs_path, "w") as f:
                json.dump(fs, f, indent=2, cls=NumpyEncoder)
            for label, info in fs.items():
                print(f"    {label}: N={info['count']} ({info['fraction']:.1%}), "
                      f"mean_U={info['mean_utility']:.3f}, U>0={info['positive_ratio']:.0%}")

    # ── Generate collection report ──
    print()
    generate_collection_report(results_by_env, output_dir)

    print()
    print("═" * 65)
    print("  ✅ Phase 1: Data Collection Complete")
    print()
    print("  Next step: Run analysis with:")
    print(f"    python experiments/phase1_analysis.py --data-dir {output_dir}")
    print("═" * 65)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

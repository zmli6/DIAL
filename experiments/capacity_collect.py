#!/usr/bin/env python3
"""
Phase 6 B1: Data Collection for New Environments
==================================================

Runs always_trigger episodes with HF engine to collect:
  - state_texts.json: prompts at each step
  - step_data.npz: hidden_states, utilities, signals
  - multi_layer_data.npz: multi-layer hidden states (9 layers)

For environments that don't have Phase 5 data (twexpress, babyai, plancraft).

Usage:
    python experiments/p6_b1_data_collection.py \
        --config configs/phase5_twexpress.yaml \
        --env twexpress --seed 42 --episodes 200
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dial.envs import make_env
from dial.inference.proposer import ActionProposer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DIAL")

# Layers to extract
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]


def compute_rollout_utility_generic(env, rollout_proposer, rollout_cfg, proposed_action):
    """Generic rollout utility for any text-based environment."""
    N = rollout_cfg.get("num_chains", 3)
    H = rollout_cfg.get("horizon", 3)

    def _single_rollout(first_action):
        env_copy = env.deepcopy()
        try:
            total = 0.0
            obs, reward, term, trunc, info = env_copy.step(first_action)
            total += reward
            if term or trunc:
                return total
            for _ in range(1, H):
                try:
                    obs_text = env_copy.get_text_description() if hasattr(env_copy, 'get_text_description') else str(obs)
                    a = rollout_proposer.choose_action(env_copy, obs_text)
                except Exception:
                    acts = env_copy.get_actions()
                    a = acts[0] if acts else 0
                obs, reward, term, trunc, info = env_copy.step(a)
                total += reward
                if term or trunc:
                    break
            return total
        except Exception:
            return 0.0
        finally:
            del env_copy

    # Value of proposed action
    proposed_values = [_single_rollout(proposed_action) for _ in range(N)]
    proposed_value = np.mean(proposed_values)

    # Value of alternative actions
    actions = env.get_actions()
    if len(actions) > 5:
        actions = actions[:5]
    if proposed_action not in actions:
        actions.append(proposed_action)

    best_value = proposed_value
    for a in actions:
        if a == proposed_action:
            continue
        vals = [_single_rollout(a) for _ in range(N)]
        v = np.mean(vals)
        if v > best_value:
            best_value = v

    utility = best_value - proposed_value
    return {"utility": float(utility)}


def extract_signals_generic(env, obs, proposer_result):
    """Extract signals for any environment."""
    signals = {}
    signals["step_count"] = getattr(env, "_step_count", 0)
    signals["token_entropy"] = 0.0

    # Token entropy from logprobs
    logprobs = proposer_result.get("token_logprobs", [])
    if logprobs:
        entropies = []
        for lp in logprobs:
            if isinstance(lp, (int, float)) and lp < 0:
                entropies.append(-lp)
        if entropies:
            signals["token_entropy"] = float(np.mean(entropies))

    # Environment-specific signals
    signals["evidence_count"] = 0
    signals["num_available_actions"] = len(env.get_actions()) if hasattr(env, 'get_actions') else 0
    signals["is_finish_proposed"] = False

    action_text = proposer_result.get("action_text", "")
    if any(w in action_text.lower() for w in ["finish", "submit", "done", "answer"]):
        signals["is_finish_proposed"] = True

    return signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_name = args.env
    env_section = cfg.get(env_name, {})
    env_cfg = env_section.get("environment", {"name": env_name, "type": env_name})
    proposer_cfg = env_section.get("proposer", cfg.get("proposer", {}))
    rollout_cfg = env_section.get("rollout", cfg.get("rollout", {"num_chains": 3, "horizon": 3}))

    # Output directories
    data_dir = f"results/phase5/data/{env_name}/seed_{args.seed}"
    multi_dir = f"results/phase6/hidden_states_multi/{env_name}/seed_{args.seed}"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(multi_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  B1 Data Collection: {env_name} seed={args.seed}")
    print(f"  Episodes: {args.episodes}")
    print(f"{'='*60}\n")

    # Create environment
    env = make_env(env_cfg)

    # Create proposers
    llm_config = dict(proposer_cfg.get("llm_config", {}))
    llm_config["temperature"] = 0.0
    vllm_endpoint = os.environ.get("DIAL_VLLM_ENDPOINT")
    if vllm_endpoint:
        llm_config["endpoint"] = vllm_endpoint
    base_proposer = ActionProposer(mode=proposer_cfg.get("mode", "llm_api"), llm_config=llm_config)

    rollout_llm_config = dict(proposer_cfg.get("llm_config", {}))
    rollout_llm_config["temperature"] = rollout_cfg.get("temperature", 0.7)
    if vllm_endpoint:
        rollout_llm_config["endpoint"] = vllm_endpoint
    rollout_proposer = ActionProposer(mode=proposer_cfg.get("mode", "llm_api"), llm_config=rollout_llm_config)

    # HF engine for hidden states
    from dial.inference.hf import HFInferenceEngine
    hf_engine = HFInferenceEngine(
        model_name=proposer_cfg.get("llm_config", {}).get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
        device=args.device, dtype="bfloat16",
    )

    # Collect data
    from tqdm import tqdm

    all_hidden_states = []
    all_hidden_multi = []
    all_utilities = []
    all_signals = []
    all_state_texts = []
    signal_keys = None

    t_start = time.time()
    total_steps = 0

    for ep_idx in tqdm(range(args.episodes), desc=f"Collecting {env_name}"):
        ep_seed = args.seed + ep_idx
        obs, info = env.reset(seed=ep_seed)
        terminated = truncated = False
        step = 0

        while not (terminated or truncated):
            # Propose action
            try:
                result = base_proposer.choose_action_with_logprobs(env, obs)
                proposed_action = result["action"]
                result["action_text"] = env.get_action_name(proposed_action)
            except Exception as e:
                actions = env.get_actions()
                proposed_action = actions[0] if actions else 0
                result = {"action": proposed_action, "token_logprobs": [],
                          "action_text": env.get_action_name(proposed_action)}

            # Extract signals
            signals = extract_signals_generic(env, obs, result)
            if signal_keys is None:
                signal_keys = sorted(signals.keys())

            # Get prompt and hidden state
            try:
                prompt = base_proposer._build_prompt(env, obs)
                hidden_state = hf_engine.encode_state(prompt)

                # Multi-layer extraction
                from dial.inference.hf import HFInferenceEngine
                encoding = hf_engine.tokenizer(
                    prompt if not hasattr(hf_engine.tokenizer, 'apply_chat_template') else
                    hf_engine.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True
                    ),
                    return_tensors="pt", padding=False, truncation=True, max_length=2048,
                )
                import torch
                with torch.no_grad():
                    input_ids = encoding["input_ids"].to(hf_engine.model.device)
                    outputs = hf_engine.model(input_ids=input_ids, output_hidden_states=True)
                    multi_hidden = []
                    for layer_idx in LAYERS:
                        if layer_idx < len(outputs.hidden_states):
                            h = outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
                        else:
                            h = outputs.hidden_states[-1][0, -1, :].cpu().float().numpy()
                        multi_hidden.append(h)
                    multi_hidden = np.stack(multi_hidden)  # (n_layers, d_model)
            except Exception as e:
                logger.warning(f"Hidden state extraction failed: {e}")
                hidden_state = np.zeros(2560)
                multi_hidden = np.zeros((len(LAYERS), 2560))

            # Compute rollout utility (always trigger)
            try:
                rollout_result = compute_rollout_utility_generic(
                    env, rollout_proposer, rollout_cfg, proposed_action
                )
                utility = rollout_result["utility"]
            except Exception as e:
                logger.warning(f"Rollout utility failed: {e}")
                utility = 0.0

            # Store
            all_hidden_states.append(hidden_state)
            all_hidden_multi.append(multi_hidden)
            all_utilities.append(utility)
            all_signals.append([float(signals.get(k, 0) or 0) for k in signal_keys])
            all_state_texts.append(prompt[:2000])  # truncate for storage

            # Execute proposed action (always trigger, but use proposed)
            obs, reward, terminated, truncated, info = env.step(proposed_action)
            step += 1
            total_steps += 1

    elapsed = time.time() - t_start

    # Save step_data.npz (Phase 5 format)
    np.savez_compressed(
        os.path.join(data_dir, "step_data.npz"),
        hidden_states=np.array(all_hidden_states),
        utilities=np.array(all_utilities),
        signals=np.array(all_signals),
        signal_keys=np.array(signal_keys),
        token_entropies=np.array([s[signal_keys.index("token_entropy")] if "token_entropy" in signal_keys else 0
                                   for s in all_signals]),
    )

    # Save state_texts.json
    with open(os.path.join(data_dir, "state_texts.json"), "w") as f:
        json.dump(all_state_texts, f)

    # Save multi_layer_data.npz (Phase 6 format)
    np.savez_compressed(
        os.path.join(multi_dir, "multi_layer_data.npz"),
        hidden_states_multi=np.array(all_hidden_multi),
        hidden_states=np.array(all_hidden_states),
        utilities=np.array(all_utilities),
        signals=np.array(all_signals),
        signal_keys=np.array(signal_keys),
        layer_indices=np.array(LAYERS),
    )

    print(f"\n  Done! {total_steps} steps in {elapsed:.0f}s")
    print(f"  Saved to {data_dir} + {multi_dir}")
    print(f"  Hidden states: ({total_steps}, {len(LAYERS)}, 2560)")


if __name__ == "__main__":
    main()

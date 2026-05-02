#!/usr/bin/env python3
"""
Phase 6 B4v3: Probe Gate with Offline / Adaptive-RL threshold.

Two strategies:
  offline:   Pre-computed threshold from B1 data, no exploration
  adaptive:  Warm-start + epsilon-greedy + gradient threshold updates

Usage:
    python experiments/p6_b4v3_probe_gate.py \
        --config configs/phase5_comparison.yaml \
        --env hotpotqa --strategy offline --seed 42

    python experiments/p6_b4v3_probe_gate.py \
        --config configs/phase5_comparison.yaml \
        --env hotpotqa --strategy adaptive --seed 42
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dial.gates.probe_gate_v2 import OfflineProbeGate, AdaptiveProbeGate
from dial.gates.probes import HiddenStateProbe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DIAL")

# Pre-computed F1-optimal thresholds from B1 offline analysis
OFFLINE_THRESHOLDS = {
    "hotpotqa": 0.362,
    "apps":     0.298,
    "webshop":  0.315,
}

# Cost ratios (C_rollout / C_base)
COST_RATIOS = {
    "hotpotqa": 35.8,
    "apps":     3.9,
    "webshop":  12.9,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--strategy", required=True, choices=["offline", "adaptive"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_name = args.env
    strategy = args.strategy
    seed = args.seed

    env_section = cfg.get(env_name, {})
    env_cfg = env_section.get("environment", {"name": env_name, "type": env_name})
    proposer_cfg = env_section.get("proposer", cfg.get("proposer", {}))
    rollout_cfg = env_section.get("rollout", {})
    probe_cfg = cfg.get("probes", {})

    method_name = f"probe_{strategy}"
    output_dir = os.path.join("results/phase6/b4v3", env_name, method_name, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    threshold = OFFLINE_THRESHOLDS[env_name]

    print(f"\n{'='*60}")
    print(f"  Phase 6 B4v3: Probe Gate — {strategy.upper()}")
    print(f"  Env: {env_name} | Seed: {seed} | Threshold: {threshold:.4f}")
    print(f"{'='*60}\n")

    # ── Create probe ──
    probe = HiddenStateProbe(input_dim=2560, d_hidden=256)
    probe_path = f"results/phase5/probes/{env_name}/hidden_state_probe_main.pt"
    if os.path.exists(probe_path):
        probe.load(probe_path, args.device)
        print(f"  Loaded probe: {probe_path}")
    else:
        print(f"  WARNING: probe not found at {probe_path}")

    # ── Create gate ──
    if strategy == "offline":
        gate = OfflineProbeGate(
            probe=probe,
            threshold=threshold,
            device=args.device,
        )
    else:  # adaptive
        gate = AdaptiveProbeGate(
            probe=probe,
            init_threshold=threshold,
            epsilon_init=0.15,
            epsilon_decay=0.998,
            epsilon_min=0.02,
            lr=0.05,
            cost_penalty=COST_RATIOS.get(env_name, 10.0),
            device=args.device,
        )

    # ── Import episode runners ──
    from experiments.p5_comparison import run_gated_episode, run_gated_episode_p4
    from dial.envs import make_env
    from dial.inference.proposer import ActionProposer

    env = make_env(env_cfg)
    env_type = getattr(env, "ENV_TYPE", env_name)

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
        model_name=probe_cfg.get("hf_model", "Qwen/Qwen3-4B-Instruct-2507"),
        device=args.device, dtype="bfloat16",
    )

    # ── Run episodes ──
    from tqdm import tqdm
    results = []
    t_start = time.time()

    for ep_idx in tqdm(range(args.episodes), desc=f"{method_name} [{env_name}] s{seed}"):
        ep_seed = seed + ep_idx
        if env_type in ("webshop", "alfworld"):
            result = run_gated_episode_p4(
                env, gate, base_proposer, rollout_proposer, rollout_cfg,
                ep_idx, ep_seed, mode="gated",
                hf_engine=hf_engine,
            )
        else:
            result = run_gated_episode(
                env, gate, base_proposer, rollout_proposer, rollout_cfg,
                ep_idx, ep_seed, mode="gated",
                hf_engine=hf_engine,
            )
        results.append(result)

    elapsed = time.time() - t_start
    successes = [r["success"] for r in results]
    rollout_counts = [r["rollout_count"] for r in results]

    summary = {
        "method": method_name,
        "strategy": strategy,
        "env_name": env_name,
        "seed": seed,
        "num_episodes": args.episodes,
        "elapsed_seconds": elapsed,
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_rollouts_per_ep": float(np.mean(rollout_counts)),
        "total_rollouts": sum(rollout_counts),
        "init_threshold": threshold,
        "final_threshold": gate._threshold if hasattr(gate, '_threshold') else threshold,
        "gate_stats": gate.get_stats(),
        "gate_pattern": gate.get_estimated_pattern(),
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    episodes_lite = [{k: v for k, v in r.items() if k != "step_records"} for r in results]
    with open(os.path.join(output_dir, "episodes.json"), "w") as f:
        json.dump(episodes_lite, f, indent=2, cls=NumpyEncoder)

    gate.save_logs(output_dir)

    # Save threshold history for adaptive
    if strategy == "adaptive" and hasattr(gate, '_threshold_history'):
        with open(os.path.join(output_dir, "threshold_history.json"), "w") as f:
            json.dump(gate._threshold_history, f)

    print(f"\n  SR={summary['success_rate']:.1%} | Ro/ep={summary['avg_rollouts_per_ep']:.2f} | "
          f"Threshold: {threshold:.4f} → {summary['final_threshold']:.4f} | Time={elapsed:.0f}s")
    print(f"  Saved to {output_dir}")


if __name__ == "__main__":
    main()

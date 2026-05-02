#!/usr/bin/env python3
"""
Run DIAL (or one of the two reference bounds) on one (env, seed) cell.

Method families
---------------
    --method dial               Full DIAL (LLM-proposed features + LASSO + online decay).
    --method dial_universal     DIAL with universal features only (no LLM layer).
    --method bound:base_only    Never trigger the optimizer (lower-cost reference bound).
    --method bound:always_trigger
                                Always trigger the optimizer (upper-bound reference).

Examples
--------
    python experiments/run_dial.py \\
        --config configs/methods/dial.yaml \\
        --env hotpotqa --method dial --seed 42 --episodes 200

    python experiments/run_dial.py \\
        --config configs/methods/dial.yaml \\
        --env webshop --method bound:base_only --seed 42

    DIAL_VLLM_ENDPOINT=http://localhost:8900/v1 \\
    python experiments/run_dial.py \\
        --config configs/methods/dial.yaml \\
        --env apps --method dial --seed 42

The script reads `proposer`, `rollout`, and `gate` sections from the
config YAML; the env section may either be top-level or under
`<env_name>:` (the second form lets one config cover all six envs).
"""
from __future__ import annotations

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DIAL")


# Per-env C_rollout / C_base ratios (cf. paper Appendix `app:cost`).
COST_RATIOS = {
    "hotpotqa":  35.8,
    "apps":       3.9,
    "webshop":   12.9,
    "fever":     22.4,
    "twexpress":  8.1,
    "plancraft": 11.5,
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


# ──────────────────────────── gate factories ────────────────────────────


def build_dial_gate(env_name: str, gate_cfg: dict, *, with_llm_features: bool):
    """Construct DIAL (full) or DIAL-universal gate."""
    cost_ratio = COST_RATIOS.get(env_name, 10.0)
    common = dict(
        explore_rate=gate_cfg.get("explore_rate", 0.5),
        min_cal_points=gate_cfg.get("min_cal_points", 50),
        window_size=gate_cfg.get("window_size", 500),
        utility_threshold=gate_cfg.get("utility_threshold", 0.05),
    )
    if with_llm_features:
        from dial.gates import DIAL
        return DIAL(
            num_cycles=gate_cfg.get("num_cycles", 1),
            use_feedback=gate_cfg.get("use_feedback", False),
            feature_strategy="selective",
            data_strategy="cumulative",
            reflect_backend=gate_cfg.get("reflect_backend", "local"),
            max_retries=gate_cfg.get("max_retries", 10),
            max_llm_features=gate_cfg.get("max_llm_features", 5),
            feature_filter=True,
            lambda_cost=cost_ratio,
            max_features=gate_cfg.get("max_features", 10),
            pca_model=None,
            **common,
        )
    from dial.gates import DIALUniversal
    return DIALUniversal(
        lambda_cost=cost_ratio,
        max_features=gate_cfg.get("max_features", 10),
        pca_model=None,
        threshold_mode=gate_cfg.get("threshold_mode", "adaptive_lambda"),
        explore_mode=gate_cfg.get("explore_mode", "random"),
        regularizer=gate_cfg.get("regularizer", "l1"),
        **common,
    )


def build_bound_gate(name: str):
    """Construct one of the two trivial reference bounds (no-gating or full-gating)."""
    if name == "base_only":
        return _BoundGate(trigger=False)
    if name == "always_trigger":
        return _BoundGate(trigger=True)
    raise ValueError(
        f"Unknown bound '{name}'. Options: ['base_only', 'always_trigger']."
    )


class _BoundGate:
    """Trivial gate used for the two reference bounds."""
    VARIANT = "bound"

    def __init__(self, trigger: bool):
        self._trigger = trigger
        self.phase = "fixed"
        self._n = 0

    def should_rollout(self, *args, **kwargs):
        self._n += 1
        return self._trigger

    def update(self, *args, **kwargs):
        pass

    def get_stats(self):
        return {
            "variant": "always_trigger" if self._trigger else "base_only",
            "phase": "fixed",
            "total_decisions": self._n,
            "rollout_count": self._n if self._trigger else 0,
            "rollout_rate": 1.0 if self._trigger else 0.0,
        }

    def get_estimated_pattern(self):
        return {"direction": "n/a", "n": self._n}

    def save_logs(self, output_dir, prefix=""):
        with open(os.path.join(output_dir, f"{prefix}stats.json"), "w") as f:
            json.dump(self.get_stats(), f, indent=2)


# ──────────────────────────── main ────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True,
                        help="YAML with env, proposer, rollout, gate sections.")
    parser.add_argument("--env", required=True,
                        choices=list(COST_RATIOS.keys()))
    parser.add_argument("--method", required=True,
                        help="'dial', 'dial_universal', or 'bound:base_only|always_trigger'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="results",
                        help="Top-level output dir; per-env/method/seed subdirs are created.")
    parser.add_argument("--reverse-weights", action="store_true",
                        help="After training DIAL, flip the sign of all gate weights "
                             "(used for the wrong-direction ablation in Table 2).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_section = cfg.get(args.env, cfg)
    env_cfg = env_section.get("environment", {"name": args.env, "type": args.env})
    proposer_cfg = env_section.get("proposer", cfg.get("proposer", {}))
    rollout_cfg = env_section.get("rollout", {})
    gate_cfg = cfg.get("gate", {})

    output_dir = Path(args.output_root) / args.env / args.method.replace(":", "_") / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  DIAL run | env={args.env} method={args.method} seed={args.seed}")
    print(f"  Episodes: {args.episodes} | Output: {output_dir}")
    print(f"{'═'*60}\n")

    # ── Gate ──
    if args.method == "dial":
        gate = build_dial_gate(args.env, gate_cfg, with_llm_features=True)
    elif args.method == "dial_universal":
        gate = build_dial_gate(args.env, gate_cfg, with_llm_features=False)
    elif args.method.startswith("bound:"):
        gate = build_bound_gate(args.method.split(":", 1)[1])
    else:
        raise SystemExit(
            f"--method must be 'dial', 'dial_universal', or "
            f"'bound:base_only|always_trigger', got '{args.method}'."
        )
    gate.VARIANT = args.method
    logger.info(f"Gate: {gate.VARIANT}")

    # ── Env + LLM ──
    from dial.envs import make_env
    from dial.inference.proposer import ActionProposer

    env = make_env(env_cfg)
    env_type = getattr(env, "ENV_TYPE", args.env)

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

    # ── Episode loop ──
    from experiments._episode_loop import run_gated_episode, run_gated_episode_p4
    from tqdm import tqdm

    results = []
    t_start = time.time()

    for ep_idx in tqdm(range(args.episodes), desc=f"{args.method}[{args.env}]"):
        ep_seed = args.seed + ep_idx
        if env_type in ("webshop", "alfworld"):
            r = run_gated_episode_p4(
                env, gate, base_proposer, rollout_proposer, rollout_cfg,
                ep_idx, ep_seed, mode="gated", hf_engine=None,
            )
        else:
            r = run_gated_episode(
                env, gate, base_proposer, rollout_proposer, rollout_cfg,
                ep_idx, ep_seed, mode="gated", hf_engine=None,
            )
        results.append(r)
        if hasattr(gate, "on_episode_end"):
            gate.on_episode_end()

        # Wrong-direction ablation: flip weights right after the gate transitions
        # to exploitation. Implemented as a post-transition hook so the explore
        # phase remains identical to standard DIAL.
        if args.reverse_weights and getattr(gate, "phase", None) == "exploitation":
            _maybe_flip_weights(gate)

    elapsed = time.time() - t_start
    successes = [r["success"] for r in results]
    rollout_counts = [r["rollout_count"] for r in results]

    summary = {
        "method": args.method,
        "env_name": args.env,
        "seed": args.seed,
        "num_episodes": args.episodes,
        "elapsed_seconds": elapsed,
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_rollouts_per_ep": float(np.mean(rollout_counts)),
        "total_rollouts": int(sum(rollout_counts)),
        "gate_stats": gate.get_stats(),
        "gate_pattern": gate.get_estimated_pattern() if hasattr(gate, "get_estimated_pattern") else {},
        "timestamp": datetime.now().isoformat(),
        "config_path": args.config,
        "reverse_weights": args.reverse_weights,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    episodes_lite = [{k: v for k, v in r.items() if k != "step_records"} for r in results]
    with open(output_dir / "episodes.json", "w") as f:
        json.dump(episodes_lite, f, indent=2, cls=NumpyEncoder)

    if hasattr(gate, "save_logs"):
        gate.save_logs(str(output_dir))

    print(f"\n  SR={summary['success_rate']:.1%} | Ro/ep={summary['avg_rollouts_per_ep']:.2f} "
          f"| Time={elapsed:.0f}s | → {output_dir}")


def _maybe_flip_weights(gate) -> None:
    """Flip linear gate weights once for the wrong-direction ablation."""
    if getattr(gate, "_weights_flipped", False):
        return
    for attr in ("_lr", "_clf", "model", "_logreg"):
        m = getattr(gate, attr, None)
        if m is None:
            continue
        if hasattr(m, "coef_"):
            m.coef_ = -m.coef_
            m.intercept_ = -m.intercept_
            gate._weights_flipped = True
            logger.info(f"[wrong-direction] Flipped weights on gate.{attr}.")
            return


if __name__ == "__main__":
    main()

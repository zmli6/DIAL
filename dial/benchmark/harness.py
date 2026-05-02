"""
Harness that runs a `GateInterface` across the six paper environments
and produces a `GateRunResult`.

Two execution modes:
    --stub: replaces the base proposer with a deterministic mock and
            uses a tiny fixture environment. Runs in seconds, no
            external deps. Use this to validate gate code.
    --real: drives the actual env adapters with a vLLM-served LLM,
            matching the paper's evaluation pipeline.

The harness is environment-agnostic: it relies only on the `BaseEnv`
interface defined in `dial.envs.base`, so adding new environments is a
matter of registering a new adapter (see `docs/extending.md`).
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dial.benchmark.gate_interface import Decision, GateInterface
from dial.benchmark.schema import EnvResult, GateRunResult
from dial.benchmark.cost import TokenLedger, compute_cost, env_cost_constants

logger = logging.getLogger("DIAL.benchmark")

DEFAULT_ENVS = ["hotpotqa", "apps", "webshop", "fever", "twexpress", "plancraft"]


# ─── public API ──────────────────────────────────────────────────


def run_benchmark(
    gate: GateInterface,
    envs: Iterable[str] = DEFAULT_ENVS,
    backbone: str = "qwen3-4b",
    seeds: Iterable[int] = (42, 123, 456),
    episodes: int = 100,
    explore_data_dir: Optional[str] = None,
    stub: bool = False,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> "BenchmarkReport":
    """
    Run `gate` across (envs × seeds) and return an aggregated report.

    Parameters
    ----------
    gate : GateInterface
        Your gate implementation.
    envs : list of str
        Subset of {hotpotqa, apps, webshop, fever, twexpress, plancraft}.
    backbone : str
        LLM identifier — passed through to gate.setup() and used for
        result naming. Real run requires a matching vLLM server.
    seeds : list of int
        Random seeds; SR/Cost are reported as mean across seeds.
    episodes : int
        Episodes per (env, seed) cell. Paper uses 100 for headline
        Pareto numbers and 200 for InfoPoor/InfoRich.
    explore_data_dir : str or None
        If set, load DIAL's pre-collected explore-phase data from this
        directory (one file per env) and pass to gate.setup(). Speeds
        up calibration-based methods and matches the paper's protocol.
    stub : bool
        If True, run a tiny fixture (no LLM, no env packages) — for
        smoke-testing gate code in seconds.
    output_dir : str or None
        Save per-cell JSON + the aggregated report under here.
    config : dict
        Stored verbatim in the report under `config`, for traceability.
    """
    envs = list(envs)
    seeds = list(seeds)
    config = dict(config or {})
    config.update({"episodes": episodes, "stub": stub})

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    runner = _StubRunner() if stub else _RealRunner(backbone=backbone)
    report = GateRunResult(
        gate_name=gate.name,
        backbone=backbone,
        config=config,
        metadata={
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "envs": envs,
            "seeds": seeds,
        },
    )

    for env_name in envs:
        explore_data = _load_explore_data(explore_data_dir, env_name, backbone)
        for seed in seeds:
            logger.info(
                f"=== {gate.name} | env={env_name} backbone={backbone} "
                f"seed={seed} episodes={episodes} ==="
            )
            gate.setup(env_name=env_name, backbone=backbone, explore_data=explore_data)
            cell = runner.run_cell(
                gate=gate,
                env_name=env_name,
                seed=seed,
                episodes=episodes,
            )
            report.per_env.append(cell)

            if output_dir:
                cell_path = Path(output_dir) / f"cell_{env_name}_seed{seed}.json"
                with open(cell_path, "w") as f:
                    json.dump(cell.__dict__, f, indent=2, default=str)

    if output_dir:
        report.save(str(Path(output_dir) / "report.json"))
    return BenchmarkReport(report)


# ─── report wrapper ──────────────────────────────────────────────


class BenchmarkReport:
    """Pretty-printing wrapper around `GateRunResult`."""

    def __init__(self, result: GateRunResult):
        self.result = result

    def save(self, path: str) -> None:
        self.result.save(path)

    def to_table(self) -> str:
        means = self.result.env_means()
        rows = []
        rows.append(f"{'Environment':<14}{'SR (%)':>10}{'Cost (×base)':>16}{'Rollout %':>14}")
        rows.append("─" * 54)
        for env in sorted(means):
            m = means[env]
            rows.append(
                f"{env:<14}"
                f"{m['success_rate']*100:>9.1f} "
                f"{m['cost_x_base']:>15.2f} "
                f"{m['rollout_rate']*100:>13.1f}"
            )
        rows.append("─" * 54)
        sr = sum(m["success_rate"] for m in means.values()) / max(len(means), 1)
        cost = sum(m["cost_x_base"] for m in means.values()) / max(len(means), 1)
        rows.append(f"{'Mean':<14}{sr*100:>9.1f} {cost:>15.2f}")
        return "\n".join(rows)

    def compare_to(self, other_path: str) -> str:
        """Diff vs. another saved benchmark report (e.g. paper_results/dial.json)."""
        with open(other_path) as f:
            other = json.load(f)
        other_means = other.get("env_means", {})
        mine = self.result.env_means()

        rows = [f"{'Environment':<14}{'ΔSR (pp)':>12}{'ΔCost (×)':>14}"]
        rows.append("─" * 40)
        for env in sorted(mine):
            if env not in other_means:
                continue
            d_sr = (mine[env]["success_rate"] - other_means[env]["success_rate"]) * 100
            d_cost = mine[env]["cost_x_base"] - other_means[env]["cost_x_base"]
            rows.append(f"{env:<14}{d_sr:>+12.1f}{d_cost:>+14.2f}")
        return "\n".join(rows)


# ─── internal runners ────────────────────────────────────────────


class _StubRunner:
    """Fixture for fast smoke-testing without external dependencies."""

    def run_cell(
        self,
        gate: GateInterface,
        env_name: str,
        seed: int,
        episodes: int,
    ) -> EnvResult:
        rng = random.Random(seed)
        cost = env_cost_constants(env_name)

        successes = 0
        rollouts = 0
        steps_total = 0
        ledger = TokenLedger()

        for ep in range(episodes):
            ep_steps = rng.randint(3, 10)
            ep_success_baseline = rng.random() < 0.4
            ep_success_boost = 0.0

            for t in range(ep_steps):
                state = f"<stub-state env={env_name} ep={ep} step={t}>"
                signals = {
                    "token_entropy": rng.uniform(0.1, 1.5),
                    "action_entropy": rng.uniform(0.0, 1.2),
                    "step_count": float(t),
                    "state_length": float(50 + 10 * t),
                    "num_avail_actions": float(rng.randint(2, 8)),
                }
                ledger.add("base", 100)

                decision = gate.should_rollout(state, signals)
                if not isinstance(decision, Decision):
                    decision = Decision(trigger=bool(decision))
                if decision.trigger:
                    rollouts += 1
                    ledger.add("rollout", int(100 * cost["rollout_per_step"]))
                    utility = 1.0 if rng.random() < 0.55 else 0.0
                    ep_success_boost += 0.05 * (1 if utility > 0 else -1)
                    gate.update(state, signals, utility)
                steps_total += 1

            if ep_success_baseline or rng.random() < ep_success_boost:
                successes += 1

        sr = successes / episodes
        base_only_tokens = 100 * steps_total
        cost_x = compute_cost(ledger.total(), base_only_tokens)
        return EnvResult(
            env=env_name,
            backbone="stub",
            seed=seed,
            episodes=episodes,
            success_rate=sr,
            cost_x_base=cost_x,
            rollout_rate=rollouts / max(steps_total, 1),
            raw_tokens=ledger.total(),
            raw_base_tokens=base_only_tokens,
            diagnostics=gate.finalize() or {},
        )


class _RealRunner:
    """Drives real env adapters via vLLM. Requires backbone server running."""

    def __init__(self, backbone: str):
        self.backbone = backbone

    def run_cell(
        self,
        gate: GateInterface,
        env_name: str,
        seed: int,
        episodes: int,
    ) -> EnvResult:
        # Import lazily so users without env deps can still run --stub.
        from dial.envs import make_env
        from dial.explore.valuator import GenericForwardValuator
        from dial.benchmark._loop import run_episode

        env = make_env({"environment": {"name": env_name, "type": env_name}})
        valuator = GenericForwardValuator(env)
        rng = random.Random(seed)

        successes = 0
        rollouts = 0
        steps_total = 0
        ledger = TokenLedger()
        base_only_tokens = 0

        for ep in range(episodes):
            ep_result = run_episode(
                env=env,
                gate=gate,
                valuator=valuator,
                episode_idx=ep,
                seed=seed,
                rng=rng,
                ledger=ledger,
            )
            successes += int(ep_result.success)
            rollouts += ep_result.n_rollouts
            steps_total += ep_result.n_steps
            base_only_tokens += ep_result.base_only_tokens_estimate

        return EnvResult(
            env=env_name,
            backbone=self.backbone,
            seed=seed,
            episodes=episodes,
            success_rate=successes / episodes,
            cost_x_base=compute_cost(ledger.total(), base_only_tokens),
            rollout_rate=rollouts / max(steps_total, 1),
            raw_tokens=ledger.total(),
            raw_base_tokens=base_only_tokens,
            diagnostics=gate.finalize() or {},
        )


# ─── helpers ─────────────────────────────────────────────────────


def _load_explore_data(
    base_dir: Optional[str],
    env_name: str,
    backbone: str,
) -> Optional[List[Dict[str, Any]]]:
    if not base_dir:
        return None
    candidates = [
        Path(base_dir) / f"{env_name}_{backbone}.json",
        Path(base_dir) / env_name / f"{backbone}.json",
        Path(base_dir) / f"{env_name}.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} explore records from {p}")
            return data
    logger.warning(
        f"No explore data found for env={env_name} backbone={backbone} "
        f"under {base_dir}; gate.setup() will receive None."
    )
    return None

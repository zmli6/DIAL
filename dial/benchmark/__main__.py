"""
`python -m dial.benchmark` CLI.

Examples
--------
Smoke-test your gate (5 minutes, no LLM needed)::

    python -m dial.benchmark \\
        --gate examples/threshold_gate.py:ThresholdGate \\
        --stub --episodes 20

Run real evaluation on three environments::

    python -m dial.benchmark \\
        --gate my_pkg.my_gate:MyGate \\
        --envs hotpotqa,webshop,fever \\
        --backbone qwen3-4b \\
        --seeds 42,123,456 \\
        --episodes 100 \\
        --output results/my_gate/

Compare against published DIAL numbers::

    python -m dial.benchmark \\
        --gate my_pkg.my_gate:MyGate \\
        --stub --output results/my_gate/ \\
        --compare paper_results/dial_qwen3-4b.json
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from dial.benchmark import GateInterface, run_benchmark


def _load_gate(spec: str) -> GateInterface:
    """
    Load a gate from a `module:Class` or `path/to/file.py:Class` spec.
    Returns an instance (calls Class() with no args).
    """
    if ":" not in spec:
        raise SystemExit(
            f"--gate expects 'module:Class' or 'path.py:Class', got '{spec}'."
        )
    target, class_name = spec.rsplit(":", 1)

    if target.endswith(".py") or "/" in target:
        path = Path(target).resolve()
        if not path.exists():
            raise SystemExit(f"Gate file not found: {path}")
        spec_obj = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[path.stem] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise SystemExit(f"Class '{class_name}' not found in {target}.")
    instance = cls()
    if not isinstance(instance, GateInterface):
        raise SystemExit(
            f"{spec} does not subclass dial.benchmark.GateInterface."
        )
    return instance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dial-bench",
        description=(
            "Evaluate any GateInterface across DIAL's 6 paper environments, "
            "outputting SR / Cost numbers compatible with the published "
            "Pareto frontier."
        ),
    )
    parser.add_argument(
        "--gate", required=True,
        help="Gate class spec: 'module.path:ClassName' or 'file.py:ClassName'.",
    )
    parser.add_argument(
        "--envs",
        default="hotpotqa,apps,webshop,fever,twexpress,plancraft",
        help="Comma-separated env names. Default: all six paper envs.",
    )
    parser.add_argument(
        "--backbone", default="qwen3-4b",
        help="LLM identifier passed to gate.setup() and used in result naming.",
    )
    parser.add_argument(
        "--seeds", default="42,123,456",
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--explore-data-dir", default=None,
        help="Directory of pre-collected explore data; "
             "see docs/extending.md for format.",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="Run a tiny in-memory fixture (no LLM, no env packages).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save per-cell JSON and the aggregated report under this dir.",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Path to another benchmark JSON to diff against.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    gate = _load_gate(args.gate)
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    report = run_benchmark(
        gate,
        envs=envs,
        backbone=args.backbone,
        seeds=seeds,
        episodes=args.episodes,
        explore_data_dir=args.explore_data_dir,
        stub=args.stub,
        output_dir=args.output,
        config={"cli_args": vars(args)},
    )

    print(report.to_table())

    if args.compare:
        print()
        print(f"Diff vs. {args.compare}:")
        print(report.compare_to(args.compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
DIAL benchmark harness.

Provides a single `run_benchmark()` entry point that takes a
`GateInterface` implementation, runs it across the six paper environments,
and returns SR/Cost numbers comparable with the published Pareto frontier.

Quick start
-----------
    from dial.benchmark import GateInterface, run_benchmark

    class MyGate(GateInterface):
        def setup(self, env_name, backbone, explore_data=None):
            self.threshold = 0.5

        def should_rollout(self, state, signals):
            return signals["token_entropy"] > self.threshold

    report = run_benchmark(
        MyGate(),
        envs=["hotpotqa", "webshop"],
        backbone="qwen3-4b",
        seeds=[42, 123, 456],
        episodes=100,
    )
    print(report.to_table())
    report.save("results/my_gate.json")
"""
from dial.benchmark.gate_interface import GateInterface, Decision
from dial.benchmark.harness import run_benchmark, BenchmarkReport
from dial.benchmark.cost import compute_cost
from dial.benchmark.schema import GateRunResult, EnvResult

__all__ = [
    "GateInterface",
    "Decision",
    "run_benchmark",
    "BenchmarkReport",
    "compute_cost",
    "GateRunResult",
    "EnvResult",
]

"""Smoke tests for the benchmark API. Run via `pytest tests/`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dial.benchmark import GateInterface, Decision, run_benchmark
from dial.benchmark.cost import compute_cost, env_cost_constants


class _AlwaysOn(GateInterface):
    name = "always_on"
    def should_rollout(self, state, signals):
        return Decision(trigger=True)


class _AlwaysOff(GateInterface):
    name = "always_off"
    def should_rollout(self, state, signals):
        return Decision(trigger=False)


def test_compute_cost_basic():
    assert compute_cost(150, 100) == pytest.approx(1.5)
    assert compute_cost(100, 100) == 1.0
    assert compute_cost(100, 0) == float("inf")


def test_env_cost_constants_known_envs():
    for env in ["hotpotqa", "apps", "webshop", "fever", "twexpress", "plancraft"]:
        c = env_cost_constants(env)
        assert "rollout_per_step" in c and c["rollout_per_step"] > 0


def test_stub_run_always_off_has_zero_rollouts():
    r = run_benchmark(
        _AlwaysOff(),
        envs=["hotpotqa"], seeds=[42], episodes=5, stub=True,
    )
    means = r.result.env_means()
    assert means["hotpotqa"]["rollout_rate"] == 0.0
    assert means["hotpotqa"]["cost_x_base"] == pytest.approx(1.0)


def test_stub_run_always_on_has_full_rollouts():
    r = run_benchmark(
        _AlwaysOn(),
        envs=["hotpotqa"], seeds=[42], episodes=5, stub=True,
    )
    means = r.result.env_means()
    assert means["hotpotqa"]["rollout_rate"] == 1.0
    # Cost > 1× because every step adds rollout tokens.
    assert means["hotpotqa"]["cost_x_base"] > 1.0


def test_report_save_and_load(tmp_path: Path):
    r = run_benchmark(_AlwaysOff(), envs=["webshop"], seeds=[42], episodes=3, stub=True)
    out = tmp_path / "report.json"
    r.save(str(out))
    loaded = json.loads(out.read_text())
    assert loaded["gate_name"] == "always_off"
    assert "env_means" in loaded
    assert "webshop" in loaded["env_means"]


def test_report_to_table_smoke():
    r = run_benchmark(_AlwaysOff(), envs=["fever", "apps"], seeds=[42], episodes=3, stub=True)
    table = r.to_table()
    assert "fever" in table and "apps" in table
    assert "SR (%)" in table

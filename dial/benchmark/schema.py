"""
Result schema. Defining these structures explicitly ensures every method's
output JSON has the same fields, which is what makes the leaderboard
trustworthy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EnvResult:
    """Per-(env, seed) result. SR and cost are the headline numbers."""
    env: str
    backbone: str
    seed: int
    episodes: int
    success_rate: float                # in [0, 1]
    cost_x_base: float                 # mean total tokens / base-only tokens
    rollout_rate: float                # fraction of steps where gate triggered
    raw_tokens: int = 0                # absolute total tokens used
    raw_base_tokens: int = 0           # absolute base-only baseline tokens
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateRunResult:
    """Aggregated result across all (env, seed) cells for a single gate."""
    gate_name: str
    backbone: str
    config: Dict[str, Any] = field(default_factory=dict)
    per_env: List[EnvResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def env_means(self) -> Dict[str, Dict[str, float]]:
        """Mean SR / Cost / rollout_rate per env across seeds."""
        out: Dict[str, Dict[str, float]] = {}
        envs = sorted({r.env for r in self.per_env})
        for env in envs:
            rows = [r for r in self.per_env if r.env == env]
            n = len(rows)
            out[env] = {
                "success_rate": sum(r.success_rate for r in rows) / n,
                "cost_x_base": sum(r.cost_x_base for r in rows) / n,
                "rollout_rate": sum(r.rollout_rate for r in rows) / n,
                "n_seeds": n,
            }
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "backbone": self.backbone,
            "config": self.config,
            "per_env": [asdict(r) for r in self.per_env],
            "env_means": self.env_means(),
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

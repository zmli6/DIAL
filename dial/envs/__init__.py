"""
Environment adapters for the six benchmarks evaluated in the DIAL paper.

All adapters subclass `BaseEnv` and expose a uniform interface
(reset/step/get_state/forward_value/...) so gates and the benchmark
harness work across environments without per-env code paths.

Each adapter falls back to a stub mode when the underlying package
is not installed, so users can validate gate code before installing
heavy dependencies.
"""
from dial.envs.base import BaseEnv, EnvState
from dial.envs.registry import make_env, list_envs, ENV_REGISTRY

__all__ = ["BaseEnv", "EnvState", "make_env", "list_envs", "ENV_REGISTRY"]

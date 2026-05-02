"""
Environment registry — factory function to create any of DIAL's six
supported environments from a config dict or string name.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("DIAL")


# The six environments evaluated in the DIAL paper.
ENV_REGISTRY: Dict[str, str] = {
    "hotpotqa":  "dial.envs.hotpotqa.HotpotQAEnv",
    "apps":      "dial.envs.apps.APPSEnv",
    "webshop":   "dial.envs.webshop.WebShopEnv",
    "fever":     "dial.envs.fever.FEVEREnv",
    "twexpress": "dial.envs.twexpress.TWExpressEnv",
    "plancraft": "dial.envs.plancraft.PlancraftEnv",
}


def _import_class(dotted_path: str):
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def detect_env_type(env_name: str) -> str:
    """Guess env type from its name string. Used when config omits 'type'."""
    low = env_name.lower()
    if "hotpot" in low:
        return "hotpotqa"
    if "apps" in low:
        return "apps"
    if "webshop" in low:
        return "webshop"
    if "fever" in low:
        return "fever"
    if "twexpress" in low or "textworld_express" in low or "tw_express" in low:
        return "twexpress"
    if "plancraft" in low or "plan_craft" in low:
        return "plancraft"
    raise ValueError(
        f"Cannot detect env type from '{env_name}'. "
        f"Set 'environment.type' explicitly. "
        f"Supported: {list(ENV_REGISTRY.keys())}"
    )


def make_env(cfg: Dict[str, Any]):
    """
    Create an environment adapter from a config dict.

        environment:
          name: "hotpotqa"
          type: "hotpotqa"          # optional, auto-detected
          max_steps: 10
          # ... env-specific keys ...
    """
    env_cfg = cfg.get("environment", cfg)
    env_name = env_cfg["name"]
    env_type = env_cfg.get("type") or detect_env_type(env_name)

    if env_type not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown env type '{env_type}'. Supported: {list(ENV_REGISTRY.keys())}"
        )

    cls = _import_class(ENV_REGISTRY[env_type])
    env = cls(env_name=env_name, **env_cfg)
    logger.info(f"Created environment: {env}")
    return env


def list_envs() -> List[str]:
    return list(ENV_REGISTRY.keys())

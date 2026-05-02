"""
Per-episode rollout loop — the engine `run_dial.py`
calls. Two flavors are kept (text-based vs. WebShop-style) because
the WebShop adapter exposes a different observation/action API; merging
them is left as future cleanup since the paper protocol depends on the
exact behavior.
"""

import os
import time
import copy
import numpy as np
import logging
logger = logging.getLogger('DIAL')

def run_gated_episode(
    env,
    gate: Optional[SCGBase],
    base_proposer: ActionProposer,
    rollout_proposer: ActionProposer,
    rollout_cfg: Dict,
    episode_idx: int,
    seed: int,
    mode: str = "gated",  # "gated" | "base_only" | "always_trigger"
    probe_phase: bool = False,
    probe_trigger_rate: float = 0.5,
    hf_engine=None,
) -> Dict[str, Any]:
    """
    Run one episode with gate-controlled rollouts.

    Modes:
      - "gated": gate decides whether to rollout
      - "base_only": never rollout (pure LLM baseline)
      - "always_trigger": always rollout (upper bound on rollout benefit)

    In probe_phase, randomly trigger rollouts to collect calibration data.
    """
    import random as _random

    env_type = getattr(env, "ENV_TYPE", "unknown")
    is_code_env = env_type in ("mbpp", "humaneval", "apps")

    obs, info = env.reset(seed=seed)
    if gate is not None and hasattr(gate, 'reset_episode'):
        gate.reset_episode()
    step_count = 0
    terminated = truncated = False
    total_reward = 0.0

    step_records = []
    rollout_count = 0
    decision_changed_count = 0

    while not (terminated or truncated):
        # Step A: LLM proposes action (temperature=0)
        proposer_result = {"action": 0, "token_logprobs": [], "text": ""}
        try:
            result = base_proposer.choose_action_with_logprobs(env, obs)
            proposed_action = result["action"]
            proposer_result = result
            proposer_result["action_text"] = env.get_action_name(proposed_action)
        except Exception as e:
            logger.warning(f"LLM error ep={episode_idx} step={step_count}: {e}")
            actions = env.get_actions()
            proposed_action = actions[0] if actions else 0
            proposer_result["action"] = proposed_action
            proposer_result["action_text"] = env.get_action_name(proposed_action)

        # Step B: Extract signals
        signals = extract_signals(env, obs, proposer_result)

        # Step C: Gate decision
        if mode == "base_only":
            should_rollout = False
        elif mode == "always_trigger":
            should_rollout = True
        elif probe_phase:
            # Random trigger during probe phase
            should_rollout = _random.random() < probe_trigger_rate
        else:
            # Use gate
            ctx = {**signals, "episode": episode_idx, "step": step_count,
                   "_env": env, "_obs": obs}
            # Pass obs_text and action_text for verbalized-confidence gates
            ctx["obs_text"] = obs if isinstance(obs, str) else str(obs)
            ctx["action_text"] = proposer_result.get("action_text", env.get_action_name(proposed_action))
            # Extract hidden state if HF engine available
            if hf_engine is not None:
                try:
                    prompt = base_proposer._build_prompt(env, obs)
                    hs = hf_engine.encode_state(prompt)
                    ctx["hidden_state"] = hs
                except Exception:
                    pass
            if gate is not None:
                should_rollout = gate.should_rollout(
                    signals.get("token_entropy", 0.0), **ctx,
                )
            else:
                should_rollout = False

        # Step D: Execute rollout if triggered
        chosen_action = proposed_action
        utility = 0.0
        decision_changed = False
        rollout_result = {}

        if should_rollout:
            rollout_count += 1

            if is_code_env:
                if env_type == "apps":
                    rollout_result = compute_apps_rollout_utility(
                        env, rollout_proposer, rollout_cfg, proposed_action,
                    )
                else:
                    rollout_result = compute_mbpp_rollout_utility(
                        env, rollout_proposer, rollout_cfg, proposed_action,
                    )
            else:
                rollout_result = compute_hotpotqa_rollout_utility(
                    env, rollout_proposer, rollout_cfg, proposed_action,
                )

            utility = rollout_result["utility"]

            # Switch to best action if rollout found a better one
            if utility > 0 and rollout_result.get("best_action") != proposed_action:
                chosen_action = rollout_result["best_action"]
                decision_changed = True
                decision_changed_count += 1

            # Update gate calibration with rollout result
            if gate is not None:
                ctx = {**signals, "episode": episode_idx, "step": step_count}
                gate.update(signals.get("token_entropy", 0.0), utility, **ctx)

        elif probe_phase and gate is not None:
            # During probe, also record skipped steps (utility=0) so the gate
            # learns that skipping was a valid choice at this signal value.
            ctx = {**signals, "episode": episode_idx, "step": step_count}
            gate.update(signals.get("token_entropy", 0.0), 0.0, **ctx)

        # Record step data
        step_records.append({
            "step": step_count,
            "proposed_action": proposed_action,
            "proposed_action_text": env.get_action_name(proposed_action),
            "chosen_action": chosen_action,
            "chosen_action_text": env.get_action_name(chosen_action) if chosen_action != proposed_action else env.get_action_name(proposed_action),
            "should_rollout": should_rollout,
            "utility": utility,
            "decision_changed": decision_changed,
            "gate_phase": "probe" if probe_phase else ("exploitation" if gate and gate.phase == "exploitation" else mode),
            **signals,
            **{f"rollout_{k}": v for k, v in rollout_result.items() if k != "utility"},
        })

        # Step E: Execute chosen action
        obs, reward, terminated, truncated, info = env.step(chosen_action)
        total_reward += reward
        step_count += 1

    success = env.is_success(reward, terminated, info)

    return {
        "episode": episode_idx,
        "seed": seed,
        "reward": float(total_reward),
        "steps": step_count,
        "success": success,
        "rollout_count": rollout_count,
        "decision_changed": decision_changed_count,
        "mode": mode,
        "gate_phase": "probe" if probe_phase else (gate.phase if gate else mode),
        "step_records": step_records,
    }


# ══════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════


def run_gated_episode_p4(
    env,
    gate,
    base_proposer: ActionProposer,
    rollout_proposer: ActionProposer,
    rollout_cfg: Dict,
    episode_idx: int,
    seed: int,
    mode: str = "gated",
    probe_phase: bool = False,
    probe_trigger_rate: float = 0.5,
    hf_engine=None,
) -> Dict[str, Any]:
    """
    Run one episode with gate-controlled rollouts for WebShop/ALFWorld.

    Uses environment-specific rollout utility computation and signal
    extraction. Otherwise identical to the Phase 2/3 run_gated_episode.
    """
    import random as _random

    env_type = getattr(env, "ENV_TYPE", "unknown")

    obs, info = env.reset(seed=seed)
    if gate is not None and hasattr(gate, 'reset_episode'):
        gate.reset_episode()
    step_count = 0
    terminated = truncated = False
    total_reward = 0.0

    step_records = []
    rollout_count = 0
    decision_changed_count = 0

    while not (terminated or truncated):
        # Step A: LLM proposes action (temperature=0)
        proposer_result = {"action": 0, "token_logprobs": [], "text": ""}
        try:
            result = base_proposer.choose_action_with_logprobs(env, obs)
            proposed_action = result["action"]
            proposer_result = result
            proposer_result["action_text"] = env.get_action_name(proposed_action)
        except Exception as e:
            logger.warning(f"LLM error ep={episode_idx} step={step_count}: {e}")
            actions = env.get_actions()
            proposed_action = actions[0] if actions else 0
            proposer_result["action"] = proposed_action
            proposer_result["action_text"] = env.get_action_name(proposed_action)

        # Step B: Extract signals (Phase 4 version)
        signals = extract_signals_phase4(env, obs, proposer_result)

        # Step C: Gate decision
        if mode == "base_only":
            should_rollout = False
        elif mode == "always_trigger":
            should_rollout = True
        elif probe_phase:
            should_rollout = _random.random() < probe_trigger_rate
        else:
            ctx = {**signals, "episode": episode_idx, "step": step_count,
                   "_env": env, "_obs": obs}
            # Pass obs_text and action_text for verbalized-confidence gates
            ctx["obs_text"] = obs if isinstance(obs, str) else str(obs)
            ctx["action_text"] = proposer_result.get("action_text", env.get_action_name(proposed_action))
            # Extract hidden state if HF engine available
            if hf_engine is not None:
                try:
                    prompt = base_proposer._build_prompt(env, obs)
                    hs = hf_engine.encode_state(prompt)
                    ctx["hidden_state"] = hs
                except Exception:
                    pass
            if gate is not None:
                should_rollout = gate.should_rollout(
                    signals.get("token_entropy", 0.0), **ctx,
                )
            else:
                should_rollout = False

        # Step D: Execute rollout if triggered
        chosen_action = proposed_action
        utility = 0.0
        decision_changed = False
        rollout_result = {}

        if should_rollout:
            rollout_count += 1

            if env_type == "webshop":
                rollout_result = compute_webshop_rollout_utility(
                    env, rollout_proposer, rollout_cfg, proposed_action,
                )
            elif env_type == "alfworld":
                rollout_result = compute_alfworld_rollout_utility(
                    env, rollout_proposer, rollout_cfg, proposed_action,
                )
            else:
                rollout_result = compute_hotpotqa_rollout_utility(
                    env, rollout_proposer, rollout_cfg, proposed_action,
                )

            utility = rollout_result["utility"]

            # For LLM-as-Simulator (ALFWorld), scores are 1-10 so small
            # differences are noise.  Require a minimum margin to override.
            utility_threshold = rollout_cfg.get("utility_threshold", 0.0)
            if env_type == "alfworld" and utility_threshold == 0.0:
                utility_threshold = 2.0  # default margin for 1-10 scale

            if utility > utility_threshold and rollout_result.get("best_action") != proposed_action:
                chosen_action = rollout_result["best_action"]
                decision_changed = True
                decision_changed_count += 1

            if gate is not None:
                ctx = {**signals, "episode": episode_idx, "step": step_count}
                gate.update(signals.get("token_entropy", 0.0), utility, **ctx)

        elif probe_phase and gate is not None:
            ctx = {**signals, "episode": episode_idx, "step": step_count}
            gate.update(signals.get("token_entropy", 0.0), 0.0, **ctx)

        # Record step data
        step_records.append({
            "step": step_count,
            "proposed_action": proposed_action,
            "proposed_action_text": env.get_action_name(proposed_action),
            "chosen_action": chosen_action,
            "chosen_action_text": env.get_action_name(chosen_action),
            "should_rollout": should_rollout,
            "utility": utility,
            "decision_changed": decision_changed,
            "gate_phase": "probe" if probe_phase else (
                "exploitation" if gate and gate.phase == "exploitation" else mode
            ),
            **signals,
            **{f"rollout_{k}": v for k, v in rollout_result.items() if k != "utility"},
        })

        # Step E: Execute chosen action
        obs, reward, terminated, truncated, info = env.step(chosen_action)
        total_reward += reward
        step_count += 1

    success = env.is_success(reward, terminated, info)

    return {
        "episode": episode_idx,
        "seed": seed,
        "reward": float(total_reward),
        "steps": step_count,
        "success": success,
        "rollout_count": rollout_count,
        "decision_changed": decision_changed_count,
        "mode": mode,
        "gate_phase": "probe" if probe_phase else (gate.phase if gate else mode),
        "step_records": step_records,
    }


# ══════════════════════════════════════════════════════════════════
# ORACLE EPISODE (Phase 4 version)
# ══════════════════════════════════════════════════════════════════

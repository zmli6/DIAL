"""
Plancraft environment adapter.

Plancraft is a text-based Minecraft crafting planning benchmark.
The agent must plan multi-step crafting recipes from an inventory of
items, using move/smelt actions on a 3x3 crafting grid and inventory
slots.

Reference: Dag et al., COLM 2025
           https://github.com/gautierdag/plancraft

Key characteristics:
  - Discrete crafting-planning tasks (5-30 steps)
  - Text-based inventory + crafting grid observations
  - Move/smelt actions with slot-level control
  - Difficulty levels: easy / medium / hard / impossible
  - Oracle planner available via optimal_planner()

Requires: pip install plancraft
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseEnv, EnvState

logger = logging.getLogger("DIAL")

# ── Try importing plancraft; fall back to stub mode if unavailable ─────
_PLANCRAFT_AVAILABLE = False
try:
    from plancraft.simple import get_plancraft_examples, PlancraftGymWrapper
    from plancraft.config import PlancraftExample
    from plancraft.environment.env import (
        PlancraftEnvironment,
        target_and_inventory_to_text_obs,
    )
    from plancraft.environment.actions import (
        MoveAction,
        SmeltAction,
        StopAction,
        MoveActionHandler,
        SmeltActionHandler,
        ImpossibleActionHandler,
        convert_from_slot_index,
    )
    from plancraft.environment.planner import (
        optimal_planner,
        get_subplans,
        get_inventory_counter,
    )
    from plancraft.environment.recipes import RECIPES

    _PLANCRAFT_AVAILABLE = True
except ImportError as _import_err:
    logger.warning(
        f"plancraft is not installed ({_import_err}). "
        "Install with: pip install plancraft\n"
        "PlancraftEnv will operate in STUB mode."
    )


class PlancraftEnv(BaseEnv):
    """
    Adapter for Plancraft Minecraft crafting planning environment.

    Each episode presents a target item to craft and an inventory of items.
    The agent must issue move/smelt actions to place items on a 3x3 crafting
    grid and collect the crafted result.

    The action space is discretized: we pre-compute a set of candidate
    actions (the next actions from the oracle planner, plus some heuristic
    move/smelt actions) and map them to integer IDs.
    """

    ENV_TYPE = "plancraft"

    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name, **kwargs)

        # Task configuration
        self.split: str = kwargs.get("split", "val.small")
        self.difficulty: str = kwargs.get("difficulty", "all")  # easy/medium/hard/all
        self.include_impossible: bool = kwargs.get("include_impossible", False)
        self.max_steps: int = kwargs.get("max_steps", 30)

        # State tracking
        self._step_count: int = 0
        self._done: bool = False
        self._success: bool = False
        self._reward: float = 0.0
        self._history: List[Dict[str, str]] = []

        # Plancraft-specific state
        self._target: str = ""
        self._inventory: Dict = {}  # slot -> {type, quantity}
        self._obs_text: str = ""
        self._example: Any = None
        self._example_idx: int = -1
        self._optimal_path: Optional[List[str]] = None
        self._optimal_path_length: int = 0
        self._crafted_items: List[str] = []  # items crafted so far

        # Action space: list of text action strings
        self._action_texts: List[str] = []

        # Plancraft wrapper
        self._wrapper: Any = None

        # Dataset of examples
        self._examples: List[Any] = []
        self._filtered_examples: List[Any] = []

        # Oracle / LLM helpers
        self._oracle_proposer = None
        self._oracle_llm_config = None

        # Load examples
        self._load_examples()

    def _load_examples(self):
        """Load plancraft examples from the dataset."""
        if not _PLANCRAFT_AVAILABLE:
            self._examples = []
            self._filtered_examples = []
            return

        try:
            self._examples = get_plancraft_examples(self.split)

            # Filter by difficulty
            if self.difficulty != "all":
                self._filtered_examples = [
                    ex for ex in self._examples
                    if ex.complexity_split == self.difficulty
                ]
            else:
                self._filtered_examples = list(self._examples)

            # Filter out impossible unless requested
            if not self.include_impossible:
                self._filtered_examples = [
                    ex for ex in self._filtered_examples
                    if not ex.impossible
                ]

            logger.info(
                f"Plancraft: loaded {len(self._filtered_examples)} examples "
                f"from split='{self.split}' difficulty='{self.difficulty}'"
            )
        except Exception as e:
            logger.warning(f"Failed to load Plancraft examples: {e}")
            self._examples = []
            self._filtered_examples = []

    # ── lifecycle ─────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        self._step_count = 0
        self._done = False
        self._success = False
        self._reward = 0.0
        self._history = []
        self._crafted_items = []

        if not _PLANCRAFT_AVAILABLE or not self._filtered_examples:
            return self._stub_reset()

        # Pick a random example
        rng = random.Random(seed)
        self._example_idx = rng.randint(0, len(self._filtered_examples) - 1)
        self._example = self._filtered_examples[self._example_idx]
        self._target = self._example.target
        self._optimal_path = self._example.optimal_path
        self._optimal_path_length = int(
            self._example.optimal_path_length or 0
        )

        try:
            # Create the gym wrapper
            self._wrapper = PlancraftGymWrapper(
                example=self._example,
                max_steps=self.max_steps,
                resolution="low",
                use_text_inventory=True,
            )

            # Get initial observation
            obs, _, _, _, info = self._wrapper.step()
            self._obs_text = obs.get("text", "")
            self._inventory = copy.deepcopy(obs.get("inventory", {}))

            # Generate candidate actions
            self._generate_action_space()

        except Exception as e:
            logger.warning(f"Plancraft reset failed: {e}. Using stub mode.")
            return self._stub_reset()

        return self._obs_text, {
            "target": self._target,
            "example_id": self._example.id,
            "difficulty": self._example.complexity_split,
            "optimal_path_length": self._optimal_path_length,
        }

    def _stub_reset(self) -> Tuple[Any, Dict]:
        """Fallback when plancraft is unavailable."""
        self._target = "oak_planks"
        self._obs_text = (
            "Plancraft Stub Mode\n"
            "Craft an item of type: oak_planks\n"
            "inventory:\n"
            " - oak_log [I1] quantity 4"
        )
        self._inventory = {}
        self._optimal_path = ["oak_planks"]
        self._optimal_path_length = 1
        self._action_texts = [
            "move: from [I1] to [A1] with quantity 1",
            "move: from [0] to [I2] with quantity 4",
            "impossible: cannot craft",
        ]
        self._wrapper = None
        return self._obs_text, {
            "target": self._target,
            "example_id": "STUB",
            "difficulty": "easy",
            "optimal_path_length": 1,
        }

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        # Get the action text
        action_text = (
            self._action_texts[action]
            if action < len(self._action_texts)
            else "impossible: no valid action"
        )
        self._step_count += 1

        if self._wrapper is not None:
            try:
                obs, reward, terminated, truncated, info = self._wrapper.step(
                    action_text
                )
                self._obs_text = obs.get("text", str(obs))
                new_inventory = obs.get("inventory", {})

                # Track crafted items by checking output slot
                if new_inventory:
                    self._inventory = copy.deepcopy(new_inventory)

                self._reward = float(reward)
                self._success = reward > 0 or info.get("reason") == "success"
                self._done = terminated or truncated

                # Regenerate action space based on new inventory
                if not self._done:
                    self._generate_action_space()

            except Exception as e:
                logger.warning(f"Plancraft step failed: {e}")
                self._obs_text = f"Error: {e}"
                reward = 0.0
                terminated = True
                truncated = False
                self._done = True

        else:
            # Stub mode
            reward = 1.0 if "move: from [0]" in action_text else 0.0
            terminated = self._step_count >= self.max_steps or reward > 0
            truncated = False
            self._obs_text = (
                f"Stub: executed '{action_text}' [step {self._step_count}]"
            )
            self._reward = reward
            self._done = terminated
            self._success = reward > 0

        self._history.append({
            "action": action_text,
            "obs": self._obs_text[:200],
            "reward": float(self._reward),
        })

        terminated = self._done
        truncated = self._step_count >= self.max_steps and not terminated

        return self._obs_text, float(self._reward), terminated, truncated, {
            "target": self._target,
            "success": self._success,
            "step_count": self._step_count,
        }

    def close(self):
        self._wrapper = None

    # ── action space generation ───────────────────────────────────

    def _generate_action_space(self):
        """
        Generate a discrete set of candidate actions from the current
        inventory state.  This includes:
          1. Oracle-suggested next actions (from the planner)
          2. Heuristic move/smelt candidates
          3. An 'impossible' action
        """
        actions = []

        if not _PLANCRAFT_AVAILABLE or not self._inventory:
            self._action_texts = [
                "impossible: no valid action available"
            ]
            return

        try:
            # Get oracle plan actions
            inventory_counter = {}
            for slot, item in self._inventory.items():
                if slot != 0 and item.get("quantity", 0) > 0:
                    item_type = item["type"]
                    inventory_counter[item_type] = (
                        inventory_counter.get(item_type, 0) + item["quantity"]
                    )

            plan = optimal_planner(self._target, copy.deepcopy(inventory_counter))

            if plan is not None and len(plan) > 0:
                # Get subplans - these give us the actual move/smelt actions
                try:
                    observation = {
                        "inventory": copy.deepcopy(self._inventory),
                        "target": self._target,
                    }
                    subplans, _ = get_subplans(observation)
                    # Flatten first subplan as immediate actions
                    if subplans and len(subplans) > 0:
                        for sp in subplans:
                            for act_str in sp:
                                if act_str not in actions:
                                    actions.append(act_str)
                except Exception as e:
                    logger.debug(f"Subplan generation failed: {e}")

            # Add some heuristic moves: move items to/from crafting grid
            occupied_slots = list(self._inventory.keys())
            inv_slots = [s for s in occupied_slots if s >= 10]
            grid_slots = [s for s in occupied_slots if 1 <= s <= 9]

            # Move from inventory to crafting grid slots
            for from_slot in inv_slots[:5]:
                item = self._inventory[from_slot]
                for to_slot in range(1, 10):
                    if to_slot not in self._inventory:
                        act = (
                            f"move: from {convert_from_slot_index(from_slot)} "
                            f"to {convert_from_slot_index(to_slot)} "
                            f"with quantity 1"
                        )
                        if act not in actions:
                            actions.append(act)
                        break  # one target slot per source

            # Move from output slot to inventory
            if 0 in self._inventory and self._inventory[0].get("quantity", 0) > 0:
                out_item = self._inventory[0]
                for to_slot in range(10, 46):
                    if to_slot not in self._inventory:
                        act = (
                            f"move: from [0] "
                            f"to {convert_from_slot_index(to_slot)} "
                            f"with quantity {out_item['quantity']}"
                        )
                        if act not in actions:
                            actions.append(act)
                        break

            # Smelt actions for smelting recipes
            for from_slot in inv_slots[:5]:
                item = self._inventory[from_slot]
                # Check if this item can be smelted
                for recipe_list in RECIPES.values():
                    for recipe in recipe_list:
                        if recipe.recipe_type == "smelting":
                            if item["type"] in recipe.ingredient:
                                for to_slot in range(10, 46):
                                    if to_slot not in self._inventory:
                                        act = (
                                            f"smelt: from "
                                            f"{convert_from_slot_index(from_slot)} "
                                            f"to {convert_from_slot_index(to_slot)} "
                                            f"with quantity {item['quantity']}"
                                        )
                                        if act not in actions:
                                            actions.append(act)
                                        break
                                break
                    else:
                        continue
                    break

        except Exception as e:
            logger.debug(f"Action generation failed: {e}")

        # Always include impossible action
        actions.append("impossible: cannot craft target item")

        # Limit to reasonable number
        self._action_texts = actions[:30]

    # ── snapshot / restore ────────────────────────────────────────

    def get_state(self) -> EnvState:
        data = {
            "obs_text": self._obs_text,
            "step_count": self._step_count,
            "done": self._done,
            "success": self._success,
            "reward": self._reward,
            "target": self._target,
            "inventory": copy.deepcopy(self._inventory),
            "history": [dict(h) for h in self._history],
            "action_texts": list(self._action_texts),
            "example_idx": self._example_idx,
            "optimal_path": self._optimal_path,
            "optimal_path_length": self._optimal_path_length,
            "crafted_items": list(self._crafted_items),
        }
        return EnvState(data=data, env_type="plancraft")

    def set_state(self, state: EnvState):
        d = state.data
        self._obs_text = d["obs_text"]
        self._step_count = d["step_count"]
        self._done = d["done"]
        self._success = d.get("success", False)
        self._reward = d["reward"]
        self._target = d.get("target", "")
        self._inventory = copy.deepcopy(d.get("inventory", {}))
        self._history = list(d.get("history", []))
        self._action_texts = list(d.get("action_texts", []))
        self._example_idx = d.get("example_idx", -1)
        self._optimal_path = d.get("optimal_path")
        self._optimal_path_length = d.get("optimal_path_length", 0)
        self._crafted_items = list(d.get("crafted_items", []))

    def deepcopy(self) -> "PlancraftEnv":
        """
        Create an independent copy.  We rebuild the wrapper from the
        same example and replay the action history.
        """
        clone = object.__new__(PlancraftEnv)
        clone.env_name = self.env_name
        clone.max_steps = self.max_steps
        clone.split = self.split
        clone.difficulty = self.difficulty
        clone.include_impossible = self.include_impossible
        clone._env = None
        clone._oracle_proposer = self._oracle_proposer
        clone._oracle_llm_config = self._oracle_llm_config

        # Share the examples list (read-only)
        clone._examples = self._examples
        clone._filtered_examples = self._filtered_examples

        # Copy state
        clone._obs_text = self._obs_text
        clone._step_count = self._step_count
        clone._done = self._done
        clone._success = self._success
        clone._reward = self._reward
        clone._target = self._target
        clone._inventory = copy.deepcopy(self._inventory)
        clone._history = [dict(h) for h in self._history]
        clone._action_texts = list(self._action_texts)
        clone._example_idx = self._example_idx
        clone._example = self._example
        clone._optimal_path = self._optimal_path
        clone._optimal_path_length = self._optimal_path_length
        clone._crafted_items = list(self._crafted_items)

        # Rebuild wrapper and replay actions
        clone._wrapper = None
        if _PLANCRAFT_AVAILABLE and self._example is not None:
            try:
                clone._wrapper = PlancraftGymWrapper(
                    example=self._example,
                    max_steps=self.max_steps,
                    resolution="low",
                    use_text_inventory=True,
                )
                # Initial step
                clone._wrapper.step()
                # Replay action history
                for h in self._history:
                    clone._wrapper.step(h["action"])
            except Exception as e:
                logger.warning(f"Plancraft deepcopy replay failed: {e}")
                clone._wrapper = None

        return clone

    # ── action space ──────────────────────────────────────────────

    def get_actions(self) -> List[int]:
        return list(range(len(self._action_texts)))

    def get_action_name(self, action: int) -> str:
        if action < len(self._action_texts):
            return self._action_texts[action]
        return f"action_{action}"

    # ── text description ──────────────────────────────────────────

    def get_text_description(self) -> str:
        lines = [
            "You are playing Plancraft, a Minecraft crafting planning game.",
            "",
            "CRAFTING RULES:",
            "- Place items from inventory onto the 3x3 crafting grid (slots [A1]-[C3])",
            "- Crafted item appears in output slot [0]",
            "- Move output from [0] to an inventory slot [I1]-[I36]",
            "- Some items need smelting instead of crafting",
            "",
            "ACTION FORMAT:",
            "- move: from [Source] to [Target] with quantity N",
            "- smelt: from [Source] to [Target] with quantity N",
            "- impossible: <reason>",
            "",
        ]

        lines.append(f"Target: Craft an item of type: {self._target}")
        lines.append(f"Step: {self._step_count}/{self.max_steps}")
        lines.append("")

        # Current observation
        if self._obs_text:
            lines.append(f"Current state:")
            lines.append(self._obs_text)

        # Show recent history
        if self._history:
            lines.append("")
            lines.append("Recent actions:")
            for h in self._history[-5:]:
                obs_short = h["obs"][:80].replace("\n", " ")
                lines.append(
                    f"  > {h['action']} -> {obs_short} (r={h['reward']})"
                )

        # Available actions
        lines.append("")
        lines.append(
            f"Available actions ({len(self._action_texts)}):"
        )
        for i, act in enumerate(self._action_texts[:15]):
            lines.append(f"  {i}: {act}")

        return "\n".join(lines)

    # ── forward value (heuristic) ─────────────────────────────────

    def forward_value(self, action: int) -> float:
        """
        Heuristic value for an action.  Favors actions that:
          - Move items from output slot [0] (collecting crafted item)
          - Move items to crafting grid (setting up for craft)
          - Are part of the oracle plan (if detectable)
          - Are smelt actions when appropriate
        Penalizes:
          - Impossible/stop actions
          - Moves away from crafting grid
        """
        if action >= len(self._action_texts):
            return -1.0

        act_text = self._action_texts[action].lower()
        score = 0.0

        # Highly reward collecting crafted output
        if "from [0]" in act_text:
            score += 1.0

        # Reward moves to crafting grid
        elif "move:" in act_text:
            if any(f"to [{slot}]" in act_text for slot in
                   ["[a1]", "[a2]", "[a3]", "[b1]", "[b2]", "[b3]",
                    "[c1]", "[c2]", "[c3]"]):
                score += 0.6
            else:
                score += 0.2  # general move

        # Reward smelting
        elif "smelt:" in act_text:
            score += 0.7

        # Penalize impossible/stop
        elif "impossible:" in act_text:
            score -= 0.5

        # Bonus for being early in the action list (oracle actions first)
        if action < 5:
            score += 0.3 * (1.0 - action / 5.0)

        return score

    # ── success ───────────────────────────────────────────────────

    def is_success(self, reward: float, terminated: bool, info: Dict) -> bool:
        return self._success or reward > 0

    # ── state classification ──────────────────────────────────────

    def classify_state(self) -> str:
        if self._success:
            return "success"
        if self._done:
            return "failed"

        # Estimate progress based on items on crafting grid and crafted items
        grid_occupied = sum(
            1 for s in self._inventory
            if 1 <= s <= 9 and self._inventory[s].get("quantity", 0) > 0
        )
        has_output = (
            0 in self._inventory
            and self._inventory[0].get("quantity", 0) > 0
        )

        if has_output:
            return "output_ready"
        if grid_occupied >= 3:
            return "grid_filling"
        if self._step_count > self.max_steps * 0.7:
            return "late_stage"
        if self._step_count > 0:
            return "in_progress"
        return "initial"

    # ── state key ─────────────────────────────────────────────────

    def make_state_key(self) -> str:
        # Create a deterministic key from target + inventory state
        inv_items = sorted(
            (s, self._inventory[s].get("type", ""), self._inventory[s].get("quantity", 0))
            for s in self._inventory
        )
        h = hashlib.md5(
            f"{self._target}:{self._step_count}:{inv_items}".encode()
        ).hexdigest()[:12]
        return f"pc_{h}"

    # ── greedy oracle action ──────────────────────────────────────

    def greedy_oracle_action(self) -> int:
        """
        Use the optimal planner to determine the best next action.
        The first actions in self._action_texts are typically from
        the planner, so action 0 is often optimal.
        """
        actions = self.get_actions()
        if not actions:
            return 0

        # If we have an LLM oracle, try it
        if self._oracle_proposer is None and self._oracle_llm_config is None:
            self._try_init_oracle_llm()

        if self._oracle_proposer is not None:
            try:
                return self._oracle_llm_choose(actions)
            except Exception as e:
                logger.debug(f"LLM oracle failed: {e}")

        # Check if output slot has item - always collect it first
        if 0 in self._inventory and self._inventory[0].get("quantity", 0) > 0:
            for i, act in enumerate(self._action_texts):
                if "from [0]" in act:
                    return i

        # Otherwise prefer the first action (usually from oracle planner)
        # But verify it's not "impossible" unless we're stuck
        for i, act in enumerate(self._action_texts):
            if "impossible" not in act.lower():
                return i

        # Fall back to action with best forward value
        values = [(a, self.forward_value(a)) for a in actions]
        values.sort(key=lambda x: -x[1])
        return values[0][0]

    def _oracle_llm_choose(self, actions) -> int:
        import openai

        state_desc = self.get_text_description()
        action_list = "\n".join(
            f"  {a}: {self._action_texts[a]}" for a in actions[:15]
        )
        prompt = (
            f"{state_desc}\n\n"
            f"Available actions:\n{action_list}\n\n"
            "Pick the best action index to craft the target item. "
            "Reply with ONLY the integer."
        )

        port = os.environ.get("VLLM_PORT", "8000")
        client = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="unused",
        )
        resp = client.chat.completions.create(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\d+", raw)
        if m:
            chosen = int(m.group())
            if chosen in actions:
                return chosen
        return actions[0]

    def _try_init_oracle_llm(self):
        try:
            import requests

            port = os.environ.get("VLLM_PORT", "8000")
            base_url = f"http://localhost:{port}"
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                from dial.inference.proposer import ActionProposer

                self._oracle_proposer = ActionProposer(
                    mode="llm_api",
                    llm_config={
                        "api_type": "vllm",
                        "endpoint": f"{base_url}/v1",
                        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
                        "temperature": 0.1,
                        "max_tokens": 200,
                    },
                )
        except Exception:
            self._oracle_llm_config = "unavailable"

    def set_oracle_proposer(self, proposer):
        self._oracle_proposer = proposer

    # ── Plancraft-specific signals ────────────────────────────────

    def get_signals(self) -> Dict[str, Any]:
        """Return Plancraft-specific signals for gate learning."""
        # Count items in inventory (excluding crafting slots)
        inventory_size = sum(
            1 for s, item in self._inventory.items()
            if s >= 10 and item.get("quantity", 0) > 0
        )

        # Count items on crafting grid
        grid_items = sum(
            1 for s in range(1, 10)
            if s in self._inventory and self._inventory[s].get("quantity", 0) > 0
        )

        # Crafting progress: fraction of optimal steps completed
        if self._optimal_path_length > 0:
            crafting_progress = min(
                1.0, self._step_count / max(self._optimal_path_length * 3, 1)
            )
        else:
            crafting_progress = 0.0

        # Check if output slot has item (we're ready to collect)
        has_output = (
            0 in self._inventory
            and self._inventory[0].get("quantity", 0) > 0
        )

        return {
            "step_count": self._step_count,
            "inventory_size": inventory_size,
            "grid_items": grid_items,
            "crafting_progress": crafting_progress,
            "has_output": float(has_output),
            "optimal_path_length": self._optimal_path_length,
            "num_actions_available": len(self._action_texts),
        }

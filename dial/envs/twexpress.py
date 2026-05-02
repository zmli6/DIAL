"""
TextWorldExpress environment adapter.

TextWorldExpress is an ultra-fast Java-based reimplementation of text game
benchmarks (CoinCollector, CookingWorld, TextWorld Commonsense, MapReader,
Arithmetic, Sorting, SimonSays, PeckingOrder).

Unlike the original TextWorld, TextWorldExpress does not compile .z8 files;
instead it runs games directly in a JVM server via py4j, making reset/step
nearly instantaneous.

Reference: Jansen et al., 2023
           https://github.com/cognitiveailab/TextWorldExpress

Requires:
  - pip install textworld-express (tested v1.1.0)
  - Java 1.8+ runtime (JRE/JDK) accessible on PATH

Key characteristics:
  - Multiple game types: coin, cookingworld, twc, mapreader, etc.
  - Configurable difficulty via gameParams string
  - Text-based admissible action space (validActions per step)
  - Normalised score in [0, 1]; success = tasksuccess flag or score >= 1.0
  - Built-in serialize/clone for state save-restore
  - Gold action sequences available for oracle play
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseEnv, EnvState

logger = logging.getLogger("DIAL")

# ── Guard import ──────────────────────────────────────────────────
try:
    from textworld_express import TextWorldExpressEnv as TWXEnv
    from textworld_express import GAME_NAMES as TWX_GAME_NAMES
    TWX_AVAILABLE = True
except ImportError:
    TWXEnv = None
    TWX_GAME_NAMES = []
    TWX_AVAILABLE = False
    logger.warning(
        "textworld-express is not installed. Install with:\n"
        "  pip install textworld-express\n"
        "TWExpressEnv will operate in STUB mode."
    )

# ── Shorthand game-name aliases ──────────────────────────────────
_GAME_ALIASES = {
    "coin": "coin",
    "coincollector": "coin",
    "coin_collector": "coin",
    "cooking": "cookingworld",
    "cookingworld": "cookingworld",
    "cooking_world": "cookingworld",
    "twc": "twc",
    "commonsense": "twc",
    "textworld_commonsense": "twc",
    "mapreader": "mapreader",
    "map_reader": "mapreader",
    "arithmetic": "arithmetic",
    "sorting": "sorting",
    "simonsays": "simonsays",
    "simon_says": "simonsays",
    "peckingorder": "peckingorder",
    "pecking_order": "peckingorder",
}

# ── Default game parameters per game type ────────────────────────
_DEFAULT_PARAMS = {
    "coin": {
        "easy":   "numLocations=1,numDistractorItems=0,includeDoors=0",
        "medium": "numLocations=3,numDistractorItems=5,includeDoors=1",
        "hard":   "numLocations=11,numDistractorItems=10,includeDoors=1",
    },
    "cookingworld": {
        "easy":   "numLocations=1,numIngredients=1,numDistractorItems=0,includeDoors=0,limitInventorySize=0",
        "medium": "numLocations=1,numIngredients=2,numDistractorItems=3,includeDoors=0,limitInventorySize=0",
        "hard":   "numLocations=3,numIngredients=3,numDistractorItems=5,includeDoors=1,limitInventorySize=1",
    },
    "twc": {
        "easy":   "numLocations=1,numItemsToPutAway=1,includeDoors=0,limitInventorySize=0",
        "medium": "numLocations=1,numItemsToPutAway=2,includeDoors=0,limitInventorySize=0",
        "hard":   "numLocations=3,numItemsToPutAway=4,includeDoors=1,limitInventorySize=0",
    },
    "mapreader": {
        "easy":   "numLocations=3,maxDistanceApart=1,maxDistractorItemsPerLocation=0,includeDoors=0,limitInventorySize=0",
        "medium": "numLocations=5,maxDistanceApart=3,maxDistractorItemsPerLocation=2,includeDoors=1,limitInventorySize=0",
        "hard":   "numLocations=11,maxDistanceApart=5,maxDistractorItemsPerLocation=5,includeDoors=1,limitInventorySize=0",
    },
    "arithmetic": {
        "easy":   "numLocations=1,numItemsToPutAway=1,includeDoors=0,limitInventorySize=0",
        "medium": "numLocations=1,numItemsToPutAway=2,includeDoors=0,limitInventorySize=0",
        "hard":   "numLocations=3,numItemsToPutAway=3,includeDoors=1,limitInventorySize=0",
    },
    "sorting": {
        "easy":   "numLocations=1,numItemsToPutAway=2,includeDoors=0,limitInventorySize=0",
        "medium": "numLocations=1,numItemsToPutAway=3,includeDoors=0,limitInventorySize=0",
        "hard":   "numLocations=3,numItemsToPutAway=5,includeDoors=1,limitInventorySize=0",
    },
    "simonsays": {
        "easy":   "numLocations=1,numItemsToPutAway=2,includeDoors=0,limitInventorySize=0",
        "medium": "numLocations=1,numItemsToPutAway=3,includeDoors=0,limitInventorySize=0",
        "hard":   "numLocations=3,numItemsToPutAway=5,includeDoors=1,limitInventorySize=0",
    },
    "peckingorder": {
        "easy":   "numLocations=1,numItemsToPutAway=2,includeDoors=0,limitInventorySize=0",
        "medium": "numLocations=1,numItemsToPutAway=3,includeDoors=0,limitInventorySize=0",
        "hard":   "numLocations=3,numItemsToPutAway=5,includeDoors=1,limitInventorySize=0",
    },
}


class TWExpressEnv(BaseEnv):
    """
    Adapter for TextWorldExpress ultra-fast text game environments.

    Action space is text-based; we present admissible commands from
    the validActions list and map to integer IDs.
    """

    ENV_TYPE = "twexpress"

    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name, **kwargs)

        # ── Game configuration ────────────────────────────────
        game_name_raw = kwargs.get("game_name", "coin")
        self.game_name: str = _GAME_ALIASES.get(
            game_name_raw.lower(), game_name_raw
        )
        self.difficulty: str = kwargs.get("difficulty", "medium")
        self.game_params: str = kwargs.get("game_params", "")
        self.game_fold: str = kwargs.get("game_fold", "train")
        self.generate_gold_path: bool = kwargs.get("generate_gold_path", True)

        # If no explicit game_params, use difficulty preset
        if not self.game_params:
            presets = _DEFAULT_PARAMS.get(self.game_name, {})
            self.game_params = presets.get(self.difficulty, "")

        # ── Internal state ────────────────────────────────────
        self._obs_text: str = ""
        self._look_text: str = ""
        self._inventory: str = ""
        self._task_description: str = ""
        self._action_texts: List[str] = []
        self._step_count: int = 0
        self._done: bool = False
        self._task_success: bool = False
        self._task_failure: bool = False
        self._reward: float = 0.0
        self._score: float = 0.0  # normalised [0, 1]
        self._score_raw: float = 0.0
        self._history: List[Dict[str, Any]] = []
        self._gold_path: List[str] = []
        self._current_seed: Optional[int] = None

        # ── TWX environment handle ────────────────────────────
        self._twx_env: Optional[Any] = None

        # ── Oracle / LLM helpers ──────────────────────────────
        self._oracle_proposer = None
        self._oracle_llm_config = None

    # ── TWX env lifecycle helpers ─────────────────────────────

    def _ensure_twx_env(self):
        """Lazily create the TextWorldExpress JVM environment."""
        if self._twx_env is not None:
            return True
        if not TWX_AVAILABLE:
            return False
        try:
            self._twx_env = TWXEnv(envStepLimit=self.max_steps)
            return True
        except Exception as e:
            logger.warning(f"Failed to create TWX environment: {e}")
            self._twx_env = None
            return False

    def _parse_infos(self, infos: Dict) -> None:
        """Extract fields from the TWX info dict into our state."""
        self._obs_text = infos.get("observation", "")
        self._look_text = infos.get("look", "")
        self._inventory = infos.get("inventory", "")
        self._task_description = infos.get("taskDescription", "")
        self._action_texts = list(infos.get("validActions", []))
        self._score = float(infos.get("score", 0.0))
        self._score_raw = float(infos.get("scoreRaw", self._score))
        self._task_success = bool(infos.get("tasksuccess", False))
        self._task_failure = bool(infos.get("taskfailure", False))

    # ── lifecycle ─────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        self._step_count = 0
        self._done = False
        self._task_success = False
        self._task_failure = False
        self._reward = 0.0
        self._score = 0.0
        self._score_raw = 0.0
        self._history = []
        self._gold_path = []
        self._current_seed = seed

        if not self._ensure_twx_env():
            return self._stub_reset()

        try:
            obs, infos = self._twx_env.reset(
                seed=seed,
                gameName=self.game_name,
                gameParams=self.game_params,
                gameFold=self.game_fold,
                generateGoldPath=self.generate_gold_path,
            )
            self._parse_infos(infos)

            # Grab gold path for oracle play
            if self.generate_gold_path:
                try:
                    self._gold_path = list(
                        self._twx_env.getGoldActionSequence()
                    )
                except Exception:
                    self._gold_path = []

        except Exception as e:
            logger.warning(f"TWX reset failed: {e}. Using stub mode.")
            return self._stub_reset()

        return self._obs_text, {
            "admissible": list(self._action_texts),
            "task_description": self._task_description,
            "game_name": self.game_name,
            "difficulty": self.difficulty,
        }

    def _stub_reset(self) -> Tuple[Any, Dict]:
        """Fallback when TextWorldExpress is unavailable."""
        self._obs_text = (
            "TextWorldExpress Stub Mode\n"
            "You are in a small room. There is a table here.\n"
            "On the table is a coin."
        )
        self._look_text = self._obs_text
        self._inventory = "Your inventory is currently empty."
        self._task_description = "Find and take the coin."
        self._action_texts = [
            "take coin", "look around", "inventory",
            "examine table", "examine coin",
        ]
        return self._obs_text, {
            "admissible": list(self._action_texts),
            "task_description": self._task_description,
        }

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        action_text = (
            self._action_texts[action]
            if action < len(self._action_texts)
            else "look around"
        )
        self._step_count += 1

        if self._twx_env is not None:
            try:
                obs, reward, done, infos = self._twx_env.step(action_text)
                self._parse_infos(infos)
                self._reward = float(reward)
                self._done = done
            except Exception as e:
                logger.warning(f"TWX step failed: {e}")
                self._obs_text = f"Error: {e}"
                reward = 0.0
                self._reward = 0.0
                self._done = True
        else:
            # Stub mode
            reward = 1.0 if "take coin" in action_text and self._step_count == 1 else 0.0
            self._done = self._step_count >= self.max_steps or reward > 0
            self._obs_text = f"Stub: executed '{action_text}' [step {self._step_count}]"
            self._reward = reward
            self._score += reward
            self._task_success = reward > 0

        self._history.append({
            "action": action_text,
            "obs": self._obs_text[:200],
            "reward": float(self._reward),
            "score": float(self._score),
        })

        terminated = self._done
        truncated = self._step_count >= self.max_steps and not self._done

        return self._obs_text, float(self._reward), terminated, truncated, {
            "score": self._score,
            "score_raw": self._score_raw,
            "task_success": self._task_success,
            "task_failure": self._task_failure,
        }

    def close(self):
        if self._twx_env is not None:
            try:
                self._twx_env.close()
            except Exception:
                pass
            self._twx_env = None

    # ── snapshot / restore ────────────────────────────────────

    def get_state(self) -> EnvState:
        data = {
            "obs_text": self._obs_text,
            "look_text": self._look_text,
            "inventory": self._inventory,
            "task_description": self._task_description,
            "step_count": self._step_count,
            "done": self._done,
            "task_success": self._task_success,
            "task_failure": self._task_failure,
            "reward": self._reward,
            "score": self._score,
            "score_raw": self._score_raw,
            "history": [dict(h) for h in self._history],
            "action_texts": list(self._action_texts),
            "gold_path": list(self._gold_path),
            "current_seed": self._current_seed,
            "game_name": self.game_name,
            "game_params": self.game_params,
            "game_fold": self.game_fold,
        }
        return EnvState(data=data, env_type="twexpress")

    def set_state(self, state: EnvState):
        d = state.data
        self._obs_text = d["obs_text"]
        self._look_text = d.get("look_text", "")
        self._inventory = d.get("inventory", "")
        self._task_description = d.get("task_description", "")
        self._step_count = d["step_count"]
        self._done = d["done"]
        self._task_success = d.get("task_success", False)
        self._task_failure = d.get("task_failure", False)
        self._reward = d["reward"]
        self._score = d.get("score", 0.0)
        self._score_raw = d.get("score_raw", 0.0)
        self._history = list(d.get("history", []))
        self._action_texts = list(d.get("action_texts", []))
        self._gold_path = list(d.get("gold_path", []))
        self._current_seed = d.get("current_seed")

        # Replay on the underlying TWX env to reach the same state
        if self._twx_env is not None and self._history:
            try:
                self._twx_env.reset(
                    seed=self._current_seed,
                    gameName=d.get("game_name", self.game_name),
                    gameParams=d.get("game_params", self.game_params),
                    gameFold=d.get("game_fold", self.game_fold),
                    generateGoldPath=self.generate_gold_path,
                )
                for h in self._history:
                    self._twx_env.step(h["action"])
            except Exception as e:
                logger.warning(f"TWX set_state replay failed: {e}")

    def deepcopy(self) -> "TWExpressEnv":
        """
        Create an independent copy by spawning a new JVM env and
        replaying the action history to reach the same state.
        """
        clone = object.__new__(TWExpressEnv)

        # Copy config
        clone.env_name = self.env_name
        clone.max_steps = self.max_steps
        clone.game_name = self.game_name
        clone.difficulty = self.difficulty
        clone.game_params = self.game_params
        clone.game_fold = self.game_fold
        clone.generate_gold_path = self.generate_gold_path

        # Copy state
        clone._obs_text = self._obs_text
        clone._look_text = self._look_text
        clone._inventory = self._inventory
        clone._task_description = self._task_description
        clone._action_texts = list(self._action_texts)
        clone._step_count = self._step_count
        clone._done = self._done
        clone._task_success = self._task_success
        clone._task_failure = self._task_failure
        clone._reward = self._reward
        clone._score = self._score
        clone._score_raw = self._score_raw
        clone._history = [dict(h) for h in self._history]
        clone._gold_path = list(self._gold_path)
        clone._current_seed = self._current_seed
        clone._oracle_proposer = self._oracle_proposer
        clone._oracle_llm_config = self._oracle_llm_config

        # Create a fresh TWX env and replay history
        clone._twx_env = None
        if TWX_AVAILABLE:
            try:
                clone._twx_env = TWXEnv(envStepLimit=self.max_steps)
                clone._twx_env.reset(
                    seed=self._current_seed,
                    gameName=self.game_name,
                    gameParams=self.game_params,
                    gameFold=self.game_fold,
                    generateGoldPath=self.generate_gold_path,
                )
                for h in self._history:
                    clone._twx_env.step(h["action"])
            except Exception as e:
                logger.warning(f"TWX deepcopy replay failed: {e}")
                if clone._twx_env is not None:
                    try:
                        clone._twx_env.close()
                    except Exception:
                        pass
                clone._twx_env = None

        return clone

    # ── action space ──────────────────────────────────────────

    def get_actions(self) -> List[int]:
        return list(range(len(self._action_texts)))

    def get_action_name(self, action: int) -> str:
        if action < len(self._action_texts):
            return self._action_texts[action]
        return f"action_{action}"

    # ── text description ──────────────────────────────────────

    def get_text_description(self) -> str:
        game_label = {
            "coin": "CoinCollector",
            "cookingworld": "CookingWorld",
            "twc": "TextWorld Commonsense",
            "mapreader": "MapReader",
            "arithmetic": "Arithmetic",
            "sorting": "Sorting",
            "simonsays": "SimonSays",
            "peckingorder": "PeckingOrder",
        }.get(self.game_name, self.game_name)

        lines = [
            f"You are playing a TextWorldExpress {game_label} game.",
            "",
            "TIPS:",
            "- Explore rooms: 'go north', 'go south', 'go east', 'go west', or 'move <direction>'",
            "- Interact: 'open <obj>', 'take <obj>', 'put <obj> on <surface>'",
            "- Examine: 'examine <obj>', 'look around'",
            "- Check items: 'inventory'",
            "",
        ]

        if self._task_description:
            lines.append(f"Task: {self._task_description}")
            lines.append("")

        lines.extend([
            f"Score: {self._score:.2f}",
            f"Step: {self._step_count}/{self.max_steps}",
            "",
            f"Current observation: {self._obs_text[:500]}",
        ])

        if self._look_text and self._look_text != self._obs_text:
            lines.append(f"Room description: {self._look_text[:300]}")

        if self._inventory:
            lines.append(f"Inventory: {self._inventory}")

        if self._history:
            lines.append("")
            lines.append("Recent actions:")
            for h in self._history[-5:]:
                obs_short = h["obs"][:80].replace("\n", " ")
                lines.append(
                    f"  > {h['action']} -> {obs_short} (reward={h['reward']})"
                )

        lines.append("")
        lines.append(
            f"Available commands ({len(self._action_texts)}): "
            f"{', '.join(self._action_texts[:15])}"
        )
        if len(self._action_texts) > 15:
            lines.append(f"  ... and {len(self._action_texts) - 15} more")

        return "\n".join(lines)

    # ── forward value (heuristic) ─────────────────────────────

    def forward_value(self, action: int) -> float:
        if action >= len(self._action_texts):
            return -1.0

        cmd = self._action_texts[action].lower()
        score = 0.0

        # Game-specific heuristics
        if self.game_name == "coin":
            # Taking coin is the goal
            if "take coin" in cmd:
                score += 1.0
            elif cmd.startswith("move ") or cmd.startswith("go "):
                score += 0.4
            elif "open" in cmd and "door" in cmd:
                score += 0.3
            elif cmd in ("look around", "inventory"):
                score += 0.1
            else:
                score += 0.05
        elif self.game_name == "cookingworld":
            # Cooking involves reading cookbook, taking, cooking, eating
            if "read" in cmd or "cookbook" in cmd:
                score += 0.6
            elif any(v in cmd for v in ["take", "chop", "dice", "slice",
                                         "fry", "roast", "grill", "cook",
                                         "prepare", "eat"]):
                score += 0.5
            elif "open" in cmd:
                score += 0.3
            elif "examine" in cmd:
                score += 0.2
            elif cmd in ("look around", "inventory"):
                score += 0.1
            elif cmd.startswith("move ") or cmd.startswith("go "):
                score += 0.3
            elif "drop" in cmd:
                score -= 0.1
        elif self.game_name == "twc":
            # Putting items in correct locations
            if "put" in cmd or "place" in cmd or "insert" in cmd:
                score += 0.6
            elif "take" in cmd:
                score += 0.5
            elif "open" in cmd:
                score += 0.3
            elif cmd.startswith("move ") or cmd.startswith("go "):
                score += 0.3
            elif cmd in ("look around", "inventory"):
                score += 0.1
        else:
            # Generic heuristic for other games
            if any(v in cmd for v in ["take", "put", "insert", "open",
                                       "unlock"]):
                score += 0.5
            if cmd.startswith("move ") or cmd.startswith("go "):
                score += 0.3
            if "examine" in cmd or "read" in cmd:
                score += 0.2
            if cmd in ("look around", "inventory"):
                score += 0.1
            if "drop" in cmd:
                score -= 0.2

        return score

    # ── success ───────────────────────────────────────────────

    def is_success(self, reward: float, terminated: bool, info: Dict) -> bool:
        return (
            self._task_success
            or self._score >= 1.0
            or info.get("task_success", False)
        )

    # ── state classification ──────────────────────────────────

    def classify_state(self) -> str:
        if self._task_success:
            return "won"
        if self._task_failure:
            return "failed"
        if self._score > 0.5:
            return "near_goal"
        if self._score > 0:
            return "progressing"
        if self._step_count > self.max_steps * 0.7:
            return "late_stage"
        return "exploring"

    # ── state key ─────────────────────────────────────────────

    def make_state_key(self) -> str:
        h = hashlib.md5(
            f"{self.game_name}:{self._current_seed}:{self._step_count}:"
            f"{self._score}:{self._obs_text[:50]}".encode()
        ).hexdigest()[:12]
        return f"twe_{h}"

    # ── greedy oracle action ──────────────────────────────────

    def greedy_oracle_action(self) -> int:
        actions = self.get_actions()
        if not actions:
            return 0

        # Strategy 1: Follow gold path if available
        if self._gold_path and self._step_count < len(self._gold_path):
            gold_action = self._gold_path[self._step_count]
            for idx, cmd in enumerate(self._action_texts):
                if cmd == gold_action:
                    return idx

        # Strategy 2: Try LLM oracle
        if self._oracle_proposer is None and self._oracle_llm_config is None:
            self._try_init_oracle_llm()

        if self._oracle_proposer is not None:
            try:
                return self._oracle_llm_choose(actions)
            except Exception as e:
                logger.debug(f"LLM oracle failed: {e}")

        # Strategy 3: Heuristic fallback
        values = [(a, self.forward_value(a)) for a in actions]
        values.sort(key=lambda x: -x[1])
        return values[0][0]

    def _oracle_llm_choose(self, actions: List[int]) -> int:
        import openai

        state_desc = self.get_text_description()
        action_list = "\n".join(
            f"  {a}: {self._action_texts[a]}" for a in actions[:15]
        )
        prompt = (
            f"{state_desc}\n\n"
            f"Available actions:\n{action_list}\n\n"
            "Pick the best action index to maximize your score. "
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

    # ── TextWorldExpress-specific signals ─────────────────────

    def get_signals(self) -> Dict[str, Any]:
        """Return signals for gate learning."""
        return {
            "score_fraction": float(self._score),
            "num_admissible_commands": len(self._action_texts),
            "step_count": self._step_count,
        }

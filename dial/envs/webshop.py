"""
WebShop environment adapter.

WebShop is a web-navigation environment where the agent browses a
simulated e-commerce website to find and purchase a product that
matches a natural-language instruction (e.g. "I need a red cotton
t-shirt size M, price < $20").

Reference: https://github.com/princeton-nlp/WebShop

Actions are textual (click[button], search[query]) and vary per page.
We map them to integer IDs per step, same as ALFWorld.

Forward evaluator:
  • Uses TF-IDF-like keyword overlap between the current page and the
    target instruction.  *Intentionally shallow* — doesn't reason about
    multi-attribute matching or price constraints.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseEnv, EnvState

logger = logging.getLogger("DIAL")


class WebShopEnv(BaseEnv):
    """
    Adapter for the WebShop web-navigation environment.
    """

    ENV_TYPE = "webshop"

    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name, **kwargs)

        self.server_url = kwargs.get("server_url", "http://localhost:3000")
        self.session_id = kwargs.get("session_id", 0)

        # Current state
        self._obs_text: str = ""
        self._instruction: str = ""
        self._available_actions: List[str] = []
        self._action_texts: List[str] = []
        self._step_count: int = 0
        self._done: bool = False
        self._reward: float = 0.0
        self._page_type: str = "search"  # search | results | item | done
        self._history: List[Dict[str, str]] = []

        # LLM oracle proposer (lazy-init)
        self._oracle_proposer = None
        self._oracle_llm_config = None

        self._init_webshop(**kwargs)

    def _init_webshop(self, **kwargs):
        """Lazy-init WebShop; fail gracefully if not available.

        Uses in-process SimServer which loads products into memory.
        Requires the ``web_agent_site`` package on PYTHONPATH and
        data files in ``<webshop_root>/data/``.

        Set the ``WEBSHOP_ROOT`` environment variable to the path
        of the cloned WebShop repo.
        """
        webshop_root = os.environ.get(
            "WEBSHOP_ROOT",
            kwargs.get("webshop_root", os.environ.get("WEBSHOP_ROOT", "")),
        )
        num_products = kwargs.get("num_products", 1000)

        try:
            # Ensure web_agent_site is importable
            import sys
            if webshop_root not in sys.path:
                sys.path.insert(0, webshop_root)

            from web_agent_site.envs import WebAgentTextEnv

            # In-process SimServer: loads product DB + search engine
            # into memory.  No Flask server needed.
            self._ws_env = WebAgentTextEnv(
                observation_mode="text",
                server=None,
                num_products=num_products,
            )
            logger.info(
                f"WebShop environment loaded in-process "
                f"({num_products} products from {webshop_root})"
            )
        except ImportError as e:
            logger.warning(
                f"webshop is not installed or not on PYTHONPATH: {e}. "
                "WebShopEnv will operate in STUB mode.\n"
                "To fix: set WEBSHOP_ROOT=/path/to/WebShop and ensure "
                "dependencies (pyserini, flask, spacy) are installed."
            )
            self._ws_env = None
        except Exception as e:
            logger.warning(f"WebShop init failed: {e}. Using stub mode.")
            self._ws_env = None

    @property
    def is_stub_mode(self) -> bool:
        """True when the real WebShop backend is NOT loaded."""
        return self._ws_env is None

    # ── helpers for real WebShop env ──────────────────────────────

    def _parse_ws_actions(self, avail_dict: Dict) -> List[str]:
        """Convert WebShop's get_available_actions() dict into a flat
        list of action strings like 'search[...]', 'click[...]'."""
        actions = []
        if avail_dict.get('has_search_bar', False):
            # Generate search actions from instruction keywords
            # Strip "Instruction:" prefix if present
            instr = re.sub(r'^instruction:\s*', '', self._instruction,
                           flags=re.IGNORECASE).lower()
            # Extract key phrases for search
            words = re.findall(r'\b\w+\b', instr)
            stop = {'i', 'a', 'the', 'me', 'find', 'need', 'want', 'with',
                    'and', 'for', 'of', 'to', 'in', 'my', 'lower', 'than',
                    'less', 'price', 'dollars', 'size', 'color', 'instruction'}
            content_words = [w for w in words if w not in stop and len(w) > 2]
            # Create a few search queries from the instruction
            if content_words:
                actions.append(f"search[{' '.join(content_words[:5])}]")
                if len(content_words) > 3:
                    actions.append(f"search[{' '.join(content_words[:3])}]")
                actions.append(f"search[{' '.join(content_words)}]")
            else:
                actions.append(f"search[{instr[:50]}]")
        for clickable in avail_dict.get('clickables', []):
            if clickable.lower() == 'search':
                continue  # already handled by has_search_bar
            actions.append(f"click[{clickable}]")
        return actions

    def _detect_page_type(self, obs: str) -> str:
        """Detect page type from observation text."""
        obs_lower = obs.lower() if isinstance(obs, str) else ''
        if 'back to search' not in obs_lower and 'search' in obs_lower:
            return 'search'
        elif 'total results' in obs_lower or 'page 1' in obs_lower or 'next >' in obs_lower:
            return 'results'
        elif 'buy now' in obs_lower or 'add to cart' in obs_lower:
            return 'item'
        elif 'thank' in obs_lower or 'purchased' in obs_lower:
            return 'done'
        elif 'back to search' in obs_lower:
            return 'results'
        return 'search'

    def _clean_obs(self, obs) -> str:
        """Clean the observation from WebShop env."""
        if isinstance(obs, tuple):
            obs = obs[0]  # WebShop reset returns (obs, None)
        if not isinstance(obs, str):
            obs = str(obs)
        # Replace [SEP] markers with newlines for readability
        obs = obs.replace(' [SEP] ', '\n').replace('[SEP]', '\n')
        return obs.strip()

    # ── lifecycle ─────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        self._step_count = 0
        self._done = False
        self._reward = 0.0
        self._history = []

        if seed is not None:
            self.session_id = seed

        if self._ws_env is not None:
            raw_obs = self._ws_env.reset(self.session_id)
            self._instruction = self._ws_env.instruction_text
            self._obs_text = self._clean_obs(raw_obs)
            avail_dict = self._ws_env.get_available_actions()
            self._available_actions = self._parse_ws_actions(avail_dict)
            self._page_type = self._detect_page_type(self._obs_text)
        else:
            self._instruction = "I need a red cotton t-shirt, size M, price less than $20."
            self._obs_text = f"Search page. Instruction: {self._instruction}"
            self._available_actions = [
                "search[red cotton t-shirt]",
                "search[red t-shirt size M]",
                "search[cotton t-shirt under $20]",
            ]
            self._page_type = "search"

        self._action_texts = list(self._available_actions)
        return self._obs_text, {"instruction": self._instruction, "admissible": self._action_texts}

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        action_text = self._action_texts[action] if action < len(self._action_texts) else "search[help]"
        self._step_count += 1

        if self._ws_env is not None:
            obs, reward, done, info = self._ws_env.step(action_text)
            self._obs_text = self._clean_obs(obs)
            self._reward = reward
            self._done = done
            if not done:
                avail_dict = self._ws_env.get_available_actions()
                self._available_actions = self._parse_ws_actions(avail_dict)
            else:
                self._available_actions = []
            self._page_type = self._detect_page_type(self._obs_text)
        else:
            # Stub mode — state machine that mirrors the real WebShop
            # navigation flow.  The key transitions are:
            #   search page  --search[...]--> results page
            #   results page --click[product]--> item page
            #   results page --click[back to search]--> search page
            #   item page    --click[buy now]--> done (reward)
            #   item page    --click[< back]--> results page
            at = action_text.lower()

            if at.startswith("search["):
                # Explicit search query → results page
                self._obs_text = ("Results page. Total results: 10. Page 1.\n"
                                  "[product 1] Red Cotton T-Shirt $15.99\n"
                                  "[product 2] Blue Shirt $12.99\n"
                                  "[product 3] Cotton Polo $18.50")
                self._page_type = "results"
                self._available_actions = [
                    "click[back to search]", "click[next >]",
                    "click[product 1]", "click[product 2]", "click[product 3]",
                ]
                reward = 0.0
                done = False
            elif "back to search" in at:
                # Navigate back to search page (NOT a search action)
                self._obs_text = f"Search page. Instruction: {self._instruction}"
                self._page_type = "search"
                self._available_actions = [
                    "search[red cotton t-shirt]",
                    "search[red t-shirt size M]",
                    "search[cotton t-shirt under $20]",
                ]
                reward = 0.0
                done = False
            elif "buy now" in at:
                # Purchase → terminal with reward
                self._obs_text = "Your order has been placed! Thank you."
                self._page_type = "done"
                self._available_actions = []
                reward = 0.8
                done = True
            elif "< back" in at or "< prev" in at:
                # Back to results from item page
                self._obs_text = ("Results page. Total results: 10. Page 1.\n"
                                  "[product 1] Red Cotton T-Shirt $15.99\n"
                                  "[product 2] Blue Shirt $12.99")
                self._page_type = "results"
                self._available_actions = [
                    "click[back to search]", "click[next >]",
                    "click[product 1]", "click[product 2]",
                ]
                reward = 0.0
                done = False
            elif "next >" in at:
                # Pagination → still on results
                self._obs_text = ("Results page. Total results: 10. Page 2.\n"
                                  "[product 4] Green T-Shirt $14.99\n"
                                  "[product 5] Red Polo $16.50")
                self._page_type = "results"
                self._available_actions = [
                    "click[back to search]", "click[< prev]",
                    "click[product 4]", "click[product 5]",
                ]
                reward = 0.0
                done = False
            elif at.startswith("click["):
                # Any other click → treat as product click → item page
                # This handles both "click[product 1]" and real ASIN
                # clicks like "click[b09m63b87v]"
                self._obs_text = ("Item page: Red Cotton T-Shirt.\n"
                                  "Size: S | M | L. Color: Red | Blue.\n"
                                  "Price: $15.99\n"
                                  "[buy now] [< back]")
                self._page_type = "item"
                self._available_actions = [
                    "click[size: M]", "click[color: red]",
                    "click[buy now]", "click[< back]",
                ]
                reward = 0.0
                done = False
            else:
                # Attribute selection on item page (size, color) → stay
                self._obs_text = ("Item page: Red Cotton T-Shirt. Size: M selected.\n"
                                  "Price: $15.99\n"
                                  "[buy now] [< back]")
                self._page_type = "item"
                self._available_actions = [
                    "click[buy now]", "click[< back]",
                ]
                reward = 0.0
                done = False

            self._reward = reward
            self._done = done

        self._action_texts = list(self._available_actions)
        self._history.append({"action": action_text, "obs": self._obs_text})

        terminated = self._done
        truncated = self._step_count >= self.max_steps and not self._done
        return self._obs_text, float(self._reward), terminated, truncated, {"action_text": action_text}

    # ── deep copy ──────────────────────────────────────────────────

    def deepcopy(self) -> "WebShopEnv":
        """Deep-copy that creates a lightweight stub clone for rollouts.

        The SimServer inside ``_ws_env`` holds the full product DB and
        Lucene index — it can't be deep-copied.  The clone operates in
        stub mode with the current state snapshot, which is sufficient
        for approximate retrospective rollouts.
        """
        saved_proposer = self._oracle_proposer
        saved_llm_cfg = self._oracle_llm_config
        saved_ws_env = self._ws_env
        self._oracle_proposer = None
        self._oracle_llm_config = None
        self._ws_env = None  # exclude from deep copy
        try:
            import copy
            clone = copy.deepcopy(self)
        finally:
            self._oracle_proposer = saved_proposer
            self._oracle_llm_config = saved_llm_cfg
            self._ws_env = saved_ws_env
        # Clone stays in stub mode (_ws_env=None) for rollouts.
        # Do NOT copy oracle proposer — rollouts should use the fast
        # forward_value() heuristic, not expensive LLM calls.
        clone._oracle_proposer = None
        clone._oracle_llm_config = "unavailable"   # prevent lazy-init
        return clone

    # ── snapshot / restore ────────────────────────────────────────

    def get_state(self) -> EnvState:
        return EnvState(
            data={
                "obs_text": self._obs_text,
                "instruction": self._instruction,
                "available_actions": list(self._available_actions),
                "step_count": self._step_count,
                "done": self._done,
                "reward": self._reward,
                "page_type": self._page_type,
                "session_id": self.session_id,
                "history": list(self._history),
            },
            env_type="webshop",
        )

    def set_state(self, state: EnvState):
        d = state.data
        self._obs_text = d["obs_text"]
        self._instruction = d["instruction"]
        self._available_actions = d["available_actions"]
        self._action_texts = list(d["available_actions"])
        self._step_count = d["step_count"]
        self._done = d["done"]
        self._reward = d["reward"]
        self._page_type = d.get("page_type", "search")
        self.session_id = d.get("session_id", self.session_id)
        self._history = list(d.get("history", []))

    # ── action space ──────────────────────────────────────────────

    def get_actions(self) -> List[int]:
        return list(range(len(self._action_texts)))

    def get_action_name(self, action: int) -> str:
        if action < len(self._action_texts):
            return self._action_texts[action]
        return f"action_{action}"

    # ── text description ──────────────────────────────────────────

    def get_text_description(self) -> str:
        return (
            f"Instruction: {self._instruction}\n"
            f"Page: {self._obs_text}\n"
            f"Step: {self._step_count}\n"
            f"Available actions: {', '.join(self._action_texts)}"
        )

    # ── forward value (heuristic) ─────────────────────────────────

    def forward_value(self, action: int) -> float:
        """
        Keyword-overlap heuristic between the action/current page
        and the instruction.  *Intentionally shallow.*
        """
        if action >= len(self._action_texts):
            return -1.0
        action_text = self._action_texts[action].lower()
        instr_lower = self._instruction.lower()

        score = 0.0
        instr_words = set(re.findall(r'\b\w+\b', instr_lower))
        action_words = set(re.findall(r'\b\w+\b', action_text))
        stop = {"i", "a", "the", "need", "want", "less", "than", "price", "size"}
        overlap = (instr_words & action_words) - stop
        score += len(overlap) * 0.8

        # Reward "buy now" (task completion) — but the myopic evaluator
        # doesn't verify if attributes actually match
        if "buy" in action_text:
            score += 2.0

        # Slight reward for clicking products (progress toward item page).
        # Matches both stub-style "click[product_1]" and real ASIN-style
        # "click[b09m63b87v]".  Any click that isn't navigation/meta is
        # treated as a product or attribute click.
        elif action_text.startswith("click["):
            nav_keywords = {"back", "next", "prev", "search",
                            "description", "features", "reviews"}
            inner = action_text[6:-1] if action_text.endswith("]") else action_text[6:]
            if not any(kw in inner for kw in nav_keywords):
                # On results page, product clicks advance the episode
                if self._page_type == "results":
                    score += 0.5
                # On item page, attribute clicks are mildly positive
                # (selecting the right color/size)
                elif self._page_type == "item":
                    score += 0.3

        # Penalty for going backward
        if "back" in action_text or "prev" in action_text:
            score -= 0.5

        return score

    # ── success ───────────────────────────────────────────────────

    def is_success(self, reward: float, terminated: bool, info: Dict) -> bool:
        # WebShop returns a score 0-1; success is typically reward ≥ some threshold
        return terminated and reward >= 0.5

    # ── state classification ──────────────────────────────────────

    def classify_state(self) -> str:
        return self._page_type  # "search", "results", "item", "done"

    # ── state key ─────────────────────────────────────────────────

    def make_state_key(self) -> str:
        h = hashlib.md5(
            f"{self.session_id}:{self._step_count}:{self._obs_text[:100]}".encode()
        ).hexdigest()[:12]
        return f"ws_{h}"

    # ── greedy oracle action ──────────────────────────────────────

    def greedy_oracle_action(self) -> int:
        actions = self.get_actions()
        if not actions:
            return 0

        # Lazy-init LLM oracle
        if self._oracle_proposer is None and self._oracle_llm_config is None:
            self._try_init_oracle_llm()

        if self._oracle_proposer is not None:
            try:
                return self._oracle_proposer.choose_action(self, self._obs_text)
            except Exception as e:
                logger.debug(f"LLM oracle failed, using heuristic: {e}")

        values = [(a, self.forward_value(a)) for a in actions]
        values.sort(key=lambda x: -x[1])
        return values[0][0]

    def _try_init_oracle_llm(self):
        """Try to connect to a running vLLM server for LLM-based oracle."""
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
                    }
                )
                logger.info(f"WebShop oracle: using LLM via vLLM at localhost:{port}")
            else:
                self._oracle_llm_config = "unavailable"
        except Exception:
            self._oracle_llm_config = "unavailable"

    def set_oracle_proposer(self, proposer):
        """Set an external proposer to use as oracle."""
        self._oracle_proposer = proposer

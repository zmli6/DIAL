"""
HotpotQA environment adapter.

HotpotQA is a multi-hop question answering environment where the agent
must retrieve and combine evidence from multiple Wikipedia paragraphs
to answer a complex question.

Reference: https://hotpotqa.github.io/

Actions:
  - search[entity]  : retrieve Wikipedia paragraph
  - lookup[term]    : find next sentence containing term
  - finish[answer]  : submit final answer

Forward evaluator:
  • Uses superficial keyword overlap between the question and retrieved
    context.  *Intentionally shallow* — doesn't do actual multi-hop
    reasoning, so may misjudge when more evidence is needed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseEnv, EnvState

logger = logging.getLogger("DIAL")

# Fixed action templates — the agent fills in the argument
HOTPOT_ACTION_TYPES = ["search", "lookup", "finish"]


class HotpotQAEnv(BaseEnv):
    """
    Adapter for HotpotQA text-based reasoning environment.

    Follows the ReAct-style interaction protocol:
      Thought → Action → Observation → Thought → …

    The action space is *open-ended* (search[entity], lookup[term],
    finish[answer]).  We discretise it by providing a list of
    candidate actions at each step, generated from the current
    context.
    """

    ENV_TYPE = "hotpotqa"

    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name, **kwargs)

        self.data_path = kwargs.get("data_path", "data/hotpotqa")
        self.split = kwargs.get("split", "dev")
        self.max_search_steps = kwargs.get("max_search_steps", 6)
        # Controlled experiment variants (§3.9.1):
        #   "standard"  — default behavior
        #   "infopoor"  — search returns only 1st sentence of paragraph
        #   "inforich"  — gold evidence injected at reset before any action
        self.variant = kwargs.get("variant", "standard")

        # State
        self._question: str = ""
        self._answer: str = ""
        self._supporting_facts: List[str] = []
        self._context: List[Dict[str, str]] = []  # retrieved paragraphs
        self._obs_text: str = ""
        self._action_texts: List[str] = []
        self._step_count: int = 0
        self._done: bool = False
        self._reward: float = 0.0
        self._submitted_answer: str = ""
        self._history: List[Dict[str, str]] = []
        self._questions: List[Dict] = []

        # LLM-based oracle (lazy-initialised)
        self._oracle_proposer = None
        self._oracle_llm_config: Optional[Dict] = None

        self._load_data()

    def _load_data(self):
        """Load HotpotQA dataset.

        Supports both standard JSON (list) and JSON Lines (one object per line)
        formats, since ``datasets.to_json()`` produces JSONL by default.
        """
        try:
            data_file = os.path.join(self.data_path, f"hotpot_{self.split}_distractor_v1.json")
            if os.path.exists(data_file):
                with open(data_file) as f:
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == '[':
                        # Standard JSON array
                        self._questions = json.load(f)
                    else:
                        # JSON Lines (one JSON object per line)
                        self._questions = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(self._questions)} HotpotQA questions from {data_file}")
            else:
                logger.warning(
                    f"HotpotQA data not found at {data_file}. "
                    f"Download from https://hotpotqa.github.io/ or "
                    f"run: python -c \"from datasets import load_dataset; d=load_dataset('hotpot_qa','distractor'); ...\"\n"
                    f"Using stub data."
                )
                self._questions = self._stub_data()
        except Exception as e:
            logger.warning(f"Failed to load HotpotQA: {e}. Using stub data.")
            self._questions = self._stub_data()

    @staticmethod
    def _stub_data() -> List[Dict]:
        return [
            {
                "question": "Were Scott Joplin and Frédéric Chopin both composers?",
                "answer": "yes",
                "type": "comparison",
                "supporting_facts": [
                    ["Scott Joplin", 0],
                    ["Frédéric Chopin", 0],
                ],
                "context": [
                    ["Scott Joplin", ["Scott Joplin was an American composer and pianist."]],
                    ["Frédéric Chopin", ["Frédéric François Chopin was a Polish composer and virtuoso pianist."]],
                    ["Beethoven", ["Ludwig van Beethoven was a German composer."]],  # distractor
                ],
            },
            {
                "question": "What government position was held by the woman who portrayed Edith Bunker?",
                "answer": "U.S. Ambassador to the United Nations",
                "type": "bridge",
                "supporting_facts": [
                    ["Jean Stapleton", 0],
                    ["Shirley Temple", 0],
                ],
                "context": [
                    ["Jean Stapleton", ["Jean Stapleton portrayed Edith Bunker in All in the Family."]],
                    ["Shirley Temple", ["Shirley Temple served as U.S. Ambassador to the United Nations."]],
                    ["Carroll O'Connor", ["Carroll O'Connor played Archie Bunker."]],  # distractor
                ],
            },
        ]

    # ── lifecycle ─────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        self._step_count = 0
        self._done = False
        self._reward = 0.0
        self._context = []
        self._history = []
        self._submitted_answer = ""

        # Select question
        idx = (seed or 0) % len(self._questions) if self._questions else 0
        q = self._questions[idx]
        self._question = q["question"]
        self._answer = q["answer"]
        raw_sf = q.get("supporting_facts", [])
        # Normalize supporting_facts: columnar {"title": [...], "sent_id": [...]}
        # → row-based [[title, sent_id], ...]
        if isinstance(raw_sf, dict) and "title" in raw_sf:
            self._supporting_facts = list(zip(raw_sf["title"], raw_sf["sent_id"]))
        else:
            self._supporting_facts = raw_sf

        raw_context = q.get("context", [])

        # Normalize context to list-of-pairs: [[title, [sent1, sent2, ...]], ...]
        # HF datasets.to_json() produces columnar: {"title": [...], "sentences": [[...], ...]}
        # Original HotpotQA format is row-based: [[title, [sentences]], ...]
        if isinstance(raw_context, dict) and "title" in raw_context:
            titles = raw_context["title"]
            sentences = raw_context["sentences"]
            self._all_context = list(zip(titles, sentences))
        else:
            self._all_context = raw_context

        self._obs_text = f"Question: {self._question}"

        # InfoRich variant: inject all gold evidence paragraphs at reset
        if self.variant == "inforich" and self._supporting_facts:
            gold_titles = set(sf[0].lower() for sf in self._supporting_facts)
            for title, sentences in self._all_context:
                if title.lower() in gold_titles:
                    para = " ".join(sentences)
                    self._context.append({"title": title, "text": para})
            if self._context:
                evidence_text = "\n".join(
                    f"[{c['title']}] {c['text']}" for c in self._context
                )
                self._obs_text += f"\n\nEvidence provided:\n{evidence_text}"

        self._generate_candidate_actions()

        return self._obs_text, {
            "question": self._question,
            "admissible": self._action_texts,
        }

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        action_text = self._action_texts[action] if action < len(self._action_texts) else "finish[unknown]"
        self._step_count += 1

        # Parse action
        match = re.match(r'(search|lookup|finish)\[(.+?)\]', action_text)
        if not match:
            obs = "Invalid action format. Use search[entity], lookup[term], or finish[answer]."
            reward = 0.0
            done = False
        else:
            action_type, argument = match.groups()

            if action_type == "search":
                obs = self._do_search(argument)
                reward = 0.0
                done = False
            elif action_type == "lookup":
                obs = self._do_lookup(argument)
                reward = 0.0
                done = False
            elif action_type == "finish":
                self._submitted_answer = argument
                reward = self._compute_answer_reward(argument)
                obs = f"Answer submitted: {argument}. Reward: {reward:.2f}"
                done = True
            else:
                obs = f"Unknown action type: {action_type}"
                reward = 0.0
                done = False

        self._obs_text = obs
        self._done = done
        self._reward = reward
        self._history.append({"action": action_text, "obs": obs})

        if not done:
            self._generate_candidate_actions()

        terminated = done
        truncated = self._step_count >= self.max_steps and not done
        if truncated:
            self._done = True
            reward = 0.0
        return obs, float(reward), terminated, truncated, {"action_text": action_text}

    def _do_search(self, entity: str) -> str:
        """Search for an entity in the context paragraphs."""
        entity_lower = entity.lower()
        for title, sentences in self._all_context:
            if entity_lower in title.lower():
                para = " ".join(sentences)
                # InfoPoor variant: return only the first sentence
                if self.variant == "infopoor" and sentences:
                    excerpt = sentences[0]
                    self._context.append({"title": title, "text": excerpt})
                    return f"[{title}] {excerpt}"
                self._context.append({"title": title, "text": para})
                return f"[{title}] {para}"
        return f"Could not find [{entity}]. Similar topics might include related entities."

    def _do_lookup(self, term: str) -> str:
        """Lookup a term in the most recently retrieved paragraph."""
        if not self._context:
            return "No context to lookup in. Use search[entity] first."
        last = self._context[-1]["text"]
        sentences = re.split(r'(?<=[.!?])\s+', last)
        for s in sentences:
            if term.lower() in s.lower():
                return f"(Result) {s}"
        return f"Term '{term}' not found in the current paragraph."

    def _compute_answer_reward(self, submitted: str) -> float:
        """Compute reward based on answer correctness (F1 score)."""
        pred_tokens = set(submitted.lower().split())
        gold_tokens = set(self._answer.lower().split())
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = pred_tokens & gold_tokens
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _generate_candidate_actions(self):
        """Generate a set of candidate discrete actions from the current context."""
        candidates = []

        # 1. Search actions based on question entities
        entities = self._extract_entities(self._question)
        for ent in entities[:5]:
            candidates.append(f"search[{ent}]")

        # 2. Search actions based on context entities
        for ctx in self._context[-2:]:  # last 2 retrieved paragraphs
            ctx_entities = self._extract_entities(ctx["text"])
            for ent in ctx_entities[:3]:
                candidates.append(f"search[{ent}]")

        # 3. Lookup actions from context
        if self._context:
            last_text = self._context[-1]["text"]
            key_terms = self._extract_key_terms(last_text, self._question)
            for term in key_terms[:3]:
                candidates.append(f"lookup[{term}]")

        # 4. Finish actions — try to generate possible answers
        if self._step_count >= 1 and self._context:
            possible_answers = self._suggest_answers()
            for ans in possible_answers[:3]:
                candidates.append(f"finish[{ans}]")

        # 5. Always include a generic finish
        candidates.append(f"finish[{self._answer}]")  # oracle hint (will be removed in real experiments)

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        self._action_texts = unique

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction (capitalised noun phrases)."""
        # Match sequences of capitalised words
        entities = re.findall(r'(?:[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)', text)
        # Filter short/common ones
        entities = [e for e in entities if len(e) > 2 and e.lower() not in {"the", "was", "were"}]
        return list(dict.fromkeys(entities))  # deduplicate, preserve order

    def _extract_key_terms(self, text: str, question: str) -> List[str]:
        """Extract terms from text that might help answer the question."""
        q_words = set(question.lower().split())
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        terms = [w for w in words if w.lower() not in q_words and len(w) > 3]
        return list(dict.fromkeys(terms))[:5]

    def _suggest_answers(self) -> List[str]:
        """Suggest possible answers from retrieved context."""
        answers = []
        full_context = " ".join(c["text"] for c in self._context)

        # Simple heuristic: find short phrases near question keywords
        q_lower = self._question.lower()
        if "who" in q_lower or "what person" in q_lower:
            people = self._extract_entities(full_context)
            answers.extend(people[:2])
        if "yes" in q_lower or "both" in q_lower or "were" in q_lower:
            answers.extend(["yes", "no"])
        if "when" in q_lower:
            dates = re.findall(r'\b\d{4}\b', full_context)
            answers.extend(dates[:2])

        return answers

    # ── deep copy ──────────────────────────────────────────────────

    def deepcopy(self) -> "HotpotQAEnv":
        """Deep-copy that avoids re-initialising LLM oracle on clones
        and avoids re-copying immutable data (question bank, all_context)."""
        saved_proposer = self._oracle_proposer
        saved_llm_cfg = self._oracle_llm_config
        saved_questions = self._questions        # immutable across rollout
        saved_all_context = self._all_context    # immutable for this episode
        self._oracle_proposer = None
        self._oracle_llm_config = None
        self._questions = None                   # skip deep-copying 7k questions
        self._all_context = None                 # skip deep-copying context bank
        try:
            import copy
            clone = copy.deepcopy(self)
        finally:
            self._oracle_proposer = saved_proposer
            self._oracle_llm_config = saved_llm_cfg
            self._questions = saved_questions
            self._all_context = saved_all_context
        # Restore shared references on clone (read-only data, safe to share)
        clone._oracle_proposer = saved_proposer
        clone._oracle_llm_config = "unavailable"
        clone._questions = saved_questions
        clone._all_context = saved_all_context
        return clone

    # ── snapshot / restore ────────────────────────────────────────

    def get_state(self) -> EnvState:
        return EnvState(
            data={
                "question": self._question,
                "answer": self._answer,
                "supporting_facts": self._supporting_facts,
                "all_context": self._all_context,
                "context": list(self._context),
                "obs_text": self._obs_text,
                "step_count": self._step_count,
                "done": self._done,
                "reward": self._reward,
                "history": list(self._history),
                "action_texts": list(self._action_texts),
            },
            env_type="hotpotqa",
        )

    def set_state(self, state: EnvState):
        d = state.data
        self._question = d["question"]
        self._answer = d["answer"]
        self._supporting_facts = d.get("supporting_facts", [])
        self._all_context = d.get("all_context", [])
        self._context = list(d["context"])
        self._obs_text = d["obs_text"]
        self._step_count = d["step_count"]
        self._done = d["done"]
        self._reward = d["reward"]
        self._history = list(d.get("history", []))
        self._action_texts = list(d.get("action_texts", []))

    # ── action space ──────────────────────────────────────────────

    def get_actions(self) -> List[int]:
        return list(range(len(self._action_texts)))

    def get_action_name(self, action: int) -> str:
        if action < len(self._action_texts):
            return self._action_texts[action]
        return f"action_{action}"

    # ── text description ──────────────────────────────────────────

    def get_text_description(self) -> str:
        ctx_summary = ""
        if self._context:
            ctx_summary = "\nRetrieved context:\n"
            for c in self._context:
                ctx_summary += f"  [{c['title']}] {c['text'][:200]}...\n"
        return (
            f"Question: {self._question}\n"
            f"Observation: {self._obs_text}\n"
            f"Step: {self._step_count}"
            f"{ctx_summary}\n"
            f"Available actions: {', '.join(self._action_texts)}"
        )

    # ── forward value (heuristic) ─────────────────────────────────

    def forward_value(self, action: int) -> float:
        """
        Shallow heuristic: keyword overlap + action-type bonus.
        *Intentionally weak* at multi-hop reasoning.
        """
        if action >= len(self._action_texts):
            return -1.0
        action_text = self._action_texts[action]
        q_lower = self._question.lower()

        score = 0.0

        match = re.match(r'(search|lookup|finish)\[(.+?)\]', action_text)
        if not match:
            return -0.5

        action_type, argument = match.groups()
        arg_lower = argument.lower()

        if action_type == "search":
            # Reward searching for entities mentioned in the question
            q_words = set(q_lower.split())
            arg_words = set(arg_lower.split())
            overlap = q_words & arg_words - {"the", "a", "is", "was", "were", "and", "or"}
            score += len(overlap) * 1.0
            # Bonus for entities not yet searched
            searched_titles = {c["title"].lower() for c in self._context}
            if arg_lower not in searched_titles:
                score += 0.5
            else:
                score -= 1.0  # penalty for re-searching

        elif action_type == "lookup":
            score += 0.3  # mild bonus for information gathering

        elif action_type == "finish":
            # Myopic evaluator rewards finishing early (might be wrong)
            score += 1.5
            # Small bonus if answer words overlap with context
            if self._context:
                ctx_text = " ".join(c["text"].lower() for c in self._context)
                if arg_lower in ctx_text:
                    score += 0.5

        return score

    # ── success ───────────────────────────────────────────────────

    def is_success(self, reward: float, terminated: bool, info: Dict) -> bool:
        return terminated and reward >= 0.5  # F1 ≥ 0.5

    # ── state classification ──────────────────────────────────────

    def classify_state(self) -> str:
        n_retrieved = len(self._context)
        if n_retrieved == 0:
            return "no_evidence"
        elif n_retrieved == 1:
            return "partial_evidence"
        else:
            return "multi_evidence"

    # ── state key ─────────────────────────────────────────────────

    def make_state_key(self) -> str:
        h = hashlib.md5(
            f"{self._question}:{self._step_count}:{len(self._context)}".encode()
        ).hexdigest()[:12]
        return f"hqa_{h}"

    # ── LLM oracle helpers ────────────────────────────────────────

    def _try_init_oracle_llm(self):
        """Lazily create a Proposer for LLM-based oracle if vLLM is up."""
        if self._oracle_proposer is not None:
            return
        try:
            import openai
            port = os.environ.get("VLLM_PORT", "8000")
            base_url = f"http://localhost:{port}/v1"
            client = openai.OpenAI(
                base_url=base_url,
                api_key="unused",
            )
            models = client.models.list()
            model_id = models.data[0].id if models.data else None
            if model_id is None:
                return
            self._oracle_llm_config = {
                "model_name": model_id,
                "api_base": base_url,
                "api_key": "unused",
                "temperature": 0.0,
                "max_tokens": 64,
            }
            from ..proposer import ActionProposer
            self._oracle_proposer = ActionProposer(
                mode="llm_api",
                llm_config=self._oracle_llm_config,
            )
            logger.info(f"HotpotQA oracle LLM initialised: {model_id} (port {port})")
        except Exception as e:
            logger.debug(f"HotpotQA oracle LLM not available: {e}")

    def set_oracle_proposer(self, proposer):
        """Inject an external Proposer for oracle use."""
        self._oracle_proposer = proposer

    # ── greedy oracle action ──────────────────────────────────────

    def greedy_oracle_action(self) -> int:
        """Oracle policy with full knowledge of the correct answer.

        Uses heuristic-first strategy for speed (avoids expensive LLM calls).
        The heuristic has full access to self._answer and self._supporting_facts
        so it is already near-optimal for oracle collection.

        Strategy:
          1. If we have enough context (≥2 paragraphs) or step ≥ 3,
             pick finish[correct_answer] if available.
          2. Otherwise, search for supporting-fact entities we haven't
             retrieved yet.
          3. Fallback: use forward_value ranking.
        """
        actions = self.get_actions()
        if not actions:
            return 0

        # ── Heuristic oracle (fast, no LLM calls) ─────────────────

        # Phase 1: try to finish with the correct answer
        searched_titles = {c["title"].lower() for c in self._context}
        have_enough = len(self._context) >= 2 or self._step_count >= 3
        if have_enough:
            answer_lower = self._answer.lower()
            for a in actions:
                txt = self._action_texts[a] if a < len(self._action_texts) else ""
                m = re.match(r'finish\[(.+?)\]', txt)
                if m and m.group(1).lower() == answer_lower:
                    return a

        # Phase 2: search for unseen supporting-fact entities
        sf_titles = [sf[0] for sf in self._supporting_facts]
        for sf_title in sf_titles:
            if sf_title.lower() not in searched_titles:
                for a in actions:
                    txt = self._action_texts[a] if a < len(self._action_texts) else ""
                    if txt.startswith("search[") and sf_title.lower() in txt.lower():
                        return a

        # Phase 3: search for any unseen context entity
        for a in actions:
            txt = self._action_texts[a] if a < len(self._action_texts) else ""
            m = re.match(r'search\[(.+?)\]', txt)
            if m and m.group(1).lower() not in searched_titles:
                return a

        # Phase 4: fallback to forward value
        values = [(a, self.forward_value(a)) for a in actions]
        values.sort(key=lambda x: -x[1])
        return values[0][0]

"""
FEVER (Fact Extraction and VERification) environment adapter.

FEVER is a claim verification environment where the agent must retrieve
evidence from Wikipedia paragraphs and classify a claim as SUPPORTS,
REFUTES, or NOT ENOUGH INFO.

Reference: https://fever.ai/

Actions:
  - search[entity]              : retrieve Wikipedia paragraph
  - lookup[term]                : find next sentence containing term
  - classify[SUPPORTS]          : submit SUPPORTS verdict
  - classify[REFUTES]           : submit REFUTES verdict
  - classify[NOT ENOUGH INFO]   : submit NOT ENOUGH INFO verdict

Forward evaluator:
  * Uses superficial keyword overlap between the claim and retrieved
    evidence.  *Intentionally shallow* — does not perform actual
    entailment reasoning, so may misjudge evidence sufficiency.
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

# Valid classification labels
FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# Fixed action templates — the agent fills in the argument
FEVER_ACTION_TYPES = ["search", "lookup", "classify"]


class FEVEREnv(BaseEnv):
    """
    Adapter for FEVER claim-verification environment.

    Follows the ReAct-style interaction protocol:
      Thought -> Action -> Observation -> Thought -> ...

    The action space is *open-ended* (search[entity], lookup[term],
    classify[label]).  We discretise it by providing a list of
    candidate actions at each step, generated from the current
    context.
    """

    ENV_TYPE = "fever"

    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name, **kwargs)

        self.data_path = kwargs.get("data_path") or "data/fever"
        self.split = kwargs.get("split", "labelled_dev")
        self.max_steps = kwargs.get("max_steps", 8)

        # State
        self._claim: str = ""
        self._label: str = ""  # ground-truth label
        self._evidence_sets: List[List] = []  # [[title, sent_id], ...]
        self._all_context: List[Tuple[str, List[str]]] = []  # [(title, [sentences]), ...]
        self._context: List[Dict[str, str]] = []  # retrieved paragraphs
        self._obs_text: str = ""
        self._action_texts: List[str] = []
        self._step_count: int = 0
        self._done: bool = False
        self._reward: float = 0.0
        self._submitted_label: str = ""
        self._history: List[Dict[str, str]] = []
        self._claims: List[Dict] = []

        # LLM-based oracle (lazy-initialised)
        self._oracle_proposer = None
        self._oracle_llm_config: Optional[Dict] = None

        self._load_data()

    def _load_data(self):
        """Load FEVER dataset.

        Tries the following sources in order:
          1. Local JSON / JSONL file at ``data_path/fever_{split}.json``
          2. HuggingFace ``fever/fever`` dataset (split ``labelled_dev``
             or ``paper_dev``)
          3. Stub data (2 example claims) as a fallback

        The FEVER dataset stores evidence at the row level: each row has
        one claim together with one evidence entry.  Multiple rows may
        share the same claim ``id``.  We group rows by ``id`` and
        aggregate evidence into a list of ``[wiki_url, sentence_id]``
        pairs.

        For the context pool (``_all_context``) we build paragraphs from
        the ``evidence_wiki_url`` -> ``evidence_sentence`` mapping and
        add distractor paragraphs from neighbouring claims.
        """
        try:
            # ── 1. Try local file ────────────────────────────────
            data_file = os.path.join(self.data_path, f"fever_{self.split}.json")
            if os.path.exists(data_file):
                self._claims = self._load_from_file(data_file)
                logger.info(
                    f"Loaded {len(self._claims)} FEVER claims from {data_file}"
                )
                return

            # ── 2. Try HuggingFace ───────────────────────────────
            try:
                from datasets import load_dataset

                # Map user-facing split names to HF dataset split names
                split_aliases = {
                    "labelled_dev": "validation",
                    "paper_dev": "validation",
                    "dev": "validation",
                }
                hf_split = split_aliases.get(self.split, self.split)

                ds = None
                for ds_name in ["fever/fever", "fever"]:
                    try:
                        ds = load_dataset(ds_name, split=hf_split)
                        break
                    except Exception:
                        continue
                if ds is None:
                    raise RuntimeError(
                        f"Could not load FEVER from HuggingFace (split={hf_split})"
                    )

                self._claims = self._parse_hf_dataset(ds)
                # Filter out claims with empty labels
                self._claims = [
                    c for c in self._claims
                    if c["label"] in FEVER_LABELS
                ]
                logger.info(
                    f"Loaded {len(self._claims)} FEVER claims from HuggingFace "
                    f"(split={hf_split})"
                )
                return
            except Exception as hf_err:
                logger.warning(f"HuggingFace load failed: {hf_err}")

            # ── 4. Fallback to stub ──────────────────────────────
            logger.warning(
                f"FEVER data not found at {data_file} and HuggingFace load "
                f"failed. Using stub data."
            )
            self._claims = self._stub_data()

        except Exception as e:
            logger.warning(f"Failed to load FEVER: {e}. Using stub data.")
            self._claims = self._stub_data()

    @staticmethod
    def _load_from_file(path: str) -> List[Dict]:
        """Load claims from a local JSON or JSONL file."""
        with open(path) as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                raw = json.load(f)
            else:
                raw = [json.loads(line) for line in f if line.strip()]

        # If the file is already in our grouped format, return as-is
        if raw and "claim" in raw[0] and "label" in raw[0] and "evidence" in raw[0]:
            return raw

        # Otherwise, group by claim id
        return FEVEREnv._group_rows(raw)

    @staticmethod
    def _parse_hf_dataset(ds) -> List[Dict]:
        """Parse a HuggingFace FEVER dataset split into grouped claims.

        The HF ``fever/fever`` dataset has columns:
          id, label, claim, evidence_annotation_id, evidence_id,
          evidence_wiki_url, evidence_sentence_id

        Note: ``evidence_sentence`` (the actual text) is NOT present in
        the standard HF dataset.  When it is absent we synthesise
        evidence text from the claim and the wiki URL title so that the
        environment can still function as a search-based agent task.
        """
        rows = []
        for row in ds:
            wiki_url = row.get("evidence_wiki_url", "") or ""
            sent_text = row.get("evidence_sentence", "") or ""

            # If no evidence sentence text, synthesise from claim + title
            if not sent_text and wiki_url:
                title = wiki_url.replace("_", " ")
                claim = row.get("claim", "")
                label = str(row.get("label", "")).upper().strip()
                # Create a synthetic evidence sentence that is informative
                if label in ("SUPPORTS", "0"):
                    sent_text = f"{title}: {claim}"
                elif label in ("REFUTES", "1"):
                    sent_text = (
                        f"{title} is a topic related to the claim. "
                        f"However, the available evidence contradicts "
                        f"the statement."
                    )
                else:
                    sent_text = f"{title} is mentioned in available sources."

            rows.append({
                "id": row.get("id"),
                "label": row.get("label", ""),
                "claim": row.get("claim", ""),
                "evidence_wiki_url": wiki_url,
                "evidence_sentence_id": row.get("evidence_sentence_id", 0),
                "evidence_sentence": sent_text,
            })
        return FEVEREnv._group_rows(rows)

    @staticmethod
    def _group_rows(rows: List[Dict]) -> List[Dict]:
        """Group flat evidence rows by claim id into structured dicts.

        Returns list of::

            {
                "id": ...,
                "claim": "...",
                "label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
                "evidence": [[wiki_url, sentence_id], ...],
                "context": [[wiki_url, [sentence_text, ...]], ...],
            }
        """
        from collections import OrderedDict

        grouped: Dict[Any, Dict] = OrderedDict()
        # Also collect a global wiki_url -> sentences mapping for context
        url_sentences: Dict[str, List[str]] = {}

        for row in rows:
            cid = row.get("id", id(row))
            if cid not in grouped:
                # Normalise label
                raw_label = str(row.get("label", "")).upper().strip()
                # HF dataset sometimes uses integer labels:
                # 0 -> SUPPORTS, 1 -> REFUTES, 2 -> NOT ENOUGH INFO
                label_map = {"0": "SUPPORTS", "1": "REFUTES", "2": "NOT ENOUGH INFO"}
                label = label_map.get(raw_label, raw_label)
                if label not in FEVER_LABELS:
                    label = "NOT ENOUGH INFO"  # safe default
                grouped[cid] = {
                    "id": cid,
                    "claim": row.get("claim", ""),
                    "label": label,
                    "evidence": [],
                    "context_urls": {},
                }

            wiki_url = row.get("evidence_wiki_url", "") or ""
            sent_id = row.get("evidence_sentence_id", 0)
            sent_text = row.get("evidence_sentence", "") or ""

            if wiki_url:
                grouped[cid]["evidence"].append([wiki_url, sent_id])
                # Accumulate sentences per URL
                if wiki_url not in grouped[cid]["context_urls"]:
                    grouped[cid]["context_urls"][wiki_url] = []
                if sent_text and sent_text not in grouped[cid]["context_urls"][wiki_url]:
                    grouped[cid]["context_urls"][wiki_url].append(sent_text)

                if wiki_url not in url_sentences:
                    url_sentences[wiki_url] = []
                if sent_text and sent_text not in url_sentences[wiki_url]:
                    url_sentences[wiki_url].append(sent_text)

        # Build final list with context (evidence + distractors)
        all_claims = list(grouped.values())
        all_urls = list(url_sentences.keys())

        result = []
        for i, claim_data in enumerate(all_claims):
            # Build context: evidence paragraphs
            context = []
            for url, sents in claim_data["context_urls"].items():
                if sents:
                    context.append([url.replace("_", " "), sents])

            # Add distractor paragraphs from neighbouring claims
            distractor_indices = []
            for offset in [1, -1, 2, -2]:
                di = i + offset
                if 0 <= di < len(all_claims) and di != i:
                    distractor_indices.append(di)
            for di in distractor_indices[:3]:
                other = all_claims[di]
                for url, sents in other.get("context_urls", {}).items():
                    title = url.replace("_", " ")
                    # Avoid duplicate titles
                    if sents and not any(c[0] == title for c in context):
                        context.append([title, sents])
                        break  # one distractor per neighbour

            result.append({
                "id": claim_data["id"],
                "claim": claim_data["claim"],
                "label": claim_data["label"],
                "evidence": claim_data["evidence"],
                "context": context,
            })

        return result

    @staticmethod
    def _stub_data() -> List[Dict]:
        """Fallback data when no dataset is available."""
        return [
            {
                "id": 1,
                "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
                "label": "SUPPORTS",
                "evidence": [
                    ["Nikolaj Coster-Waldau", 0],
                ],
                "context": [
                    [
                        "Nikolaj Coster-Waldau",
                        [
                            "Nikolaj William Coster-Waldau is a Danish actor and producer.",
                            "He graduated from the Danish National School of Performing Arts in Copenhagen in 1993.",
                            "He appeared in the Fox television series New Amsterdam in 2008.",
                        ],
                    ],
                    [
                        "Fox Broadcasting Company",
                        [
                            "The Fox Broadcasting Company is an American commercial broadcast television network.",
                            "Fox Broadcasting Company is owned by Fox Corporation.",
                        ],
                    ],
                    [
                        "Game of Thrones",
                        [
                            "Game of Thrones is an American fantasy drama television series on HBO.",
                        ],
                    ],
                ],
            },
            {
                "id": 2,
                "claim": "Stranger Things is set in Bloomington, Indiana.",
                "label": "REFUTES",
                "evidence": [
                    ["Stranger Things", 0],
                ],
                "context": [
                    [
                        "Stranger Things",
                        [
                            "Stranger Things is an American science fiction horror drama television series.",
                            "Set in the 1980s in the fictional town of Hawkins, Indiana, the series centers on supernatural events.",
                        ],
                    ],
                    [
                        "Bloomington, Indiana",
                        [
                            "Bloomington is a city in and the county seat of Monroe County in southern Indiana.",
                        ],
                    ],
                    [
                        "Indiana",
                        [
                            "Indiana is a U.S. state in the Midwestern United States.",
                        ],
                    ],
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
        self._submitted_label = ""

        # Select claim
        idx = (seed or 0) % len(self._claims) if self._claims else 0
        c = self._claims[idx]
        self._claim = c["claim"]
        self._label = c["label"]
        self._evidence_sets = c.get("evidence", [])

        raw_context = c.get("context", [])
        # Normalise context: list of [title, [sentences]]
        if isinstance(raw_context, dict) and "title" in raw_context:
            titles = raw_context["title"]
            sentences = raw_context["sentences"]
            self._all_context = list(zip(titles, sentences))
        else:
            self._all_context = raw_context

        self._obs_text = f"Claim: {self._claim}"
        self._generate_candidate_actions()

        return self._obs_text, {
            "claim": self._claim,
            "admissible": self._action_texts,
        }

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        action_text = (
            self._action_texts[action]
            if action < len(self._action_texts)
            else "classify[NOT ENOUGH INFO]"
        )
        self._step_count += 1

        # Parse action
        match = re.match(r'(search|lookup|classify)\[(.+?)\]', action_text)
        if not match:
            obs = (
                "Invalid action format. Use search[entity], lookup[term], "
                "or classify[SUPPORTS/REFUTES/NOT ENOUGH INFO]."
            )
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
            elif action_type == "classify":
                self._submitted_label = argument.upper().strip()
                reward = self._compute_classification_reward(self._submitted_label)
                obs = f"Classification submitted: {self._submitted_label}. Reward: {reward:.2f}"
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

    def _compute_classification_reward(self, submitted_label: str) -> float:
        """Compute reward based on classification correctness.

        Exact match: 1.0 if the submitted label matches the ground truth,
        0.0 otherwise.
        """
        if submitted_label.upper().strip() == self._label.upper().strip():
            return 1.0
        return 0.0

    def _generate_candidate_actions(self):
        """Generate a set of candidate discrete actions from the current context."""
        candidates = []

        # 1. Search actions based on claim entities
        entities = self._extract_entities(self._claim)
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
            key_terms = self._extract_key_terms(last_text, self._claim)
            for term in key_terms[:3]:
                candidates.append(f"lookup[{term}]")

        # 4. Classify actions — only propose after at least one step with context
        #    NOTE: The oracle correct label is NOT included here. It is only
        #    accessible via greedy_oracle_action(), which is used exclusively
        #    by the 'oracle' rollout method. This prevents other rollout
        #    methods from finding the ground-truth label by evaluating
        #    candidates with forward_value().
        if self._step_count >= 1 and self._context:
            for lbl in FEVER_LABELS:
                candidates.append(f"classify[{lbl}]")

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
        entities = [
            e for e in entities
            if len(e) > 2 and e.lower() not in {"the", "was", "were", "are", "not"}
        ]
        return list(dict.fromkeys(entities))  # deduplicate, preserve order

    def _extract_key_terms(self, text: str, query: str) -> List[str]:
        """Extract terms from text that might help verify the claim."""
        q_words = set(query.lower().split())
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        terms = [w for w in words if w.lower() not in q_words and len(w) > 3]
        return list(dict.fromkeys(terms))[:5]

    # ── deep copy ──────────────────────────────────────────────────

    def deepcopy(self) -> "FEVEREnv":
        """Deep-copy that avoids re-initialising LLM oracle on clones
        and avoids re-copying immutable data (claim bank, all_context)."""
        saved_proposer = self._oracle_proposer
        saved_llm_cfg = self._oracle_llm_config
        saved_claims = self._claims          # immutable across rollout
        saved_all_context = self._all_context  # immutable for this episode
        self._oracle_proposer = None
        self._oracle_llm_config = None
        self._claims = None                  # skip deep-copying full dataset
        self._all_context = None             # skip deep-copying context bank
        try:
            import copy
            clone = copy.deepcopy(self)
        finally:
            self._oracle_proposer = saved_proposer
            self._oracle_llm_config = saved_llm_cfg
            self._claims = saved_claims
            self._all_context = saved_all_context
        # Restore shared references on clone (read-only data, safe to share)
        clone._oracle_proposer = saved_proposer
        clone._oracle_llm_config = "unavailable"
        clone._claims = saved_claims
        clone._all_context = saved_all_context
        return clone

    # ── snapshot / restore ────────────────────────────────────────

    def get_state(self) -> EnvState:
        return EnvState(
            data={
                "claim": self._claim,
                "label": self._label,
                "evidence_sets": self._evidence_sets,
                "all_context": self._all_context,
                "context": list(self._context),
                "obs_text": self._obs_text,
                "step_count": self._step_count,
                "done": self._done,
                "reward": self._reward,
                "submitted_label": self._submitted_label,
                "history": list(self._history),
                "action_texts": list(self._action_texts),
            },
            env_type="fever",
        )

    def set_state(self, state: EnvState):
        d = state.data
        self._claim = d["claim"]
        self._label = d["label"]
        self._evidence_sets = d.get("evidence_sets", [])
        self._all_context = d.get("all_context", [])
        self._context = list(d["context"])
        self._obs_text = d["obs_text"]
        self._step_count = d["step_count"]
        self._done = d["done"]
        self._reward = d["reward"]
        self._submitted_label = d.get("submitted_label", "")
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
        evidence_summary = ""
        if self._context:
            evidence_summary = "\nRetrieved evidence:\n"
            for c in self._context:
                evidence_summary += f"  [{c['title']}] {c['text'][:200]}...\n"
        return (
            f"Claim: {self._claim}\n"
            f"Observation: {self._obs_text}\n"
            f"Step: {self._step_count}"
            f"{evidence_summary}\n"
            f"Available actions: {', '.join(self._action_texts)}"
        )

    # ── forward value (heuristic) ─────────────────────────────────

    def forward_value(self, action: int) -> float:
        """
        Shallow heuristic: keyword overlap + action-type bonus.
        *Intentionally weak* at entailment reasoning.
        """
        if action >= len(self._action_texts):
            return -1.0
        action_text = self._action_texts[action]
        claim_lower = self._claim.lower()

        score = 0.0

        match = re.match(r'(search|lookup|classify)\[(.+?)\]', action_text)
        if not match:
            return -0.5

        action_type, argument = match.groups()
        arg_lower = argument.lower()

        if action_type == "search":
            # Reward searching for entities mentioned in the claim
            claim_words = set(claim_lower.split())
            arg_words = set(arg_lower.split())
            overlap = claim_words & arg_words - {
                "the", "a", "is", "was", "were", "and", "or", "in", "of",
            }
            score += len(overlap) * 1.0
            # Bonus for entities not yet searched
            searched_titles = {c["title"].lower() for c in self._context}
            if arg_lower not in searched_titles:
                score += 0.5
            else:
                score -= 1.0  # penalty for re-searching

        elif action_type == "lookup":
            score += 0.3  # mild bonus for information gathering

        elif action_type == "classify":
            # Mild bonus for classifying after gathering evidence
            if self._context:
                score += 1.5
                # Additional small bonus based on evidence count
                score += min(len(self._context), 3) * 0.2
            else:
                # Discourage premature classification without evidence
                score += 0.2

        return score

    # ── success ───────────────────────────────────────────────────

    def is_success(self, reward: float, terminated: bool, info: Dict) -> bool:
        return terminated and reward >= 1.0

    # ── state classification ──────────────────────────────────────

    def classify_state(self) -> str:
        n_retrieved = len(self._context)
        if n_retrieved == 0:
            return "no_evidence"
        elif n_retrieved == 1:
            return "partial_evidence"
        else:
            return "sufficient_evidence"

    # ── state key ─────────────────────────────────────────────────

    def make_state_key(self) -> str:
        h = hashlib.md5(
            f"{self._claim}:{self._step_count}:{len(self._context)}".encode()
        ).hexdigest()[:12]
        return f"fever_{h}"

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
            logger.info(
                f"FEVER oracle LLM initialised: {model_id} (port {port})"
            )
        except Exception as e:
            logger.debug(f"FEVER oracle LLM not available: {e}")

    def set_oracle_proposer(self, proposer):
        """Inject an external Proposer for oracle use."""
        self._oracle_proposer = proposer

    # ── greedy oracle action ──────────────────────────────────────

    def greedy_oracle_action(self) -> int:
        """Oracle policy with full knowledge of the correct label.

        Uses heuristic-first strategy for speed (avoids expensive LLM calls).
        The heuristic has full access to self._label and self._evidence_sets
        so it is already near-optimal for oracle collection.

        The oracle correct-label classify action is NOT in the regular
        candidate list (to prevent oracle leakage to non-oracle rollout
        methods). When this method decides to classify, it injects the
        correct classify action into _action_texts and returns its index.

        Strategy (3-phase):
          1. If we have enough evidence (>= 1 paragraph) or step >= 2,
             inject classify[correct_label] and return it.
          2. Otherwise, search for evidence entities we haven't
             retrieved yet.
          3. Fallback: use forward_value ranking.
        """
        actions = self.get_actions()
        if not actions:
            return 0

        # ── Phase 1: classify with correct label if we have evidence ──
        searched_titles = {c["title"].lower() for c in self._context}
        have_enough = len(self._context) >= 1 or self._step_count >= 2
        if have_enough:
            oracle_action = f"classify[{self._label}]"
            # Check if it already exists in candidate list
            for a in actions:
                txt = self._action_texts[a] if a < len(self._action_texts) else ""
                if txt == oracle_action:
                    return a
            # Not in candidate list — inject it
            self._action_texts.append(oracle_action)
            return len(self._action_texts) - 1

        # ── Phase 2: search for unseen evidence entities ──────────────
        evidence_titles = [ev[0].replace("_", " ") for ev in self._evidence_sets]
        for ev_title in evidence_titles:
            if ev_title.lower() not in searched_titles:
                for a in actions:
                    txt = (
                        self._action_texts[a]
                        if a < len(self._action_texts)
                        else ""
                    )
                    if txt.startswith("search[") and ev_title.lower() in txt.lower():
                        return a

        # ── Phase 2b: search for any unseen context entity ────────────
        for a in actions:
            txt = self._action_texts[a] if a < len(self._action_texts) else ""
            m = re.match(r'search\[(.+?)\]', txt)
            if m and m.group(1).lower() not in searched_titles:
                return a

        # ── Phase 3: fallback to forward value ────────────────────────
        values = [(a, self.forward_value(a)) for a in actions]
        values.sort(key=lambda x: -x[1])
        return values[0][0]

    # ── signals for gate learning ─────────────────────────────────

    def get_signals(self) -> Dict[str, Any]:
        """Return FEVER-specific signals for gate learning."""
        return {
            "evidence_count": len(self._context),
            "is_classify_proposed": any(
                a.startswith("classify[") for a in self._action_texts
            ),
            "claim_length": len(self._claim),
        }

    # ── trajectory ────────────────────────────────────────────────

    def get_trajectory(self) -> Dict[str, Any]:
        """
        Return the full trajectory of the current (or just-finished) episode.

        Designed for post-hoc analysis of claim verification paths:
          - claim metadata (claim text, ground-truth label, evidence sets)
          - per-step records: action chosen, observation returned
          - outcome: success, final reward, total steps, submitted label
        """
        return {
            "claim": self._claim,
            "ground_truth_label": self._label,
            "evidence_sets": self._evidence_sets,
            "num_evidence_titles": len(
                {ev[0] for ev in self._evidence_sets}
            ) if self._evidence_sets else 0,
            "claim_length": len(self._claim),
            # Episode outcome
            "steps": self._step_count,
            "done": self._done,
            "final_reward": self._reward,
            "submitted_label": self._submitted_label,
            "success": self._done and self._reward >= 1.0,
            "evidence_retrieved": len(self._context),
            "state_category": self.classify_state(),
            # Per-step trajectory (the core data)
            "history": list(self._history),
        }

    # ── convenience ──────────────────────────────────────────────

    def get_num_episodes(self) -> int:
        """Total number of available claims."""
        return len(self._claims)

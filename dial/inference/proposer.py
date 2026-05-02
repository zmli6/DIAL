"""
Action proposer module for generating candidate actions.

Supported modes:
  - "all": return all discrete actions (no model needed)
  - "filtered": heuristic filtering of invalid actions
  - "llm_local": local transformers LLM/VLM (e.g. Qwen2-VL, LLaMA, etc.)
  - "llm_api": remote API via OpenAI-compatible interface
      api_type:
        - "openrouter": OpenRouter (https://openrouter.ai)
        - "vllm": self-hosted vLLM OpenAI-compatible server
        - "openai": native OpenAI
        - "anthropic": Anthropic Claude
"""
from typing import List, Optional, Dict, Any
import numpy as np
import logging
import json
import re
import base64
import time
import threading
from io import BytesIO
from PIL import Image


logger = logging.getLogger("DIAL")


# ── Global Token Counter ──────────────────────────────────────────────
class TokenCounter:
    """Thread-safe global token usage counter."""
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock if hasattr(self, '_lock') else threading.Lock():
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_calls = 0

    def record(self, usage):
        """Record usage from an OpenAI-compatible API response."""
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += getattr(usage, 'prompt_tokens', 0) or 0
            self.completion_tokens += getattr(usage, 'completion_tokens', 0) or 0
            self.total_calls += 1

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def snapshot(self):
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "total_calls": self.total_calls,
            }

# Singleton instance — import and use from anywhere
token_counter = TokenCounter()


class ActionProposer:
    """Generates candidate actions for a given state."""

    # MiniGrid action mapping (standard)
    ACTION_NAMES = {
        0: "left", 1: "right", 2: "forward",
        3: "pickup", 4: "drop", 5: "toggle", 6: "done",
    }
    ACTION_DESCRIPTIONS = {
        0: "Turn left",
        1: "Turn right",
        2: "Move forward",
        3: "Pick up an object",
        4: "Drop the object being carried",
        5: "Toggle/activate an object (door, switch, etc.)",
        6: "Complete the task (done signal)",
    }

    def __init__(
        self,
        mode: str = "all",
        k: Optional[int] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            mode: "all" | "filtered" | "llm_local" | "llm_api"
            k: return at most *k* actions (None -> all)
            llm_config: backend-specific keys, see config.yaml for full schema
        """
        self.mode = mode
        self.k = k
        self.llm_config = llm_config or {}

        # Lazy-init model / client
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._client = None

        if mode == "llm_local":
            self._init_local_model()
        elif mode == "llm_api":
            self._init_api_client()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_local_model(self):
        """Load a HuggingFace transformers model locally.

        If model_path is a HuggingFace model ID (not a local directory),
        transformers will automatically download it on first use.
        Set hf_cache_dir in llm_config (or HF_HOME env var) to control
        where downloaded models are stored.
        """
        import os

        model_path = self.llm_config.get("model_path")
        if not model_path:
            raise ValueError("llm_local mode requires 'model_path' in llm_config")

        # Set HuggingFace cache directory if specified
        hf_cache_dir = self.llm_config.get("hf_cache_dir")
        if hf_cache_dir:
            os.environ["HF_HOME"] = hf_cache_dir
            logger.info(f"HF_HOME set to {hf_cache_dir}")

        model_type = self.llm_config.get("model_type", "llm")  # "llm" or "vlm"
        device = self.llm_config.get("device", "cuda")
        dtype = self.llm_config.get("dtype", "auto")  # "auto", "float16", "bfloat16"

        if os.path.isdir(model_path):
            logger.info(f"Loading local {model_type.upper()} from disk: {model_path}")
        else:
            logger.info(
                f"Loading {model_type.upper()} '{model_path}' "
                f"(will auto-download if not cached, "
                f"cache={os.environ.get('HF_HOME', '~/.cache/huggingface')})"
            )
        logger.info(f"  device={device}, dtype={dtype}")

        try:
            import torch
            torch_dtype_map = {
                "auto": "auto",
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = torch_dtype_map.get(dtype, "auto")

            if model_type == "vlm":
                self._init_local_vlm(model_path, device, torch_dtype)
            else:
                self._init_local_llm(model_path, device, torch_dtype)

            logger.info("Local model loaded successfully")

        except ImportError:
            logger.error(
                "transformers / torch not installed. "
                "Install with: pip install transformers torch accelerate"
            )
            raise

    def _init_local_llm(self, model_path: str, device: str, torch_dtype):
        """Text-only LLM via transformers."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()

    def _init_local_vlm(self, model_path: str, device: str, torch_dtype):
        """
        VLM via transformers.
        Supports common VLM architectures:
          - Qwen2-VL / Qwen2.5-VL  (AutoModelForCausalLM + processor)
          - LLaVA-style             (AutoModelForVision2Seq + processor)
          - InternVL               (AutoModelForCausalLM + processor)
        Falls back to AutoModelForCausalLM + AutoProcessor (covers most cases).
        """
        from transformers import AutoProcessor, AutoModelForCausalLM

        # Load processor
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Try Vision2Seq first (LLaVA), fall back to CausalLM (Qwen-VL, InternVL)
        try:
            from transformers import AutoModelForVision2Seq
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )
        except Exception:
            logger.info("AutoModelForVision2Seq failed, trying AutoModelForCausalLM for VLM...")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )

        self._model.eval()

        # Keep a reference to a tokenizer for decoding
        if hasattr(self._processor, "tokenizer"):
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = self._processor

    def _init_api_client(self):
        """
        Initialize an API client.
        Supported api_type values:
          - "openrouter" -> OpenRouter (OpenAI-compatible, base_url=https://openrouter.ai/api/v1)
          - "vllm"       -> self-hosted vLLM server (OpenAI-compatible, user-provided endpoint)
          - "openai"     -> native OpenAI
          - "anthropic"  -> Anthropic Claude
        """
        import os

        api_type = self.llm_config.get("api_type", "openai")
        api_key = self.llm_config.get("api_key")
        model_name = self.llm_config.get("model_name", "gpt-4o")
        endpoint = self.llm_config.get("endpoint")

        self.api_type = api_type
        self.model_name = model_name

        if api_type in ("openai", "openrouter", "vllm"):
            # All three use the OpenAI SDK with different base_url
            import openai

            if api_type == "openrouter":
                base_url = endpoint or "https://openrouter.ai/api/v1"
                api_key = api_key or os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OpenRouter requires api_key in config or OPENROUTER_API_KEY env var"
                    )
                logger.info(f"Initialized OpenRouter client -> model={model_name}")

            elif api_type == "vllm":
                # Allow DIAL_VLLM_ENDPOINT env var to override config
                # (SLURM scripts set this to the dynamically chosen port)
                base_url = os.getenv("DIAL_VLLM_ENDPOINT") or endpoint
                if not base_url:
                    raise ValueError(
                        "vLLM mode requires 'endpoint' in config or DIAL_VLLM_ENDPOINT env var"
                    )
                api_key = api_key or "EMPTY"  # vLLM doesn't need a real key
                logger.info(f"Initialized vLLM client at {base_url} -> model={model_name}")

            else:  # openai
                base_url = endpoint  # None -> use default
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OpenAI requires api_key in config or OPENAI_API_KEY env var"
                    )
                logger.info(f"Initialized OpenAI client -> model={model_name}")

            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

        elif api_type == "anthropic":
            import anthropic

            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic requires api_key in config or ANTHROPIC_API_KEY env var"
                )
            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic client -> model={model_name}")

        else:
            raise ValueError(f"Unsupported api_type: {api_type}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _num_actions(self, env) -> int:
        """Get action count from either BaseEnv or gym env."""
        if hasattr(env, 'get_actions'):
            return len(env.get_actions())
        return env.action_space.n

    def _all_actions(self, env) -> List[int]:
        """Get all valid action indices."""
        if hasattr(env, 'get_actions'):
            return env.get_actions()
        return list(range(env.action_space.n))

    def _is_text_env(self, env) -> bool:
        """Check if this is a text-based environment (ALFWorld/WebShop/HotpotQA/ScienceWorld/AppWorld)."""
        env_type = getattr(env, 'ENV_TYPE', '')
        return env_type in ('alfworld', 'webshop', 'hotpotqa', 'scienceworld', 'appworld')

    def propose_actions(self, env, obs) -> List[int]:
        """Return a list of candidate action indices."""
        if self.mode == "all":
            actions = self._all_actions(env)
        elif self.mode == "filtered":
            actions = self._filter_invalid_actions(env, obs)
        elif self.mode == "llm_local":
            actions = self._propose_local(env, obs)
        elif self.mode == "llm_api":
            actions = self._propose_api(env, obs)
        else:
            logger.warning(f"Unknown proposer mode '{self.mode}', falling back to 'all'")
            actions = self._all_actions(env)

        if self.k is not None and self.k < len(actions):
            actions = actions[: self.k]
        return actions

    def choose_action(self, env, obs) -> int:
        """
        Use the LLM to directly choose the best action for text-based envs.
        Returns a single action index.
        """
        if self.mode not in ("llm_api", "llm_local"):
            # Fallback: return first action
            actions = self._all_actions(env)
            return actions[0] if actions else 0

        prompt = self._build_text_env_prompt(env, obs, choose_one=True)
        max_retries = self.llm_config.get("max_retries", 2)
        num_actions = self._num_actions(env)
        # If actions were truncated, parse against truncated count
        parse_count = len(self._action_index_map) if getattr(self, '_action_index_map', None) else num_actions

        for attempt in range(max_retries + 1):
            try:
                if self.mode == "llm_api":
                    response = self._call_openai_compatible(env, prompt)
                else:
                    response = self._local_llm_generate(
                        prompt, self.llm_config.get("max_new_tokens", 200),
                        self.llm_config.get("temperature", 0.1)
                    )
                actions = self._parse_response(response, parse_count)
                if actions:
                    chosen = actions[0]
                    # Remap truncated index → original env index
                    if getattr(self, '_action_index_map', None) and chosen in self._action_index_map:
                        chosen = self._action_index_map[chosen]
                    return chosen
            except Exception as e:
                logger.warning(f"choose_action attempt {attempt+1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))

        # Fallback: random action
        actions = self._all_actions(env)
        return actions[0] if actions else 0

    def choose_action_with_logprobs(self, env, obs) -> Dict[str, Any]:
        """
        Like choose_action, but also returns per-token logprobs.

        Works with vLLM / OpenAI-compatible APIs that support
        ``logprobs=True, top_logprobs=K``.

        Returns dict:
            action: int             — chosen action index
            token_logprobs: list    — [{token, logprob, top_logprobs}, ...]
            text: str               — raw response text
        """
        fallback = {
            "action": (self._all_actions(env) or [0])[0],
            "token_logprobs": [],
            "text": "",
        }
        if self.mode != "llm_api" or self.api_type not in ("vllm", "openai"):
            logger.debug("choose_action_with_logprobs: mode/api not supported, no logprobs")
            action = self.choose_action(env, obs)
            return {**fallback, "action": action}

        prompt = self._build_text_env_prompt(env, obs, choose_one=True)
        max_retries = self.llm_config.get("max_retries", 2)
        num_actions = self._num_actions(env)
        # If actions were truncated, parse against truncated count
        parse_count = len(self._action_index_map) if getattr(self, '_action_index_map', None) else num_actions
        temperature = self.llm_config.get("temperature", 0.1)
        max_tokens = self.llm_config.get("max_tokens", 200)
        top_k_logprobs = self.llm_config.get("top_logprobs", 5)

        messages = [{"role": "user", "content": prompt}]

        # Disable Qwen3 thinking mode if configured
        extra_kwargs = {}
        if self.llm_config.get("disable_thinking", False):
            extra_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        for attempt in range(max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=top_k_logprobs,
                    **extra_kwargs,
                )
                token_counter.record(getattr(resp, 'usage', None))
                choice = resp.choices[0]
                text = choice.message.content or ""
                actions = self._parse_response(text, parse_count)
                action = actions[0] if actions else fallback["action"]
                # Remap truncated index → original env index
                if getattr(self, '_action_index_map', None) and action in self._action_index_map:
                    action = self._action_index_map[action]

                # Extract per-token logprobs
                token_logprobs = []
                lp_data = getattr(choice, "logprobs", None)
                if lp_data and hasattr(lp_data, "content") and lp_data.content:
                    for tok_info in lp_data.content:
                        entry = {
                            "token": tok_info.token,
                            "logprob": tok_info.logprob,
                        }
                        if hasattr(tok_info, "top_logprobs") and tok_info.top_logprobs:
                            entry["top_logprobs"] = [
                                {"token": t.token, "logprob": t.logprob}
                                for t in tok_info.top_logprobs
                            ]
                        token_logprobs.append(entry)

                return {
                    "action": action,
                    "token_logprobs": token_logprobs,
                    "text": text,
                }
            except Exception as e:
                logger.warning(f"choose_action_with_logprobs attempt {attempt+1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))

        action = self.choose_action(env, obs)
        return {**fallback, "action": action}

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_invalid_actions(self, env, obs) -> List[int]:
        """Heuristic filtering of clearly invalid actions."""
        all_actions = self._all_actions(env)
        carrying = getattr(env, "carrying", None)
        filtered = []
        for a in all_actions:
            if a == 4 and carrying is None:
                continue  # drop without object
            if a == 3 and carrying is not None:
                continue  # pickup while carrying
            filtered.append(a)
        return filtered or all_actions

    # ------------------------------------------------------------------
    # State description helpers
    # ------------------------------------------------------------------

    def _get_state_description(self, env, obs) -> str:
        mission = getattr(env, "mission", "Navigate the environment")
        agent_pos = getattr(env, "agent_pos", None)
        agent_dir = getattr(env, "agent_dir", None)
        carrying = getattr(env, "carrying", None)

        lines = [f"Mission: {mission}"]
        if agent_pos is not None:
            lines.append(f"Agent position: {tuple(agent_pos)}")
        if agent_dir is not None:
            dirs = ["right", "down", "left", "up"]
            lines.append(f"Agent facing: {dirs[agent_dir]}")
        if carrying:
            ctype = carrying.type if hasattr(carrying, "type") else "object"
            lines.append(f"Carrying: {ctype}")
        else:
            lines.append("Carrying: nothing")
        return "\n".join(lines)

    def _get_state_image(self, env) -> Image.Image:
        try:
            rgb = env.render()
            if isinstance(rgb, np.ndarray):
                return Image.fromarray(rgb)
        except Exception as e:
            logger.debug(f"render() failed: {e}")
        return Image.new("RGB", (256, 256), color="gray")

    def _image_to_base64(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _build_prompt(self, env, obs) -> str:
        """Build prompt — dispatches to text-env or grid-env version."""
        if self._is_text_env(env):
            return self._build_text_env_prompt(env, obs, choose_one=False)
        # MiniGrid / grid-based
        state = self._get_state_description(env, obs)
        n = self._num_actions(env)
        actions = "\n".join(
            f"  {i}: {self.ACTION_NAMES.get(i, f'action_{i}')} - {self.ACTION_DESCRIPTIONS.get(i, '')}"
            for i in range(n)
        )
        return (
            "You are helping an agent navigate a grid-based environment.\n\n"
            f"{state}\n\n"
            f"Available actions:\n{actions}\n\n"
            "Which actions are most promising? Return a ranked list.\n"
            'Respond ONLY with JSON: {"actions": [id1, id2, ...]}\n'
        )

    # Maximum actions to include in prompt to avoid token overflow.
    # ScienceWorld can have 200-400 valid action-object combos which
    # would blow past vLLM's max_model_len (e.g. 4096).
    MAX_PROMPT_ACTIONS = 50

    def _build_text_env_prompt(self, env, obs, choose_one: bool = False) -> str:
        """Build a prompt for text-based environments (ALFWorld/WebShop/HotpotQA/ScienceWorld/AppWorld)."""
        state_desc = env.get_text_description()
        all_actions = env.get_actions()
        action_texts = [env.get_action_name(a) for a in all_actions]

        truncation_note = ""
        if len(action_texts) > self.MAX_PROMPT_ACTIONS:
            # Keep the most relevant actions using forward_value heuristic
            if hasattr(env, 'forward_value'):
                scored = [(a, env.forward_value(a)) for a in all_actions]
                scored.sort(key=lambda x: -x[1])
                kept_actions = [a for a, _ in scored[:self.MAX_PROMPT_ACTIONS]]
                kept_actions.sort()  # restore index order for readability
                action_texts = [env.get_action_name(a) for a in kept_actions]
                # Re-index from 0 so the model sees contiguous indices
                index_map = {new_i: orig_a for new_i, orig_a in enumerate(kept_actions)}
                # Store the mapping for response parsing
                self._action_index_map = index_map
            else:
                action_texts = action_texts[:self.MAX_PROMPT_ACTIONS]
                self._action_index_map = None
            truncation_note = (
                f"\n(Showing top {self.MAX_PROMPT_ACTIONS} of "
                f"{len(all_actions)} available actions, ranked by relevance.)\n"
            )
        else:
            self._action_index_map = None

        actions_str = "\n".join(f"  {i}: {t}" for i, t in enumerate(action_texts))

        if choose_one:
            return (
                "You are an expert agent solving a task step by step.\n\n"
                f"{state_desc}\n\n"
                f"Available actions (choose by index):\n{actions_str}\n"
                f"{truncation_note}\n"
                "Think briefly about which action best advances the task, then respond "
                'with ONLY a JSON object: {"actions": [best_action_index]}\n'
            )
        else:
            return (
                "You are an expert agent solving a task step by step.\n\n"
                f"{state_desc}\n\n"
                f"Available actions (choose by index):\n{actions_str}\n"
                f"{truncation_note}\n"
                "Rank the most promising actions for progressing the task.\n"
                'Respond ONLY with JSON: {"actions": [id1, id2, ...]}\n'
            )

    # ------------------------------------------------------------------
    # Local model inference
    # ------------------------------------------------------------------

    def _propose_local(self, env, obs) -> List[int]:
        """Run inference on a local transformers model."""
        try:
            prompt = self._build_prompt(env, obs)
            model_type = self.llm_config.get("model_type", "llm")
            max_new_tokens = self.llm_config.get("max_new_tokens", 100)
            temperature = self.llm_config.get("temperature", 0.1)
            num_actions = self._num_actions(env)
            parse_count = len(self._action_index_map) if getattr(self, '_action_index_map', None) else num_actions

            if model_type == "vlm" and self._processor is not None:
                response = self._local_vlm_generate(prompt, env, max_new_tokens, temperature)
            else:
                response = self._local_llm_generate(prompt, max_new_tokens, temperature)

            actions = self._parse_response(response, parse_count)
            # Remap truncated indices → original env indices
            if getattr(self, '_action_index_map', None):
                actions = [self._action_index_map.get(a, a) for a in actions]
            logger.debug(f"Local model proposed: {actions}")
            return actions

        except Exception as e:
            logger.error(f"Local model inference error: {e}", exc_info=True)
            return self._all_actions(env)

    def _local_llm_generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate text with a local text-only LLM."""
        import torch

        # Build chat messages if the tokenizer supports it
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=(temperature > 0),
            )

        # Only decode the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _local_vlm_generate(
        self, prompt: str, env, max_new_tokens: int, temperature: float
    ) -> str:
        """Generate text with a local VLM."""
        import torch

        image = self._get_state_image(env)

        # Build messages in the format expected by most VLM processors
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Try chat template path first (Qwen2-VL, InternVL, etc.)
        try:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[image], return_tensors="pt", padding=True
            )
        except Exception:
            # Fallback: simple processor call
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=(temperature > 0),
            )

        # Decode only new tokens
        input_len = inputs.get("input_ids", outputs[:, :0]).shape[-1]
        new_tokens = outputs[0][input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # API-based inference
    # ------------------------------------------------------------------

    def _propose_api(self, env, obs) -> List[int]:
        """Call a remote API to get proposed actions."""
        prompt = self._build_prompt(env, obs)
        max_retries = self.llm_config.get("max_retries", 2)
        num_actions = self._num_actions(env)
        parse_count = len(self._action_index_map) if getattr(self, '_action_index_map', None) else num_actions

        for attempt in range(max_retries + 1):
            try:
                if self.api_type in ("openai", "openrouter", "vllm"):
                    response = self._call_openai_compatible(env, prompt)
                elif self.api_type == "anthropic":
                    response = self._call_anthropic(env, prompt)
                else:
                    raise ValueError(f"Unknown api_type: {self.api_type}")

                actions = self._parse_response(response, parse_count)
                # Remap truncated indices → original env indices
                if getattr(self, '_action_index_map', None):
                    actions = [self._action_index_map.get(a, a) for a in actions]
                logger.debug(f"API ({self.api_type}) proposed: {actions}")
                return actions

            except Exception as e:
                logger.warning(
                    f"API call attempt {attempt + 1}/{max_retries + 1} failed: {e}"
                )
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))

        logger.error("All API retries exhausted, falling back to all actions")
        return self._all_actions(env)

    def _call_openai_compatible(self, env, prompt: str) -> str:
        """
        Works for OpenAI, OpenRouter, and vLLM (all OpenAI-compatible).
        Automatically includes image if use_vision is True.
        """
        use_vision = self.llm_config.get("use_vision", False)
        temperature = self.llm_config.get("temperature", 0.1)
        max_tokens = self.llm_config.get("max_tokens", 200)

        # Build extra headers for OpenRouter
        extra_kwargs = {}
        if self.api_type == "openrouter":
            extra_kwargs["extra_headers"] = {
                "HTTP-Referer": self.llm_config.get("site_url", "https://dial.example.com"),
                "X-Title": self.llm_config.get("app_name", "DIAL"),
            }

        # Disable Qwen3 thinking mode if configured (saves ~90% generation tokens)
        if self.llm_config.get("disable_thinking", False):
            extra_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        if use_vision:
            image = self._get_state_image(env)
            b64 = self._image_to_base64(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )
        token_counter.record(getattr(resp, 'usage', None))
        return resp.choices[0].message.content

    def _call_anthropic(self, env, prompt: str) -> str:
        """Call Anthropic Claude API with optional vision."""
        use_vision = self.llm_config.get("use_vision", False)
        temperature = self.llm_config.get("temperature", 0.1)
        max_tokens = self.llm_config.get("max_tokens", 200)

        if use_vision:
            image = self._get_state_image(env)
            b64 = self._image_to_base64(image)
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str, num_actions: int) -> List[int]:
        """Extract action list from model output."""
        # 1. Try JSON object extraction  {"actions": [...]}
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
                actions = data.get("actions", [])
                valid = [a for a in actions if isinstance(a, int) and 0 <= a < num_actions]
                if valid:
                    return valid
        except json.JSONDecodeError:
            pass

        # 2. Try bare JSON array  [2, 0, 1]
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > start:
                actions = json.loads(text[start:end])
                valid = [a for a in actions if isinstance(a, int) and 0 <= a < num_actions]
                if valid:
                    return valid
        except json.JSONDecodeError:
            pass

        # 3. Regex fallback: extract all integers
        numbers = re.findall(r"\b(\d+)\b", text)
        valid = [int(n) for n in numbers if int(n) < num_actions]
        if valid:
            # deduplicate while preserving order
            seen = set()
            deduped = []
            for a in valid:
                if a not in seen:
                    seen.add(a)
                    deduped.append(a)
            return deduped[:5]

        logger.warning("Could not parse actions from response, falling back to all")
        return list(range(num_actions))

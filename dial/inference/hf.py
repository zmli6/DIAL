"""
HuggingFace Transformers Inference Engine — Hidden State Extraction.

Single-forward-pass inference that returns text + logprobs + hidden_state
for Phase 5 automatic feature discovery.  Replaces vLLM only for data
collection; vLLM remains the production backend for rollout evaluations.

Key class: ``HFInferenceEngine``
  - generate_with_hidden(prompt, max_new_tokens) → dict with text, logprobs, hidden_state, token_entropy
  - choose_action_with_hidden(env, obs) → same as ActionProposer.choose_action_with_logprobs + hidden_state
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("DIAL")


class HFInferenceEngine:
    """
    HuggingFace Transformers inference engine with hidden state extraction.

    Uses ``model.generate(output_hidden_states=True, return_dict_in_generate=True)``
    to extract the last hidden layer, then mean-pools over generated tokens
    to produce a fixed-size ``(d_model,)`` representation (e.g. 2560 for Qwen3-4B).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"Qwen/Qwen3-4B-Instruct-2507"``).
    device : str
        Device string (``"cuda"`` or ``"cpu"``).
    dtype : str
        Data type: ``"auto"``, ``"float16"``, ``"bfloat16"``, ``"float32"``.
    max_memory : dict, optional
        Per-device memory limits for ``device_map="auto"`` multi-GPU loading.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "auto",
        max_memory: Optional[Dict] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, "auto")

        logger.info(f"[HFInference] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"[HFInference] Loading model: {model_name} (dtype={dtype})")
        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if max_memory is not None:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = max_memory
        else:
            load_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs,
        )
        if max_memory is None:
            self.model = self.model.to(device)
        self.model.eval()

        # Determine d_model from config
        config = self.model.config
        self.d_model = getattr(config, "hidden_size", None)
        if self.d_model is None:
            # Fallback for unusual architectures
            self.d_model = getattr(config, "d_model", 2560)
        logger.info(
            f"[HFInference] Model loaded: d_model={self.d_model}, "
            f"device={next(self.model.parameters()).device}"
        )

    @torch.no_grad()
    def generate_with_hidden(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        pooling_strategy: str = "mean",
        layer_selection: str = "last",
    ) -> Dict[str, Any]:
        """
        Generate text from *prompt* and extract hidden states + logprobs.

        Parameters
        ----------
        pooling_strategy : str
            How to aggregate hidden states across generated tokens:
            ``"mean"`` (default), ``"last_token"``, ``"weighted_mean"``.
        layer_selection : str
            Which hidden layer(s) to use:
            ``"last"`` (default), ``"second_to_last"``, ``"avg_last4"``.

        Returns
        -------
        dict with keys:
            text : str
                Generated text (new tokens only).
            logprobs : list[dict]
                Per-token log probabilities with top-k alternatives.
            hidden_state : np.ndarray, shape ``(d_model,)``
                Pooled hidden state over generated tokens.
            token_entropy : float
                Average Shannon entropy across generated tokens.
        """
        # Tokenize
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                text_input = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                text_input = prompt
        else:
            text_input = prompt

        encoding = self.tokenizer(
            text_input, return_tensors="pt", padding=False, truncation=True,
            max_length=2048,
        )
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)
        input_len = input_ids.shape[1]

        # Generate with hidden states
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if temperature > 0 and do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        outputs = self.model.generate(**gen_kwargs)

        # Extract generated token IDs
        generated_ids = outputs.sequences[0, input_len:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        )

        # Extract per-token logprobs from scores
        logprobs_list = []
        entropies = []
        if outputs.scores:
            for step_logits in outputs.scores:
                # step_logits: (1, vocab_size)
                logits = step_logits[0].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)

                # Shannon entropy
                entropy = -(probs * log_probs).sum().item()
                # Filter out -inf contributions
                if math.isnan(entropy) or math.isinf(entropy):
                    entropy = 0.0
                entropies.append(entropy)

                # Top-5 tokens
                top_vals, top_idx = torch.topk(log_probs, k=min(5, log_probs.shape[-1]))
                top_tokens = []
                for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                    token_str = self.tokenizer.decode([idx])
                    top_tokens.append({"token": token_str, "logprob": val})

                logprobs_list.append({
                    "token": self.tokenizer.decode([generated_ids[len(logprobs_list)].item()])
                    if len(logprobs_list) < len(generated_ids) else "",
                    "logprob": top_vals[0].item() if len(top_vals) > 0 else 0.0,
                    "top_logprobs": top_tokens,
                })

        token_entropy = float(np.mean(entropies)) if entropies else 0.0

        # Extract hidden states — pool over generated tokens
        # outputs.hidden_states is a tuple of (num_gen_steps,) where each
        # element is a tuple of (num_layers,) tensors of shape (1, 1, d_model)
        # for greedy/beam search, or (1, seq_len, d_model) for the first step.
        hidden_state = self._extract_hidden_state(
            outputs, input_len,
            pooling_strategy=pooling_strategy,
            layer_selection=layer_selection,
        )

        return {
            "text": generated_text,
            "logprobs": logprobs_list,
            "hidden_state": hidden_state,
            "token_entropy": token_entropy,
        }

    def _extract_hidden_state(
        self, outputs, input_len: int,
        pooling_strategy: str = "mean",
        layer_selection: str = "last",
    ) -> np.ndarray:
        """
        Extract and pool hidden states over generated tokens.

        For ``model.generate(..., output_hidden_states=True)``, the hidden_states
        structure depends on the generation mode:
          - ``outputs.hidden_states`` is a tuple over generation steps
          - Each element is a tuple over layers
          - Each layer tensor has shape ``(batch, seq_len, d_model)``

        Parameters
        ----------
        pooling_strategy : str
            ``"mean"`` — average over all generated tokens (default).
            ``"last_token"`` — use only the last generated token.
            ``"weighted_mean"`` — linearly weighted mean (later tokens weighted more).
        layer_selection : str
            ``"last"`` — last hidden layer (default).
            ``"second_to_last"`` — second-to-last hidden layer.
            ``"avg_last4"`` — average of last 4 hidden layers.
        """
        try:
            hidden_states_per_step = outputs.hidden_states
            if not hidden_states_per_step:
                return np.zeros(self.d_model, dtype=np.float32)

            # Collect hidden states for each generated token
            gen_hiddens = []
            for step_hidden in hidden_states_per_step:
                # step_hidden is a tuple of (num_layers,) tensors
                # Each tensor shape: (batch, seq_len, d_model)
                n_layers = len(step_hidden)

                if layer_selection == "last":
                    layer_h = step_hidden[-1]
                elif layer_selection == "second_to_last":
                    idx = max(-2, -n_layers)
                    layer_h = step_hidden[idx]
                elif layer_selection == "avg_last4":
                    k = min(4, n_layers)
                    layers = [step_hidden[-i] for i in range(1, k + 1)]
                    layer_h = torch.stack(layers, dim=0).mean(dim=0)
                else:
                    layer_h = step_hidden[-1]

                # Take the last token's hidden state from this step
                token_hidden = layer_h[0, -1, :]  # (d_model,)
                gen_hiddens.append(token_hidden)

            if not gen_hiddens:
                return np.zeros(self.d_model, dtype=np.float32)

            stacked = torch.stack(gen_hiddens, dim=0)  # (T, d_model)
            T = stacked.shape[0]

            if pooling_strategy == "mean":
                pooled = stacked.mean(dim=0)
            elif pooling_strategy == "last_token":
                pooled = stacked[-1]
            elif pooling_strategy == "weighted_mean":
                # Linearly increasing weights: 1, 2, ..., T
                weights = torch.arange(1, T + 1, dtype=stacked.dtype,
                                       device=stacked.device).unsqueeze(1)
                pooled = (stacked * weights).sum(dim=0) / weights.sum()
            else:
                pooled = stacked.mean(dim=0)

            return pooled.cpu().float().numpy()

        except Exception as e:
            logger.warning(f"[HFInference] Hidden state extraction failed: {e}")
            return np.zeros(self.d_model, dtype=np.float32)

    def choose_action_with_hidden(
        self,
        env,
        obs,
        max_new_tokens: int = 256,
        pooling_strategy: str = "mean",
        layer_selection: str = "last",
    ) -> Dict[str, Any]:
        """
        Choose an action for a text-based environment, returning the same
        dict as ``ActionProposer.choose_action_with_logprobs`` plus
        ``hidden_state`` and ``token_entropy``.

        Parameters
        ----------
        env : BaseEnv
            Environment instance with ``get_text_description()`` and
            ``get_actions()`` / ``get_action_name()``.
        obs : str
            Current observation text.
        max_new_tokens : int
            Max generation length.
        pooling_strategy : str
            Pooling over generated tokens: ``"mean"``, ``"last_token"``,
            ``"weighted_mean"``.
        layer_selection : str
            Which layer(s): ``"last"``, ``"second_to_last"``, ``"avg_last4"``.

        Returns
        -------
        dict with keys: action, token_logprobs, text, hidden_state, token_entropy
        """
        # Build prompt (same as ActionProposer._build_text_env_prompt)
        state_desc = env.get_text_description()
        actions = env.get_actions()
        action_names = [env.get_action_name(a) for a in actions]

        action_list = "\n".join(
            f"  {i}: {name}" for i, name in enumerate(action_names)
        )

        prompt = (
            "You are an expert agent solving a task step by step.\n\n"
            f"{state_desc}\n\n"
            "Available actions (choose by index):\n"
            f"{action_list}\n\n"
            "Choose the single best action. Respond ONLY with JSON: "
            '{"action": <index>}'
        )

        result = self.generate_with_hidden(
            prompt, max_new_tokens=max_new_tokens,
            pooling_strategy=pooling_strategy,
            layer_selection=layer_selection,
        )

        # Parse action from generated text
        chosen_action = self._parse_action(result["text"], len(actions))

        return {
            "action": chosen_action,
            "token_logprobs": result["logprobs"],
            "text": result["text"],
            "hidden_state": result["hidden_state"],
            "token_entropy": result["token_entropy"],
        }

    def _parse_action(self, text: str, num_actions: int) -> int:
        """Parse action index from generated text."""
        # Try JSON
        try:
            m = re.search(r'\{[^}]*"action"\s*:\s*(\d+)[^}]*\}', text)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < num_actions:
                    return idx
        except (ValueError, AttributeError):
            pass

        # Try bare integer
        try:
            m = re.search(r'\b(\d+)\b', text)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < num_actions:
                    return idx
        except (ValueError, AttributeError):
            pass

        return 0  # fallback

    @torch.no_grad()
    def encode_state(self, prompt: str) -> np.ndarray:
        """
        Encode a state description into a hidden-state vector without
        generation.  Uses the last-layer hidden of the last input token.

        Useful for encoding state texts after the fact (e.g., for
        TextEmbeddingProbe comparison).
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                text_input = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                text_input = prompt
        else:
            text_input = prompt

        encoding = self.tokenizer(
            text_input, return_tensors="pt", padding=False, truncation=True,
            max_length=2048,
        )
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last layer, last token
        last_hidden = outputs.hidden_states[-1][0, -1, :]
        return last_hidden.cpu().float().numpy()

"""Stateless Transformers backend.

The backend performs one full-prompt generation per call. It deliberately
does not keep causal state or persisted sessions; token accounting is the
actual prompt tokens sent to the model plus the generated output tokens.
"""

from __future__ import annotations

import inspect
import os
from typing import Any


class TransformersModel:
    """Local Transformers callable using full prompts for every invocation."""

    def __init__(self, model_path: str, *, max_new_tokens: int | None = None, **_: Any) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs["dtype" if "dtype" in signature.parameters else "torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        self.max_new_tokens = None if max_new_tokens is None else int(max_new_tokens)
        self.last_call_metrics: dict[str, Any] = {}
        self.total_call_metrics: dict[str, int] = {
            "physical_input_tokens": 0,
            "output_tokens": 0,
            "forward_calls": 0,
        }

    def __call__(self, request: Any) -> str:
        prompt = str(getattr(request, "session_prompt", "") or getattr(request, "prompt", ""))
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = encoded.get("attention_mask") if isinstance(encoded, dict) else encoded.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = self.max_new_tokens
        else:
            max_context = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
            if max_context is None:
                max_context = getattr(self.tokenizer, "model_max_length", None)
            if not isinstance(max_context, int) or max_context <= 0 or max_context > 10_000_000:
                raise RuntimeError("cannot determine the model context window for uncapped generation")
            generation_kwargs["max_length"] = max_context
        with self.torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)
        new_ids = output_ids[0, input_ids.shape[-1]:]
        output = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        metrics = {
            "physical_input_tokens": int(input_ids.shape[-1]),
            "output_tokens": int(new_ids.shape[-1]),
            "forward_calls": 1,
        }
        self.last_call_metrics = metrics
        for key in self.total_call_metrics:
            self.total_call_metrics[key] += int(metrics.get(key, 0))
        return output

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "backend": "stateless_transformers",
            "last_call_metrics": dict(self.last_call_metrics),
            "total_call_metrics": dict(self.total_call_metrics),
        }

    def rollback_session(self, session_id: str) -> None:
        return None


DirectTransformersModel = TransformersModel


def model_path_from_env(default: str = "") -> str:
    return os.getenv("MODEL_PATH", default).strip()

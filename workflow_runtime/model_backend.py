"""Local Transformers backend with optional cross-call prefix KV caching.

Logical prompt accounting remains unchanged. Prefix caching only reuses
Transformer KV states for an identical token prefix and reports the physical
prefill work separately.
"""

from __future__ import annotations

import inspect
import os
import time
import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class _PrefixCacheEntry:
    namespace: str
    input_ids: tuple[int, ...]
    past_key_values: Any
    last_access: float


class TransformersModel:
    """Local Transformers callable using full prompts for every invocation."""

    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int | None = None,
        use_chat_template: bool = False,
        prefix_cache: bool = True,
        prefix_cache_max_entries: int = 16,
        **_: Any,
    ) -> None:
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
        self.use_chat_template = bool(use_chat_template)
        self.prefix_cache_enabled = bool(prefix_cache)
        self.prefix_cache_max_entries = max(1, int(prefix_cache_max_entries))
        self._prefix_cache: list[_PrefixCacheEntry] = []
        self.last_call_metrics: dict[str, Any] = {}
        self.total_call_metrics: dict[str, int] = {
            "physical_input_tokens": 0,
            "logical_input_tokens": 0,
            "physical_prefill_tokens": 0,
            "prefix_cache_hit_tokens": 0,
            "prefix_cache_miss_tokens": 0,
            "prefix_cache_warmup_tokens": 0,
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 0,
            "transformer_forward_calls": 0,
            "output_tokens": 0,
            "forward_calls": 0,
        }

    def clear_prefix_cache(self, namespace: str | None = None) -> None:
        """Release cached KV states globally or for one logical session."""
        if namespace is None:
            self._prefix_cache.clear()
        else:
            key = str(namespace)
            self._prefix_cache = [item for item in self._prefix_cache if item.namespace != key]

    def _find_prefix_cache(
        self,
        namespace: str,
        input_ids: tuple[int, ...],
    ) -> _PrefixCacheEntry | None:
        candidates = [
            item
            for item in self._prefix_cache
            if item.namespace == namespace
            and len(item.input_ids) < len(input_ids)
            and input_ids[: len(item.input_ids)] == item.input_ids
        ]
        if not candidates:
            return None
        entry = max(candidates, key=lambda item: len(item.input_ids))
        entry.last_access = time.monotonic()
        return entry

    def _warm_prefix_cache(
        self,
        namespace: str,
        input_ids: Any,
        attention_mask: Any,
    ) -> int:
        """Materialize a prompt prefix before the current generation."""
        prefix_length = int(input_ids.shape[-1]) - 1
        if not self.prefix_cache_enabled or not namespace or prefix_length <= 0:
            return 0
        prefix_ids = input_ids[:, :prefix_length]
        prefix_mask = attention_mask[:, :prefix_length] if attention_mask is not None else None
        with self.torch.no_grad():
            outputs = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
                return_dict=True,
            )
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            return 0
        self._prefix_cache = [
            item for item in self._prefix_cache
            if not (item.namespace == namespace and item.input_ids == tuple(prefix_ids[0].tolist()))
        ]
        self._prefix_cache.append(
            _PrefixCacheEntry(
                namespace=namespace,
                input_ids=tuple(prefix_ids[0].tolist()),
                past_key_values=past,
                last_access=time.monotonic(),
            )
        )
        self._prefix_cache.sort(key=lambda item: item.last_access, reverse=True)
        self._prefix_cache = self._prefix_cache[: self.prefix_cache_max_entries]
        return prefix_length

    def render_prompt(self, prompt: str) -> str:
        prompt = str(prompt)
        if not self.use_chat_template:
            return prompt
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("the tokenizer does not define a chat template")
        return str(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def __call__(self, request: Any) -> str:
        from transformers import StoppingCriteria, StoppingCriteriaList

        raw_prompt = str(getattr(request, "session_prompt", "") or getattr(request, "prompt", ""))
        prompt = self.render_prompt(raw_prompt)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = encoded.get("attention_mask") if isinstance(encoded, dict) else encoded.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logical_input_length = int(input_ids.shape[-1])
        namespace = str(
            getattr(request, "cache_namespace", "")
            or getattr(request, "session_id", "")
            or ""
        )
        input_id_tuple = tuple(input_ids[0].tolist())
        cache_entry = (
            self._find_prefix_cache(namespace, input_id_tuple)
            if self.prefix_cache_enabled and namespace
            else None
        )
        # Keep the full logical input in ``generate``.  Transformers 5 uses
        # the full attention mask plus the cache length to slice the already
        # cached prefix. Passing only the suffix bypasses that logic and can
        # give Qwen incorrect position handling.
        generation_input_ids = input_ids
        generation_attention_mask = attention_mask
        cache_was_reused = cache_entry is not None
        prefix_prefill_tokens = 0
        if cache_entry is None and self.prefix_cache_enabled and namespace:
            prefix_prefill_tokens = self._warm_prefix_cache(namespace, input_ids, attention_mask)
            cache_entry = self._find_prefix_cache(namespace, input_id_tuple)
        cache_hit_tokens = 0
        cache_miss_tokens = logical_input_length
        generation_prefill_tokens = logical_input_length
        cache_hit = cache_was_reused
        if cache_entry is not None:
            prefix_length = len(cache_entry.input_ids)
            generation_prefill_tokens = logical_input_length - prefix_length
            if attention_mask is None:
                generation_attention_mask = self.torch.ones(
                    (1, logical_input_length), dtype=self.torch.long, device=input_ids.device
                )
            else:
                generation_attention_mask = attention_mask
            if cache_was_reused:
                cache_hit_tokens = prefix_length
                cache_miss_tokens = logical_input_length - prefix_length

        generation_kwargs = {
            "input_ids": generation_input_ids,
            "attention_mask": generation_attention_mask,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if cache_entry is not None:
            generation_kwargs["past_key_values"] = copy.deepcopy(cache_entry.past_key_values)
        stopping_criteria: list[Any] = []
        constraint = getattr(request, "action_constraint", None)
        if constraint is not None:
            from transformers import LogitsProcessorList

            constraint.bind(self.tokenizer)
            prompt_length = logical_input_length

            class _ActionConstraintProcessor:
                def __call__(self, current_ids: Any, scores: Any) -> Any:
                    generated = current_ids[0, prompt_length:].tolist()
                    return constraint.mask_logits(scores, generated)

            class _ActionConstraintStop(StoppingCriteria):
                def __call__(self, current_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    generated = current_ids[0, prompt_length:].tolist()
                    return bool(constraint.is_complete(generated))

            generation_kwargs["logits_processor"] = LogitsProcessorList([_ActionConstraintProcessor()])
            stopping_criteria.append(_ActionConstraintStop())

        timeout_seconds = getattr(request, "timeout_seconds", None)
        deadline = None
        timeout_state = {"triggered": False}
        if timeout_seconds is not None and float(timeout_seconds) > 0:
            deadline = time.monotonic() + float(timeout_seconds)

            class _DeadlineStop(StoppingCriteria):
                def __call__(self, current_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    timeout_state["triggered"] = time.monotonic() >= deadline
                    return timeout_state["triggered"]

            stopping_criteria.append(_DeadlineStop())
        if stopping_criteria:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)
        if self.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = self.max_new_tokens
        else:
            max_context = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
            if max_context is None:
                max_context = getattr(self.tokenizer, "model_max_length", None)
            if not isinstance(max_context, int) or max_context <= 0 or max_context > 10_000_000:
                raise RuntimeError("cannot determine the model context window for uncapped generation")
            # Preserve the original uncapped-generation behavior.  The cache
            # must not introduce a new completion-token limit.
            generation_kwargs["max_length"] = max_context
        started = time.monotonic()
        warmup_tokens = 0
        transformer_forward_calls = 1 + int(prefix_prefill_tokens > 0)
        try:
            with self.torch.no_grad():
                output_ids = self.model.generate(**generation_kwargs)
        except Exception:
            if cache_entry is None:
                raise
            # A Transformers version/model implementation may reject passing
            # a cached state to generate. Preserve the original behavior and
            # disable only this failed cache entry.
            self._prefix_cache = [item for item in self._prefix_cache if item is not cache_entry]
            cache_hit = False
            cache_hit_tokens = 0
            cache_miss_tokens = logical_input_length
            generation_input_ids = input_ids
            generation_kwargs.pop("past_key_values", None)
            generation_kwargs["input_ids"] = input_ids
            generation_kwargs["attention_mask"] = attention_mask
            if self.max_new_tokens is None:
                max_context = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
                if max_context is None:
                    max_context = getattr(self.tokenizer, "model_max_length", None)
                if not isinstance(max_context, int) or max_context <= 0 or max_context > 10_000_000:
                    raise RuntimeError("cannot determine the model context window for uncapped generation")
                generation_kwargs["max_length"] = max_context
            with self.torch.no_grad():
                output_ids = self.model.generate(**generation_kwargs)
            transformer_forward_calls = 1 + int(prefix_prefill_tokens > 0)
        new_ids = output_ids[0, logical_input_length:]
        output = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        metrics = {
            # Keep this field as the logical prompt size for compatibility
            # with existing experiment tables.
            "physical_input_tokens": logical_input_length,
            "logical_input_tokens": logical_input_length,
            "physical_prefill_tokens": prefix_prefill_tokens + generation_prefill_tokens,
            "prefix_cache_hit_tokens": cache_hit_tokens,
            "prefix_cache_miss_tokens": cache_miss_tokens,
            "prefix_cache_warmup_tokens": warmup_tokens,
            "prefix_cache_hit": cache_hit,
            "prefix_cache_hits": int(cache_hit),
            "prefix_cache_misses": int(not cache_hit),
            "transformer_forward_calls": transformer_forward_calls,
            "output_tokens": int(new_ids.shape[-1]),
            "forward_calls": 1,
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": timeout_state["triggered"],
            "chat_template_applied": self.use_chat_template,
        }
        self.last_call_metrics = metrics
        for key in self.total_call_metrics:
            self.total_call_metrics[key] += int(metrics.get(key, 0))
        return output

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "backend": "transformers_prefix_cache" if self.prefix_cache_enabled else "stateless_transformers",
            "last_call_metrics": dict(self.last_call_metrics),
            "total_call_metrics": dict(self.total_call_metrics),
        }

    def rollback_session(self, session_id: str) -> None:
        return None


DirectTransformersModel = TransformersModel


def model_path_from_env(default: str = "") -> str:
    return os.getenv("MODEL_PATH", default).strip()

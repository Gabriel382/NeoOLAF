from __future__ import annotations

"""Prompt capture utilities for layer-wise ablation and prompt-size review."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import json
import re
import time
import threading


@dataclass
class PromptRecord:
    """One prompt call captured from a layer LLM backend."""

    layer_name: str
    call_index: int
    timestamp: str
    model: str
    temperature: float
    system_prompt: str
    user_prompt: str
    full_prompt: str
    prompt_chars: int
    estimated_tokens: int
    message_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptCaptureBackend:
    """Wrap an LLM backend and record every chat prompt sent by a layer.

    The wrapper preserves the old NeoOLAF `.chat(...)` API and delegates JSON
    extraction to the wrapped backend when available.
    """

    def __init__(self, backend: Any, layer_name: str) -> None:
        self.backend = backend
        self.layer_name = layer_name
        self.records: list[PromptRecord] = []
        self._lock = threading.Lock()

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        with self._lock:
            record = self._build_record(model=model, messages=messages, temperature=temperature)
            self.records.append(record)
        return self.backend.chat(model=model, messages=messages, temperature=temperature, **kwargs)

    def _build_record(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> PromptRecord:
        system_parts: list[str] = []
        user_parts: list[str] = []
        full_parts: list[str] = []

        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            full_parts.append(f"[{role}]\n{content}")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)

        system_prompt = "\n\n".join(system_parts)
        user_prompt = "\n\n".join(user_parts)
        full_prompt = "\n\n".join(full_parts)

        return PromptRecord(
            layer_name=self.layer_name,
            call_index=len(self.records) + 1,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            full_prompt=full_prompt,
            prompt_chars=len(full_prompt),
            estimated_tokens=max(1, len(full_prompt) // 4),
            message_count=len(messages),
        )

    def extract_json(self, text: str) -> Any:
        if hasattr(self.backend, "extract_json"):
            return self.backend.extract_json(text)
        if text is None:
            raise ValueError("extract_json received None from the LLM backend.")
        text = text.strip()
        if not text:
            raise ValueError(
                "Could not parse JSON from model output because the model returned an empty string. "
                "Check saved raw responses and prompts in the layer artifact folder."
            )
        fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1).strip())
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for i, char in enumerate(text):
            if char in "[{":
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                    return obj
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not parse JSON from model output: {text[:1000]}")

    def __getattr__(self, item: str) -> Any:
        return getattr(self.backend, item)


def summarize_prompt_records(records: list[PromptRecord]) -> dict[str, Any]:
    """Return compact prompt statistics for a layer."""
    if not records:
        return {
            "prompt_call_count": 0,
            "total_prompt_chars": 0,
            "total_estimated_tokens": 0,
            "max_prompt_chars": 0,
            "max_estimated_tokens": 0,
        }

    return {
        "prompt_call_count": len(records),
        "total_prompt_chars": sum(r.prompt_chars for r in records),
        "total_estimated_tokens": sum(r.estimated_tokens for r in records),
        "max_prompt_chars": max(r.prompt_chars for r in records),
        "max_estimated_tokens": max(r.estimated_tokens for r in records),
        "average_prompt_chars": round(sum(r.prompt_chars for r in records) / len(records), 2),
        "average_estimated_tokens": round(sum(r.estimated_tokens for r in records) / len(records), 2),
    }

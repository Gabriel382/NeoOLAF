from __future__ import annotations

"""Compatibility wrapper so layers can call `.chat(...)` on different LLM backends."""

import json
import re
from typing import Any, Dict, List


class ChatCompatBackend:
    """
    Wrap a backend that exposes either `.chat(...)` or `.generate(...)`.

    Existing NeoOLAF layers expect the old Ollama/OpenAI-compatible method:
        chat(model, messages, temperature=...)

    The LiteLLM backend introduced for the ablation work may expose either
    method. This wrapper keeps the layer code unchanged.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        if hasattr(self.backend, "chat"):
            return self.backend.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )

        if hasattr(self.backend, "generate"):
            system_prompt = None
            user_parts: list[str] = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system" and system_prompt is None:
                    system_prompt = content
                else:
                    user_parts.append(content)

            return self.backend.generate(
                prompt="\n\n".join(user_parts),
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs,
            )

        raise TypeError("Backend must expose either chat(...) or generate(...).")

    @staticmethod
    def extract_json(text: str) -> Any:
        if text is None:
            raise ValueError("extract_json received None from the LLM backend.")

        text = text.strip()
        if not text:
            raise ValueError(
                "Could not parse JSON from model output because the model returned an empty string. "
                "This is usually an LLM/provider issue, a max_tokens issue, or a prompt/output-format issue. "
                "Check the saved raw_response_*.txt and prompt_*.txt files in the layer artifact folder."
            )

        fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1).strip())

        fenced2 = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if fenced2:
            return json.loads(fenced2.group(1).strip())

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for i, char in enumerate(text):
            if char in "{[":
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                    return obj
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse JSON from model output:\n{text[:1500]}")

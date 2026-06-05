from __future__ import annotations

"""LiteLLM backend for NeoOLAF.

Keys stay in `.env` and are loaded through python-dotenv.  The backend accepts
an optional `response_format` so layers can use structured output when the
provider/model supports it, while still falling back to normal JSON prompting.
"""

from typing import Any

from dotenv import load_dotenv
from litellm import completion


class LiteLLMBackend:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        load_dotenv()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        model: str | None,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        fallback_to_json_parse: bool = True,
        **kwargs: Any,
    ) -> str:
        selected_model = model or self.model
        if not selected_model:
            raise ValueError("LiteLLMBackend requires a model name.")

        completion_kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        completion_kwargs.update(kwargs)
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        try:
            response = completion(**completion_kwargs)
        except Exception:
            # Some OpenRouter/Ollama/local backends do not support OpenAI-style
            # response_format. Fall back to normal JSON prompting when allowed.
            if not response_format or not fallback_to_json_parse:
                raise
            completion_kwargs.pop("response_format", None)
            response = completion(**completion_kwargs)

        content = response.choices[0].message.content
        return content or ""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        fallback_to_json_parse: bool = True,
        **kwargs: Any,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            fallback_to_json_parse=fallback_to_json_parse,
            **kwargs,
        )

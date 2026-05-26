# Central LLM backend based on LiteLLM.
# This keeps provider-specific keys in .env and avoids hardcoding providers in layers.

import os
from typing import Any

from dotenv import load_dotenv
from litellm import completion


class LiteLLMBackend:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        load_dotenv()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        return response.choices[0].message.content or ""
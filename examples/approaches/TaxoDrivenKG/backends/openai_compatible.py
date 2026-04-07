"""OpenAI-compatible backend wrapper."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI


class OpenAICompatibleBackend:
    """Simple wrapper around OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: float = 300.0,
    ) -> None:
        """Initialize the backend."""
        self.base_url = base_url
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            default_headers=self.extra_headers,
        )

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Run a chat completion and return the assistant text."""
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
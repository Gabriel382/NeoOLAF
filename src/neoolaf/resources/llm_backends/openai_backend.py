from __future__ import annotations

# Standard library imports
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

# Third-party imports
import requests
from requests.exceptions import RequestException, ReadTimeout, ConnectionError


class OpenAIBackend:
    """
    Generic OpenAI-compatible backend.

    Supports:
    - OpenRouter
    - OpenAI
    - local OpenAI-compatible servers
    - vLLM-style endpoints if needed

    Notes:
    - For OpenRouter, use host="https://openrouter.ai/api"
    - For OpenAI, use host="https://api.openai.com"
    - This backend expects /v1/chat/completions under the host
    """

    def __init__(
        self,
        host: str,
        api_key: Optional[str] = None,
        timeout: int = 900,
        max_retries: int = 3,
        retry_wait_seconds: float = 3.0,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        env_var_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the backend.

        Args:
            host:
                Base URL of the OpenAI-compatible server, without trailing slash.
            api_key:
                Explicit API key. If None, try env_var_name, then OPENROUTER_API_KEY,
                then OPENAI_API_KEY.
            timeout:
                Default request timeout in seconds.
            max_retries:
                Number of retry attempts for transient failures.
            retry_wait_seconds:
                Wait time between retries.
            referer:
                Optional HTTP-Referer header, useful for OpenRouter.
            title:
                Optional X-Title header, useful for OpenRouter.
            env_var_name:
                Optional environment variable name to resolve the API key from.
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.referer = referer
        self.title = title

        # Resolve API key with fallback order.
        if api_key:
            self.api_key = api_key
        elif env_var_name and os.getenv(env_var_name):
            self.api_key = os.getenv(env_var_name)
        elif os.getenv("OPENROUTER_API_KEY"):
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY", "")

        if not self.api_key:
            raise ValueError(
                "No API key found. Provide api_key explicitly or set an environment variable."
            )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to an OpenAI-compatible endpoint.

        Args:
            model:
                Model name to use.
            messages:
                OpenAI-style message list.
            temperature:
                Sampling temperature.
            timeout:
                Optional per-call timeout override.
            max_tokens:
                Optional max completion tokens.

        Returns:
            The generated assistant text.

        Raises:
            RuntimeError:
                If all retry attempts fail.
        """
        url = f"{self.host}/v1/chat/completions"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Add optional OpenRouter-friendly headers.
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title

        effective_timeout = timeout if timeout is not None else self.timeout
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=effective_timeout,
                )
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"]

            except (ReadTimeout, ConnectionError, RequestException) as exc:
                last_error = exc

                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_seconds)
                    continue

        raise RuntimeError(
            f"OpenAI-compatible request failed after {self.max_retries} attempts "
            f"(host={self.host}, timeout={effective_timeout}s). "
            f"Last error: {last_error}"
        )

    @staticmethod
    def extract_json(text: str) -> Any:
        """
        Extract JSON from raw model output.

        Supports:
        - fenced ```json ... ```
        - fenced ``` ... ```
        - direct JSON object or array
        - first array/object found in text
        """
        text = text.strip()

        fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        fenced2 = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if fenced2:
            return json.loads(fenced2.group(1))

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        arr_match = re.search(r"(\[.*\])", text, re.DOTALL)
        if arr_match:
            return json.loads(arr_match.group(1))

        obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(1))

        raise ValueError("Could not parse JSON from OpenAI-compatible output.")
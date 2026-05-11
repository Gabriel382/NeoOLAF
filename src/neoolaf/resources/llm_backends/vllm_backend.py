from __future__ import annotations

# Standard library imports
import json
import re
import time
from typing import Any, Dict, List, Optional

# Third-party imports
import requests
from requests.exceptions import RequestException, ReadTimeout, ConnectionError


class VLLMBackend:
    """
    Simple vLLM OpenAI-compatible backend.

    Improvements over the previous version:
    - higher default timeout
    - retry logic for transient failures
    - optional per-call timeout override
    - clearer error messages
    """

    def __init__(
        self,
        host: str = "http://localhost:8000",
        api_key: str = "dummy",
        timeout: int = 900,
        max_retries: int = 3,
        retry_wait_seconds: float = 3.0,
    ) -> None:
        """
        Initialize the backend.

        Args:
            host:
                Base URL of the vLLM server.
            api_key:
                Dummy or real API key depending on the local setup.
            timeout:
                Default request timeout in seconds.
            max_retries:
                Number of retry attempts for transient failures.
            retry_wait_seconds:
                Wait time between retries.
        """
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to vLLM.

        Args:
            model:
                Model name served by vLLM.
            messages:
                OpenAI-style message list.
            temperature:
                Sampling temperature.
            timeout:
                Optional per-call timeout override.

        Returns:
            The generated assistant text.

        Raises:
            RuntimeError:
                If all retry attempts fail.
        """
        url = f"{self.host}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

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

                # If we still have retries left, wait a bit and try again.
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_seconds)
                    continue

        raise RuntimeError(
            f"vLLM request failed after {self.max_retries} attempts "
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

        raise ValueError("Could not parse JSON from vLLM output.")

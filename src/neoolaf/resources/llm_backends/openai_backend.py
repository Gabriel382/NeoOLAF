from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException, ReadTimeout, ConnectionError


class OpenAIBackend:
    def __init__(
        self,
        host: str,
        api_key: Optional[str] = None,
        timeout: int = 900,
        max_retries: int = 5,
        retry_wait_seconds: float = 4.0,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        env_var_name: Optional[str] = None,
        retry_on_empty: bool = True,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.referer = referer
        self.title = title
        self.retry_on_empty = retry_on_empty

        if api_key:
            self.api_key = api_key
        elif env_var_name and os.getenv(env_var_name):
            self.api_key = os.getenv(env_var_name)
        elif os.getenv("OPENROUTER_API_KEY"):
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY", "")

        if not self.api_key:
            raise ValueError("No API key found.")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
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
                choices = data.get("choices")

                if not choices:
                    raise RuntimeError(f"No choices in response: {data}")

                message = choices[0].get("message", {})
                content = message.get("content")

                # Retry if empty content and retry_on_empty is enabled
                if content is None or not isinstance(content, str) or not content.strip():
                    if self.retry_on_empty:
                        raise RuntimeError(f"Empty or missing message content: {data}")
                    return ""

                return content

            except (ReadTimeout, ConnectionError, RequestException, RuntimeError, ValueError) as exc:
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
        Extract the first valid JSON object or array from model output.

        Supports:
        - fenced ```json ... ```
        - fenced ``` ... ```
        - raw JSON
        - first balanced JSON object/array inside extra text
        """
        import json

        if text is None:
            raise ValueError("extract_json received None from backend.chat().")

        text = text.strip()

        # 1. Try fenced json block
        fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1).strip())

        # 2. Try generic fenced block
        fenced2 = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if fenced2:
            return json.loads(fenced2.group(1).strip())

        # 3. Try direct full parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 4. Try first balanced JSON object/array
        decoder = json.JSONDecoder()

        for i, ch in enumerate(text):
            if ch in "{[":
                try:
                    obj, end = decoder.raw_decode(text[i:])
                    return obj
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse JSON from output:\n{text[:1500]}")
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import requests


class OllamaBackend:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 300) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": "low",
            "keep_alive": "10m",
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }

        #print(f"[Ollama] URL: {url}")
        #print(f"[Ollama] Model: {model}")
        #print(f"[Ollama] Payload: {payload}")
        #print(f"[Ollama] Number of messages: {len(messages)}")

        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            #print(f"[Ollama] Message {i} role={role} length={len(content)}")
            #print(content[:1000])
            #print("-" * 80)

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed before streaming: {e}") from e

        final_content_parts = []
        thinking_preview_parts = []
        last_done_reason = None

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                data = json.loads(line)

                message = data.get("message", {}) or {}
                chunk_content = message.get("content", "")
                chunk_thinking = message.get("thinking", "")

                if chunk_thinking and len("".join(thinking_preview_parts)) < 1200:
                    thinking_preview_parts.append(chunk_thinking)

                if chunk_content:
                    final_content_parts.append(chunk_content)

                if data.get("done"):
                    last_done_reason = data.get("done_reason")
                    break

        except Exception as e:
            raise RuntimeError(f"Ollama streaming parse failed: {e}") from e

        final_content = "".join(final_content_parts).strip()

        #print(f"[Ollama] Done reason: {last_done_reason}")
        #print(f"[Ollama] Final content length: {len(final_content)}")

        if not final_content:
            thinking_preview = "".join(thinking_preview_parts)[:1200]
            raise RuntimeError(
                "Ollama returned no final content. "
                f"done_reason={last_done_reason}. "
                f"Thinking preview: {thinking_preview}"
            )

        return final_content

    @staticmethod
    def extract_json(text: str) -> Any:
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

        raise ValueError("Could not parse JSON from Ollama output.")
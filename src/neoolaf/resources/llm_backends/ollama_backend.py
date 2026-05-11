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
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

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
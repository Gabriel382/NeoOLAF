"""Native Ollama backend using its local HTTP API."""

from typing import Dict, List
import requests

from .base import BaseChatBackend


class OllamaBackend(BaseChatBackend):
    """Call Ollama's chat endpoint directly."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host.rstrip("/")

    def chat(self, messages: List[Dict[str, str]], model_name: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

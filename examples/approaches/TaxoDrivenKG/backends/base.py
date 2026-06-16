"""Abstract backend interface used by the extractor."""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseChatBackend(ABC):
    """Minimal chat backend interface."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], model_name: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        """Return the assistant text response for the given messages."""
        raise NotImplementedError

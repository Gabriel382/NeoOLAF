from __future__ import annotations

"""Stable RAG backend interface for NeoOLAF ablation runs.

The goal is to let NeoOLAF layers call one generic retrieval interface today,
while keeping RAGTree replaceable as a future backend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RAGRequest:
    """Generic retrieval request produced by a NeoOLAF layer."""

    query: str
    layer_name: str
    document_id: str | None = None
    allowed_spaces: list[str] | None = None
    top_k: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Generic retrieval result returned to a NeoOLAF layer."""

    context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RAGBackend(ABC):
    """Backend-independent RAG contract used by future NeoOLAF layers."""

    name: str = "base"

    @abstractmethod
    def retrieve(self, request: RAGRequest) -> RAGResult:
        """Retrieve context and sources for a layer request."""
        raise NotImplementedError

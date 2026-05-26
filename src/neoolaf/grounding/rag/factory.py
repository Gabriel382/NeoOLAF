from __future__ import annotations

"""RAG backend factory used by CLI, notebooks, and future layers."""

from typing import Any

from neoolaf.grounding.rag.agentic_rag_backend import AgenticRAGBackend
from neoolaf.grounding.rag.base import RAGBackend
from neoolaf.grounding.rag.ragtree_backend import RAGTreeBackend


class NullRAGBackend(AgenticRAGBackend):
    """Explicit no-RAG backend for ablation variants."""

    name = "none"

    def __init__(self) -> None:
        super().__init__(engine=None)


def build_rag_backend(name: str = "agentic", **kwargs: Any) -> RAGBackend:
    normalized = (name or "agentic").strip().lower()
    if normalized in {"agentic", "neoolaf", "semantic", "semanticrag"}:
        return AgenticRAGBackend(engine=kwargs.get("engine"))
    if normalized in {"none", "null", "off"}:
        return NullRAGBackend()
    if normalized in {"ragtree", "rag_tree"}:
        return RAGTreeBackend(
            config_path=kwargs.get("config_path"),
            backend=kwargs.get("backend"),
        )
    raise ValueError(f"Unknown RAG backend: {name}")

from __future__ import annotations

"""Adapter for the current internal NeoOLAF SemanticRAG/agentic RAG logic."""

from typing import Any

from neoolaf.grounding.rag.base import RAGBackend, RAGRequest, RAGResult


class AgenticRAGBackend(RAGBackend):
    """Use the current NeoOLAF SemanticRAG engine when available.

    The implementation is deliberately defensive because existing NeoOLAF code
    has used several internal RAG request/result shapes.  If no engine is given,
    it returns an empty but valid result, which is useful for ablation runs with
    `--rag-backend none` or while RAGTree is not yet integrated.
    """

    name = "agentic"

    def __init__(self, engine: Any | None = None) -> None:
        self.engine = engine

    def retrieve(self, request: RAGRequest) -> RAGResult:
        if self.engine is None:
            return RAGResult(
                context="",
                sources=[],
                confidence=None,
                metadata={"backend": self.name, "note": "No internal RAG engine configured."},
            )

        # Preferred future API.
        if hasattr(self.engine, "retrieve"):
            raw = self.engine.retrieve(request)
            return self._coerce_result(raw)

        # Existing internal API used by SemanticRAGEngine.
        if hasattr(self.engine, "ground"):
            raw = self.engine.ground(request)
            return self._coerce_result(raw)

        raise TypeError("Configured agentic RAG engine must expose retrieve(...) or ground(...).")

    def _coerce_result(self, raw: Any) -> RAGResult:
        if isinstance(raw, RAGResult):
            return raw
        if raw is None:
            return RAGResult(metadata={"backend": self.name, "note": "Engine returned None."})
        if isinstance(raw, str):
            return RAGResult(context=raw, metadata={"backend": self.name})
        context = getattr(raw, "context", None) or getattr(raw, "grounding_context", None) or ""
        sources = getattr(raw, "sources", None) or getattr(raw, "items", None) or []
        confidence = getattr(raw, "confidence", None)
        if not isinstance(sources, list):
            sources = [sources]
        normalized_sources = []
        for source in sources:
            if isinstance(source, dict):
                normalized_sources.append(source)
            else:
                normalized_sources.append({"raw": repr(source)})
        return RAGResult(
            context=str(context),
            sources=normalized_sources,
            confidence=confidence if isinstance(confidence, (float, int)) else None,
            metadata={"backend": self.name, "raw_type": type(raw).__name__},
        )

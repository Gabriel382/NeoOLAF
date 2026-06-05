from __future__ import annotations

"""Future RAGTree backend adapter.

RAGTree can be injected here without changing NeoOLAF layer code.  The class
exposes both the new ``retrieve`` API and the legacy ``ground`` API used by
Layer 2, so switching backends does not require editing layers.
"""

from neoolaf.grounding.rag.base import RAGBackend, RAGRequest, RAGResult
from neoolaf.grounding.rag.types import GroundingRequest, GroundingResult, RetrievedItem


class RAGTreeBackend(RAGBackend):
    name = "ragtree"

    def __init__(self, config_path: str | None = None, backend: object | None = None) -> None:
        self.config_path = config_path
        self.backend = backend

    def retrieve(self, request: RAGRequest) -> RAGResult:
        if self.backend is None:
            raise NotImplementedError(
                "RAGTree backend is declared but not connected yet. "
                "Use backend='agentic' for the current NeoOLAF RAG stub, or pass "
                "a RAGTree-compatible backend when the library is integrated."
            )
        if not hasattr(self.backend, "retrieve"):
            raise TypeError("RAGTree backend object must expose retrieve(...).")
        raw = self.backend.retrieve(request)
        if isinstance(raw, RAGResult):
            return raw
        return RAGResult(context=str(raw), sources=[], metadata={"backend": self.name})

    def ground(self, request: GroundingRequest | RAGRequest) -> GroundingResult:
        """Compatibility wrapper for legacy layers using ``ground``."""
        if isinstance(request, GroundingRequest):
            rag_request = RAGRequest(
                query=request.query,
                layer_name=request.layer_name,
                allowed_spaces=list(request.preferred_sources or []),
                top_k=request.top_k,
                metadata=dict(request.payload or {}),
            )
            grounding_request = request
        else:
            rag_request = request
            grounding_request = GroundingRequest(
                layer_name=request.layer_name,
                query=request.query,
                payload=dict(request.metadata or {}),
                preferred_sources=list(request.allowed_spaces or []),
                top_k=request.top_k,
            )

        result = self.retrieve(rag_request)
        items = []
        sources = []
        for source in result.sources or []:
            if isinstance(source, dict):
                name = str(source.get("space") or source.get("source") or self.name)
                text = str(source.get("text") or source.get("content") or source.get("raw") or "")
                metadata = {k: v for k, v in source.items() if k not in {"space", "source", "text", "content"}}
            else:
                name = self.name
                text = str(source)
                metadata = {}
            sources.append(name)
            if text:
                items.append(RetrievedItem(source=name, content=text, metadata=metadata))
        if result.context and not items:
            sources.append(self.name)
            items.append(RetrievedItem(source=self.name, content=result.context))
        return GroundingResult(
            request=grounding_request,
            selected_sources=list(dict.fromkeys(sources)),
            retrieved_items=items,
            grounding_summary=result.context or "",
            merged_context={"metadata": result.metadata, "confidence": result.confidence},
        )

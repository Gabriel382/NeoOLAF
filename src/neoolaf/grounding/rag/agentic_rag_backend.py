from __future__ import annotations

"""Adapter for the current internal NeoOLAF SemanticRAG/agentic RAG logic.

This backend intentionally exposes both APIs used in NeoOLAF:

- ``retrieve(RAGRequest) -> RAGResult`` for the newer generic RAG interface.
- ``ground(GroundingRequest) -> GroundingResult`` for older layers that still
  expect the grounding API, especially Layer 2.

The two interfaces are bridged here so that layers can be migrated gradually
without breaking ablation runs.
"""

from typing import Any

from neoolaf.grounding.rag.base import RAGBackend, RAGRequest, RAGResult
from neoolaf.grounding.rag.types import GroundingRequest, GroundingResult, RetrievedItem


class AgenticRAGBackend(RAGBackend):
    """Use the current NeoOLAF SemanticRAG engine when available.

    If no engine is configured, the backend returns an empty but valid result.
    This is useful for ablations and for profiles where RAG is enabled as a
    light guidance channel but the full RAG engine is not connected yet.
    """

    name = "agentic"

    def __init__(self, engine: Any | None = None) -> None:
        self.engine = engine

    def retrieve(self, request: RAGRequest | GroundingRequest) -> RAGResult:
        """Retrieve context using the generic NeoOLAF RAG contract.

        The method also accepts ``GroundingRequest`` defensively, because some
        legacy layer code may still pass that object directly.
        """
        normalized_request = self._to_rag_request(request)

        if self.engine is None:
            # Lightweight built-in guidance mode.  This is intentionally small:
            # it lets profiles provide short retrieval snippets before a full
            # SemanticRAG/RAGTree backend is wired in.
            snippets = normalized_request.metadata.get("lightweight_profile_context") or []
            if snippets:
                top_k = max(0, int(normalized_request.top_k or 0))
                selected = snippets[:top_k] if top_k else []
                return RAGResult(
                    context="\n".join(str(item) for item in selected if item),
                    sources=[
                        {"space": "profile_guidance", "rank": idx + 1, "text": str(item)}
                        for idx, item in enumerate(selected)
                    ],
                    confidence=None,
                    metadata={"backend": self.name, "mode": "lightweight_profile_context"},
                )
            return RAGResult(
                context="",
                sources=[],
                confidence=None,
                metadata={"backend": self.name, "note": "No internal RAG engine configured."},
            )

        # Preferred future API.
        if hasattr(self.engine, "retrieve"):
            raw = self.engine.retrieve(normalized_request)
            return self._coerce_result(raw)

        # Existing internal API used by SemanticRAGEngine.
        if hasattr(self.engine, "ground"):
            raw = self.engine.ground(self._to_grounding_request(normalized_request))
            return self._coerce_result(raw)

        raise TypeError("Configured agentic RAG engine must expose retrieve(...) or ground(...).")

    def ground(self, request: GroundingRequest | RAGRequest) -> GroundingResult:
        """Ground a request using the legacy grounding contract.

        Layer 2 currently calls ``rag_adapter.ground(...)``.  This bridge keeps
        that layer compatible with the newer ``RAGBackend`` implementations.
        """
        grounding_request = self._to_grounding_request(request)

        # If a real engine exposes the legacy API, preserve it.
        if self.engine is not None and hasattr(self.engine, "ground"):
            raw = self.engine.ground(grounding_request)
            return self._coerce_grounding_result(raw, grounding_request)

        # Otherwise, route through the generic retrieval API and convert the
        # result to the GroundingResult object expected by older layers.
        rag_result = self.retrieve(self._to_rag_request(grounding_request))
        return self._rag_result_to_grounding_result(rag_result, grounding_request)

    def _to_rag_request(self, request: RAGRequest | GroundingRequest) -> RAGRequest:
        """Normalize a RAG/Grounding request to ``RAGRequest``."""
        if isinstance(request, RAGRequest):
            return request

        # GroundingRequest has payload/preferred_sources instead of
        # metadata/allowed_spaces.
        return RAGRequest(
            query=getattr(request, "query", ""),
            layer_name=getattr(request, "layer_name", "unknown_layer"),
            document_id=(getattr(request, "payload", {}) or {}).get("document_id"),
            allowed_spaces=list(getattr(request, "preferred_sources", []) or []),
            top_k=int(getattr(request, "top_k", 5) or 5),
            metadata=dict(getattr(request, "payload", {}) or {}),
        )

    def _to_grounding_request(self, request: GroundingRequest | RAGRequest) -> GroundingRequest:
        """Normalize a RAG/Grounding request to ``GroundingRequest``."""
        if isinstance(request, GroundingRequest):
            return request

        payload = dict(getattr(request, "metadata", {}) or {})
        if getattr(request, "document_id", None):
            payload.setdefault("document_id", request.document_id)

        return GroundingRequest(
            layer_name=getattr(request, "layer_name", "unknown_layer"),
            query=getattr(request, "query", ""),
            payload=payload,
            preferred_sources=list(getattr(request, "allowed_spaces", []) or []),
            top_k=int(getattr(request, "top_k", 5) or 5),
        )

    def _coerce_result(self, raw: Any) -> RAGResult:
        """Coerce diverse backend result shapes to ``RAGResult``."""
        if isinstance(raw, RAGResult):
            return raw
        if raw is None:
            return RAGResult(metadata={"backend": self.name, "note": "Engine returned None."})
        if isinstance(raw, str):
            return RAGResult(context=raw, metadata={"backend": self.name})
        if isinstance(raw, GroundingResult):
            return RAGResult(
                context=raw.grounding_summary or "\n".join(item.content for item in raw.retrieved_items),
                sources=[
                    {
                        "space": item.source,
                        "text": item.content,
                        "metadata": item.metadata,
                        "score": item.score,
                        "reference": item.reference,
                    }
                    for item in raw.retrieved_items
                ],
                confidence=None,
                metadata={"backend": self.name, "raw_type": type(raw).__name__},
            )

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

    def _coerce_grounding_result(self, raw: Any, request: GroundingRequest) -> GroundingResult:
        """Coerce diverse backend result shapes to ``GroundingResult``."""
        if isinstance(raw, GroundingResult):
            return raw
        if isinstance(raw, RAGResult):
            return self._rag_result_to_grounding_result(raw, request)
        if raw is None:
            return GroundingResult(request=request, selected_sources=[], retrieved_items=[])
        if isinstance(raw, str):
            return GroundingResult(
                request=request,
                selected_sources=[self.name],
                retrieved_items=[RetrievedItem(source=self.name, content=raw)],
                grounding_summary=raw,
            )
        return GroundingResult(
            request=request,
            selected_sources=[self.name],
            retrieved_items=[RetrievedItem(source=self.name, content=str(raw))],
            grounding_summary=str(raw),
        )

    def _rag_result_to_grounding_result(
        self,
        rag_result: RAGResult,
        request: GroundingRequest,
    ) -> GroundingResult:
        """Convert a ``RAGResult`` to the legacy ``GroundingResult`` shape."""
        retrieved_items: list[RetrievedItem] = []
        selected_sources: list[str] = []

        for source in rag_result.sources or []:
            if isinstance(source, dict):
                space = str(source.get("space") or source.get("source") or self.name)
                content = str(source.get("text") or source.get("content") or source.get("raw") or "")
                score = source.get("score")
                reference = source.get("reference") or source.get("url")
                metadata = {k: v for k, v in source.items() if k not in {"space", "source", "text", "content", "score", "reference", "url"}}
            else:
                space = self.name
                content = str(source)
                score = None
                reference = None
                metadata = {}

            selected_sources.append(space)
            if content:
                retrieved_items.append(
                    RetrievedItem(
                        source=space,
                        content=content,
                        metadata=metadata,
                        score=score if isinstance(score, (float, int)) else None,
                        reference=str(reference) if reference else None,
                    )
                )

        if rag_result.context and not retrieved_items:
            retrieved_items.append(RetrievedItem(source=self.name, content=rag_result.context))
            selected_sources.append(self.name)

        # Deduplicate sources while preserving order.
        deduped_sources = list(dict.fromkeys(selected_sources))

        return GroundingResult(
            request=request,
            selected_sources=deduped_sources,
            retrieved_items=retrieved_items,
            grounding_summary=rag_result.context or "",
            merged_context={"metadata": rag_result.metadata, "confidence": rag_result.confidence},
        )

from __future__ import annotations

"""Future RAGTree backend adapter.

RAGTree can later be injected here without changing NeoOLAF layer code.  For
now this class is explicit about not being implemented, which prevents silent
fake RAG results during experiments.
"""

from neoolaf.grounding.rag.base import RAGBackend, RAGRequest, RAGResult


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

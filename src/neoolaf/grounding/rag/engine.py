from __future__ import annotations

# Local imports
from neoolaf.grounding.rag.types import GroundingRequest, GroundingResult
from neoolaf.grounding.rag.graph import SemanticRAGGraphFactory


class SemanticRAGEngine:
    """
    Placeholder SemanticRAG engine for NeoOLAF.

    This is the current internal grounding backend.
    Later it should be replaceable by a RAGTree-backed implementation.
    """

    def __init__(self, registry, ollama_backend, model_name: str) -> None:
        """
        Initialize the engine.
        """
        self.registry = registry
        self.ollama_backend = ollama_backend
        self.model_name = model_name

    def ground(self, request: GroundingRequest) -> GroundingResult:
        """
        Run the agentic grounding workflow and return a standardized result.
        """
        graph = SemanticRAGGraphFactory(
            registry=self.registry,
            ollama_backend=self.ollama_backend,
            model_name=self.model_name,
        ).build()

        final_state = graph.invoke(
            {
                "request": request,
                "available_sources": self.registry.available_sources(),
            }
        )

        return GroundingResult(
            request=request,
            selected_sources=final_state.get("selected_sources", []),
            retrieved_items=final_state.get("retrieved_items", []),
            grounding_summary=final_state.get("grounding_summary", ""),
            merged_context=final_state.get("merged_context", {}),
        )
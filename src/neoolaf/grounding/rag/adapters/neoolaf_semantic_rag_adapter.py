from __future__ import annotations

# Local imports
from neoolaf.grounding.rag.engine import SemanticRAGEngine


class NeoOLAFSemanticRAGAdapter:
    """
    Adapter exposing the current internal SemanticRAG engine.

    This keeps the external interface stable so that RAGTree can replace it later.
    """

    def __init__(self, engine: SemanticRAGEngine) -> None:
        """
        Initialize the adapter.
        """
        self.engine = engine

    def ground(self, request):
        """
        Proxy grounding to the internal SemanticRAG engine.
        """
        return self.engine.ground(request)
from __future__ import annotations

# Standard library imports
from typing import Dict, List

# Local imports
from neoolaf.grounding.rag.interface import RetrievalSpace


class RetrievalRegistry:
    """
    Registry of retrieval spaces.
    """

    def __init__(self) -> None:
        """
        Initialize an empty registry.
        """
        self._spaces: Dict[str, RetrievalSpace] = {}

    def register(self, space: RetrievalSpace) -> None:
        """
        Register one retrieval space.
        """
        self._spaces[space.source_name] = space

    def get(self, source_name: str) -> RetrievalSpace | None:
        """
        Get a retrieval space by name.
        """
        return self._spaces.get(source_name)

    def available_sources(self) -> List[str]:
        """
        Return all available source names.
        """
        return list(self._spaces.keys())
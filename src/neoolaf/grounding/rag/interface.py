from __future__ import annotations

# Standard library imports
from abc import ABC, abstractmethod
from typing import List

# Local imports
from neoolaf.grounding.rag.types import RetrievedItem


class RetrievalSpace(ABC):
    """
    Abstract retrieval space.

    Every retrieval backend must implement this interface.
    """

    # Unique source name
    source_name: str

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """
        Retrieve evidence items for a given query.
        """
        raise NotImplementedError
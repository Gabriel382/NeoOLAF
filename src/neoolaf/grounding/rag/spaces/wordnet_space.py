from __future__ import annotations

# Standard library imports
from typing import List

# Local imports
from neoolaf.grounding.rag.interface import RetrievalSpace
from neoolaf.grounding.rag.types import RetrievedItem
from neoolaf.resources.knowledge_sources.wordnet_source import WordNetSource


class WordNetSpace(RetrievalSpace):
    """
    Retrieval space over WordNet.
    """

    source_name = "wordnet"

    def __init__(self, source: WordNetSource | None = None) -> None:
        """
        Initialize WordNet retrieval space.
        """
        self.source = source or WordNetSource()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """
        Retrieve WordNet lexical evidence.
        """
        result = self.source.search(query, max_synsets=top_k)
        items: List[RetrievedItem] = []

        for definition in result.get("definitions", []):
            items.append(
                RetrievedItem(
                    source=self.source_name,
                    content=definition,
                    metadata=result,
                    reference=None,
                )
            )

        return items[:top_k]
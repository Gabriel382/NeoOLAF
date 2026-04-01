from __future__ import annotations

# Standard library imports
from typing import List

# Local imports
from neoolaf.grounding.rag.interface import RetrievalSpace
from neoolaf.grounding.rag.types import RetrievedItem
from neoolaf.resources.knowledge_sources.wikipedia_source import WikipediaSource


class WikipediaSpace(RetrievalSpace):
    """
    Retrieval space over Wikipedia.
    """

    source_name = "wikipedia"

    def __init__(self, source: WikipediaSource | None = None) -> None:
        """
        Initialize Wikipedia retrieval space.
        """
        self.source = source or WikipediaSource(language="en")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """
        Retrieve Wikipedia evidence.
        """
        result = self.source.search(query)
        items: List[RetrievedItem] = []

        if result.get("found"):
            items.append(
                RetrievedItem(
                    source=self.source_name,
                    content=result.get("summary", ""),
                    metadata=result,
                    reference=result.get("url"),
                )
            )

        return items[:top_k]
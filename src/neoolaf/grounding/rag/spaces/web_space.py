from __future__ import annotations

# Standard library imports
from typing import List

# Local imports
from neoolaf.grounding.rag.interface import RetrievalSpace
from neoolaf.grounding.rag.types import RetrievedItem
from neoolaf.resources.knowledge_sources.web_search_source import WebSearchSource


class WebSpace(RetrievalSpace):
    """
    Retrieval space over web search.
    """

    source_name = "web"

    def __init__(self, source: WebSearchSource | None = None) -> None:
        """
        Initialize web search space.
        """
        self.source = source or WebSearchSource()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """
        Retrieve web results.
        """
        result = self.source.search(query, max_results=top_k)
        items: List[RetrievedItem] = []

        for item in result.get("results", []):
            items.append(
                RetrievedItem(
                    source=self.source_name,
                    content=f"{item.get('title', '')} -- {item.get('body', '')}".strip(),
                    metadata=item,
                    reference=item.get("href"),
                )
            )

        return items
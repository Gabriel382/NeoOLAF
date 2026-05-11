from __future__ import annotations

# Standard library imports
from typing import List

# Local imports
from neoolaf.grounding.rag.interface import RetrievalSpace
from neoolaf.grounding.rag.types import RetrievedItem
from neoolaf.resources.knowledge_sources.wikidata_source import WikidataSource


class WikidataSpace(RetrievalSpace):
    """
    Retrieval space over Wikidata.
    """

    source_name = "wikidata"

    def __init__(self, source: WikidataSource | None = None) -> None:
        """
        Initialize Wikidata retrieval space.
        """
        self.source = source or WikidataSource()

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """
        Retrieve Wikidata evidence.
        """
        result = self.source.search(query, limit=top_k)
        items: List[RetrievedItem] = []

        for item in result.get("results", []):
            content = f"{', '.join(item.get('labels', []))} -- {item.get('description', '')}".strip()
            items.append(
                RetrievedItem(
                    source=self.source_name,
                    content=content,
                    metadata=item,
                    reference=item.get("url"),
                )
            )

        return items
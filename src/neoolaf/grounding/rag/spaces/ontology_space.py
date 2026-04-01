from __future__ import annotations

# Standard library imports
from typing import List

# Local imports
from neoolaf.grounding.rag.interface import RetrievalSpace
from neoolaf.grounding.rag.types import RetrievedItem
from neoolaf.domain.seed_ontology import SeedOntology
from neoolaf.ontology.retrieval import SeedOntologyRetriever


class OntologySpace(RetrievalSpace):
    """
    Retrieval space over the seed/source ontology.
    """

    source_name = "ontology"

    def __init__(self, seed_ontology: SeedOntology | None) -> None:
        """
        Initialize the ontology retrieval space.
        """
        self.seed_ontology = seed_ontology
        self.retriever = SeedOntologyRetriever(seed_ontology) if seed_ontology is not None else None

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """
        Retrieve nearest classes and properties from the seed ontology.
        """
        if self.retriever is None:
            return []

        items: List[RetrievedItem] = []

        for cls in self.retriever.nearest_classes(query, top_k=top_k):
            items.append(
                RetrievedItem(
                    source=self.source_name,
                    content=f"Class: {cls.label}. {cls.description or ''}".strip(),
                    metadata={
                        "type": "class",
                        "uri": cls.uri,
                        "label": cls.label,
                        "alt_labels": getattr(cls, "alt_labels", []),
                        "parents": cls.parent_uris,
                        "children": cls.child_uris,
                    },
                    reference=cls.uri,
                )
            )

        for prop in self.retriever.nearest_properties(query, top_k=top_k):
            items.append(
                RetrievedItem(
                    source=self.source_name,
                    content=f"Property: {prop.label}. {prop.description or ''}".strip(),
                    metadata={
                        "type": "property",
                        "uri": prop.uri,
                        "label": prop.label,
                        "alt_labels": getattr(prop, "alt_labels", []),
                        "domain_uris": prop.domain_uris,
                        "range_uris": prop.range_uris,
                        "parents": prop.parent_uris,
                        "children": prop.child_uris,
                    },
                    reference=prop.uri,
                )
            )

        return items[:top_k]
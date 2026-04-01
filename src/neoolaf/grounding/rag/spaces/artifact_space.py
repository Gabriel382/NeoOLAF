from __future__ import annotations

# Standard library imports
from typing import List

# Local imports
from neoolaf.grounding.rag.interface import RetrievalSpace
from neoolaf.grounding.rag.types import RetrievedItem


class ArtifactSpace(RetrievalSpace):
    """
    Retrieval space over prior NeoOLAF artifacts already stored in PipelineState.

    This is intentionally simple for the placeholder version.
    """

    source_name = "artifacts"

    def __init__(self, state) -> None:
        """
        Initialize with the current pipeline state.
        """
        self.state = state

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedItem]:
        """
        Retrieve simple text evidence from prior artifacts.
        """
        query_norm = query.lower().strip()
        items: List[RetrievedItem] = []

        # Search concept candidates
        for concept in getattr(self.state, "concept_candidates", []):
            if query_norm in concept.label.lower():
                items.append(
                    RetrievedItem(
                        source=self.source_name,
                        content=f"Concept candidate: {concept.label}. {concept.description or ''}".strip(),
                        metadata={"type": "concept_candidate", "id": concept.concept_id},
                        reference=concept.concept_id,
                    )
                )

        # Search relation candidates
        for relation in getattr(self.state, "ontology_relation_candidates", []):
            if query_norm in relation.label.lower():
                items.append(
                    RetrievedItem(
                        source=self.source_name,
                        content=f"Ontology relation candidate: {relation.label}. {relation.description or ''}".strip(),
                        metadata={"type": "ontology_relation_candidate", "id": relation.relation_id},
                        reference=relation.relation_id,
                    )
                )

        # Search triples
        for triple in getattr(self.state, "candidate_triples", []):
            joined = f"{triple.subject_label} {triple.predicate_label} {triple.object_label}".lower()
            if query_norm in joined:
                items.append(
                    RetrievedItem(
                        source=self.source_name,
                        content=f"Triple: ({triple.subject_label}, {triple.predicate_label}, {triple.object_label})",
                        metadata={"type": "candidate_triple", "id": triple.triple_id},
                        reference=triple.triple_id,
                    )
                )

        return items[:top_k]
from __future__ import annotations

# Local imports
from neoolaf.domain.seed_ontology import SeedOntology
from neoolaf.ontology.retrieval import SeedOntologyRetriever


def build_seed_ontology_context(
    seed_ontology: SeedOntology | None,
    query: str,
    top_k_classes: int = 3,
    top_k_properties: int = 3,
) -> str:
    """
    Build a compact ontology context snippet for prompts.

    This can be injected into layer prompts to bias extraction, enrichment,
    typing, promotion, hierarchy placement, and schema extraction.
    """
    if seed_ontology is None:
        return ""

    retriever = SeedOntologyRetriever(seed_ontology)
    nearest_classes = retriever.nearest_classes(query, top_k=top_k_classes)
    nearest_properties = retriever.nearest_properties(query, top_k=top_k_properties)

    lines = ["Seed ontology context:"]

    if nearest_classes:
        lines.append("Nearest classes:")
        for cls in nearest_classes:
            lines.append(f"- {cls.label}: {cls.description or 'no description'}")

    if nearest_properties:
        lines.append("Nearest properties:")
        for prop in nearest_properties:
            lines.append(f"- {prop.label}: {prop.description or 'no description'}")

    return "\n".join(lines) + "\n"
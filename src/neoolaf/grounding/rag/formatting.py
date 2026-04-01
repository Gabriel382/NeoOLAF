from __future__ import annotations

# Local imports
from neoolaf.grounding.rag.types import GroundingResult


def build_grounding_context(grounding_result: GroundingResult | None, max_items: int = 5) -> str:
    """
    Convert a GroundingResult into a compact prompt-ready text block.
    """
    if grounding_result is None:
        return ""

    lines = []

    if grounding_result.selected_sources:
        lines.append(f"Grounding sources: {', '.join(grounding_result.selected_sources)}")

    if grounding_result.grounding_summary:
        lines.append(f"Grounding summary: {grounding_result.grounding_summary}")

    if grounding_result.retrieved_items:
        lines.append("Retrieved evidence:")
        for item in grounding_result.retrieved_items[:max_items]:
            snippet = item.content.strip().replace("\n", " ")
            lines.append(f"- [{item.source}] {snippet}")

    if not lines:
        return ""

    return "\n".join(lines) + "\n\n"
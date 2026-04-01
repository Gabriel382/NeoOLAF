from __future__ import annotations

# Standard library imports
from typing import Dict, List, TypedDict

# Local imports
from neoolaf.grounding.rag.types import GroundingRequest, RetrievedItem


class GroundingGraphState(TypedDict, total=False):
    """
    State for the agentic semantic RAG workflow.
    """

    # Input request
    request: GroundingRequest

    # Available registry sources
    available_sources: List[str]

    # Selected sources for this request
    selected_sources: List[str]

    # Retrieved items by source
    retrieved_by_source: Dict[str, List[RetrievedItem]]

    # Flattened retrieved items
    retrieved_items: List[RetrievedItem]

    # Final grounding summary
    grounding_summary: str

    # Final merged context
    merged_context: Dict
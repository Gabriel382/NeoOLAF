from __future__ import annotations

# Standard library imports
from typing import TypedDict, List, Dict, Any

# Local imports
from neoolaf.domain.linguistic_expression import LinguisticExpression


class EnrichmentGraphState(TypedDict, total=False):
    """
    State passed through the Layer 2 LangGraph workflow.
    """

    # Input expression to enrich
    expression: LinguisticExpression

    # Which sources are selected
    selected_sources: List[str]

    # Raw source outputs
    wordnet_result: Dict[str, Any]
    wikipedia_result: Dict[str, Any]
    wikidata_result: Dict[str, Any]
    web_result: Dict[str, Any]

    # Merged evidence
    gathered_evidence: Dict[str, Any]

    # Final LLM synthesis
    enrichment_result: Dict[str, Any]
from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Local imports
from neoolaf.domain.linguistic_expression import LinguisticExpression


@dataclass
class EnrichmentEvidence:
    """
    Evidence object used to justify Layer 2 enrichment.

    This stores source-level evidence such as:
    - a WordNet definition
    - a Wikipedia summary
    - a Wikidata label/description
    - a web snippet
    - the raw LLM enrichment output
    """

    # Name of the source that produced the evidence
    source: str

    # Human-readable extracted content from the source
    content: str

    # Optional URL, entity ID, or model name
    reference: Optional[str] = None


@dataclass
class EnrichedExpression:
    """
    Layer 2 output: a linguistic expression enriched with lexical and semantic cues.

    In addition to the final aliases/synonyms/variants, this object stores
    provenance maps so we know where each lexical item came from.
    """

    # Original Layer 1 expression
    base_expression: LinguisticExpression

    # Final merged lexical enrichments
    aliases: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    lexical_variants: List[str] = field(default_factory=list)

    # Provenance maps for lexical items
    # Example:
    #   "stop" -> ["wordnet", "llm"]
    alias_sources: Dict[str, List[str]] = field(default_factory=dict)
    synonym_sources: Dict[str, List[str]] = field(default_factory=dict)
    lexical_variant_sources: Dict[str, List[str]] = field(default_factory=dict)

    # Final semantic enrichment
    definition: Optional[str] = None
    ontology_hints: List[str] = field(default_factory=list)

    # Full evidence list used during enrichment
    enrichment_evidence: List[EnrichmentEvidence] = field(default_factory=list)
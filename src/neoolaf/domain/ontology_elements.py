from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.linguistic_expression import Evidence


@dataclass
class ConceptCandidate:
    """
    Layer 6 output: candidate ontology concept induced from Layer 3/5 artifacts.
    """

    # Stable identifier for the concept candidate
    concept_id: str

    # Ontology-oriented label
    label: str

    # Optional normalized label
    normalized_label: str

    # Optional short semantic description
    description: Optional[str] = None

    # Optional concept kind, for example:
    # component, failure, symptom, process, resource, event-type
    concept_kind: Optional[str] = None

    # Optional parent concept hint for later hierarchisation
    parent_hint: Optional[str] = None

    # Provenance candidate IDs that led to this concept
    source_candidate_ids: List[str] = field(default_factory=list)

    # Optional supporting triple IDs
    source_triple_ids: List[str] = field(default_factory=list)

    # Optional confidence score
    confidence: Optional[float] = None

    # Short explanation of why promotion was proposed
    justification: str = ""

    # Supporting evidence snippets
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class OntologyRelationCandidate:
    """
    Layer 6 output: candidate ontology relation induced from Layer 3/5 artifacts.
    """

    # Stable identifier for the ontology relation candidate
    relation_id: str

    # Ontology-oriented relation label
    label: str

    # Optional normalized label
    normalized_label: str

    # Optional semantic description
    description: Optional[str] = None

    # Optional domain/range hints for later axiom layers
    domain_hint: Optional[str] = None
    range_hint: Optional[str] = None

    # Provenance candidate IDs that led to this relation
    source_candidate_ids: List[str] = field(default_factory=list)

    # Supporting triple IDs
    source_triple_ids: List[str] = field(default_factory=list)

    # Optional confidence score
    confidence: Optional[float] = None

    # Short explanation of why promotion was proposed
    justification: str = ""

    # Supporting evidence snippets
    evidence: List[Evidence] = field(default_factory=list)
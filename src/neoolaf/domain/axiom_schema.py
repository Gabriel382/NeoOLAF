from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.linguistic_expression import Evidence


@dataclass
class AxiomSchemaCandidate:
    """
    Layer 8 output: reusable structural axiom schema candidate.

    Examples:
    - causes domain -> FailureEvent
    - causes range -> State
    - BearingFailure subclassOf MechanicalFailure
    """

    # Stable identifier for the axiom schema
    schema_id: str

    # Schema type, for example:
    # relation_domain, relation_range, subclass
    schema_type: str

    # Main subject of the schema
    subject_id: str
    subject_label: str

    # Optional predicate / role inside the schema
    predicate: str

    # Main object / target of the schema
    object_id: str
    object_label: str

    # Short explanation of the schema
    justification: str

    # Optional confidence score
    confidence: Optional[float] = None

    # Provenance source IDs
    source_relation_ids: List[str] = field(default_factory=list)
    source_concept_ids: List[str] = field(default_factory=list)
    source_triple_ids: List[str] = field(default_factory=list)

    # Supporting evidence
    evidence: List[Evidence] = field(default_factory=list)
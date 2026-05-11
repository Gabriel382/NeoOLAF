from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.linguistic_expression import Evidence


@dataclass
class GeneralAxiomCandidate:
    """
    Layer 9 output: candidate general ontology axiom.

    Examples:
    - OverheatingEvent SubClassOf FailureEvent
    - causes domain FailureEvent
    - causes range State
    - ThermalFailure rdfs:description "A failure concept related to abnormal heating."
    """

    # Stable identifier for the candidate axiom
    axiom_id: str

    # Axiom family, for example:
    # subclass, relation_domain, relation_range, description
    axiom_type: str

    # Subject of the axiom
    subject_id: str
    subject_label: str

    # Predicate / ontology operator
    predicate: str

    # Object of the axiom, if any
    object_id: Optional[str] = None
    object_label: Optional[str] = None

    # Literal description if the axiom is textual
    literal_value: Optional[str] = None

    # Explanation of why this axiom was generated
    justification: str = ""

    # Optional confidence score
    confidence: Optional[float] = None

    # Provenance source IDs
    source_schema_ids: List[str] = field(default_factory=list)
    source_concept_ids: List[str] = field(default_factory=list)
    source_relation_ids: List[str] = field(default_factory=list)

    # Supporting evidence
    evidence: List[Evidence] = field(default_factory=list)
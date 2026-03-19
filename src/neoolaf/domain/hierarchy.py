from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.linguistic_expression import Evidence


@dataclass
class ConceptHierarchyLink:
    """
    Layer 7 output: one concept hierarchy relation.

    This represents a candidate subclass-style relation between two
    promoted concept candidates.
    """

    # Stable identifier of the hierarchy link
    link_id: str

    # Child concept
    child_concept_id: str
    child_label: str

    # Parent concept
    parent_concept_id: str
    parent_label: str

    # Explanation for the hierarchy decision
    justification: str

    # Optional confidence score
    confidence: Optional[float] = None

    # Supporting evidence
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class RelationHierarchyLink:
    """
    Layer 7 output: one relation hierarchy relation.

    This represents a candidate subrelation-style relation between two
    promoted ontology relation candidates.
    """

    # Stable identifier of the hierarchy link
    link_id: str

    # Child relation
    child_relation_id: str
    child_label: str

    # Parent relation
    parent_relation_id: str
    parent_label: str

    # Explanation for the hierarchy decision
    justification: str

    # Optional confidence score
    confidence: Optional[float] = None

    # Supporting evidence
    evidence: List[Evidence] = field(default_factory=list)
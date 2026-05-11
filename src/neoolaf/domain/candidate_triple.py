from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.linguistic_expression import Evidence


@dataclass
class CandidateTriple:
    """
    Layer 5 output: candidate triple with provenance and confidence.

    This is the first explicit graph-like structure produced by NeoOLAF.
    It represents a candidate factual assertion of the form:

        (subject, predicate, object, provenance, confidence)
    """

    # Stable identifier of the candidate triple
    triple_id: str

    # Subject node
    subject_id: str
    subject_label: str
    subject_type: str

    # Predicate / relation
    predicate_id: str
    predicate_label: str

    # Object node
    object_id: str
    object_label: str
    object_type: str

    # Chunk where this triple was extracted
    chunk_id: str

    # Explanation of why this triple exists
    justification: str

    # Optional confidence score
    confidence: Optional[float] = None

    # Provenance evidence from the original text
    provenance: List[Evidence] = field(default_factory=list)
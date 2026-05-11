from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.linguistic_expression import Evidence


@dataclass
class CandidateRelationAssertion:
    """
    Layer 4 output: a candidate relation assertion connecting canonical candidates.

    This object represents a relation candidate linked to a source candidate
    and a target candidate, with supporting evidence and confidence.

    It is not yet the final graph triple representation. It is the structured
    relation extraction output that will later feed triple generation.
    """

    # Stable identifier for the relation assertion
    assertion_id: str

    # Candidate relation used in this assertion
    relation_candidate_id: str
    relation_label: str

    # Source candidate
    source_candidate_id: str
    source_candidate_label: str
    source_candidate_type: str

    # Target candidate
    target_candidate_id: str
    target_candidate_label: str
    target_candidate_type: str

    # Chunk where this relation assertion was detected
    chunk_id: str

    # Short explanation for why this relation was extracted
    justification: str

    # Optional confidence score
    confidence: Optional[float] = None

    # Supporting evidence from the original text
    evidence: List[Evidence] = field(default_factory=list)
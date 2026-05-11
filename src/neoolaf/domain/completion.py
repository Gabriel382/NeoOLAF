from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.linguistic_expression import Evidence
from neoolaf.domain.candidate_triple import CandidateTriple
from neoolaf.domain.general_axiom import GeneralAxiomCandidate


@dataclass
class CompletionCandidate:
    """
    Layer 11 output: one completion candidate added after validation/reasoning.

    This can represent either:
    - a completed graph triple
    - a completed ontology/general axiom
    """

    # Stable identifier for the completion candidate
    completion_id: str

    # Completion family, for example:
    # graph_completion, ontology_completion, type_link_completion
    completion_type: str

    # Short explanation of the completion
    justification: str

    # Optional confidence score
    confidence: Optional[float] = None

    # Optional completed triple
    completed_triple: Optional[CandidateTriple] = None

    # Optional completed ontology/general axiom
    completed_axiom: Optional[GeneralAxiomCandidate] = None

    # Supporting evidence
    evidence: List[Evidence] = field(default_factory=list)
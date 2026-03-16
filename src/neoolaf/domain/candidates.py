from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.enriched_expression import EnrichedExpression
from neoolaf.domain.linguistic_expression import Evidence


@dataclass
class CandidateMention:
    """
    One mention attached to a canonical candidate.

    This preserves traceability from the canonical candidate back to the
    original text positions extracted earlier in the pipeline.
    """

    # ID of the source linguistic expression
    expr_id: str

    # Original mention text
    text: str

    # Evidence collected at Layer 1
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class BaseCandidate:
    """
    Base class shared by all typed semantic candidates.
    """

    # Stable candidate identifier
    candidate_id: str

    # Canonical label selected for this candidate
    canonical_label: str

    # Normalized form used for matching and resolution
    normalized_label: str

    # Candidate type: entity / relation / attribute / event
    candidate_type: str

    # Mentions merged into this canonical candidate
    mentions: List[CandidateMention] = field(default_factory=list)

    # Optional confidence assigned by the typing layer
    confidence: Optional[float] = None

    # Optional ontology hints propagated from Layer 2
    ontology_hints: List[str] = field(default_factory=list)

    # Optional definition propagated from Layer 2
    definition: Optional[str] = None

    # Optional aliases and synonyms from Layer 2
    aliases: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    lexical_variants: List[str] = field(default_factory=list)


@dataclass
class EntityCandidate(BaseCandidate):
    """
    Canonical entity candidate.
    """
    pass


@dataclass
class RelationCandidate(BaseCandidate):
    """
    Canonical relation candidate.
    """
    pass


@dataclass
class AttributeCandidate(BaseCandidate):
    """
    Canonical attribute/value candidate.
    """
    pass


@dataclass
class EventCandidate(BaseCandidate):
    """
    Canonical event/state candidate.
    """
    pass
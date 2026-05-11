from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class TypingExample:
    """
    Example used to guide candidate typing decisions.
    """

    # Example text span
    text: str

    # Expected type: entity / relation / attribute / event
    expected_type: str

    # Optional short explanation
    explanation: Optional[str] = None


@dataclass
class RelationExample:
    """
    Example used to guide relation extraction decisions.
    """

    # Source text or local context
    text: str

    # Example source label
    source_label: str

    # Example relation label
    relation_label: str

    # Example target label
    target_label: str

    # Optional explanation
    explanation: Optional[str] = None


@dataclass
class PromotionExample:
    """
    Example used to guide concept / relation promotion decisions.
    """

    # Candidate label or expression
    text: str

    # Whether it should be promoted
    promote: bool

    # Optional expected promoted label
    promoted_label: Optional[str] = None

    # Optional explanation
    explanation: Optional[str] = None


@dataclass
class NegativeExample:
    """
    Example used to show what should not be extracted / typed / promoted.
    """

    # Example text
    text: str

    # Reason why it is negative
    explanation: Optional[str] = None

    # Optional layer name where it matters most
    target_layer: Optional[str] = None


@dataclass
class UserGuidance:
    """
    User guidance injected into NeoOLAF layers.

    This object now supports:
    - semantic instructions
    - examples
    - ontology depth preference
    - executable policy settings
    """

    # Domain and semantic framing
    domain_focus: Optional[str] = None
    abstraction_level: Optional[str] = None
    priority_relations: List[str] = field(default_factory=list)
    population_policy: Optional[str] = None
    event_modeling_preference: Optional[str] = None

    # Example-based guidance
    typing_examples: List[TypingExample] = field(default_factory=list)
    relation_examples: List[RelationExample] = field(default_factory=list)
    promotion_examples: List[PromotionExample] = field(default_factory=list)
    negative_examples: List[NegativeExample] = field(default_factory=list)

    # Ontology depth preference
    # shallow = fewer promoted concepts, flatter schema
    # balanced = default
    # deep = more abstraction, more hierarchy, more schema promotion
    ontology_depth: Literal["shallow", "balanced", "deep"] = "balanced"

    # Executable policy parameters
    # Minimum confidence to accept promotion in concept/relation induction
    promotion_min_confidence: float = 0.50

    # Hierarchy aggressiveness:
    # lower -> fewer hierarchy links
    # higher -> more hierarchy links
    hierarchy_min_confidence: float = 0.50

    # Concept-vs-instance bias:
    # higher -> more concept promotion
    # lower -> more conservative, instance-like interpretation
    concept_promotion_bias: float = 0.50
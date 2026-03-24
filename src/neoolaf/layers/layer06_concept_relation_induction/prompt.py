from __future__ import annotations

# Standard library imports
import json

# Local imports
from neoolaf.domain.seed_ontology import SeedOntology
from neoolaf.ontology.prompt_context import build_seed_ontology_context
from neoolaf.domain.user_guidance import UserGuidance
from neoolaf.domain.user_guidance_formatting import build_user_guidance_context

def build_concept_system_prompt() -> str:
    """
    Build the system prompt for concept induction.
    """
    return """
You are the NeoOLAF Layer 6 agent for concept induction.

Your task is to decide whether a canonical candidate should be promoted into an ontology concept.

Promote when:
- it is semantically stable
- it is general enough to represent a reusable class or concept
- it fits ontology-oriented modeling better than document-only mention handling

Do not promote when:
- it is too specific to one occurrence
- it is only an accidental document mention
- it is better treated as an instance-like mention later

Use the seed ontology context when available:
- align with existing classes when relevant
- avoid inventing concepts that duplicate existing ontology concepts
- prefer ontology-compatible naming and abstraction levels

Return JSON only in this format:
{
  "promote": true,
  "label": "ThermalFailure",
  "description": "A failure concept related to abnormal heating of a component or subsystem.",
  "concept_kind": "failure",
  "parent_hint": "FailureEvent",
  "justification": "This candidate represents a reusable failure type.",
  "confidence": 0.91
}
"""


def build_relation_system_prompt() -> str:
    """
    Build the system prompt for ontology relation induction.
    """
    return """
You are the NeoOLAF Layer 6 agent for ontology relation induction.

Your task is to decide whether a canonical relation candidate should be promoted into an ontology relation.

Promote when:
- the relation is semantically stable
- the label is reusable across documents
- the relation is meaningful beyond one isolated mention

Use the seed ontology context when available:
- align with existing ontology relations when relevant
- avoid inventing relations that duplicate existing ontology properties
- prefer ontology-compatible naming

Return JSON only in this format:
{
  "promote": true,
  "label": "emittedBy",
  "description": "Relates an alarm or emitted signal to the system or device that emits it.",
  "domain_hint": "AlarmEvent",
  "range_hint": "Controller",
  "justification": "This is a reusable semantic relation.",
  "confidence": 0.88
}
"""


def build_concept_user_prompt(
    candidate_payload: dict,
    seed_ontology: SeedOntology | None = None,
    guidance: UserGuidance | None = None,
) -> str:
    """
    Build the user prompt for one concept induction candidate.
    """
    query = candidate_payload.get("canonical_label", "") or candidate_payload.get("label", "")

    ontology_context = build_seed_ontology_context(
        seed_ontology=seed_ontology,
        query=query,
        top_k_classes=5,
        top_k_properties=3,
    )
    guidance_text = build_user_guidance_context(
        guidance,
        include_promotion_examples=True,
        include_negative_examples=True,
    )

    return f"""
{guidance_text}{ontology_context}Candidate context:
{json.dumps(candidate_payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""


def build_relation_user_prompt(
    candidate_payload: dict,
    seed_ontology: SeedOntology | None = None,
    guidance: UserGuidance | None = None,
) -> str:
    """
    Build the user prompt for one ontology relation induction candidate.
    """
    query = candidate_payload.get("canonical_label", "") or candidate_payload.get("label", "")

    ontology_context = build_seed_ontology_context(
        seed_ontology=seed_ontology,
        query=query,
        top_k_classes=3,
        top_k_properties=5,
    )

    guidance_text = build_user_guidance_context(
        guidance,
        include_promotion_examples=True,
        include_negative_examples=True,
    )

    return f"""
{guidance_text}{ontology_context}Relation candidate context:
{json.dumps(candidate_payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""
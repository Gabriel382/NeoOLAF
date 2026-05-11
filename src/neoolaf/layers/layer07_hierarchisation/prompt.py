from __future__ import annotations

# Standard library imports
import json

# Local imports
from neoolaf.domain.seed_ontology import SeedOntology
from neoolaf.ontology.prompt_context import build_seed_ontology_context


def build_concept_hierarchy_system_prompt() -> str:
    """
    Build the system prompt for concept hierarchisation.
    """
    return """
You are the NeoOLAF Layer 7 agent for concept hierarchisation.

Your task is to decide whether one concept candidate should be placed under another
in a concept hierarchy.

Interpret this as:
    child ⊑ parent

Use a semantic subclass interpretation:
- the child must be more specific than the parent
- the parent must be more general than the child
- the relation must make sense as a reusable ontology hierarchy link

Use the seed ontology context when available:
- prefer hierarchy placements compatible with the source ontology
- avoid hierarchy links that contradict or duplicate obvious ontology structure
- use ontology context to choose more plausible parent placement

Return JSON only in this format:
{
  "is_subclass": true,
  "justification": "BearingFailure is a more specific kind of MechanicalFailure.",
  "confidence": 0.92
}
"""


def build_relation_hierarchy_system_prompt() -> str:
    """
    Build the system prompt for relation hierarchisation.
    """
    return """
You are the NeoOLAF Layer 7 agent for relation hierarchisation.

Your task is to decide whether one ontology relation candidate should be placed under another
in a relation hierarchy.

Interpret this as:
    child ⊑ parent

Use a semantic subrelation interpretation:
- the child relation must be more specific than the parent relation
- the parent relation must be more general than the child relation
- the relation must make sense as a reusable ontology hierarchy link

Use the seed ontology context when available:
- prefer hierarchy placements compatible with the source ontology
- avoid hierarchy links that contradict or duplicate existing ontology property structure

Return JSON only in this format:
{
  "is_subrelation": true,
  "justification": "emittedBy is more specific than producedBy.",
  "confidence": 0.88
}
"""


def build_concept_hierarchy_user_prompt(
    child_payload: dict,
    parent_payload: dict,
    seed_ontology=None,
    grounding_context: str = "",
) -> str:
    """
    Build the user prompt for one concept hierarchy decision.
    """
    query = f"{child_payload.get('label', '')} {parent_payload.get('label', '')}"

    ontology_context = build_seed_ontology_context(
        seed_ontology=seed_ontology,
        query=query,
        top_k_classes=6,
        top_k_properties=2,
    )

    payload = {
        "child_concept": child_payload,
        "parent_concept": parent_payload,
    }

    return f"""
{ontology_context}{grounding_context}Evaluate the following candidate concept hierarchy relation.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""


def build_relation_hierarchy_user_prompt(
    child_payload: dict,
    parent_payload: dict,
    seed_ontology=None,
    grounding_context: str = "",
) -> str:
    """
    Build the user prompt for one relation hierarchy decision.
    """
    query = f"{child_payload.get('label', '')} {parent_payload.get('label', '')}"

    ontology_context = build_seed_ontology_context(
        seed_ontology=seed_ontology,
        query=query,
        top_k_classes=2,
        top_k_properties=6,
    )

    payload = {
        "child_relation": child_payload,
        "parent_relation": parent_payload,
    }

    return f"""
{ontology_context}{grounding_context}Evaluate the following candidate relation hierarchy relation.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""
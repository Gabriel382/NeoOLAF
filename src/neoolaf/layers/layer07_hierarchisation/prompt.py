from __future__ import annotations

# Standard library imports
import json


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

Return JSON only in this format:
{
  "is_subrelation": true,
  "justification": "emittedBy is more specific than producedBy.",
  "confidence": 0.88
}
"""


def build_concept_hierarchy_user_prompt(child_payload: dict, parent_payload: dict) -> str:
    """
    Build the user prompt for one concept hierarchy decision.
    """
    payload = {
        "child_concept": child_payload,
        "parent_concept": parent_payload,
    }

    return f"""
Evaluate the following candidate concept hierarchy relation.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""


def build_relation_hierarchy_user_prompt(child_payload: dict, parent_payload: dict) -> str:
    """
    Build the user prompt for one relation hierarchy decision.
    """
    payload = {
        "child_relation": child_payload,
        "parent_relation": parent_payload,
    }

    return f"""
Evaluate the following candidate relation hierarchy relation.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""
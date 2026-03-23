from __future__ import annotations

# Standard library imports
import json

# Local imports
from neoolaf.domain.seed_ontology import SeedOntology
from neoolaf.ontology.prompt_context import build_seed_ontology_context


def build_relation_schema_system_prompt() -> str:
    """
    Build the system prompt for relation-based axiom schemata extraction.
    """
    return """
You are the NeoOLAF Layer 8 agent for axiom schemata extraction.

Your task is to extract reusable structural patterns from ontology relation candidates
and their supporting triple patterns.

Focus on reusable schemata such as:
- relation domain
- relation range

Use the seed ontology context when available:
- prefer schema labels compatible with the source ontology
- align domain/range patterns with existing ontology classes when possible
- avoid producing schema patterns that obviously contradict the ontology

Return JSON only in this format:
{
  "emit_domain_schema": true,
  "domain_label": "FailureEvent",
  "emit_range_schema": true,
  "range_label": "State",
  "justification": "The relation causes repeatedly connects failure events to resulting states.",
  "confidence": 0.89
}
"""


def build_subclass_schema_system_prompt() -> str:
    """
    Build the system prompt for subclass schema extraction.
    """
    return """
You are the NeoOLAF Layer 8 agent for axiom schemata extraction.

Your task is to convert a concept hierarchy relation into a reusable subclass schema
when it is semantically valid.

Use the seed ontology context when available:
- prefer subclass schemata that are compatible with known source ontology class structure
- avoid subclass schemata that contradict obvious ontology organization

Return JSON only in this format:
{
  "emit_subclass_schema": true,
  "justification": "BearingFailure is a reusable subclass of MechanicalFailure.",
  "confidence": 0.93
}
"""


def build_relation_schema_user_prompt(
    payload: dict,
    seed_ontology: SeedOntology | None = None,
) -> str:
    """
    Build the user prompt for relation schema extraction.
    """
    relation_label = payload.get("relation_candidate", {}).get("label", "")

    ontology_context = build_seed_ontology_context(
        seed_ontology=seed_ontology,
        query=relation_label,
        top_k_classes=5,
        top_k_properties=5,
    )

    return f"""
{ontology_context}Extract reusable structural schemata from the following relation context.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""


def build_subclass_schema_user_prompt(
    payload: dict,
    seed_ontology: SeedOntology | None = None,
) -> str:
    """
    Build the user prompt for subclass schema extraction.
    """
    query = f"{payload.get('child_label', '')} {payload.get('parent_label', '')}"

    ontology_context = build_seed_ontology_context(
        seed_ontology=seed_ontology,
        query=query,
        top_k_classes=6,
        top_k_properties=2,
    )

    return f"""
{ontology_context}Extract reusable subclass schemata from the following hierarchy context.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""
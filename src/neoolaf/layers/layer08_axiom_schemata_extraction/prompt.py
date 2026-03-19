from __future__ import annotations

# Standard library imports
import json


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

Return JSON only in this format:
{
  "emit_subclass_schema": true,
  "justification": "BearingFailure is a reusable subclass of MechanicalFailure.",
  "confidence": 0.93
}
"""


def build_relation_schema_user_prompt(payload: dict) -> str:
    """
    Build the user prompt for relation schema extraction.
    """
    return f"""
Extract reusable structural schemata from the following relation context.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""


def build_subclass_schema_user_prompt(payload: dict) -> str:
    """
    Build the user prompt for subclass schema extraction.
    """
    return f"""
Extract reusable subclass schemata from the following hierarchy context.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""
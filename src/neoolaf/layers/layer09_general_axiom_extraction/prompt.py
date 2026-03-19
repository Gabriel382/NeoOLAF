from __future__ import annotations

# Standard library imports
import json


def build_axiom_system_prompt() -> str:
    """
    Build the system prompt for general axiom extraction.
    """
    return """
You are the NeoOLAF Layer 9 agent for general axiom extraction.

Your task is to transform reusable structural schemata and induced ontology candidates
into candidate general ontology axioms.

Typical outputs include:
- subclass axioms
- relation domain axioms
- relation range axioms
- textual description axioms using rdfs:description

Important rule:
all ontology entities, concepts, and relations should receive a meaningful rdfs:description.

Return JSON only in this format:
{
  "emit_axiom": true,
  "axiom_type": "subclass",
  "predicate": "subClassOf",
  "object_label": "FailureEvent",
  "literal_value": null,
  "justification": "OverheatingEvent is a reusable subclass of FailureEvent.",
  "confidence": 0.93
}

For description axioms:
{
  "emit_axiom": true,
  "axiom_type": "description",
  "predicate": "rdfs:description",
  "object_label": null,
  "literal_value": "A failure concept related to abnormal heating of a component or subsystem.",
  "justification": "This textual description improves interpretability.",
  "confidence": 0.95
}
"""


def build_axiom_user_prompt(payload: dict) -> str:
    """
    Build the user prompt for one general axiom extraction decision.
    """
    return f"""
Generate a candidate general ontology axiom from the following context.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""
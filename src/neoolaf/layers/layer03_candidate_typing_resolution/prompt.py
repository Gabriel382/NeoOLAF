from __future__ import annotations

# Standard library imports
import json

# Local imports
from neoolaf.domain.enriched_expression import EnrichedExpression
from neoolaf.domain.user_guidance import UserGuidance


def build_system_prompt() -> str:
    """
    Build the system prompt for Layer 3 typing and resolution.

    This version gives much stronger guidance for recognizing relation candidates.
    """
    return """
You are the NeoOLAF Layer 3 agent: Candidate Typing and Resolution.

Your task is to assign one provisional semantic type to an enriched expression.

Allowed types:
- entity
- relation
- attribute
- event

Type definitions:
- entity:
  a machine, component, resource, actor, device, object, or identifiable thing
- relation:
  a linking phrase, semantic connector, verbal predicate, verbal group,
  prepositional relation, classification phrase, causality phrase,
  part-whole phrase, or dependency-like expression that can connect entities or events
- attribute:
  a measurable value, property, threshold, quality, or state-value expression
- event:
  a failure, alarm, degradation, shutdown, detection, occurrence, process event, or state occurrence

Important guidance for relations:
Relation candidates often look like:
- is divided into
- emitted by
- indicates
- compromises
- classified in
- detected by
- caused by
- part of
- belongs to
- located in

If an expression mainly serves to connect things, classify it as relation.

You must also provide:
- a canonical label
- a short justification
- an optional confidence score

Return JSON only in this format:
{
  "candidate_type": "relation",
  "canonical_label": "emitted by",
  "justification": "This is a linking phrase that connects alarms to the PLC.",
  "confidence": 0.91
}
"""


def build_user_prompt(
    enriched_expression: EnrichedExpression,
    guidance: UserGuidance | None = None,
) -> str:
    """
    Build the user prompt for one enriched expression.
    """
    guidance_text = ""
    if guidance:
        parts = []
        if guidance.domain_focus:
            parts.append(f"Domain focus: {guidance.domain_focus}")
        if guidance.abstraction_level:
            parts.append(f"Abstraction level: {guidance.abstraction_level}")
        if guidance.priority_relations:
            parts.append(f"Priority relations: {', '.join(guidance.priority_relations)}")
        if guidance.population_policy:
            parts.append(f"Population policy: {guidance.population_policy}")
        if guidance.event_modeling_preference:
            parts.append(f"Event modeling preference: {guidance.event_modeling_preference}")
        if parts:
            guidance_text = "\n".join(parts) + "\n\n"

    payload = {
        "base_expression_text": enriched_expression.base_expression.text,
        "base_expression_label": enriched_expression.base_expression.label,
        "aliases": enriched_expression.aliases,
        "synonyms": enriched_expression.synonyms,
        "lexical_variants": enriched_expression.lexical_variants,
        "definition": enriched_expression.definition,
        "ontology_hints": enriched_expression.ontology_hints,
    }

    return f"""
{guidance_text}Enriched expression:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""
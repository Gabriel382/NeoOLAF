from __future__ import annotations

# Standard library imports
import json

# Local imports
from neoolaf.domain.linguistic_expression import LinguisticExpression
from neoolaf.domain.user_guidance import UserGuidance


def build_system_prompt() -> str:
    """
    System prompt for the Layer 2 enrichment agent.
    """
    return """
You are the NeoOLAF Layer 2 agent: Candidate Enrichment.

Your role is to enrich one linguistic expression using external lexical and conceptual evidence.

You must produce:
- aliases
- synonyms
- lexical variants
- a short definition
- ontology-compatible hints

Important rules:
- prefer lexical items directly supported by the evidence
- keep only semantically relevant enrichments
- do not invent unsupported aliases or synonyms
- if multiple sources disagree, keep the safer and more general options
- avoid noisy web-specific phrases unless they clearly describe the same concept

Return JSON only with this format:
{
  "aliases": ["..."],
  "synonyms": ["..."],
  "lexical_variants": ["..."],
  "definition": "...",
  "ontology_hints": ["..."]
}
"""


def build_user_prompt(
    expression: LinguisticExpression,
    gathered_evidence: dict,
    guidance: UserGuidance | None = None,
) -> str:
    """
    Build the user prompt for one expression enrichment.
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

    return f"""
{guidance_text}Expression:
- text: {expression.text}
- label: {expression.label}
- justification: {expression.justification}

Evidence:
{json.dumps(gathered_evidence, indent=2, ensure_ascii=False)}

Return JSON only.
"""
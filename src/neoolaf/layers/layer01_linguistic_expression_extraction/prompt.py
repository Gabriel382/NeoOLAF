from __future__ import annotations

# Local imports
from neoolaf.domain.documents import DocumentChunk
from neoolaf.domain.user_guidance import UserGuidance


def build_system_prompt() -> str:
    """
    Build the system prompt for Layer 1 linguistic expression extraction.

    This version explicitly asks for:
    - entity-like expressions
    - event/state expressions
    - attribute/value expressions
    - relation-bearing expressions

    The goal is to avoid missing verbal and linking phrases that will later
    become relation candidates in Layer 3 and Layer 4.
    """
    return """
You are the NeoOLAF Layer 1 agent: Linguistic Expression Extraction.

Your goal is to extract linguistically relevant expressions from technical text that may later become ontology elements, graph assertions, or both.

You must extract not only important terms, but also relation-bearing expressions.

Focus on expressions related to:
- components
- resources
- machines
- failures
- alarms
- actions
- measurements
- operational states
- symptoms
- events
- process-relevant terms
- linking phrases and relation-bearing expressions

Relation-bearing expressions are especially important.
These include:
- verbal predicates
- verbal groups
- prepositional linking expressions
- classification phrases
- causality phrases
- part-whole phrases
- observed-by / emitted-by / caused-by / located-in style expressions

Examples of relation-bearing expressions:
- is divided into
- emitted by
- indicates
- causes
- compromises
- belongs to
- classified in
- detected by
- located in
- part of

Return JSON only.
Use exactly this format:

{
  "expressions": [
    {
      "text": "bearing overheating",
      "label": "failure symptom",
      "justification": "This expression describes an important abnormal state of a machine component."
    },
    {
      "text": "emitted by",
      "label": "relation-bearing expression",
      "justification": "This expression links alarms to the PLC."
    }
  ]
}

Rules:
- extract useful semantic expressions
- include relation-bearing phrases when present
- keep expressions as close as possible to the source text
- do not paraphrase unless necessary
- do not return duplicates
- keep expressions short and meaningful
- do not explain outside JSON
"""


def build_user_prompt(chunk: DocumentChunk, guidance: UserGuidance | None = None) -> str:
    """
    Build the user prompt for one chunk.

    Optional user guidance is included when available.
    """
    guidance_text = ""
    if guidance:
        guidance_lines = []
        if guidance.domain_focus:
            guidance_lines.append(f"Domain focus: {guidance.domain_focus}")
        if guidance.abstraction_level:
            guidance_lines.append(f"Abstraction level: {guidance.abstraction_level}")
        if guidance.priority_relations:
            guidance_lines.append(f"Priority relations: {', '.join(guidance.priority_relations)}")
        if guidance.population_policy:
            guidance_lines.append(f"Population policy: {guidance.population_policy}")
        if guidance.event_modeling_preference:
            guidance_lines.append(f"Event modeling preference: {guidance.event_modeling_preference}")

        if guidance_lines:
            guidance_text = "\n".join(guidance_lines) + "\n\n"

    return f"""
{guidance_text}Chunk ID: {chunk.chunk_id}

Text:
\"\"\"
{chunk.text}
\"\"\"

Extract the most relevant linguistic expressions from this chunk.

Important:
- include entities, events, states, attributes, and relation-bearing expressions
- relation-bearing expressions should be extracted explicitly when present

Return JSON only.
"""
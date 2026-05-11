from __future__ import annotations

# Local imports
from neoolaf.domain.documents import DocumentChunk
from neoolaf.domain.user_guidance import UserGuidance
from neoolaf.domain.seed_ontology import SeedOntology
from neoolaf.ontology.prompt_context import build_seed_ontology_context
from neoolaf.domain.user_guidance import UserGuidance
from neoolaf.domain.user_guidance_formatting import build_user_guidance_context

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


def build_user_prompt(
    chunk: DocumentChunk,
    guidance: UserGuidance | None = None,
    seed_ontology: SeedOntology | None = None,
) -> str:
    """
    Build the user prompt for one chunk.

    Optional user guidance is included when available.
    """
    guidance_text = ""
    if guidance:
        guidance_text = build_user_guidance_context(
            guidance,
            include_negative_examples=True,
        )


    ontology_context = build_seed_ontology_context(
        seed_ontology=seed_ontology,
        query=chunk.text[:300],
        top_k_classes=3,
        top_k_properties=3,
    )

    

    return f"""
{guidance_text}{ontology_context}Chunk ID: {chunk.chunk_id}

Text:
\"\"\"
{chunk.text}
\"\"\"

Extract the most relevant linguistic expressions from this chunk.

Important:
- include entities, events, states, attributes, and relation-bearing expressions
- relation-bearing expressions should be extracted explicitly when present
- do not extract expressions similar to the provided negative examples when they are semantically unhelpful

Return JSON only.
"""
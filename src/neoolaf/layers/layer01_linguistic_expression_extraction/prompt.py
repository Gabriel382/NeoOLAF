from __future__ import annotations

from neoolaf.domain.documents import DocumentChunk
from neoolaf.domain.user_guidance import UserGuidance


def build_system_prompt() -> str:
    return """
You are the NeoOLAF Layer 1 agent: Linguistic Expression Extraction.

Your goal is to extract linguistically relevant expressions from technical text that may later become ontology elements, graph assertions, or both.

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

Return JSON only.
Use exactly this format:

{
  "expressions": [
    {
      "text": "bearing overheating",
      "label": "failure symptom",
      "justification": "This expression describes an important abnormal state of a machine component."
    }
  ]
}

Rules:
- extract only useful semantic expressions
- do not return duplicates
- keep expressions short and meaningful
- do not explain outside JSON
"""


def build_user_prompt(chunk: DocumentChunk, guidance: UserGuidance | None = None) -> str:
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
Return JSON only.
"""
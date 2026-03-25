from __future__ import annotations

# Standard library imports
import json
from typing import List


def build_source_selection_system_prompt() -> str:
    """
    Prompt for choosing retrieval sources.
    """
    return """
You are a NeoOLAF SemanticRAG source-selection agent.

Choose the most relevant retrieval sources for the given request.

Possible sources include:
- ontology
- artifacts
- web
- wikidata
- wikipedia
- wordnet

Use only the sources that are useful for the request.

Return JSON only:
{
  "selected_sources": ["ontology", "wikidata", "artifacts"]
}
"""


def build_source_selection_user_prompt(request_payload: dict, available_sources: List[str]) -> str:
    """
    User prompt for source selection.
    """
    payload = {
        "request": request_payload,
        "available_sources": available_sources,
    }

    return f"""
Select the best retrieval sources for this grounding request.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""


def build_grounding_summary_system_prompt() -> str:
    """
    Prompt for grounding summary generation.
    """
    return """
You are a NeoOLAF SemanticRAG grounding summarizer.

Summarize the retrieved evidence in a concise way that helps the downstream layer.
Preserve:
- ontology-compatible hints
- lexical grounding
- relation grounding
- structural grounding

Return JSON only:
{
  "grounding_summary": "...",
  "merged_context": {
    "candidate_hints": ["..."],
    "relation_hints": ["..."],
    "ontology_hints": ["..."]
  }
}
"""
from __future__ import annotations

# Standard library imports
import json
from typing import List, Dict


def build_system_prompt() -> str:
    """
    Build the system prompt for Layer 4 candidate relation extraction.
    """
    return """
You are the NeoOLAF Layer 4 agent: Candidate Relation Extraction.

Your task is to identify whether a relation candidate connects two canonical candidates in a chunk.

Inputs:
- one chunk of text
- one relation candidate
- a set of canonical entity/event candidates present in the same chunk

Your job:
- decide whether the relation candidate connects any pair of candidates
- if yes, identify the best source candidate and target candidate
- keep the direction meaningful
- return one JSON object if a good relation is found
- return {"found": false} if no good relation can be established

Important:
- only use candidates explicitly provided
- prefer candidates that are clearly mentioned in the chunk
- prefer relations that are semantically plausible
- do not invent missing nodes
- focus on entities and events
- ignore weak or ambiguous links if the chunk does not support them

Return JSON only in this format:

{
  "found": true,
  "source_candidate_id": "cand_e_00001",
  "target_candidate_id": "cand_s_00003",
  "justification": "The text states that the alarm is emitted by the PLC.",
  "confidence": 0.91
}

or

{
  "found": false
}
"""


def build_user_prompt(
    chunk_text: str,
    chunk_id: str,
    relation_candidate: Dict,
    local_candidates: List[Dict],
) -> str:
    """
    Build the user prompt for one relation candidate within one chunk.
    """
    payload = {
        "chunk_id": chunk_id,
        "relation_candidate": relation_candidate,
        "local_candidates": local_candidates,
        "chunk_text": chunk_text,
    }

    return f"""
Analyze the following relation extraction context.

{json.dumps(payload, indent=2, ensure_ascii=False)}

Return JSON only.
"""
"""
Domain objects for Layer 1 linguistic expressions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Evidence:
    """
    Evidence object used to trace where an extracted expression came from.
    """

    # Identifier of the chunk containing the expression
    chunk_id: str

    # Local position inside the chunk text
    chunk_start_char: int
    chunk_end_char: int

    # Global position inside the cleaned document text
    doc_start_char: int
    doc_end_char: int

    # Local textual snippet for quick inspection
    snippet: str
    
@dataclass
class LinguisticExpression:
    """
    Structured linguistic expression extracted from the document.
    This is the main output of Layer 1.
    """

    # Unique identifier for the expression
    expr_id: str

    # Surface form extracted from the text
    text: str

    # Short semantic label assigned by the LLM
    label: str

    # Justification explaining why the expression is relevant
    justification: str

    # Provenance evidence showing where the expression appears
    evidence: List[Evidence] = field(default_factory=list)

    # Optional confidence score
    confidence: Optional[float] = None
from __future__ import annotations

# Pydantic imports
from pydantic import BaseModel, Field
from typing import Literal


class TypedCandidateResponse(BaseModel):
    """
    Expected LLM output for candidate typing.
    """

    # One of the four provisional semantic types
    candidate_type: Literal["entity", "relation", "attribute", "event"] = Field(
        ...,
        description="Semantic type assigned to the enriched expression."
    )

    # Canonical label proposed by the LLM
    canonical_label: str = Field(
        ...,
        description="Short canonical label for the candidate."
    )

    # Short justification of the decision
    justification: str = Field(
        ...,
        description="Short explanation for the typing and canonicalization decision."
    )

    # Optional confidence in [0,1]
    confidence: float | None = Field(
        default=None,
        description="Confidence score for the typing decision."
    )
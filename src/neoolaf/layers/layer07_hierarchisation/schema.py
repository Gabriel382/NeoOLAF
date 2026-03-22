from __future__ import annotations

# Pydantic imports
from pydantic import BaseModel, Field
from typing import Optional


class ConceptHierarchyResponse(BaseModel):
    """
    Expected LLM output for one concept hierarchy decision.
    """

    # Whether the child concept is a subclass of the parent concept
    is_subclass: bool = Field(..., description="Whether the child is a subclass of the parent.")

    # Explanation of the decision
    justification: str = Field(..., description="Why this hierarchy decision is correct or not.")

    # Optional confidence score
    confidence: Optional[float] = Field(default=None, description="Confidence score.")


class RelationHierarchyResponse(BaseModel):
    """
    Expected LLM output for one relation hierarchy decision.
    """

    # Whether the child relation is a subrelation of the parent relation
    is_subrelation: bool = Field(..., description="Whether the child is a subrelation of the parent.")

    # Explanation of the decision
    justification: str = Field(..., description="Why this hierarchy decision is correct or not.")

    # Optional confidence score
    confidence: Optional[float] = Field(default=None, description="Confidence score.")
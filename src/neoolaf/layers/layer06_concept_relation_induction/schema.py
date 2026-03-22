from __future__ import annotations

# Pydantic imports
from pydantic import BaseModel, Field
from typing import Literal, Optional


class ConceptInductionResponse(BaseModel):
    """
    Expected LLM output for concept induction.
    """

    promote: bool = Field(..., description="Whether this candidate should be promoted to a concept.")
    label: str = Field(..., description="Ontology-oriented concept label.")
    description: Optional[str] = Field(default=None, description="Short concept description.")
    concept_kind: Optional[str] = Field(default=None, description="Optional concept category.")
    parent_hint: Optional[str] = Field(default=None, description="Optional parent concept hint.")
    justification: str = Field(..., description="Why this concept should or should not be promoted.")
    confidence: Optional[float] = Field(default=None, description="Confidence score.")


class RelationInductionResponse(BaseModel):
    """
    Expected LLM output for ontology relation induction.
    """

    promote: bool = Field(..., description="Whether this candidate should be promoted to an ontology relation.")
    label: str = Field(..., description="Ontology-oriented relation label.")
    description: Optional[str] = Field(default=None, description="Short relation description.")
    domain_hint: Optional[str] = Field(default=None, description="Optional domain hint.")
    range_hint: Optional[str] = Field(default=None, description="Optional range hint.")
    justification: str = Field(..., description="Why this relation should or should not be promoted.")
    confidence: Optional[float] = Field(default=None, description="Confidence score.")
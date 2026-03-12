from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class ExtractedExpressionItem(BaseModel):
    text: str = Field(..., description="The extracted linguistic expression.")
    label: str = Field(..., description="Short semantic label.")
    justification: str = Field(..., description="Why this expression is relevant.")


class ExtractedExpressionResponse(BaseModel):
    expressions: List[ExtractedExpressionItem]
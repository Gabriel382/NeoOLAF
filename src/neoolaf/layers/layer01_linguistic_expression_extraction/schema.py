from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List


class ExtractedExpressionItem(BaseModel):
    text: str = Field(..., description="The extracted linguistic expression.")
    label: str = Field(..., description="Short semantic label.")
    justification: str = Field(..., description="Why this expression is relevant.")


class ExtractedExpressionResponse(BaseModel):
    expressions: List[ExtractedExpressionItem]
# Structured output schemas used when profile-configured Pydantic validation is enabled.
# They are imported here for discoverability/backward compatibility with layer-local imports.
from neoolaf.schemas.structured_output import (  # noqa: E402,F401
    ExtractedItem,
    ReferenceItem,
    Layer01AlarmRecord,
    Layer01AlarmRecordOutput,
)

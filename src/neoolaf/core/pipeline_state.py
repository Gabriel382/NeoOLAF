from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

from neoolaf.domain.documents import Document
from neoolaf.domain.linguistic_expression import LinguisticExpression
from neoolaf.domain.user_guidance import UserGuidance


@dataclass
class PipelineState:
    """
    Shared pipeline state passed through all NeoOLAF layers.
    This object is the main contract between layers.
    """

    document: Document
    llm_model: str
    user_guidance: Optional[UserGuidance] = None

    # Directory where intermediate artifacts are stored
    artifact_dir: Optional[str] = None

    # Main layer outputs currently implemented
    linguistic_expressions: List[LinguisticExpression] = field(default_factory=list)

    # Execution logs
    logs: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """
        Append a message to the execution log.
        """
        self.logs.append(message)
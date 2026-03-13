from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.documents import Document
from neoolaf.domain.linguistic_expression import LinguisticExpression
from neoolaf.domain.enriched_expression import EnrichedExpression
from neoolaf.domain.user_guidance import UserGuidance


@dataclass
class PipelineState:
    """
    Shared pipeline state passed through all NeoOLAF layers.
    """

    # Current document being processed
    document: Document

    # LLM model name used by the pipeline
    llm_model: str

    # Optional semantic guidance
    user_guidance: Optional[UserGuidance] = None

    # Directory where intermediate artifacts are stored
    artifact_dir: Optional[str] = None

    # Layer 1 outputs
    linguistic_expressions: List[LinguisticExpression] = field(default_factory=list)

    # Layer 2 outputs
    enriched_expressions: List[EnrichedExpression] = field(default_factory=list)

    # Execution logs
    logs: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """
        Append a message to the execution log.
        """
        self.logs.append(message)
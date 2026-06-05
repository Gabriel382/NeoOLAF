from __future__ import annotations

"""Small execution-plan objects used by the NeoOLAF orchestrator.

The plan is intentionally simple.  It records which layer interval should run,
which layers should be skipped, and which execution options should be applied.
It does not replace the existing Pipeline/Runner implementation; it gives the
project an explicit orchestrator-facing object without changing the semantics of
existing ablation runs.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ExecutionPlan:
    """A minimal layer-execution plan for one NeoOLAF run."""

    from_layer: int = 0
    to_layer: int | None = None
    skip_layers: list[int | str] = field(default_factory=list)
    mode: str = "pipeline"
    max_concurrency_layer01: int = 1
    max_concurrency_layer02: int = 1
    max_concurrency_layer03: int = 1
    max_concurrency_layer04: int = 1
    max_concurrency_layer05: int = 1
    retry_failed_calls: int = 0
    retry_sleep_seconds: float = 2.0
    rag_backend: str = "agentic"
    rag_layer01_enabled: bool = False
    rag_top_k: int = 0
    rag_max_chars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the execution plan."""
        return asdict(self)

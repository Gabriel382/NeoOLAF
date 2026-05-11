from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class ExecutionConfig:
    """
    Execution configuration for NeoOLAF runner.

    Supported modes:
    - document_mode:
        run the whole pipeline once on the full document
    - chunk_iterative_mode:
        preprocess the full document, then iterate over chunks for selected layers,
        merge the intermediate outputs, and finally run global layers
    """

    # Execution mode
    mode: Literal["document_mode", "chunk_iterative_mode"] = "document_mode"

    # Whether chunk iteration is enabled
    chunk_loop_enabled: bool = False

    # Layers executed per chunk
    chunk_layer_names: List[str] = field(default_factory=list)

    # Layers executed globally after chunk aggregation
    global_layer_names: List[str] = field(default_factory=list)

    # Optional limit for number of chunks processed
    max_chunks: Optional[int] = None
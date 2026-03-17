from __future__ import annotations

# Abstract base class utilities
from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Any

from neoolaf.core.pipeline_state import PipelineState
# Standard library imports
import time

class BaseLayer(ABC):
    """
    Abstract base class for all NeoOLAF layers.

    Every layer must:
    - define a unique name
    - implement `_run`
    - optionally save intermediate results
    - receive and return a PipelineState
    """

    name: str = "base_layer"

    def __init__(self, save_intermediate: bool = True, verbose: bool = False) -> None:
        """
        Initialize the layer.

        Args:
            save_intermediate:
                If True, save intermediate results to the pipeline artifact directory.
            verbose:
                If True, print progress information for this layer.
        """
        self.save_intermediate = save_intermediate
        self.verbose = verbose

    def run(self, state: PipelineState) -> PipelineState:
        """
        Public entrypoint used by the pipeline.

        This wrapper:
        1. logs the layer start
        2. executes the internal `_run`
        3. optionally saves intermediate artifacts
        4. logs the layer end
        """
        start_time = time.time()

        state.log(f"[{self.name}] started")
        if self.verbose:
            print(f"\n[NeoOLAF] Starting layer: {self.name}")

        state = self._run(state)

        if self.save_intermediate and state.artifact_dir is not None:
            self._save_state(state)

        elapsed = time.time() - start_time
        state.log(f"[{self.name}] finished in {elapsed:.2f}s")

        if self.verbose:
            print(f"[NeoOLAF] Finished layer: {self.name} in {elapsed:.2f}s")

        return state

    @abstractmethod
    def _run(self, state: PipelineState) -> PipelineState:
        """
        Internal implementation of the layer.
        Must be implemented by each concrete layer.
        """
        raise NotImplementedError

    def _save_state(self, state: PipelineState) -> None:
        """
        Save an intermediate JSON snapshot for this layer.

        Each layer writes inside:
            <artifact_dir>/<layer_name>/state.json
        """
        layer_dir = Path(state.artifact_dir) / self.name
        layer_dir.mkdir(parents=True, exist_ok=True)

        payload = self.build_artifact_payload(state)

        with open(layer_dir / "state.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def build_artifact_payload(self, state: PipelineState) -> dict[str, Any]:
        """
        Build a serializable payload for intermediate saving.

        Layers can override this if they need richer exports.
        """
        return {
            "layer": self.name,
            "logs": state.logs,
        }
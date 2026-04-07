"""
Pipeline orchestration logic.
Layers are executed in sequence and each one receives/returns PipelineState.
"""
from __future__ import annotations

# Standard library imports
import time
from typing import List, Set

# Local imports
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.base_layer import BaseLayer


class Pipeline:
    """
    Ordered execution container for NeoOLAF layers.

    New feature:
    - continue_from_last:
      If True, skip layers that already appear as completed in state.logs.
    """

    def __init__(
        self,
        layers: List[BaseLayer],
        verbose: bool = False,
        continue_from_last: bool = False,
    ) -> None:
        """
        Args:
            layers:
                Ordered list of layer objects.
            verbose:
                If True, print pipeline-level progress.
            continue_from_last:
                If True, skip layers already marked as completed in state.logs.
        """
        self.layers = layers
        self.verbose = verbose
        self.continue_from_last = continue_from_last

    def _get_completed_layer_names(self, state: PipelineState) -> Set[str]:
        """
        Extract completed layer names from state.logs.

        Expected flexible log formats:
        - {"layer": "layer_name", "status": "completed"}
        - {"layer_name": "layer_name", "status": "completed"}
        """
        completed: Set[str] = set()

        if not hasattr(state, "logs") or not state.logs:
            return completed

        for log_item in state.logs:
            if not isinstance(log_item, dict):
                continue

            status = str(log_item.get("status", "")).lower()
            layer_name = log_item.get("layer") or log_item.get("layer_name")

            if status == "completed" and layer_name:
                completed.add(str(layer_name))

        return completed

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute all layers in order.
        """
        total_layers = len(self.layers)
        pipeline_start = time.time()

        completed_layers = self._get_completed_layer_names(state)

        if self.verbose:
            print(f"[NeoOLAF] Pipeline started with {total_layers} layers")
            if self.continue_from_last:
                print(f"[NeoOLAF] Resume mode enabled")
                print(f"[NeoOLAF] Completed layers found: {sorted(completed_layers)}")

        for idx, layer in enumerate(self.layers, start=1):
            if self.verbose:
                print(f"[NeoOLAF] Layer {idx-1}/{total_layers-1}: {layer.name}")

            # Skip already completed layers if resume mode is enabled.
            if self.continue_from_last and layer.name in completed_layers:
                if self.verbose:
                    print(f"[NeoOLAF] Skipping already completed layer: {layer.name}")
                continue

            state = layer.run(state)

        total_elapsed = time.time() - pipeline_start

        if self.verbose:
            print(f"[NeoOLAF] Pipeline finished in {total_elapsed:.2f}s")

        return state
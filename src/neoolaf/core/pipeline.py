"""
Pipeline orchestration logic.
Layers are executed in sequence and each one receives/returns PipelineState.
"""
from __future__ import annotations

from typing import List

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.base_layer import BaseLayer

# Standard library imports
import time

class Pipeline:
    """
    Ordered execution container for NeoOLAF layers.
    """

    def __init__(self, layers: List[BaseLayer], verbose: bool = False) -> None:
        """
        Args:
            layers:
                Ordered list of layer objects.
            verbose:
                If True, print pipeline-level progress.
        """
        self.layers = layers
        self.verbose = verbose

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute all layers in order.
        """
        total_layers = len(self.layers)
        pipeline_start = time.time()

        if self.verbose:
            print(f"[NeoOLAF] Pipeline started with {total_layers} layers")

        for idx, layer in enumerate(self.layers, start=1):
            if self.verbose:
                print(f"[NeoOLAF] Layer {idx-1}/{total_layers-1}: {layer.name}")

            state = layer.run(state)

        total_elapsed = time.time() - pipeline_start

        if self.verbose:
            print(f"[NeoOLAF] Pipeline finished in {total_elapsed:.2f}s")

        return state
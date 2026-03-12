"""
Pipeline orchestration logic.
Layers are executed in sequence and each one receives/returns PipelineState.
"""
from __future__ import annotations

from typing import List

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.base_layer import BaseLayer


class Pipeline:
    """
    Ordered execution container for NeoOLAF layers.
    """

    def __init__(self, layers: List[BaseLayer]) -> None:
        """
        Args:
            layers:
                Ordered list of layer objects.
        """
        self.layers = layers

    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute all layers in order.
        """
        for layer in self.layers:
            state = layer.run(state)
        return state
"""
Pipeline orchestration logic.
Layers are executed in sequence and each one receives/returns PipelineState.
"""
from __future__ import annotations

import time
from typing import List, Set

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.base_layer import BaseLayer


class Pipeline:
    """
    Ordered execution container for NeoOLAF layers.

    Ablation features:
    - run only an interval with from_layer / to_layer
    - skip selected layers with skip_layers
    - keep completed-layer resume support
    """

    def __init__(
        self,
        layers: List[BaseLayer],
        verbose: bool = False,
        continue_from_last: bool = False,
    ) -> None:
        self.layers = layers
        self.verbose = verbose
        self.continue_from_last = continue_from_last

    def _get_completed_layer_names(self, state: PipelineState) -> Set[str]:
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

    @staticmethod
    def _normalize_skip_layers(skip_layers: set[int | str] | list[int | str] | None) -> set[int | str]:
        if skip_layers is None:
            return set()
        return set(skip_layers)

    def selected_layers(
        self,
        *,
        from_layer: int = 0,
        to_layer: int | None = None,
        skip_layers: set[int | str] | list[int | str] | None = None,
    ) -> list[tuple[int, BaseLayer]]:
        """Return indexed layers selected for this run."""
        if to_layer is None:
            to_layer = len(self.layers) - 1

        if from_layer < 0:
            raise ValueError("from_layer must be >= 0")
        if to_layer >= len(self.layers):
            raise ValueError(f"to_layer={to_layer} is out of range for {len(self.layers)} layers")
        if from_layer > to_layer:
            raise ValueError("from_layer cannot be greater than to_layer")

        skipped = self._normalize_skip_layers(skip_layers)
        selected: list[tuple[int, BaseLayer]] = []

        for index, layer in enumerate(self.layers):
            if index < from_layer or index > to_layer:
                continue
            if index in skipped or layer.name in skipped:
                continue
            selected.append((index, layer))

        return selected

    def run(
        self,
        state: PipelineState,
        *,
        from_layer: int = 0,
        to_layer: int | None = None,
        skip_layers: set[int | str] | list[int | str] | None = None,
    ) -> PipelineState:
        """
        Execute selected layers in order.
        """
        selected = self.selected_layers(
            from_layer=from_layer,
            to_layer=to_layer,
            skip_layers=skip_layers,
        )
        total_layers = len(self.layers)
        pipeline_start = time.time()

        completed_layers = self._get_completed_layer_names(state)

        if self.verbose:
            selected_names = [layer.name for _, layer in selected]
            print(f"[NeoOLAF] Pipeline has {total_layers} layers")
            print(f"[NeoOLAF] Selected layers: {selected_names}")
            if self.continue_from_last:
                print("[NeoOLAF] Resume mode enabled")
                print(f"[NeoOLAF] Completed layers found: {sorted(completed_layers)}")

        for index, layer in selected:
            if self.verbose:
                print(f"[NeoOLAF] Layer {index}/{total_layers-1}: {layer.name}")

            if self.continue_from_last and layer.name in completed_layers:
                if self.verbose:
                    print(f"[NeoOLAF] Skipping already completed layer: {layer.name}")
                continue

            state = layer.run(state)

        total_elapsed = time.time() - pipeline_start

        if self.verbose:
            print(f"[NeoOLAF] Pipeline finished in {total_elapsed:.2f}s")

        return state

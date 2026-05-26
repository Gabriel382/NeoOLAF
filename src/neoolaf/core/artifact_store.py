from __future__ import annotations

"""
Layer-level artifact store used for ablation runs.

Each layer receives a deterministic folder containing:
- state.json: full PipelineState after the layer
- output.json: layer-specific payload
- metadata.json: compact execution and output counters

This makes every layer restartable and inspectable.
"""

from pathlib import Path
from typing import Any
import json
import time

from neoolaf.core.state_serialization import dump_json, load_json, to_jsonable


class ArtifactStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def layer_dir(self, layer_name: str) -> Path:
        path = self.run_dir / layer_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, layer_name: str, filename: str, data: Any) -> Path:
        path = self.layer_dir(layer_name) / filename
        dump_json(path, data)
        return path

    def load_json(self, layer_name: str, filename: str) -> Any:
        return load_json(self.layer_dir(layer_name) / filename)

    def save_text(self, layer_name: str, filename: str, content: str) -> Path:
        path = self.layer_dir(layer_name) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def save_layer_artifacts(
        self,
        *,
        layer_name: str,
        state: Any,
        output_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        output_payload = output_payload or {}
        metadata = metadata or {}

        self.save_json(layer_name, "state.json", state)
        self.save_json(layer_name, "output.json", output_payload)
        self.save_json(
            layer_name,
            "metadata.json",
            {
                "layer": layer_name,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **metadata,
            },
        )

    @staticmethod
    def write_run_config(run_dir: str | Path, config: dict[str, Any]) -> None:
        path = Path(run_dir) / "run_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(to_jsonable(config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

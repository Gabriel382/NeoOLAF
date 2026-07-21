from __future__ import annotations

"""Immutable snapshots returned by the read-only run inspection API."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class LayerSnapshot:
    index: int
    name: str
    status: str
    directory: Path | None = None
    state_path: Path | None = None
    output_path: Path | None = None
    metadata_path: Path | None = None
    legacy_payload_path: Path | None = None
    elapsed_seconds: float | None = None
    counts: dict[str, int] = field(default_factory=dict)
    saved_at: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ExportSnapshot:
    name: str
    path: Path
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class RunSnapshot:
    run_dir: Path
    run_config: dict[str, Any]
    run_manifest: dict[str, Any]
    layers: tuple[LayerSnapshot, ...]
    exports: tuple[ExportSnapshot, ...]
    final_state_path: Path | None
    checkpoint_path: Path | None
    status: str
    is_complete: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

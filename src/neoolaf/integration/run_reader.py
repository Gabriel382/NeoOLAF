from __future__ import annotations

"""Read-only discovery and loading of NeoOLAF run artifacts.

The functions in this module never create, update, rename, or delete files in a
run directory.  They support both the current canonical artifact layout and
older KES-era runs that may contain document-named JSON payloads plus trusted
local checkpoints.
"""

from dataclasses import dataclass
import gzip
import json
from pathlib import Path, PureWindowsPath
import pickle
from typing import Any, Iterable

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.integration.integrity import hash_file
from neoolaf.integration.run_contract import (
    CHECKPOINT_MANIFEST,
    EXPORT_NAMES,
    EXPORT_SEARCH_ROOTS,
    RESERVED_LAYER_JSON_NAMES,
    RUN_MANIFEST_FILENAMES,
    SCIENTIFIC_LAYERS,
)
from neoolaf.integration.run_snapshot import ExportSnapshot, LayerSnapshot, RunSnapshot


_PIPELINE_STATE_CLASS = "neoolaf.core.pipeline_state.PipelineState"


@dataclass(frozen=True)
class _JsonRead:
    data: Any | None
    warning: str | None = None


def _read_json(path: Path) -> _JsonRead:
    if not path.is_file():
        return _JsonRead(None)
    try:
        return _JsonRead(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _JsonRead(None, f"Could not read JSON {path}: {exc}")


def _first_existing_json(root: Path, names: Iterable[str]) -> tuple[dict[str, Any], Path | None, str | None]:
    for name in names:
        path = root / name
        result = _read_json(path)
        if result.warning:
            return {}, path, result.warning
        if isinstance(result.data, dict):
            return result.data, path, None
    return {}, None, None


def _legacy_payload_path(layer_dir: Path) -> Path | None:
    if not layer_dir.is_dir():
        return None
    candidates = sorted(
        path
        for path in layer_dir.glob("*.json")
        if path.name not in RESERVED_LAYER_JSON_NAMES
    )
    return candidates[0] if candidates else None


def _extract_counts(metadata: Any, output: Any) -> dict[str, int]:
    if isinstance(metadata, dict) and isinstance(metadata.get("counts"), dict):
        return {
            str(key): int(value)
            for key, value in metadata["counts"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    result: dict[str, int] = {}
    if isinstance(output, dict):
        for key, value in output.items():
            if key.startswith("num_") and isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key[4:]] = int(value)
    return result


def _skip_sets(run_config: dict[str, Any]) -> tuple[set[int], set[str]]:
    raw = run_config.get("skip_layers", [])
    if not isinstance(raw, list):
        return set(), set()
    indexes: set[int] = set()
    names: set[str] = set()
    for value in raw:
        if isinstance(value, int):
            indexes.add(value)
        elif isinstance(value, str):
            try:
                indexes.add(int(value))
            except ValueError:
                names.add(value)
    return indexes, names


def _discover_layer(root: Path, index: int, name: str, run_config: dict[str, Any]) -> LayerSnapshot:
    directory = root / name
    state_path = directory / "state.json" if (directory / "state.json").is_file() else None
    output_path = directory / "output.json" if (directory / "output.json").is_file() else None
    metadata_path = directory / "metadata.json" if (directory / "metadata.json").is_file() else None
    legacy_payload = _legacy_payload_path(directory)

    warnings: list[str] = []
    metadata_result = _read_json(metadata_path) if metadata_path else _JsonRead(None)
    output_result = _read_json(output_path) if output_path else _JsonRead(None)
    if metadata_result.warning:
        warnings.append(metadata_result.warning)
    if output_result.warning:
        warnings.append(output_result.warning)

    skip_indexes, skip_names = _skip_sets(run_config)
    explicitly_skipped = index in skip_indexes or name in skip_names

    if explicitly_skipped:
        status = "skipped"
    elif not directory.exists():
        status = "not_started"
    elif state_path or output_path or metadata_path or legacy_payload:
        status = "completed"
    else:
        status = "started"

    metadata = metadata_result.data if isinstance(metadata_result.data, dict) else {}
    output = output_result.data if isinstance(output_result.data, dict) else {}
    elapsed = metadata.get("elapsed_seconds")
    elapsed_seconds = float(elapsed) if isinstance(elapsed, (int, float)) else None
    saved_at = metadata.get("saved_at") if isinstance(metadata.get("saved_at"), str) else None

    return LayerSnapshot(
        index=index,
        name=name,
        status=status,
        directory=directory if directory.exists() else None,
        state_path=state_path,
        output_path=output_path,
        metadata_path=metadata_path,
        legacy_payload_path=legacy_payload,
        elapsed_seconds=elapsed_seconds,
        counts=_extract_counts(metadata, output),
        saved_at=saved_at,
        warnings=tuple(warnings),
    )


def _find_export(root: Path, name: str) -> Path | None:
    for relative_root in EXPORT_SEARCH_ROOTS:
        path = root / relative_root / name
        if path.is_file():
            return path
    matches = sorted(
        (path for path in root.rglob(name) if path.is_file()),
        key=lambda path: (len(path.relative_to(root).parts), path.as_posix()),
    )
    return matches[0] if matches else None


def _discover_exports(root: Path) -> tuple[ExportSnapshot, ...]:
    exports: list[ExportSnapshot] = []
    for name in EXPORT_NAMES:
        path = _find_export(root, name)
        if path is not None:
            exports.append(
                ExportSnapshot(
                    name=name,
                    path=path,
                    size_bytes=path.stat().st_size,
                    sha256=hash_file(path),
                )
            )
    return tuple(exports)


def _is_pipeline_state_json(path: Path) -> bool:
    result = _read_json(path)
    data = result.data
    return isinstance(data, dict) and data.get("__neoolaf_class__") == _PIPELINE_STATE_CLASS


def _discover_final_state_path(root: Path, layers: tuple[LayerSnapshot, ...]) -> Path | None:
    by_index = {layer.index: layer for layer in layers}
    for index in (12, 11):
        path = by_index[index].state_path
        if path is not None:
            return path

    for layer in reversed(layers):
        if layer.state_path is not None:
            return layer.state_path

    # Historical runs sometimes used a document-named JSON file for a complete
    # state.  Only accept it when the type marker proves it is a PipelineState.
    for layer in reversed(layers):
        candidate = layer.legacy_payload_path
        if candidate is not None and _is_pipeline_state_json(candidate):
            return candidate
    return None


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _discover_checkpoint_path(root: Path) -> tuple[Path | None, str | None]:
    checkpoint_dir = root / "checkpoints"
    manifest_path = root / CHECKPOINT_MANIFEST
    warning: str | None = None

    if manifest_path.is_file():
        result = _read_json(manifest_path)
        if result.warning:
            warning = result.warning
        elif isinstance(result.data, dict):
            raw = result.data.get("latest_checkpoint")
            if isinstance(raw, str) and raw:
                manifest_candidate = Path(raw).expanduser()
                candidates = [manifest_candidate]
                if not manifest_candidate.is_absolute():
                    candidates.append(root / manifest_candidate)
                candidates.append(checkpoint_dir / manifest_candidate.name)
                candidates.append(checkpoint_dir / PureWindowsPath(raw).name)
                for candidate in candidates:
                    if candidate.is_file() and _within(candidate, checkpoint_dir):
                        return candidate.resolve(), warning

    if checkpoint_dir.is_dir():
        candidates = sorted(
            checkpoint_dir.glob("*.pkl.gz"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve(), warning
    return None, warning


def _derive_run_status(
    layers: tuple[LayerSnapshot, ...],
    exports: tuple[ExportSnapshot, ...],
    run_manifest: dict[str, Any],
) -> tuple[str, bool]:
    manifest_status = run_manifest.get("status")
    if isinstance(manifest_status, str) and manifest_status.lower() in {"failed", "cancelled"}:
        return manifest_status.lower(), False

    layer12 = next(layer for layer in layers if layer.index == 12)
    export_names = {artifact.name for artifact in exports}
    complete = layer12.status == "completed" and set(EXPORT_NAMES).issubset(export_names)
    if complete:
        return "completed", True

    completed_count = sum(layer.status == "completed" for layer in layers)
    started_count = sum(layer.status == "started" for layer in layers)
    if completed_count or started_count or exports:
        return "partial", False
    return "created", False


def inspect_run(run_dir: str | Path) -> RunSnapshot:
    """Inspect a NeoOLAF run directory without modifying any file."""

    root = Path(run_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    warnings: list[str] = []
    run_config_result = _read_json(root / "run_config.json")
    if run_config_result.warning:
        warnings.append(run_config_result.warning)
    run_config = run_config_result.data if isinstance(run_config_result.data, dict) else {}

    run_manifest, _manifest_path, manifest_warning = _first_existing_json(root, RUN_MANIFEST_FILENAMES)
    if manifest_warning:
        warnings.append(manifest_warning)

    layers = tuple(
        _discover_layer(root, contract.index, contract.name, run_config)
        for contract in SCIENTIFIC_LAYERS
    )
    for layer in layers:
        warnings.extend(layer.warnings)

    exports = _discover_exports(root)
    final_state_path = _discover_final_state_path(root, layers)
    checkpoint_path, checkpoint_warning = _discover_checkpoint_path(root)
    if checkpoint_warning:
        warnings.append(checkpoint_warning)

    status, is_complete = _derive_run_status(layers, exports, run_manifest)
    found_exports = {export.name for export in exports}
    missing_exports = [name for name in EXPORT_NAMES if name not in found_exports]
    if missing_exports:
        warnings.append("Missing canonical exports: " + ", ".join(missing_exports))
    if final_state_path is None and checkpoint_path is None:
        warnings.append("No complete PipelineState JSON or trusted local checkpoint was discovered.")
    elif final_state_path is None and checkpoint_path is not None:
        warnings.append(
            "Only a pickle checkpoint is available for the final state; loading it requires explicit trust."
        )

    return RunSnapshot(
        run_dir=root,
        run_config=run_config,
        run_manifest=run_manifest,
        layers=layers,
        exports=exports,
        final_state_path=final_state_path,
        checkpoint_path=checkpoint_path,
        status=status,
        is_complete=is_complete,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _load_trusted_checkpoint(path: Path, run_root: Path) -> PipelineState:
    checkpoint_root = (run_root / "checkpoints").resolve()
    if not path.is_file() or not _within(path, checkpoint_root):
        raise ValueError("Checkpoint must be an existing file inside the inspected run/checkpoints directory.")

    # Pickle is inherently unsafe.  This helper is intentionally private and is
    # called only after the caller has explicitly opted in to trusting a local
    # NeoOLAF checkpoint.
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - explicit trusted-local opt-in

    state = payload.get("state") if isinstance(payload, dict) else payload
    if not isinstance(state, PipelineState):
        raise ValueError(f"Checkpoint at {path} does not contain a PipelineState.")
    return state


def load_final_state(
    run_dir: str | Path,
    *,
    allow_trusted_checkpoint: bool = False,
) -> PipelineState:
    """Load the latest usable PipelineState from a run.

    JSON states are always preferred.  Gzipped pickle checkpoints are loaded
    only when ``allow_trusted_checkpoint=True`` and only when the checkpoint is
    physically located inside the inspected run's ``checkpoints`` directory.
    Never enable this option for arbitrary uploaded archives.
    """

    snapshot = inspect_run(run_dir)
    if snapshot.final_state_path is not None:
        return PipelineState.load_json(str(snapshot.final_state_path))

    if snapshot.checkpoint_path is not None and allow_trusted_checkpoint:
        return _load_trusted_checkpoint(snapshot.checkpoint_path, snapshot.run_dir)

    if snapshot.checkpoint_path is not None:
        raise PermissionError(
            "A local pickle checkpoint is available, but loading it is disabled by default. "
            "Pass allow_trusted_checkpoint=True only for a checkpoint produced locally by this NeoOLAF run."
        )

    raise FileNotFoundError(f"No loadable PipelineState was found under {snapshot.run_dir}")

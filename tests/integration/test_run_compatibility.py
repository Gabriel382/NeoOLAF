from __future__ import annotations

import gzip
import json
from pathlib import Path
import pickle

import pytest

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.documents import Document
from neoolaf.integration import inspect_run, load_final_state
from neoolaf.integration.run_contract import EXPORT_NAMES


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _legacy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "legacy_run"
    _write_json(
        run_dir / "layer00_preprocessing" / "legacy_document.json",
        {"layer": "layer00_preprocessing", "doc_id": "legacy"},
    )
    _write_json(
        run_dir / "layer12_serialization" / "legacy_document.json",
        {"layer": "layer12_serialization", "output_subdir": "data/exports"},
    )

    export_dir = run_dir / "data" / "exports"
    export_dir.mkdir(parents=True)
    for name in EXPORT_NAMES:
        (export_dir / name).write_text(f"legacy:{name}\n", encoding="utf-8")

    state = PipelineState(
        document=Document(doc_id="legacy", source_path="legacy.pdf", raw_text="legacy"),
        llm_model="legacy/model",
        artifact_dir=str(run_dir),
    )
    checkpoint = run_dir / "checkpoints" / "after_layer12_serialization.pkl.gz"
    checkpoint.parent.mkdir(parents=True)
    with gzip.open(checkpoint, "wb") as handle:
        pickle.dump({"checkpoint_name": "after_layer12_serialization", "state": state}, handle)

    # Historical manifests can contain Windows absolute paths even when the run
    # is copied to another operating system.
    _write_json(
        run_dir / "checkpoints" / "manifest.json",
        {
            "latest_checkpoint": (
                "C:\\research\\NeoOLAF\\runs\\legacy_run\\checkpoints\\"
                "after_layer12_serialization.pkl.gz"
            )
        },
    )
    return run_dir


def test_legacy_document_payloads_and_windows_checkpoint_path(tmp_path: Path) -> None:
    run_dir = _legacy_run(tmp_path)

    snapshot = inspect_run(run_dir)

    assert snapshot.layers[0].legacy_payload_path is not None
    assert snapshot.layers[12].legacy_payload_path is not None
    assert snapshot.final_state_path is None
    assert snapshot.checkpoint_path == (
        run_dir / "checkpoints" / "after_layer12_serialization.pkl.gz"
    ).resolve()
    assert snapshot.is_complete is True


def test_pickle_loading_requires_explicit_trust(tmp_path: Path) -> None:
    run_dir = _legacy_run(tmp_path)

    with pytest.raises(PermissionError):
        load_final_state(run_dir)

    state = load_final_state(run_dir, allow_trusted_checkpoint=True)
    assert state.document.doc_id == "legacy"


def test_interrupted_run_is_partial(tmp_path: Path) -> None:
    run_dir = tmp_path / "interrupted"
    (run_dir / "layer00_preprocessing").mkdir(parents=True)
    _write_json(
        run_dir / "layer00_preprocessing" / "metadata.json",
        {"layer": "layer00_preprocessing", "counts": {"chunks": 2}},
    )

    snapshot = inspect_run(run_dir)

    assert snapshot.status == "partial"
    assert snapshot.is_complete is False
    assert snapshot.layers[0].status == "completed"
    assert snapshot.layers[1].status == "not_started"
    assert snapshot.warnings

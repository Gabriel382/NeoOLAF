from __future__ import annotations

import json
from pathlib import Path

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.documents import Document
from neoolaf.integration import inspect_run, load_final_state
from neoolaf.integration.run_contract import EXPORT_NAMES


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _state(run_dir: Path) -> PipelineState:
    return PipelineState(
        document=Document(
            doc_id="demo",
            source_path="demo.pdf",
            raw_text="Alarm 1001 is caused by an emergency stop.",
        ),
        llm_model="test/model",
        artifact_dir=str(run_dir),
        profile_name="test-profile",
    )


def _complete_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_complete"
    _write_json(
        run_dir / "run_config.json",
        {
            "model": "test/model",
            "profile": "test-profile",
            "resolved_profile_name": "test-profile",
            "skip_layers": [],
        },
    )
    _write_json(
        run_dir / "orchestration_manifest.json",
        {"status": "completed", "elapsed_seconds": 1.25},
    )
    _write_json(
        run_dir / "layer00_preprocessing" / "metadata.json",
        {
            "layer": "layer00_preprocessing",
            "saved_at": "2026-07-21 10:00:00",
            "elapsed_seconds": 0.5,
            "counts": {"linguistic_expressions": 0},
        },
    )
    _write_json(
        run_dir / "layer00_preprocessing" / "output.json",
        {"layer": "layer00_preprocessing", "num_chunks": 1},
    )
    final_state = _state(run_dir)
    final_state.save_json(str(run_dir / "layer12_serialization" / "state.json"))
    _write_json(
        run_dir / "layer12_serialization" / "metadata.json",
        {
            "layer": "layer12_serialization",
            "saved_at": "2026-07-21 10:00:01",
            "elapsed_seconds": 0.1,
            "counts": {},
        },
    )
    _write_json(
        run_dir / "layer12_serialization" / "output.json",
        {"layer": "layer12_serialization"},
    )
    export_dir = run_dir / "data" / "exports"
    export_dir.mkdir(parents=True)
    for name in EXPORT_NAMES:
        (export_dir / name).write_text(f"fixture:{name}\n", encoding="utf-8")
    return run_dir


def test_inspect_complete_run(tmp_path: Path) -> None:
    run_dir = _complete_run(tmp_path)

    snapshot = inspect_run(run_dir)

    assert snapshot.status == "completed"
    assert snapshot.is_complete is True
    assert snapshot.final_state_path == run_dir.resolve() / "layer12_serialization" / "state.json"
    assert {item.name for item in snapshot.exports} == set(EXPORT_NAMES)
    assert snapshot.layers[0].status == "completed"
    assert snapshot.layers[0].counts["linguistic_expressions"] == 0
    assert snapshot.layers[12].status == "completed"


def test_load_final_json_state(tmp_path: Path) -> None:
    run_dir = _complete_run(tmp_path)

    state = load_final_state(run_dir)

    assert state.document.doc_id == "demo"
    assert state.llm_model == "test/model"
    assert state.profile_name == "test-profile"


def test_inspect_missing_directory() -> None:
    missing = Path("this-run-does-not-exist")
    try:
        inspect_run(missing)
    except FileNotFoundError as exc:
        assert exc.filename is None or "this-run-does-not-exist" in str(exc)
    else:
        raise AssertionError("inspect_run should reject a missing directory")

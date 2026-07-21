from __future__ import annotations

from pathlib import Path

from neoolaf.integration import hash_exports, inspect_run, verify_exports_unchanged
from neoolaf.integration.run_contract import EXPORT_NAMES


def test_read_only_integration_preserves_original_export_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "publication_compatible_run"
    export_dir = run_dir / "data" / "exports"
    export_dir.mkdir(parents=True)
    for index, name in enumerate(EXPORT_NAMES):
        (export_dir / name).write_bytes(f"publication-fixture-{index}\n".encode("utf-8"))

    before = hash_exports(run_dir)
    inspect_run(run_dir)
    after = hash_exports(run_dir)

    assert verify_exports_unchanged(before, after)


def test_publication_reference_hash_manifest_is_complete() -> None:
    import json

    manifest_path = Path(__file__).resolve().parents[1] / "fixtures" / "publication_export_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["scientific_pipeline"] == "neoolaf-kes-2026"
    assert set(manifest["exports"]) == set(EXPORT_NAMES)
    assert all(len(digest) == 64 for digest in manifest["exports"].values())

    repository_root = Path(__file__).resolve().parents[2]
    reference_run = repository_root / manifest["reference"]
    if reference_run.is_dir():
        assert hash_exports(reference_run) == manifest["exports"]

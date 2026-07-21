from __future__ import annotations

import json
from pathlib import Path

from neoolaf.integration import (
    build_integrity_report,
    hash_directory,
    hash_exports,
    hash_file,
    verify_exports_unchanged,
)
from neoolaf.integration.run_contract import EXPORT_NAMES


def test_file_and_directory_hashes_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    first = root / "a.txt"
    second = root / "nested" / "b.txt"
    second.parent.mkdir()
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    assert hash_file(first) == hash_file(first)
    before = hash_directory(root)
    after = hash_directory(root)
    assert before == after

    second.write_text("changed", encoding="utf-8")
    assert hash_directory(root) != before


def test_export_integrity_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    export_dir = run_dir / "data" / "exports"
    export_dir.mkdir(parents=True)
    for name in EXPORT_NAMES:
        (export_dir / name).write_text(name, encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps({"input_pdf": "missing.pdf"}), encoding="utf-8"
    )

    before = hash_exports(run_dir)
    report = build_integrity_report(run_dir)
    after = hash_exports(run_dir)

    assert set(before) == set(EXPORT_NAMES)
    assert verify_exports_unchanged(before, after)
    assert report["exports"] == before
    assert report["run_config_sha256"] is not None
    assert report["input_sha256"] is None

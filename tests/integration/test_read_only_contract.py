from __future__ import annotations

import json
from pathlib import Path

from neoolaf.integration import hash_directory, inspect_run


def test_inspection_does_not_modify_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    layer_dir = run_dir / "layer01_linguistic_expression_extraction"
    layer_dir.mkdir(parents=True)
    (layer_dir / "output.json").write_text(
        json.dumps({"layer": "layer01_linguistic_expression_extraction", "num_items": 2}),
        encoding="utf-8",
    )

    before = hash_directory(run_dir)
    inspect_run(run_dir)
    after = hash_directory(run_dir)

    assert before == after

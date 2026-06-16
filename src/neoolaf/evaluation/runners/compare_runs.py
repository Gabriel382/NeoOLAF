"""Comparison and batch evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

from neoolaf.evaluation.reports.writers import ensure_dir, flatten_summary, write_csv, write_json


def find_summary_files(runs_dir: str | Path) -> list[Path]:
    """Find metrics.summary.json files recursively."""
    return sorted(Path(runs_dir).glob("**/metrics.summary.json"))


def compare_runs(runs_dir: str | Path, output_dir: str | Path) -> list[dict]:
    """Aggregate multiple run summaries into comparison files."""
    rows = []
    summaries = []
    for path in find_summary_files(runs_dir):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append(summary)
        row = flatten_summary(summary)
        row["summary_path"] = str(path)
        rows.append(row)

    out = ensure_dir(output_dir)
    write_json(out / "method_comparison.json", summaries)
    write_csv(out / "method_comparison.csv", rows)
    write_csv(out / "paper_table.csv", rows)

    md_lines = ["# Method Comparison", "", "| Method | Profile | Entity F1 | Relation F1 | STR | CR | PC | OC | CV | DVS |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md_lines.append(
            f"| {row.get('method')} | {row.get('profile')} | {row.get('entity_f1', 0):.4f} | {row.get('relation_f1', 0):.4f} | {row.get('STR', 0):.4f} | {row.get('CR', 0):.4f} | {row.get('PC', 0):.4f} | {row.get('OC', 0):.4f} | {row.get('CV', 0):.4f} | {row.get('DVS', 0):.4f} |"
        )
    (out / "method_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
    return rows

"""Report writers for evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON with UTF-8 and indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a summary into one CSV-friendly row."""
    extraction = summary.get("extraction", {})
    validation = summary.get("validation", {}).get("summary", {})
    ontology = summary.get("ontology", {})
    counts = extraction.get("counts", {})
    entity = extraction.get("entity", {})
    relation = extraction.get("relation", {})

    row = {
        "method": summary.get("method"),
        "dataset": summary.get("dataset"),
        "profile": summary.get("profile"),
        "run_id": summary.get("run_id"),
        "entity_p": entity.get("precision", 0.0),
        "entity_r": entity.get("recall", 0.0),
        "entity_f1": entity.get("f1", 0.0),
        "entity_tp": entity.get("tp", 0),
        "entity_fp": entity.get("fp", 0),
        "entity_fn": entity.get("fn", 0),
        "relation_p": relation.get("precision", 0.0),
        "relation_r": relation.get("recall", 0.0),
        "relation_f1": relation.get("f1", 0.0),
        "relation_tp": relation.get("tp", 0),
        "relation_fp": relation.get("fp", 0),
        "relation_fn": relation.get("fn", 0),
        "pred_entities": counts.get("pred_entities", 0),
        "gold_entities": counts.get("gold_entities", 0),
        "pred_relations": counts.get("pred_relations", 0),
        "gold_relations": counts.get("gold_relations", 0),
        "STR": validation.get("STR", 0.0),
        "CR": validation.get("CR", 0.0),
        "PC": validation.get("PC", 0.0),
        "OC": validation.get("OC", 0.0),
        "CV": validation.get("CV", 0.0),
        "DVS": validation.get("DVS", 0.0),
        "ontology_available": ontology.get("available", False),
        "class_count": ontology.get("class_count", 0),
        "property_count": ontology.get("property_count", 0),
        "hierarchy_link_count": ontology.get("hierarchy_link_count", 0),
        "axiom_count": ontology.get("axiom_count", 0),
        "description_coverage": ontology.get("description_coverage", 0.0),
        "domain_coverage": ontology.get("domain_coverage", 0.0),
        "range_coverage": ontology.get("range_coverage", 0.0),
        "ontology_delta_size": ontology.get("ontology_delta_size", 0),
        "ontology_growth_rate": ontology.get("ontology_growth_rate", 0.0),
    }
    return row


def write_markdown_report(path: str | Path, summary: dict[str, Any]) -> None:
    """Write a concise Markdown report."""
    flat = flatten_summary(summary)
    lines = [
        f"# Evaluation Report: {flat['method']} / {flat['dataset']} / {flat['profile']}",
        "",
        "## Main Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Entity Precision | {flat['entity_p']:.4f} |",
        f"| Entity Recall | {flat['entity_r']:.4f} |",
        f"| Entity F1 | {flat['entity_f1']:.4f} |",
        f"| Relation Precision | {flat['relation_p']:.4f} |",
        f"| Relation Recall | {flat['relation_r']:.4f} |",
        f"| Relation F1 | {flat['relation_f1']:.4f} |",
        f"| STR | {flat['STR']:.4f} |",
        f"| CR | {flat['CR']:.4f} |",
        f"| PC | {flat['PC']:.4f} |",
        f"| OC | {flat['OC']:.4f} |",
        f"| CV | {flat['CV']:.4f} |",
        f"| DVS | {flat['DVS']:.4f} |",
        "",
        "## Counts",
        "",
        "| Count | Value |",
        "|---|---:|",
        f"| Predicted entities | {flat['pred_entities']} |",
        f"| Gold entities | {flat['gold_entities']} |",
        f"| Predicted relations | {flat['pred_relations']} |",
        f"| Gold relations | {flat['gold_relations']} |",
        "",
    ]

    per_relation = summary.get("extraction", {}).get("per_relation", [])
    if per_relation:
        lines.extend(["## Per-relation Metrics", "", "| Relation | P | R | F1 | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in per_relation:
            lines.append(
                f"| {row['relation']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} |"
            )
        lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_run_outputs(output_dir: str | Path, artifact: Any, summary: dict[str, Any]) -> None:
    """Write all standard output files for one evaluation run."""
    out = ensure_dir(output_dir)
    write_json(out / "artifact.normalized.json", artifact.to_dict())
    write_json(out / "metrics.summary.json", summary)
    write_csv(out / "metrics.flat.csv", [flatten_summary(summary)])
    write_csv(out / "per_relation_metrics.csv", summary.get("extraction", {}).get("per_relation", []))
    write_csv(out / "validation_metrics.csv", summary.get("validation", {}).get("per_document", []))
    write_json(out / "ontology_metrics.json", summary.get("ontology", {}))
    write_json(out / "matched_entities.json", summary.get("extraction", {}).get("matches", {}).get("entities", []))
    write_json(out / "matched_relations.json", summary.get("extraction", {}).get("matches", {}).get("relations", []))
    write_json(out / "unmatched_entities.json", summary.get("extraction", {}).get("unmatched", {}).get("entities_pred", []))
    write_json(out / "unmatched_relations.json", summary.get("extraction", {}).get("unmatched", {}).get("relations_pred", []))
    write_json(out / "config.resolved.json", summary.get("config", {}))
    write_json(out / "errors.json", summary.get("errors", []))
    write_markdown_report(out / "report.md", summary)

"""Robust prefix stop-after-layer ablation evaluator.

This runner is designed for experiments of the form:

    stop after Layer X -> finalize the available prefix into triples/KG/ontology
    -> evaluate the generated final KG against XQuality gold.

It complements relaxed relation F1 with output-efficiency metrics so noisy
prefix-finalized outputs are not rewarded only for over-generating triples.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from neoolaf.evaluation.domains.xquality import (
    build_neoolaf_json_method_config,
    build_xquality_domain_config,
)
from neoolaf.evaluation.metrics.ablation_robust import (
    add_robust_ablation_metrics_to_summary,
    compute_robust_ablation_metrics,
    harmonic_mean,
    safe_div,
)
from neoolaf.evaluation.runners.evaluate_domain_kg import EvaluationInput, evaluate_domain_kg


PREFERRED_TRIPLE_FILES = [
    "triples.json",
    "kg_inferred.json",
    "kg_local.json",
    "kg.json",
    "output.json",
]


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def get_nested(data: Any, path: list[str], default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def extract_label(value: Any) -> str:
    """Extract a readable label from a raw value or nested JSON object."""
    if isinstance(value, dict):
        for key in ["label", "text", "name", "value", "id"]:
            if value.get(key):
                return str(value[key]).strip()
        return ""
    return str(value or "").strip()


def find_first_existing(base: Path, names: list[str] = PREFERRED_TRIPLE_FILES) -> Path | None:
    if base.is_file():
        return base
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def load_triples_any(path_or_folder: str | Path) -> tuple[list[dict[str, Any]], Path | None]:
    """Load triples from common NeoOLAF/prefix-finalizer layouts."""
    path = Path(path_or_folder)
    source = find_first_existing(path)
    if source is None:
        return [], None

    data = load_json(source)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)], source

    if not isinstance(data, dict):
        return [], source

    for key in ["triples", "relations", "predictions", "items"]:
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)], source

    return [], source


def raw_triple_to_neoolaf_export_triple(triple: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one raw triple to the nested NeoOLAF JSON schema."""
    subject = (
        triple.get("subject_label")
        or triple.get("subject")
        or triple.get("head")
        or triple.get("s")
        or get_nested(triple, ["subject", "label"])
        or get_nested(triple, ["head", "label"])
    )
    predicate = (
        triple.get("predicate_label")
        or triple.get("predicate")
        or triple.get("relation")
        or triple.get("rel")
        or triple.get("p")
        or get_nested(triple, ["predicate", "label"])
        or get_nested(triple, ["relation", "label"])
    )
    object_ = (
        triple.get("object_label")
        or triple.get("object")
        or triple.get("tail")
        or triple.get("o")
        or get_nested(triple, ["object", "label"])
        or get_nested(triple, ["tail", "label"])
    )

    subject_label = extract_label(subject)
    predicate_label = extract_label(predicate).upper()
    object_label = extract_label(object_)

    if not subject_label or not predicate_label or not object_label:
        return None

    evidence = (
        triple.get("justification")
        or triple.get("evidence")
        or triple.get("support_text")
        or triple.get("source_text")
        or triple.get("context")
        or ""
    )
    chunk_id = triple.get("chunk_id") or triple.get("chunkid") or triple.get("source_id") or ""
    confidence = triple.get("confidence")
    provenance = triple.get("provenance") or {}

    out = {
        "subject": {"label": subject_label},
        "predicate": {"label": predicate_label},
        "object": {"label": object_label},
        "justification": str(evidence or ""),
        "chunk_id": str(chunk_id or ""),
        "provenance": provenance,
        "raw": triple,
    }
    if isinstance(confidence, (int, float)):
        out["confidence"] = float(confidence)
    return out


def write_normalized_neoolaf_export_folder(
    *,
    source_folder: str | Path,
    output_folder: str | Path,
    document_id: str = "xquality",
) -> dict[str, Any]:
    """Write a CLI/domain-evaluator compatible NeoOLAF export folder.

    Only ``kg_inferred.json`` is written to avoid accidental duplication between
    local and inferred variants.
    """
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    raw_triples, source_path = load_triples_any(source_folder)
    normalized = []
    skipped = 0

    for triple in raw_triples:
        converted = raw_triple_to_neoolaf_export_triple(triple)
        if converted is None:
            skipped += 1
            continue
        normalized.append(converted)

    kg = {
        "document_id": document_id,
        "triples": normalized,
        "metadata": {
            "source_folder": str(source_folder),
            "source_path": str(source_path) if source_path else None,
            "raw_triple_count": len(raw_triples),
            "usable_triple_count": len(normalized),
            "skipped_triple_count": skipped,
            "normalization": "robust_ablation_neoolaf_export_schema",
        },
    }
    save_json(output_folder / "kg_inferred.json", kg)
    save_json(output_folder / "normalization_report.json", kg["metadata"])

    return {
        "source_folder": str(source_folder),
        "source_path": str(source_path) if source_path else None,
        "normalized_folder": str(output_folder),
        "raw_triple_count": len(raw_triples),
        "usable_triple_count": len(normalized),
        "skipped_triple_count": skipped,
        "kg_inferred_json": str(output_folder / "kg_inferred.json"),
    }


def parse_stop_index(name: str) -> int | None:
    match = re.search(r"prefix_stop_after_(\d+)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"layer(?:_)?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def is_completed_marker_ok(folder: Path) -> bool | None:
    """Return True/False when a marker exists, otherwise None."""
    marker = folder / "COMPLETED.json"
    if not marker.exists():
        return None

    try:
        data = load_json(marker)
    except Exception:
        return False

    if isinstance(data, dict):
        status = str(data.get("status") or data.get("state") or "").lower()
        if status in {"complete", "completed", "ok", "success", "done"}:
            return True
        if data.get("completed") is True or data.get("success") is True:
            return True
        return False

    return bool(data)


def folder_has_incomplete_marker(folder: Path) -> bool:
    for name in ["INCOMPLETE.json", "FAILED.json", "RUNNING.json", ".incomplete", ".failed", ".running"]:
        if (folder / name).exists():
            return True
    return False


def discover_prefix_folders(
    prefix_runs_dir: str | Path,
    *,
    min_triples: int = 1,
    strict_completed_marker: bool = False,
) -> list[dict[str, Any]]:
    """Discover complete-enough prefix output folders.

    If ``strict_completed_marker`` is False, older folders without a marker are
    accepted if they contain a parseable non-empty triple file and no explicit
    incomplete marker.
    """
    root = Path(prefix_runs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Prefix runs directory does not exist: {root}")

    out = []
    for folder in sorted(item for item in root.iterdir() if item.is_dir()):
        if not folder.name.startswith("prefix_stop_after_"):
            continue

        marker_status = is_completed_marker_ok(folder)
        if strict_completed_marker and marker_status is not True:
            continue
        if marker_status is False or folder_has_incomplete_marker(folder):
            continue

        triples, source_path = load_triples_any(folder)
        if len(triples) < min_triples:
            continue

        stop_index = parse_stop_index(folder.name)
        out.append(
            {
                "folder": folder,
                "layer_name": folder.name,
                "stop_index": stop_index,
                "source_path": source_path,
                "triple_count": len(triples),
            }
        )

    out.sort(key=lambda x: (x["stop_index"] if x["stop_index"] is not None else 10_000, x["layer_name"]))
    return out


def extract_flat_row(
    *,
    summary: dict[str, Any],
    robust: dict[str, Any],
    series: str,
    layer_name: str,
    stop_index: int | None,
    source_folder: str | Path,
    normalized_folder: str | Path,
    raw_triple_count: int,
    usable_triple_count: int,
) -> dict[str, Any]:
    entity = summary.get("entity", {}) or {}
    relation = summary.get("relation", {}) or {}

    row = {
        "series": series,
        "layer_name": layer_name,
        "stop_index": stop_index,
        "profile": summary.get("profile"),
        "raw_triple_count": raw_triple_count,
        "usable_triple_count": usable_triple_count,
        # Backward-compatible alias for older notebooks.
        "triple_count": raw_triple_count,
        "source_folder": str(source_folder),
        "normalized_input_folder": str(normalized_folder),
        "pred_relations_seen_by_evaluator": summary.get("pred_relations_count", 0),
        "gold_relations_seen_by_evaluator": summary.get("gt_relations_count", 0),
        "pred_entities_seen_by_evaluator": summary.get("pred_entities_count", 0),
        "gold_entities_seen_by_evaluator": summary.get("gt_entities_count", 0),
        "entity_tp": entity.get("tp", 0),
        "entity_fp": entity.get("fp", 0),
        "entity_fn": entity.get("fn", 0),
        "entity_precision": entity.get("precision", 0.0),
        "entity_recall": entity.get("recall", 0.0),
        "entity_f1": entity.get("f1", 0.0),
        "relation_tp": relation.get("tp", 0),
        "relation_fp": relation.get("fp", 0),
        "relation_fn": relation.get("fn", 0),
        "relation_precision": relation.get("precision", 0.0),
        "relation_recall": relation.get("recall", 0.0),
        "relation_f1": relation.get("f1", 0.0),
    }
    row.update(robust)
    return row


def per_relation_robust_rows(
    *,
    summary: dict[str, Any],
    series: str,
    layer_name: str,
    stop_index: int | None,
) -> list[dict[str, Any]]:
    rows = []
    for item in summary.get("per_relation", []) or []:
        if not isinstance(item, dict):
            continue
        tp = item.get("tp", 0)
        gt_count = item.get("gt_count", 0)
        pred_count = item.get("pred_count", 0)
        coverage = safe_div(tp, gt_count)
        canonical_eff = safe_div(tp, pred_count)
        rows.append(
            {
                "series": series,
                "layer_name": layer_name,
                "stop_index": stop_index,
                **item,
                "relation_coverage": coverage,
                "canonical_output_efficiency": canonical_eff,
                "coverage_canonical_efficiency_f1": harmonic_mean([coverage, canonical_eff]),
            }
        )
    return rows


def evaluate_one_output_folder(
    *,
    source_folder: str | Path,
    output_dir: str | Path,
    gold_path: str | Path,
    profile: str,
    series: str,
    layer_name: str,
    stop_index: int | None,
    document_id: str = "xquality",
) -> dict[str, Any]:
    """Normalize, evaluate, and compute robust metrics for one output folder."""
    output_dir = Path(output_dir)
    normalized_folder = output_dir / "normalized_inputs" / layer_name
    eval_output_dir = output_dir / "domain_eval" / layer_name

    norm_report = write_normalized_neoolaf_export_folder(
        source_folder=source_folder,
        output_folder=normalized_folder,
        document_id=document_id,
    )

    if norm_report["usable_triple_count"] == 0:
        raise ValueError(f"No usable triples found for {layer_name}: {source_folder}")

    summary = evaluate_domain_kg(
        input_data=EvaluationInput(
            local_json_path=None,
            inferred_json_path=Path(norm_report["kg_inferred_json"]),
            ontology_local_path=None,
            ontology_inferred_path=None,
            gold_path=gold_path,
            output_dir=eval_output_dir,
        ),
        method=build_neoolaf_json_method_config(),
        domain=build_xquality_domain_config(),
        profile=profile,
    )

    summary = add_robust_ablation_metrics_to_summary(
        summary,
        raw_triple_count=norm_report["raw_triple_count"],
    )
    robust = summary["robust_ablation"]

    row = extract_flat_row(
        summary=summary,
        robust=robust,
        series=series,
        layer_name=layer_name,
        stop_index=stop_index,
        source_folder=source_folder,
        normalized_folder=normalized_folder,
        raw_triple_count=norm_report["raw_triple_count"],
        usable_triple_count=norm_report["usable_triple_count"],
    )

    per_relation_rows = per_relation_robust_rows(
        summary=summary,
        series=series,
        layer_name=layer_name,
        stop_index=stop_index,
    )

    save_json(eval_output_dir / "metrics.robust_ablation.json", robust)
    return {"summary": summary, "row": row, "per_relation_rows": per_relation_rows, "normalization": norm_report}


def maybe_plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Write a small set of default plots. Matplotlib is optional."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    plot_df = summary_df.copy()
    plot_df = plot_df[plot_df["stop_index"].notna()].sort_values("stop_index")
    if plot_df.empty:
        return

    x = plot_df["stop_index"].astype(int)

    for y_col, title, ylabel, filename in [
        ("gold_relation_coverage", "Relaxed gold relation coverage by stop layer", "Coverage", "gold_relation_coverage.png"),
        ("raw_output_efficiency", "Raw output efficiency by stop layer", "Matched gold relations / raw triples", "raw_output_efficiency.png"),
        ("coverage_efficiency_f1", "Coverage-efficiency F1 by stop layer", "Coverage-efficiency F1", "coverage_efficiency_f1.png"),
        ("entity_relation_balanced_f1", "Entity-adjusted KG score by stop layer", "Entity-adjusted KG score", "entity_adjusted_kg_score.png"),
        ("raw_triple_count", "Raw triple count by stop layer", "Raw triples", "raw_triple_count.png"),
    ]:
        if y_col not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, plot_df[y_col], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Stop layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def evaluate_prefix_stop_after_layer_ablation(
    *,
    prefix_runs_dir: str | Path,
    gold_path: str | Path,
    output_dir: str | Path,
    profile: str = "xquality_relaxed_recall",
    final_export_dir: str | Path | None = None,
    min_triples: int = 1,
    strict_completed_marker: bool = False,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Evaluate all completed prefix outputs plus an optional final export."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_folders = discover_prefix_folders(
        prefix_runs_dir,
        min_triples=min_triples,
        strict_completed_marker=strict_completed_marker,
    )

    rows: list[dict[str, Any]] = []
    per_relation_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for rec in prefix_folders:
        try:
            result = evaluate_one_output_folder(
                source_folder=rec["folder"],
                output_dir=output_dir,
                gold_path=gold_path,
                profile=profile,
                series="prefix_stop_after_layer_generated_output",
                layer_name=rec["layer_name"],
                stop_index=rec["stop_index"],
            )
            rows.append(result["row"])
            per_relation_rows.extend(result["per_relation_rows"])
        except Exception as exc:  # keep batch runs useful
            errors.append({"layer_name": rec["layer_name"], "source_folder": str(rec["folder"]), "error": repr(exc)})

    if final_export_dir is not None:
        try:
            result = evaluate_one_output_folder(
                source_folder=final_export_dir,
                output_dir=output_dir,
                gold_path=gold_path,
                profile=profile,
                series="native_neoolaf_final_export",
                layer_name="native_neoolaf_final_export",
                stop_index=12,
            )
            rows.append(result["row"])
            per_relation_rows.extend(result["per_relation_rows"])
        except Exception as exc:
            errors.append({"layer_name": "native_neoolaf_final_export", "source_folder": str(final_export_dir), "error": repr(exc)})

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["series", "stop_index", "layer_name"], na_position="last")
    per_relation_df = pd.DataFrame(per_relation_rows)

    summary_csv = output_dir / "robust_ablation_summary.csv"
    per_relation_csv = output_dir / "robust_ablation_per_relation.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_relation_df.to_csv(per_relation_csv, index=False)
    save_json(output_dir / "robust_ablation_errors.json", errors)
    save_json(output_dir / "robust_ablation_manifest.json", {
        "prefix_runs_dir": str(prefix_runs_dir),
        "gold_path": str(gold_path),
        "final_export_dir": str(final_export_dir) if final_export_dir else None,
        "profile": profile,
        "min_triples": min_triples,
        "strict_completed_marker": strict_completed_marker,
        "evaluated_rows": len(rows),
        "errors": errors,
    })

    if make_plots and not summary_df.empty:
        maybe_plot_summary(summary_df, output_dir)

    return {
        "summary_df": summary_df,
        "per_relation_df": per_relation_df,
        "errors": errors,
        "summary_csv": summary_csv,
        "per_relation_csv": per_relation_csv,
        "output_dir": output_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust prefix stop-after-layer ablation evaluation")
    parser.add_argument("--prefix-runs-dir", required=True, help="Directory containing prefix_stop_after_* folders")
    parser.add_argument("--gold", required=True, help="XQuality gold JSON path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--profile", default="xquality_relaxed_recall", help="Evaluation profile label")
    parser.add_argument("--final-export-dir", default=None, help="Optional native final export folder")
    parser.add_argument("--min-triples", type=int, default=1, help="Minimum raw triples required to evaluate a folder")
    parser.add_argument(
        "--strict-completed-marker",
        action="store_true",
        help="Evaluate only folders with a positive COMPLETED.json marker",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib plot generation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_prefix_stop_after_layer_ablation(
        prefix_runs_dir=args.prefix_runs_dir,
        gold_path=args.gold,
        output_dir=args.output,
        profile=args.profile,
        final_export_dir=args.final_export_dir,
        min_triples=args.min_triples,
        strict_completed_marker=args.strict_completed_marker,
        make_plots=not args.no_plots,
    )
    print(f"Wrote summary: {result['summary_csv']}")
    print(f"Wrote per-relation metrics: {result['per_relation_csv']}")
    if result["errors"]:
        print(f"Warnings/errors: {len(result['errors'])}. See robust_ablation_errors.json")


if __name__ == "__main__":
    main()

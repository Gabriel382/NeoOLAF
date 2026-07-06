"""Evaluate cumulative, merged, budgeted prefix consolidation outputs.

This runner is meant for NeoOLAF stop-after-layer diagnostics where each stop
point should include the accumulated evidence from previous prefix outputs.
It creates a compact consolidated KG for each stop layer and evaluates it with
NeoOLAF's existing XQuality/domain evaluator and the robust ablation metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from neoolaf.evaluation.metrics.cumulative_consolidation import (
    DEFAULT_ALLOWED_RELATIONS,
    IncrementalConsolidator,
    build_triple_record,
)
from neoolaf.evaluation.runners.evaluate_ablation_robust import (
    discover_prefix_folders,
    evaluate_one_output_folder,
    load_triples_any,
    save_json,
)


def write_consolidated_export_folder(
    *,
    output_folder: str | Path,
    triples: list[dict[str, Any]],
    metadata: dict[str, Any],
    document_id: str = "xquality",
) -> None:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    kg = {
        "document_id": document_id,
        "triples": triples,
        "metadata": {
            **metadata,
            "normalization": "cumulative_prefix_consolidation",
            "raw_triple_count": len(triples),
            "usable_triple_count": len(triples),
        },
    }
    # Write multiple common layouts for easy inspection; the robust evaluator
    # will use kg_inferred.json first after its own normalization step.
    save_json(output_folder / "triples.json", triples)
    save_json(output_folder / "kg.json", kg)
    save_json(output_folder / "kg_inferred.json", kg)
    save_json(output_folder / "consolidation_metadata.json", kg["metadata"])
    save_json(output_folder / "COMPLETED.json", {"status": "completed", "success": True})


def build_cumulative_consolidated_outputs(
    *,
    prefix_runs_dir: str | Path,
    output_dir: str | Path,
    budget: int | None = 226,
    similarity_threshold: float = 0.86,
    min_score: float | None = None,
    min_triples: int = 1,
    strict_completed_marker: bool = False,
    allowed_relations: set[str] | None = None,
    document_id: str = "xquality",
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Create cumulative consolidated KG folders for every discovered stop layer."""
    output_dir = Path(output_dir)
    cumulative_root = output_dir / "cumulative_consolidated_outputs"
    cumulative_root.mkdir(parents=True, exist_ok=True)

    prefix_folders = discover_prefix_folders(
        prefix_runs_dir,
        min_triples=min_triples,
        strict_completed_marker=strict_completed_marker,
    )
    if not prefix_folders:
        raise ValueError(f"No prefix folders found in {prefix_runs_dir}")

    if verbose:
        print(f"Discovered {len(prefix_folders)} prefix folders in {prefix_runs_dir}")
        for rec in prefix_folders:
            print(f"  L{int(rec['stop_index']):02d}: {rec['layer_name']} | triples={rec.get('triple_count', '?')}")

    consolidator = IncrementalConsolidator(
        similarity_threshold=similarity_threshold,
        allowed_relations=allowed_relations or set(DEFAULT_ALLOWED_RELATIONS),
    )

    outputs: list[dict[str, Any]] = []
    load_reports: list[dict[str, Any]] = []

    layer_iter = prefix_folders
    if show_progress and tqdm is not None:
        layer_iter = tqdm(prefix_folders, desc="Cumulative stop layers")

    for rec in layer_iter:
        stop_index = int(rec["stop_index"])
        layer_label = f"L{stop_index:02d} {rec['layer_name']}"
        if verbose:
            print(f"\n[{layer_label}] loading triples...")

        triples, source_path = load_triples_any(rec["folder"])
        usable_records = []
        skipped = 0
        parse_iter = enumerate(triples)
        if show_progress and tqdm is not None:
            parse_iter = enumerate(tqdm(triples, desc=f"Parsing {layer_label}", leave=False))
        for idx, triple in parse_iter:
            record = build_triple_record(
                triple,
                source_stop_index=stop_index,
                source_layer_name=str(rec["layer_name"]),
                raw_index=idx,
            )
            if record is None:
                skipped += 1
                continue
            usable_records.append(record)

        if verbose:
            print(f"[{layer_label}] raw={len(triples)} usable={len(usable_records)} skipped={skipped}; existing_clusters={len(consolidator.clusters)}")

        consolidator.add_records(
            usable_records,
            show_progress=show_progress,
            progress_desc=f"Merging {layer_label}",
        )

        if verbose:
            print(f"[{layer_label}] clusters_after_merge={len(consolidator.clusters)}; selecting/exporting...")

        selected_triples, metadata = consolidator.selected_export_triples(
            stop_index=int(rec["stop_index"]),
            budget=budget,
            min_score=min_score,
        )
        metadata.update({
            "source_prefix_folder": str(rec["folder"]),
            "source_prefix_path": str(source_path) if source_path else None,
            "source_prefix_raw_triples": len(triples),
            "source_prefix_usable_records": len(usable_records),
            "source_prefix_skipped_records": skipped,
            "cumulative_raw_records_seen": sum(item["usable_records"] for item in load_reports) + len(usable_records),
            "similarity_threshold": similarity_threshold,
        })

        layer_name = f"cumulative_stop_after_{int(rec['stop_index']):02d}_{rec['layer_name']}"
        out_folder = cumulative_root / layer_name
        write_consolidated_export_folder(
            output_folder=out_folder,
            triples=selected_triples,
            metadata=metadata,
            document_id=document_id,
        )

        outputs.append({
            "folder": out_folder,
            "layer_name": layer_name,
            "stop_index": int(rec["stop_index"]),
            "raw_triple_count": len(selected_triples),
            "candidate_cluster_count": metadata["candidate_cluster_count"],
            "selected_cluster_count": metadata["selected_cluster_count"],
            "avg_cluster_score": metadata["avg_cluster_score"],
            "avg_support_count": metadata["avg_support_count"],
            "avg_support_layers": metadata["avg_support_layers"],
            "max_support_layers": metadata["max_support_layers"],
            "source_prefix_folder": str(rec["folder"]),
        })
        load_reports.append({
            "stop_index": int(rec["stop_index"]),
            "layer_name": str(rec["layer_name"]),
            "raw_triples": len(triples),
            "usable_records": len(usable_records),
            "skipped_records": skipped,
        })

        if verbose:
            print(
                f"[{layer_label}] wrote {len(selected_triples)} triples | "
                f"candidate_clusters={metadata['candidate_cluster_count']} | "
                f"selected_clusters={metadata['selected_cluster_count']}"
            )

    save_json(output_dir / "cumulative_consolidation_load_report.json", load_reports)
    save_json(output_dir / "cumulative_consolidation_manifest.json", {
        "prefix_runs_dir": str(prefix_runs_dir),
        "output_dir": str(output_dir),
        "budget": budget,
        "similarity_threshold": similarity_threshold,
        "min_score": min_score,
        "strict_completed_marker": strict_completed_marker,
        "min_triples": min_triples,
        "allowed_relations": sorted(allowed_relations or set(DEFAULT_ALLOWED_RELATIONS)),
        "outputs": [{**x, "folder": str(x["folder"])} for x in outputs],
    })

    return {
        "outputs": outputs,
        "load_reports": load_reports,
        "cumulative_root": cumulative_root,
    }


def evaluate_cumulative_prefix_consolidation(
    *,
    prefix_runs_dir: str | Path,
    gold_path: str | Path,
    output_dir: str | Path,
    profile: str = "xquality_relaxed_recall",
    final_export_dir: str | Path | None = None,
    budget: int | None = 226,
    similarity_threshold: float = 0.86,
    min_score: float | None = None,
    min_triples: int = 1,
    strict_completed_marker: bool = False,
    allowed_relations: set[str] | None = None,
    make_plots: bool = True,
    show_progress: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build cumulative consolidated outputs, evaluate them, and save tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build = build_cumulative_consolidated_outputs(
        prefix_runs_dir=prefix_runs_dir,
        output_dir=output_dir,
        budget=budget,
        similarity_threshold=similarity_threshold,
        min_score=min_score,
        min_triples=min_triples,
        strict_completed_marker=strict_completed_marker,
        allowed_relations=allowed_relations,
        show_progress=show_progress,
        verbose=verbose,
    )

    rows: list[dict[str, Any]] = []
    per_relation_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    by_layer = {item["layer_name"]: item for item in build["outputs"]}

    eval_iter = build["outputs"]
    if show_progress and tqdm is not None:
        eval_iter = tqdm(build["outputs"], desc="Evaluating consolidated outputs")

    for rec in eval_iter:
        if verbose:
            print(f"Evaluating {rec['layer_name']} ...")
        try:
            result = evaluate_one_output_folder(
                source_folder=rec["folder"],
                output_dir=output_dir / "robust_eval",
                gold_path=gold_path,
                profile=profile,
                series="cumulative_consolidated_prefix_output",
                layer_name=rec["layer_name"],
                stop_index=rec["stop_index"],
            )
            row = result["row"]
            for key in [
                "candidate_cluster_count",
                "selected_cluster_count",
                "avg_cluster_score",
                "avg_support_count",
                "avg_support_layers",
                "max_support_layers",
                "source_prefix_folder",
            ]:
                row[key] = rec.get(key)
            rows.append(row)
            per_relation_rows.extend(result["per_relation_rows"])
        except Exception as exc:
            errors.append({"layer_name": rec["layer_name"], "source_folder": str(rec["folder"]), "error": repr(exc)})

    if final_export_dir is not None:
        try:
            result = evaluate_one_output_folder(
                source_folder=final_export_dir,
                output_dir=output_dir / "robust_eval",
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

    summary_csv = output_dir / "cumulative_consolidated_robust_summary.csv"
    per_relation_csv = output_dir / "cumulative_consolidated_robust_per_relation.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_relation_df.to_csv(per_relation_csv, index=False)
    save_json(output_dir / "cumulative_consolidated_errors.json", errors)

    if make_plots and not summary_df.empty:
        try:
            import matplotlib.pyplot as plt
            plot_df = summary_df[summary_df["series"] == "cumulative_consolidated_prefix_output"].copy()
            plot_df = plot_df[plot_df["stop_index"].notna()].sort_values("stop_index")
            final_df = summary_df[summary_df["series"] == "native_neoolaf_final_export"].copy()
            final_row = final_df.iloc[0] if len(final_df) else None
            for metric, title, ylabel, fname in [
                ("gold_relation_coverage", "Cumulative consolidated gold relation coverage", "Coverage", "cumulative_gold_relation_coverage.png"),
                ("raw_output_efficiency", "Cumulative consolidated output efficiency", "Efficiency", "cumulative_raw_output_efficiency.png"),
                ("coverage_efficiency_f1", "Cumulative consolidated coverage-efficiency F1", "CE-F1", "cumulative_coverage_efficiency_f1.png"),
                ("entity_relation_balanced_f1", "Cumulative consolidated entity-adjusted KG score", "Score", "cumulative_entity_adjusted_kg_score.png"),
            ]:
                if metric not in plot_df:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(plot_df["stop_index"], plot_df[metric], marker="o", label="Cumulative consolidated prefix")
                if final_row is not None and metric in final_row:
                    ax.scatter([12], [final_row[metric]], marker="*", s=220, label="Native final export")
                    ax.axhline(final_row[metric], linestyle="--", alpha=0.6, label="Final export reference")
                ax.set_title(title)
                ax.set_xlabel("Stop layer")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / fname, dpi=180)
                plt.close(fig)
        except Exception:
            pass

    return {
        "summary_df": summary_df,
        "per_relation_df": per_relation_df,
        "errors": errors,
        "summary_csv": summary_csv,
        "per_relation_csv": per_relation_csv,
        "output_dir": output_dir,
        "build": build,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate cumulative consolidated prefix outputs")
    parser.add_argument("--prefix-runs-dir", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="xquality_relaxed_recall")
    parser.add_argument("--final-export-dir", default=None)
    parser.add_argument("--budget", type=int, default=226)
    parser.add_argument("--similarity-threshold", type=float, default=0.86)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--min-triples", type=int, default=1)
    parser.add_argument("--strict-completed-marker", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_cumulative_prefix_consolidation(
        prefix_runs_dir=args.prefix_runs_dir,
        gold_path=args.gold,
        output_dir=args.output,
        profile=args.profile,
        final_export_dir=args.final_export_dir,
        budget=args.budget,
        similarity_threshold=args.similarity_threshold,
        min_score=args.min_score,
        min_triples=args.min_triples,
        strict_completed_marker=args.strict_completed_marker,
        make_plots=not args.no_plots,
        show_progress=not args.no_progress,
        verbose=not args.quiet,
    )
    print(f"Wrote summary: {result['summary_csv']}")
    print(f"Wrote per-relation: {result['per_relation_csv']}")
    if result["errors"]:
        print(f"Errors/warnings: {len(result['errors'])}")


if __name__ == "__main__":
    main()

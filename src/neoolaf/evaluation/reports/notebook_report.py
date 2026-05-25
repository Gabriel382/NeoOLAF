from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file if it exists."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file if it exists."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save_current_fig(output_path: Path | None) -> None:
    """Save the current matplotlib figure if a path is provided."""
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")


def print_main_metrics(summary: dict[str, Any]) -> None:
    """Print main extraction and validation metrics in a readable format."""
    entity = summary.get("entity", {})
    relation = summary.get("relation", {})
    validation = summary.get("validation_metrics_mean", {})

    print("=" * 70)
    print("NEOOLAF EVALUATION SUMMARY")
    print("=" * 70)

    print("\n[Entity Extraction]")
    print(f"Precision: {entity.get('precision', 0.0):.4f}")
    print(f"Recall:    {entity.get('recall', 0.0):.4f}")
    print(f"F1:        {entity.get('f1', 0.0):.4f}")
    print(f"TP/FP/FN:  {entity.get('tp', 0)} / {entity.get('fp', 0)} / {entity.get('fn', 0)}")

    print("\n[Relation Extraction]")
    print(f"Precision: {relation.get('precision', 0.0):.4f}")
    print(f"Recall:    {relation.get('recall', 0.0):.4f}")
    print(f"F1:        {relation.get('f1', 0.0):.4f}")
    print(f"TP/FP/FN:  {relation.get('tp', 0)} / {relation.get('fp', 0)} / {relation.get('fn', 0)}")

    print("\n[Validation-Oriented Metrics]")
    print(f"STR: {validation.get('STR', 0.0):.4f}")
    print(f"CR:  {validation.get('CR', 0.0):.4f}")
    print(f"PC:  {validation.get('PC', 0.0):.4f}")
    print(f"OC:  {validation.get('OC', 0.0):.4f}")
    print(f"CV:  {validation.get('CV', 0.0):.4f}")
    print(f"DVS: {validation.get('DVS', 0.0):.4f}")

    print("\n[Counts]")
    print(f"Total documents:      {summary.get('total_docs', 0)}")
    print(f"Missing predictions:  {summary.get('missing_predictions', 0)}")
    print(f"Parsed failures:      {summary.get('parsed_failures', 0)}")
    print(f"Predicted entities:   {summary.get('pred_entities_count', 0)}")
    print(f"Gold entities:        {summary.get('gt_entities_count', 0)}")
    print(f"Predicted relations:  {summary.get('pred_relations_count', 0)}")
    print(f"Gold relations:       {summary.get('gt_relations_count', 0)}")
    print("=" * 70)


def make_summary_dataframe(summary: dict[str, Any]) -> pd.DataFrame:
    """Create a compact summary dataframe."""
    entity = summary.get("entity", {})
    relation = summary.get("relation", {})
    validation = summary.get("validation_metrics_mean", {})

    rows = [
        {
            "Metric Group": "Entity",
            "Precision": entity.get("precision", 0.0),
            "Recall": entity.get("recall", 0.0),
            "F1": entity.get("f1", 0.0),
            "TP": entity.get("tp", 0),
            "FP": entity.get("fp", 0),
            "FN": entity.get("fn", 0),
        },
        {
            "Metric Group": "Relation",
            "Precision": relation.get("precision", 0.0),
            "Recall": relation.get("recall", 0.0),
            "F1": relation.get("f1", 0.0),
            "TP": relation.get("tp", 0),
            "FP": relation.get("fp", 0),
            "FN": relation.get("fn", 0),
        },
    ]

    df = pd.DataFrame(rows)

    validation_df = pd.DataFrame(
        [
            {
                "STR": validation.get("STR", 0.0),
                "CR": validation.get("CR", 0.0),
                "PC": validation.get("PC", 0.0),
                "OC": validation.get("OC", 0.0),
                "CV": validation.get("CV", 0.0),
                "DVS": validation.get("DVS", 0.0),
            }
        ]
    )

    return df, validation_df


def plot_extraction_metrics(
    summary: dict[str, Any],
    save_path: Path | None = None,
) -> None:
    """Plot entity and relation precision/recall/F1."""
    entity = summary.get("entity", {})
    relation = summary.get("relation", {})

    df = pd.DataFrame(
        [
            {
                "Target": "Entity",
                "Precision": entity.get("precision", 0.0),
                "Recall": entity.get("recall", 0.0),
                "F1": entity.get("f1", 0.0),
            },
            {
                "Target": "Relation",
                "Precision": relation.get("precision", 0.0),
                "Recall": relation.get("recall", 0.0),
                "F1": relation.get("f1", 0.0),
            },
        ]
    )

    ax = df.set_index("Target").plot(kind="bar", figsize=(8, 5))
    ax.set_title("NeoOLAF Extraction Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    _save_current_fig(save_path)
    plt.show()


def plot_validation_metrics(
    summary: dict[str, Any],
    save_path: Path | None = None,
) -> None:
    """Plot STR, CR, PC, OC, CV, and DVS."""
    validation = summary.get("validation_metrics_mean", {})

    df = pd.DataFrame(
        [
            {
                "Metric": "STR",
                "Value": validation.get("STR", 0.0),
            },
            {
                "Metric": "CR",
                "Value": validation.get("CR", 0.0),
            },
            {
                "Metric": "PC",
                "Value": validation.get("PC", 0.0),
            },
            {
                "Metric": "OC",
                "Value": validation.get("OC", 0.0),
            },
            {
                "Metric": "CV",
                "Value": validation.get("CV", 0.0),
            },
            {
                "Metric": "DVS",
                "Value": validation.get("DVS", 0.0),
            },
        ]
    )

    ax = df.plot(x="Metric", y="Value", kind="bar", legend=False, figsize=(8, 5))
    ax.set_title("NeoOLAF Validation-Oriented Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    _save_current_fig(save_path)
    plt.show()


def plot_per_relation_metrics(
    per_relation_df: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """Plot per-relation F1 scores."""
    if per_relation_df.empty:
        print("[WARNING] per_relation_metrics.csv is empty or missing.")
        return

    df = per_relation_df.copy()

    if "relation" not in df.columns or "f1" not in df.columns:
        print("[WARNING] per-relation dataframe must contain 'relation' and 'f1'.")
        return

    ax = df.plot(x="relation", y="f1", kind="bar", legend=False, figsize=(9, 5))
    ax.set_title("NeoOLAF Per-Relation F1")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    _save_current_fig(save_path)
    plt.show()


def generate_markdown_report(
    summary: dict[str, Any],
    per_relation_df: pd.DataFrame,
    ontology_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a simple printable Markdown report."""
    entity = summary.get("entity", {})
    relation = summary.get("relation", {})
    validation = summary.get("validation_metrics_mean", {})

    lines = []
    lines.append("# NeoOLAF Evaluation Report")
    lines.append("")
    lines.append("## Main Extraction Metrics")
    lines.append("")
    lines.append("| Target | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Entity | {entity.get('precision', 0.0):.4f} | {entity.get('recall', 0.0):.4f} | "
        f"{entity.get('f1', 0.0):.4f} | {entity.get('tp', 0)} | {entity.get('fp', 0)} | {entity.get('fn', 0)} |"
    )
    lines.append(
        f"| Relation | {relation.get('precision', 0.0):.4f} | {relation.get('recall', 0.0):.4f} | "
        f"{relation.get('f1', 0.0):.4f} | {relation.get('tp', 0)} | {relation.get('fp', 0)} | {relation.get('fn', 0)} |"
    )

    lines.append("")
    lines.append("## Validation-Oriented Metrics")
    lines.append("")
    lines.append("| STR | CR | PC | OC | CV | DVS |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {validation.get('STR', 0.0):.4f} | {validation.get('CR', 0.0):.4f} | "
        f"{validation.get('PC', 0.0):.4f} | {validation.get('OC', 0.0):.4f} | "
        f"{validation.get('CV', 0.0):.4f} | {validation.get('DVS', 0.0):.4f} |"
    )

    if not per_relation_df.empty:
        lines.append("")
        lines.append("## Per-Relation Metrics")
        lines.append("")
        lines.append(per_relation_df.to_markdown(index=False))

    if ontology_metrics:
        lines.append("")
        lines.append("## Ontology Metrics")
        lines.append("")
        ontology_df = pd.DataFrame([ontology_metrics])
        lines.append(ontology_df.to_markdown(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def show_neoolaf_evaluation_report(
    output_dir: str | Path,
    save_figures: bool = True,
    generate_markdown: bool = True,
) -> dict[str, Any]:
    """
    Load and display a printable NeoOLAF evaluation report from an output directory.

    Expected files:
    - metrics.summary.json
    - metrics.flat.csv
    - per_relation_metrics.csv
    - validation_metrics.csv
    - ontology_metrics.json
    """
    output_dir = Path(output_dir)

    summary = _load_json(output_dir / "metrics.summary.json")
    ontology_metrics = _load_json(output_dir / "ontology_metrics.json")

    flat_df = _load_csv(output_dir / "metrics.flat.csv")
    per_relation_df = _load_csv(output_dir / "per_relation_metrics.csv")
    validation_df = _load_csv(output_dir / "validation_metrics.csv")

    if not summary:
        raise FileNotFoundError(f"Could not find metrics.summary.json in {output_dir}")

    figures_dir = output_dir / "figures" if save_figures else None

    print_main_metrics(summary)

    summary_df, validation_summary_df = make_summary_dataframe(summary)

    print("\nMain metrics table:")
    display(summary_df)

    print("\nValidation metrics table:")
    display(validation_summary_df)

    if not per_relation_df.empty:
        print("\nPer-relation metrics:")
        display(per_relation_df)

    if not validation_df.empty:
        print("\nPer-document validation metrics:")
        display(validation_df)

    if ontology_metrics:
        print("\nOntology metrics:")
        display(pd.DataFrame([ontology_metrics]))

    plot_extraction_metrics(
        summary,
        save_path=(figures_dir / "extraction_metrics.png") if figures_dir else None,
    )

    plot_validation_metrics(
        summary,
        save_path=(figures_dir / "validation_metrics.png") if figures_dir else None,
    )

    plot_per_relation_metrics(
        per_relation_df,
        save_path=(figures_dir / "per_relation_f1.png") if figures_dir else None,
    )

    if generate_markdown:
        generate_markdown_report(
            summary=summary,
            per_relation_df=per_relation_df,
            ontology_metrics=ontology_metrics,
            output_path=output_dir / "printable_report.md",
        )
        print(f"\nPrintable Markdown report saved to: {output_dir / 'printable_report.md'}")

    if save_figures:
        print(f"Figures saved to: {output_dir / 'figures'}")

    return {
        "summary": summary,
        "summary_df": summary_df,
        "validation_summary_df": validation_summary_df,
        "flat_df": flat_df,
        "per_relation_df": per_relation_df,
        "validation_df": validation_df,
        "ontology_metrics": ontology_metrics,
    }
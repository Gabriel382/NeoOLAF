from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table without requiring tabulate."""
    if df.empty:
        return "_No data available._"

    columns = [str(col) for col in df.columns]

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]

            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file safely."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV file safely."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save_figure(path: Path | None) -> None:
    """Save current figure if path is provided."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")


def print_summary(summary: dict[str, Any]) -> None:
    """Print main evaluation summary."""
    entity = summary.get("entity", {})
    relation = summary.get("relation", {})
    validation = summary.get("validation_metrics_mean", {})

    print("=" * 80)
    print("NEOOLAF EVALUATION REPORT")
    print("=" * 80)

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

    print("=" * 80)


def build_main_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Build compact extraction summary table."""
    entity = summary.get("entity", {})
    relation = summary.get("relation", {})

    return pd.DataFrame(
        [
            {
                "Target": "Entity",
                "Precision": entity.get("precision", 0.0),
                "Recall": entity.get("recall", 0.0),
                "F1": entity.get("f1", 0.0),
                "TP": entity.get("tp", 0),
                "FP": entity.get("fp", 0),
                "FN": entity.get("fn", 0),
            },
            {
                "Target": "Relation",
                "Precision": relation.get("precision", 0.0),
                "Recall": relation.get("recall", 0.0),
                "F1": relation.get("f1", 0.0),
                "TP": relation.get("tp", 0),
                "FP": relation.get("fp", 0),
                "FN": relation.get("fn", 0),
            },
        ]
    )


def build_validation_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Build compact validation metrics table."""
    validation = summary.get("validation_metrics_mean", {})

    return pd.DataFrame(
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


def plot_extraction(summary: dict[str, Any], save_path: Path | None = None) -> None:
    """Plot entity and relation P/R/F1."""
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
    ax.set_title("Extraction Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    save_figure(save_path)
    plt.show()


def plot_validation(summary: dict[str, Any], save_path: Path | None = None) -> None:
    """Plot validation-oriented metrics."""
    validation = summary.get("validation_metrics_mean", {})

    df = pd.DataFrame(
        [
            {"Metric": "STR", "Value": validation.get("STR", 0.0)},
            {"Metric": "CR", "Value": validation.get("CR", 0.0)},
            {"Metric": "PC", "Value": validation.get("PC", 0.0)},
            {"Metric": "OC", "Value": validation.get("OC", 0.0)},
            {"Metric": "CV", "Value": validation.get("CV", 0.0)},
            {"Metric": "DVS", "Value": validation.get("DVS", 0.0)},
        ]
    )

    ax = df.plot(x="Metric", y="Value", kind="bar", legend=False, figsize=(8, 5))
    ax.set_title("Validation-Oriented Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    save_figure(save_path)
    plt.show()


def plot_per_relation(per_relation_df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Plot per-relation F1."""
    if per_relation_df.empty:
        print("[WARNING] per_relation_metrics.csv is missing or empty.")
        return

    if "relation" not in per_relation_df.columns or "f1" not in per_relation_df.columns:
        print("[WARNING] per_relation_metrics.csv must contain columns: relation, f1.")
        return

    ax = per_relation_df.plot(
        x="relation",
        y="f1",
        kind="bar",
        legend=False,
        figsize=(9, 5),
    )
    ax.set_title("Per-Relation F1")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    save_figure(save_path)
    plt.show()


def write_markdown_report(
    output_dir: Path,
    summary: dict[str, Any],
    main_table: pd.DataFrame,
    validation_table: pd.DataFrame,
    per_relation_df: pd.DataFrame,
    ontology_metrics: dict[str, Any],
) -> Path:
    """Write printable Markdown report."""
    report_path = output_dir / "printable_report.md"

    lines = [
        "# NeoOLAF Evaluation Report",
        "",
        "## Main Metrics",
        "",
        dataframe_to_markdown(main_table),
        "",
        "## Validation-Oriented Metrics",
        "",
        dataframe_to_markdown(validation_table),
    ]

    if not per_relation_df.empty:
        lines.extend(
            [
                "",
                "## Per-Relation Metrics",
                "",
                dataframe_to_markdown(per_relation_df),
            ]
        )

    if ontology_metrics:
        ontology_df = pd.DataFrame([ontology_metrics])
        lines.extend(
            [
                "",
                "## Ontology Metrics",
                "",
                dataframe_to_markdown(ontology_df),
            ]
        )

    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Total documents: {summary.get('total_docs', 0)}",
            f"- Missing predictions: {summary.get('missing_predictions', 0)}",
            f"- Parsed failures: {summary.get('parsed_failures', 0)}",
            f"- Predicted entities: {summary.get('pred_entities_count', 0)}",
            f"- Gold entities: {summary.get('gt_entities_count', 0)}",
            f"- Predicted relations: {summary.get('pred_relations_count', 0)}",
            f"- Gold relations: {summary.get('gt_relations_count', 0)}",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print and plot an evaluation report from an evaluation output directory."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Evaluation output directory containing metrics.summary.json.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save figures under <input>/figures/.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Generate printable_report.md.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not display matplotlib plots.",
    )

    args = parser.parse_args()

    output_dir = Path(args.input)
    summary = load_json(output_dir / "metrics.summary.json")
    per_relation_df = load_csv(output_dir / "per_relation_metrics.csv")
    validation_df = load_csv(output_dir / "validation_metrics.csv")
    ontology_metrics = load_json(output_dir / "ontology_metrics.json")

    if not summary:
        raise FileNotFoundError(f"Could not find metrics.summary.json in {output_dir}")

    figures_dir = output_dir / "figures" if args.save_figures else None

    print_summary(summary)

    main_table = build_main_table(summary)
    validation_table = build_validation_table(summary)

    print("\nMain metrics table:")
    print(main_table.to_string(index=False))

    print("\nValidation metrics table:")
    print(validation_table.to_string(index=False))

    if not per_relation_df.empty:
        print("\nPer-relation metrics:")
        print(per_relation_df.to_string(index=False))

    if not validation_df.empty:
        print("\nPer-document validation metrics:")
        print(validation_df.to_string(index=False))

    if ontology_metrics:
        print("\nOntology metrics:")
        print(pd.DataFrame([ontology_metrics]).to_string(index=False))

    if not args.no_plots:
        plot_extraction(
            summary,
            save_path=(figures_dir / "extraction_metrics.png") if figures_dir else None,
        )
        plot_validation(
            summary,
            save_path=(figures_dir / "validation_metrics.png") if figures_dir else None,
        )
        plot_per_relation(
            per_relation_df,
            save_path=(figures_dir / "per_relation_f1.png") if figures_dir else None,
        )

    if args.markdown:
        report_path = write_markdown_report(
            output_dir=output_dir,
            summary=summary,
            main_table=main_table,
            validation_table=validation_table,
            per_relation_df=per_relation_df,
            ontology_metrics=ontology_metrics,
        )
        print(f"\nMarkdown report saved to: {report_path}")

    if args.save_figures:
        print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
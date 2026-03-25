"""
Generate all reports (text, JSON, tables, plots) from evaluation results.

Usage:
    python -m tests.run_reports
"""
from __future__ import annotations

import json
import os

from neoolaf.evaluation.reports.summary import (
    generate_summary_text,
    generate_summary_dict,
)
from neoolaf.evaluation.reports.tables import all_tables
from neoolaf.evaluation.reports.plots import generate_all_plots

from tests.run_evaluation import run_all_metrics

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "evaluation_output")


def generate_all_reports():
    """Run metrics then generate all reports and save to evaluation_output/."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run all evaluation metrics first
    data = run_all_metrics()
    state = data["state"]
    validation = data["validation"]
    faithfulness = data["faithfulness"]
    bleu = data["bleu"]
    alignment = data["alignment"]
    results = data["results"]

    # 1. Text summary
    print("\n8. Generating text summary report...")
    summary_text = generate_summary_text(state, validation, faithfulness, bleu, alignment)
    summary_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"   Saved to {summary_path}")

    # 2. JSON summary
    print("\n9. Generating JSON summary report...")
    summary_dict = generate_summary_dict(state, validation, faithfulness, bleu, alignment)
    json_path = os.path.join(OUTPUT_DIR, "summary_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
    print(f"   Saved to {json_path}")

    # 3. Tables
    print("\n10. Generating tables...")
    tables_text = all_tables(validation, faithfulness, bleu, alignment)
    tables_path = os.path.join(OUTPUT_DIR, "tables_report.txt")
    with open(tables_path, "w", encoding="utf-8") as f:
        f.write(tables_text)
    print(f"   Saved to {tables_path}")

    # 4. Plots
    print("\n11. Generating plots...")
    confidences = [t.confidence for t in state.candidate_triples if t.confidence is not None]
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    plot_paths = generate_all_plots(validation, faithfulness, bleu, confidences, plots_dir, alignment)
    for p in plot_paths:
        print(f"   Saved: {p}")

    # 5. All metrics JSON
    all_metrics_path = os.path.join(OUTPUT_DIR, "all_metrics.json")
    with open(all_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n12. All metrics saved to {all_metrics_path}")

    print("\n" + "=" * 60)
    print("All evaluations and reports completed successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_reports()

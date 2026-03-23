from __future__ import annotations

# Standard library imports
import os
from typing import Optional

# Third-party imports
import seaborn as sns

# Local imports
from neoolaf.evaluation.no_gold.validation_outcomes import ValidationOutcomes
from neoolaf.evaluation.no_gold.faithfulness import FaithfulnessReport
from neoolaf.evaluation.no_gold.bleu_score import BleuReport
from neoolaf.evaluation.no_gold.ontology_alignment import OntologyAlignmentReport

# Set seaborn theme globally for all plots
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def plot_issue_distribution(
    validation: ValidationOutcomes,
    output_dir: str,
    filename: str = "issues_distribution.png",
) -> str:
    """
    Pie chart of validation issue types.

    Returns path to the saved file.
    """
    import matplotlib.pyplot as plt

    if not validation.issues_by_type:
        return ""

    _ensure_dir(output_dir)

    labels = list(validation.issues_by_type.keys())
    sizes = list(validation.issues_by_type.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Validation Issue Distribution")

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confidence_histogram(
    validation: ValidationOutcomes,
    confidences: list[float],
    output_dir: str,
    filename: str = "confidence_histogram.png",
) -> str:
    """
    Histogram of triple confidence scores using seaborn.

    Args:
        validation:  ValidationOutcomes (used for title stats).
        confidences: List of confidence floats from candidate triples.
        output_dir:  Directory to save the plot.

    Returns path to the saved file.
    """
    import matplotlib.pyplot as plt

    if not confidences:
        return ""

    _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(confidences, bins=20, kde=True, ax=ax)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Triple Confidence Distribution "
        f"(n={len(confidences)}, avg={validation.avg_triple_confidence:.3f})"
        if validation.avg_triple_confidence is not None
        else f"Triple Confidence Distribution (n={len(confidences)})"
    )
    if validation.avg_triple_confidence is not None:
        ax.axvline(
            x=validation.avg_triple_confidence,
            color="red",
            linestyle="--",
            label="Mean",
        )
        ax.legend()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_quality_radar(
    validation: ValidationOutcomes,
    faithfulness: FaithfulnessReport,
    bleu: BleuReport,
    alignment: Optional[OntologyAlignmentReport],
    output_dir: str,
    filename: str = "quality_radar.png",
) -> str:
    """
    Radar chart showing the overall quality profile.

    Axes: provenance coverage, textual grounding, 1-contradiction rate,
          avg BLEU, dedup ratio, and optionally concept alignment.

    Returns path to the saved file.
    """
    import matplotlib.pyplot as plt
    import math

    _ensure_dir(output_dir)

    labels = []
    values = []

    if faithfulness.provenance_coverage is not None:
        labels.append("Provenance\nCoverage")
        values.append(faithfulness.provenance_coverage)

    if faithfulness.textual_grounding_rate is not None:
        labels.append("Textual\nGrounding")
        values.append(faithfulness.textual_grounding_rate)

    if faithfulness.contradiction_rate is not None:
        labels.append("1 - Contradiction\nRate")
        values.append(1.0 - faithfulness.contradiction_rate)

    if bleu.avg_bleu is not None:
        labels.append("Avg BLEU")
        values.append(bleu.avg_bleu)

    if validation.dedup_ratio is not None:
        labels.append("Dedup\nRatio")
        values.append(validation.dedup_ratio)

    if alignment is not None and alignment.concept_alignment_rate is not None:
        labels.append("Concept\nAlignment")
        values.append(alignment.concept_alignment_rate)

    if len(labels) < 3:
        return ""

    # Close the radar polygon
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, "o-", linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Quality Profile", size=14, pad=20)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_faithfulness_bar(
    faithfulness: FaithfulnessReport,
    output_dir: str,
    filename: str = "faithfulness_bar.png",
) -> str:
    """
    Bar chart of faithfulness metrics using seaborn.

    Returns path to the saved file.
    """
    import matplotlib.pyplot as plt

    _ensure_dir(output_dir)

    names = []
    values = []
    colors = []

    if faithfulness.provenance_coverage is not None:
        names.append("Provenance\nCoverage")
        values.append(faithfulness.provenance_coverage)
        colors.append("steelblue")
    if faithfulness.textual_grounding_rate is not None:
        names.append("Textual\nGrounding")
        values.append(faithfulness.textual_grounding_rate)
        colors.append("steelblue")
    if faithfulness.contradiction_rate is not None:
        names.append("Contradiction\nRate")
        values.append(faithfulness.contradiction_rate)
        colors.append("salmon")

    if not names:
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=names, y=values, palette=colors, ax=ax)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Faithfulness Metrics")

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(
    validation: ValidationOutcomes,
    faithfulness: FaithfulnessReport,
    bleu: BleuReport,
    confidences: list[float],
    output_dir: str,
    alignment: Optional[OntologyAlignmentReport] = None,
) -> list[str]:
    """
    Generate all plots and return the list of saved file paths.

    Args:
        validation:   ValidationOutcomes from the pipeline.
        faithfulness:  FaithfulnessReport from the pipeline.
        bleu:         BleuReport from the pipeline.
        confidences:  List of triple confidence scores.
        output_dir:   Directory to save plots.
        alignment:    Optional OntologyAlignmentReport.

    Returns:
        List of paths to generated PNG files.
    """
    paths = []

    path = plot_issue_distribution(validation, output_dir)
    if path:
        paths.append(path)

    path = plot_confidence_histogram(validation, confidences, output_dir)
    if path:
        paths.append(path)

    path = plot_quality_radar(validation, faithfulness, bleu, alignment, output_dir)
    if path:
        paths.append(path)

    path = plot_faithfulness_bar(faithfulness, output_dir)
    if path:
        paths.append(path)

    return paths

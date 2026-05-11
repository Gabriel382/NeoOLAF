from __future__ import annotations

# Standard library imports
from typing import List, Optional

# Third-party imports
from tabulate import tabulate

# Local imports
from neoolaf.evaluation.no_gold.validation_outcomes import ValidationOutcomes
from neoolaf.evaluation.no_gold.faithfulness import FaithfulnessReport
from neoolaf.evaluation.no_gold.bleu_score import BleuReport
from neoolaf.evaluation.no_gold.ontology_alignment import OntologyAlignmentReport


def _format_table(
    headers: List[str],
    rows: List[List[str]],
    title: str = "",
    fmt: str = "grid",
) -> str:
    """
    Format a table using tabulate.

    Args:
        headers: Column headers.
        rows:    List of rows, each row is a list of string values.
        title:   Optional table title.
        fmt:     Tabulate format (grid, pipe, github, fancy_grid, etc.).

    Returns:
        Formatted table string.
    """
    table = tabulate(rows, headers=headers, tablefmt=fmt)
    if title:
        return f"{title}\n\n{table}"
    return table


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _fmt_float(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


# ------------------------------------------------------------------
# Table generators
# ------------------------------------------------------------------

def issues_table(validation: ValidationOutcomes) -> str:
    """
    Table of validation issues grouped by type and severity.
    """
    if not validation.issues_by_type:
        return "No validation issues detected."

    rows = []
    for issue_type, count in sorted(
        validation.issues_by_type.items(), key=lambda x: x[1], reverse=True
    ):
        rows.append([issue_type, str(count)])

    rows.append(["---", "---"])
    rows.append(["TOTAL errors", str(validation.errors_count)])
    rows.append(["TOTAL warnings", str(validation.warnings_count)])

    return _format_table(
        headers=["Issue Type", "Count"],
        rows=rows,
        title="Validation Issues",
    )


def faithfulness_table(report: FaithfulnessReport) -> str:
    """
    Table of faithfulness metrics.
    """
    rows = [
        ["Total triples", str(report.total_triples)],
        ["With provenance", str(report.triples_with_provenance)],
        ["Provenance coverage", _fmt_pct(report.provenance_coverage)],
        ["Grounded triples", str(report.triples_grounded)],
        ["Textual grounding rate", _fmt_pct(report.textual_grounding_rate)],
        ["Contradiction pairs", str(len(report.contradiction_pairs))],
        ["Contradiction rate", _fmt_pct(report.contradiction_rate)],
        ["Ungrounded triples", str(len(report.ungrounded_triple_ids))],
    ]

    return _format_table(
        headers=["Metric", "Value"],
        rows=rows,
        title="Faithfulness",
    )


def bleu_table(report: BleuReport) -> str:
    """
    Table of BLEU score statistics.
    """
    rows = [
        ["Pairs evaluated", str(report.scores_count)],
        ["Avg BLEU", _fmt_float(report.avg_bleu)],
        ["Median BLEU", _fmt_float(report.median_bleu)],
        ["Min BLEU", _fmt_float(report.min_bleu)],
        ["Max BLEU", _fmt_float(report.max_bleu)],
        ["Low BLEU items", str(len(report.low_bleu_ids))],
        ["Threshold", _fmt_float(report.low_bleu_threshold)],
    ]

    return _format_table(
        headers=["Statistic", "Value"],
        rows=rows,
        title="BLEU Scores",
    )


def completions_table(validation: ValidationOutcomes) -> str:
    """
    Table of completion statistics.
    """
    if not validation.completions_by_type:
        return "No completions generated."

    rows = []
    for ctype, count in sorted(validation.completions_by_type.items()):
        rows.append([ctype, str(count)])

    rows.append(["---", "---"])
    rows.append(["TOTAL", str(validation.total_completions)])
    rows.append(["Avg confidence", _fmt_float(validation.avg_completion_confidence)])

    return _format_table(
        headers=["Completion Type", "Count"],
        rows=rows,
        title="Completions",
    )


def ontology_health_table(validation: ValidationOutcomes) -> str:
    """
    Table of ontology health metrics.
    """
    rows = [
        ["Total concepts", str(validation.total_concepts)],
        ["Total relations", str(validation.total_relations)],
        ["Total axioms", str(validation.total_axioms)],
        ["Orphan concept ratio", _fmt_pct(validation.orphan_concept_ratio)],
        ["Domain/range coverage", _fmt_pct(validation.domain_range_coverage)],
    ]

    return _format_table(
        headers=["Metric", "Value"],
        rows=rows,
        title="Ontology Health",
    )


def alignment_table(report: OntologyAlignmentReport) -> str:
    """
    Table of ontology alignment metrics.
    """
    rows = [
        [
            "Concepts",
            str(report.total_concepts),
            str(report.aligned_concepts),
            _fmt_pct(report.concept_alignment_rate),
        ],
        [
            "Relations",
            str(report.total_relations),
            str(report.aligned_relations),
            _fmt_pct(report.relation_alignment_rate),
        ],
        [
            "Hierarchy links",
            str(report.total_hierarchy_links),
            str(report.aligned_hierarchy_links),
            _fmt_pct(report.hierarchy_alignment_rate),
        ],
    ]

    return _format_table(
        headers=["Category", "Total", "Aligned", "Rate"],
        rows=rows,
        title="Ontology Alignment",
    )


def all_tables(
    validation: ValidationOutcomes,
    faithfulness_report: FaithfulnessReport,
    bleu_report: BleuReport,
    alignment_report: Optional[OntologyAlignmentReport] = None,
) -> str:
    """
    Generate all tables concatenated into a single string.
    """
    sections = [
        issues_table(validation),
        faithfulness_table(faithfulness_report),
        bleu_table(bleu_report),
        completions_table(validation),
        ontology_health_table(validation),
    ]

    if alignment_report is not None:
        sections.append(alignment_table(alignment_report))

    return "\n\n".join(sections)

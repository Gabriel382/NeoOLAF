from __future__ import annotations

# Standard library imports
from datetime import datetime
from typing import Optional

# Local imports
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.evaluation.no_gold.validation_outcomes import (
    ValidationOutcomes,
    compute_validation_outcomes,
    outcomes_to_dict,
)
from neoolaf.evaluation.no_gold.faithfulness import (
    FaithfulnessReport,
    compute_faithfulness,
    faithfulness_to_dict,
)
from neoolaf.evaluation.no_gold.bleu_score import (
    BleuReport,
    compute_bleu_scores,
    bleu_to_dict,
)
from neoolaf.evaluation.no_gold.ontology_alignment import (
    OntologyAlignmentReport,
    alignment_to_dict,
)


def _fmt_pct(value: Optional[float]) -> str:
    """Format a float as percentage or N/A."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _fmt_float(value: Optional[float], decimals: int = 4) -> str:
    """Format a float or N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def generate_summary_text(
    state: PipelineState,
    validation: Optional[ValidationOutcomes] = None,
    faithfulness: Optional[FaithfulnessReport] = None,
    bleu: Optional[BleuReport] = None,
    alignment: Optional[OntologyAlignmentReport] = None,
) -> str:
    """
    Generate a full text evaluation report.

    If metric objects are not provided, they are computed from state.

    Args:
        state:        Completed PipelineState.
        validation:   Pre-computed ValidationOutcomes (optional).
        faithfulness:  Pre-computed FaithfulnessReport (optional).
        bleu:         Pre-computed BleuReport (optional).
        alignment:    Pre-computed OntologyAlignmentReport (optional).

    Returns:
        Multi-line report string.
    """
    if validation is None:
        validation = compute_validation_outcomes(state)
    if faithfulness is None:
        faithfulness = compute_faithfulness(state)
    if bleu is None:
        bleu = compute_bleu_scores(state)

    lines = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines.append("=" * 60)
    lines.append("NeoOLAF Evaluation Report")
    lines.append("=" * 60)
    lines.append(f"Document:  {state.document.doc_id}")
    lines.append(f"Source:    {state.document.source_path}")
    lines.append(f"LLM:       {state.llm_model}")
    lines.append(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ------------------------------------------------------------------
    # Pipeline Stats
    # ------------------------------------------------------------------
    lines.append("--- Pipeline Stats ---")
    lines.append(
        f"Entities: {len(state.entity_candidates)}  |  "
        f"Relations: {len(state.relation_candidates)}  |  "
        f"Attributes: {len(state.attribute_candidates)}  |  "
        f"Events: {len(state.event_candidates)}"
    )
    lines.append(
        f"Triples: {len(state.candidate_triples)}  |  "
        f"Concepts: {len(state.concept_candidates)}  |  "
        f"Ontology Relations: {len(state.ontology_relation_candidates)}"
    )
    lines.append(
        f"Hierarchy Links: {len(state.concept_hierarchy_links)} concept, "
        f"{len(state.relation_hierarchy_links)} relation"
    )
    lines.append(
        f"Axiom Schemata: {len(state.axiom_schema_candidates)}  |  "
        f"General Axioms: {len(state.general_axiom_candidates)}"
    )
    lines.append(
        f"Completions: {len(state.completion_candidates)}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    lines.append("--- Validation ---")
    status = "VALID" if validation.is_valid else "INVALID"
    lines.append(f"Status: {status}")
    lines.append(
        f"Issues: {validation.errors_count} errors, "
        f"{validation.warnings_count} warnings "
        f"({validation.total_issues} total)"
    )
    if validation.issues_by_type:
        sorted_issues = sorted(
            validation.issues_by_type.items(), key=lambda x: x[1], reverse=True
        )
        top = sorted_issues[:5]
        lines.append(
            "Top issues: "
            + ", ".join(f"{k} ({v})" for k, v in top)
        )
    lines.append(f"Dedup ratio:           {_fmt_pct(validation.dedup_ratio)}")
    lines.append(f"Avg triple confidence: {_fmt_float(validation.avg_triple_confidence)}")
    lines.append("")

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------
    lines.append("--- Faithfulness ---")
    lines.append(f"Provenance coverage:   {_fmt_pct(faithfulness.provenance_coverage)}")
    lines.append(f"Textual grounding:     {_fmt_pct(faithfulness.textual_grounding_rate)}")
    lines.append(f"Contradiction rate:    {_fmt_pct(faithfulness.contradiction_rate)}")
    lines.append(
        f"Ungrounded triples:    {len(faithfulness.ungrounded_triple_ids)}"
    )
    lines.append(
        f"Contradiction pairs:   {len(faithfulness.contradiction_pairs)}"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # BLEU
    # ------------------------------------------------------------------
    lines.append("--- BLEU Scores ---")
    lines.append(f"Pairs evaluated:       {bleu.scores_count}")
    lines.append(f"Avg BLEU:              {_fmt_float(bleu.avg_bleu)}")
    lines.append(f"Median BLEU:           {_fmt_float(bleu.median_bleu)}")
    lines.append(f"Min / Max:             {_fmt_float(bleu.min_bleu)} / {_fmt_float(bleu.max_bleu)}")
    lines.append(f"Low BLEU items:        {len(bleu.low_bleu_ids)}")
    lines.append("")

    # ------------------------------------------------------------------
    # Ontology Alignment (only if provided)
    # ------------------------------------------------------------------
    if alignment is not None:
        lines.append("--- Ontology Alignment ---")
        lines.append(f"Concept alignment:     {_fmt_pct(alignment.concept_alignment_rate)}")
        lines.append(
            f"  ({alignment.aligned_concepts}/{alignment.total_concepts} concepts)"
        )
        lines.append(f"Relation alignment:    {_fmt_pct(alignment.relation_alignment_rate)}")
        lines.append(
            f"  ({alignment.aligned_relations}/{alignment.total_relations} relations)"
        )
        lines.append(f"Hierarchy alignment:   {_fmt_pct(alignment.hierarchy_alignment_rate)}")
        lines.append(
            f"  ({alignment.aligned_hierarchy_links}/{alignment.total_hierarchy_links} links)"
        )
        if alignment.unaligned_concepts:
            lines.append(
                f"Unaligned concepts:    {', '.join(alignment.unaligned_concepts[:10])}"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------
    if validation.total_completions > 0:
        lines.append("--- Completions ---")
        for ctype, count in sorted(validation.completions_by_type.items()):
            lines.append(f"  {ctype}: {count}")
        lines.append(
            f"Avg completion confidence: {_fmt_float(validation.avg_completion_confidence)}"
        )
        lines.append("")

    # ------------------------------------------------------------------
    # Ontology Health
    # ------------------------------------------------------------------
    lines.append("--- Ontology Health ---")
    lines.append(f"Orphan concept ratio:  {_fmt_pct(validation.orphan_concept_ratio)}")
    lines.append(f"Domain/range coverage: {_fmt_pct(validation.domain_range_coverage)}")
    lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_summary_dict(
    state: PipelineState,
    validation: Optional[ValidationOutcomes] = None,
    faithfulness: Optional[FaithfulnessReport] = None,
    bleu: Optional[BleuReport] = None,
    alignment: Optional[OntologyAlignmentReport] = None,
) -> dict:
    """
    Generate the full evaluation report as a JSON-serializable dict.
    """
    if validation is None:
        validation = compute_validation_outcomes(state)
    if faithfulness is None:
        faithfulness = compute_faithfulness(state)
    if bleu is None:
        bleu = compute_bleu_scores(state)

    result = {
        "document": state.document.doc_id,
        "source": state.document.source_path,
        "llm_model": state.llm_model,
        "date": datetime.now().isoformat(),
        "pipeline_stats": {
            "entities": len(state.entity_candidates),
            "relations": len(state.relation_candidates),
            "attributes": len(state.attribute_candidates),
            "events": len(state.event_candidates),
            "triples": len(state.candidate_triples),
            "concepts": len(state.concept_candidates),
            "ontology_relations": len(state.ontology_relation_candidates),
            "axiom_schemata": len(state.axiom_schema_candidates),
            "general_axioms": len(state.general_axiom_candidates),
            "completions": len(state.completion_candidates),
        },
        "validation": outcomes_to_dict(validation),
        "faithfulness": faithfulness_to_dict(faithfulness),
        "bleu": bleu_to_dict(bleu),
    }

    if alignment is not None:
        result["ontology_alignment"] = alignment_to_dict(alignment)

    return result

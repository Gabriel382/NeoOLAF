from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from typing import Dict, Optional

# Local imports
from neoolaf.core.pipeline_state import PipelineState


@dataclass
class ValidationOutcomes:
    """
    Aggregated metrics derived from Layer 10 validation/reasoning
    and Layer 11 completion results.
    """

    # --- Issue summary ---
    total_issues: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    issues_by_type: Dict[str, int] = field(default_factory=dict)
    is_valid: bool = True

    # --- Graph health ---
    total_candidate_triples: int = 0
    total_inferred_triples: int = 0
    dedup_ratio: Optional[float] = None
    avg_triple_confidence: Optional[float] = None

    # --- Completion stats ---
    total_completions: int = 0
    completions_by_type: Dict[str, int] = field(default_factory=dict)
    avg_completion_confidence: Optional[float] = None

    # --- Ontology health ---
    total_concepts: int = 0
    total_relations: int = 0
    total_axioms: int = 0
    orphan_concept_ratio: Optional[float] = None
    domain_range_coverage: Optional[float] = None


def compute_validation_outcomes(state: PipelineState) -> ValidationOutcomes:
    """
    Compute all validation outcome metrics from a completed pipeline state.

    Args:
        state: PipelineState after Layer 10 and Layer 11 have run.

    Returns:
        ValidationOutcomes with all metrics populated.
    """
    outcomes = ValidationOutcomes()

    # ------------------------------------------------------------------
    # 1. Issue summary (from Layer 10 ValidationReport)
    # ------------------------------------------------------------------
    if state.validation_report is not None:
        outcomes.is_valid = state.validation_report.is_valid
        outcomes.total_issues = len(state.validation_report.issues)

        for issue in state.validation_report.issues:
            if issue.severity == "error":
                outcomes.errors_count += 1
            else:
                outcomes.warnings_count += 1

            outcomes.issues_by_type[issue.issue_type] = (
                outcomes.issues_by_type.get(issue.issue_type, 0) + 1
            )

    # ------------------------------------------------------------------
    # 2. Graph health (candidate triples vs inferred triples)
    # ------------------------------------------------------------------
    outcomes.total_candidate_triples = len(state.candidate_triples)

    if state.reasoning_report is not None:
        outcomes.total_inferred_triples = len(state.reasoning_report.inferred_triples)

    if outcomes.total_candidate_triples > 0:
        outcomes.dedup_ratio = (
            outcomes.total_inferred_triples / outcomes.total_candidate_triples
        )

    confidences = [
        t.confidence
        for t in state.candidate_triples
        if t.confidence is not None
    ]
    if confidences:
        outcomes.avg_triple_confidence = sum(confidences) / len(confidences)

    # ------------------------------------------------------------------
    # 3. Completion stats (from Layer 11)
    # ------------------------------------------------------------------
    outcomes.total_completions = len(state.completion_candidates)

    for comp in state.completion_candidates:
        outcomes.completions_by_type[comp.completion_type] = (
            outcomes.completions_by_type.get(comp.completion_type, 0) + 1
        )

    comp_confidences = [
        c.confidence
        for c in state.completion_candidates
        if c.confidence is not None
    ]
    if comp_confidences:
        outcomes.avg_completion_confidence = sum(comp_confidences) / len(comp_confidences)

    # ------------------------------------------------------------------
    # 4. Ontology health
    # ------------------------------------------------------------------
    outcomes.total_concepts = len(state.concept_candidates)
    outcomes.total_relations = len(state.ontology_relation_candidates)
    outcomes.total_axioms = len(state.general_axiom_candidates)

    # Orphan concept ratio: concepts without a parent in hierarchy
    if outcomes.total_concepts > 0:
        child_ids = {
            link.child_concept_id for link in state.concept_hierarchy_links
        }
        concept_ids = {c.concept_id for c in state.concept_candidates}
        orphans = concept_ids - child_ids
        outcomes.orphan_concept_ratio = len(orphans) / len(concept_ids)

    # Domain/range coverage: relations that have both domain AND range axioms
    if outcomes.total_relations > 0:
        relations_with_domain = set()
        relations_with_range = set()

        for axiom in state.general_axiom_candidates:
            if axiom.axiom_type == "relation_domain":
                relations_with_domain.add(axiom.subject_id)
            elif axiom.axiom_type == "relation_range":
                relations_with_range.add(axiom.subject_id)

        fully_specified = relations_with_domain & relations_with_range
        outcomes.domain_range_coverage = len(fully_specified) / outcomes.total_relations

    return outcomes


def outcomes_to_dict(outcomes: ValidationOutcomes) -> dict:
    """
    Serialize ValidationOutcomes to a JSON-compatible dictionary.
    """
    return {
        "is_valid": outcomes.is_valid,
        "issues": {
            "total": outcomes.total_issues,
            "errors": outcomes.errors_count,
            "warnings": outcomes.warnings_count,
            "by_type": outcomes.issues_by_type,
        },
        "graph_health": {
            "candidate_triples": outcomes.total_candidate_triples,
            "inferred_triples": outcomes.total_inferred_triples,
            "dedup_ratio": outcomes.dedup_ratio,
            "avg_triple_confidence": outcomes.avg_triple_confidence,
        },
        "completions": {
            "total": outcomes.total_completions,
            "by_type": outcomes.completions_by_type,
            "avg_confidence": outcomes.avg_completion_confidence,
        },
        "ontology_health": {
            "total_concepts": outcomes.total_concepts,
            "total_relations": outcomes.total_relations,
            "total_axioms": outcomes.total_axioms,
            "orphan_concept_ratio": outcomes.orphan_concept_ratio,
            "domain_range_coverage": outcomes.domain_range_coverage,
        },
    }
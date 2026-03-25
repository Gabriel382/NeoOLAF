"""
Run all 7 evaluation metrics (benchmark + no-gold) and return results.

Usage:
    python -m tests.run_evaluation
"""
from __future__ import annotations

from neoolaf.evaluation.benchmark.entity_metrics import (
    compute_entity_metrics,
    entity_metrics_to_dict,
)
from neoolaf.evaluation.benchmark.relation_metrics import (
    compute_relation_metrics,
    relation_metrics_to_dict,
)
from neoolaf.evaluation.benchmark.ontology_conformance import (
    compute_ontology_conformance,
    conformance_to_dict,
)
from neoolaf.evaluation.no_gold.validation_outcomes import (
    compute_validation_outcomes,
    outcomes_to_dict,
)
from neoolaf.evaluation.no_gold.faithfulness import (
    compute_faithfulness,
    faithfulness_to_dict,
)
from neoolaf.evaluation.no_gold.bleu_score import (
    compute_bleu_scores,
    bleu_to_dict,
)
from neoolaf.evaluation.no_gold.ontology_alignment import (
    compute_ontology_alignment,
    alignment_to_dict,
)

from tests.fake_data import (
    build_fake_state,
    build_gold_entities,
    build_gold_triples,
    build_gold_ontology,
    build_reference_ontology,
)


def run_all_metrics() -> dict:
    """
    Run all 7 evaluation metrics and return a dict with results + report objects.

    Returns dict with keys:
        - "results": JSON-serializable dict of all metrics
        - "state": PipelineState used
        - "validation": ValidationOutcomes report object
        - "faithfulness": FaithfulnessReport object
        - "bleu": BleuReport object
        - "alignment": OntologyAlignmentReport object
    """
    state = build_fake_state()
    gold_entities = build_gold_entities()
    gold_triples = build_gold_triples()
    gold_ontology = build_gold_ontology()
    reference_ontology = build_reference_ontology()

    results = {}

    # 1. Benchmark: Entity metrics
    print("1. Computing entity metrics...")
    entity_report = compute_entity_metrics(state, gold_entities)
    results["entity_metrics"] = entity_metrics_to_dict(entity_report)
    print(f"   Exact F1: {entity_report.exact.f1:.4f}")
    print(f"   TP: {entity_report.true_positives_exact}")
    print(f"   FP: {entity_report.false_positives_exact}")
    print(f"   FN: {entity_report.false_negatives_exact}")

    # 2. Benchmark: Relation metrics
    print("\n2. Computing relation metrics...")
    relation_report = compute_relation_metrics(state, gold_triples)
    results["relation_metrics"] = relation_metrics_to_dict(relation_report)
    print(f"   Strict F1: {relation_report.strict.f1:.4f}")
    print(f"   Matched: {relation_report.matched_triples}")
    print(f"   Missed:  {relation_report.missed_triples}")

    # 3. Benchmark: Ontology conformance
    print("\n3. Computing ontology conformance...")
    conformance_report = compute_ontology_conformance(state, gold_ontology)
    results["ontology_conformance"] = conformance_to_dict(conformance_report)
    print(f"   Concept coverage F1: {conformance_report.concept_coverage.f1:.4f}")
    print(f"   Hierarchy F1:        {conformance_report.hierarchy.f1:.4f}")

    # 4. No-gold: Validation outcomes
    print("\n4. Computing validation outcomes...")
    validation = compute_validation_outcomes(state)
    results["validation_outcomes"] = outcomes_to_dict(validation)
    print(f"   Valid: {validation.is_valid}")
    print(f"   Issues: {validation.total_issues} ({validation.errors_count} errors, {validation.warnings_count} warnings)")

    # 5. No-gold: Faithfulness
    print("\n5. Computing faithfulness...")
    faithfulness = compute_faithfulness(state)
    results["faithfulness"] = faithfulness_to_dict(faithfulness)
    print(f"   Provenance coverage:   {faithfulness.provenance_coverage:.2%}")
    print(f"   Textual grounding:     {faithfulness.textual_grounding_rate:.2%}")
    print(f"   Contradiction pairs:   {len(faithfulness.contradiction_pairs)}")

    # 6. No-gold: BLEU scores
    print("\n6. Computing BLEU scores...")
    bleu = compute_bleu_scores(state)
    results["bleu_scores"] = bleu_to_dict(bleu)
    print(f"   Pairs evaluated: {bleu.scores_count}")
    print(f"   Avg BLEU:  {bleu.avg_bleu:.4f}" if bleu.avg_bleu else "   Avg BLEU:  N/A")

    # 7. No-gold: Ontology alignment
    print("\n7. Computing ontology alignment...")
    alignment = compute_ontology_alignment(state, reference_ontology)
    results["ontology_alignment"] = alignment_to_dict(alignment)
    print(f"   Concept alignment:  {alignment.concept_alignment_rate:.2%}" if alignment.concept_alignment_rate else "   Concept alignment: N/A")
    print(f"   Relation alignment: {alignment.relation_alignment_rate:.2%}" if alignment.relation_alignment_rate else "   Relation alignment: N/A")

    return {
        "results": results,
        "state": state,
        "validation": validation,
        "faithfulness": faithfulness,
        "bleu": bleu,
        "alignment": alignment,
    }


if __name__ == "__main__":
    run_all_metrics()

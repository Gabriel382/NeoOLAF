"""Extraction metrics over canonical artifacts."""

from __future__ import annotations

from neoolaf.evaluation.matching.entity_matching import greedy_entity_matching
from neoolaf.evaluation.matching.relation_matching import greedy_relation_matching, per_relation_metrics
from neoolaf.evaluation.schema.artifact import EvaluationArtifact, EvalEntity, EvalRelation
from neoolaf.evaluation.schema.config import EvaluationProfile


def flatten_entities(entities_by_doc: dict[str, list[EvalEntity]]) -> set[str]:
    """Flatten entities from all documents into a label set."""
    return {entity.label for entities in entities_by_doc.values() for entity in entities if entity.label}


def flatten_relations(relations_by_doc: dict[str, list[EvalRelation]]) -> list[EvalRelation]:
    """Flatten relations from all documents into a list."""
    return [relation for relations in relations_by_doc.values() for relation in relations]


def evaluate_extraction(pred: EvaluationArtifact, gold: EvaluationArtifact, profile: EvaluationProfile) -> dict:
    """Compute entity, relation, and per-relation extraction metrics."""
    pred_entities = flatten_entities(pred.entities_by_doc)
    gold_entities = flatten_entities(gold.entities_by_doc)
    pred_relations = flatten_relations(pred.relations_by_doc)
    gold_relations = flatten_relations(gold.relations_by_doc)

    entity_result = greedy_entity_matching(pred_entities, gold_entities, profile)
    relation_result = greedy_relation_matching(pred_relations, gold_relations, profile)
    per_relation = per_relation_metrics(pred_relations, gold_relations, profile)

    return {
        "entity": entity_result.prf.to_dict(),
        "relation": relation_result.prf.to_dict(),
        "per_relation": per_relation,
        "counts": {
            "pred_entities": len(pred_entities),
            "gold_entities": len(gold_entities),
            "pred_relations": len(pred_relations),
            "gold_relations": len(gold_relations),
        },
        "matches": {
            "entities": entity_result.matches,
            "relations": relation_result.matches,
        },
        "unmatched": {
            "entities_pred": entity_result.unmatched_pred,
            "entities_gold": entity_result.unmatched_gold,
            "relations_pred": relation_result.unmatched_pred,
            "relations_gold": relation_result.unmatched_gold,
        },
    }

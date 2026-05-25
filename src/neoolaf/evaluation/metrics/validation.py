"""Validation-oriented metrics: STR, CR, PC, OC, CV, DVS."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict

from neoolaf.evaluation.matching.normalization import normalize_text
from neoolaf.evaluation.matching.similarity import loose_similarity
from neoolaf.evaluation.metrics.prf import safe_div
from neoolaf.evaluation.schema.artifact import EvalRelation, EvaluationArtifact
from neoolaf.evaluation.schema.config import EvaluationProfile

ALLOWED_XQUALITY_RELATIONS = {"TRIGGERS", "CAUSES", "REQUIRES", "HANDLED_BY", "REFERENCES"}
GENERIC_BAD = {"alarm", "device", "failure", "part", "machine", "check"}


def _fair_supported(pred: EvalRelation, gold: EvalRelation, profile: EvaluationProfile) -> bool:
    """Return whether a predicted triple is supported by a gold triple."""
    if normalize_text(pred.relation).upper() != normalize_text(gold.relation).upper():
        return False
    head_score = loose_similarity(pred.head, gold.head)
    tail_score = loose_similarity(pred.tail, gold.tail)
    return head_score >= profile.fair_str_head_threshold and tail_score >= profile.fair_tail_threshold(pred.relation)


def compute_STR(pred_relations: list[EvalRelation], gold_relations: list[EvalRelation], profile: EvaluationProfile) -> tuple[float, list[dict]]:
    """Compute Supported Triple Ratio."""
    if not pred_relations:
        return 0.0, []
    examples = []
    supported_count = 0
    for pred in pred_relations:
        ok = any(_fair_supported(pred, gold, profile) for gold in gold_relations)
        if ok:
            supported_count += 1
            if len(examples) < 12:
                examples.append(asdict(pred))
    return safe_div(supported_count, len(pred_relations)), examples


def compute_CR(pred_relations: list[EvalRelation], gold_relations: list[EvalRelation], profile: EvaluationProfile) -> tuple[float, list[dict]]:
    """Compute a pragmatic contradiction rate."""
    if not pred_relations:
        return 0.0, []

    gold_by_rel: dict[str, list[EvalRelation]] = defaultdict(list)
    for gold in gold_relations:
        gold_by_rel[normalize_text(gold.relation).upper()].append(gold)

    examples = []
    contradiction_count = 0
    for pred in pred_relations:
        if any(_fair_supported(pred, gold, profile) for gold in gold_relations):
            continue

        pred_rel = normalize_text(pred.relation).upper()
        contradiction = False
        for gold in gold_by_rel.get(pred_rel, []):
            head_sim = loose_similarity(pred.head, gold.head)
            tail_sim = loose_similarity(pred.tail, gold.tail)
            same_headish = head_sim >= profile.contradiction_same_relation_threshold
            same_tailish = tail_sim >= profile.contradiction_same_relation_threshold
            if same_headish and tail_sim < profile.contradiction_low_sim:
                contradiction = True
                break
            if same_tailish and head_sim < profile.contradiction_low_sim:
                contradiction = True
                break

        if contradiction:
            contradiction_count += 1
            if len(examples) < 12:
                examples.append(asdict(pred))

    return safe_div(contradiction_count, len(pred_relations)), examples


def compute_PC(pred_relations: list[EvalRelation]) -> tuple[float, list[dict]]:
    """Compute Provenance Coverage."""
    if not pred_relations:
        return 0.0, []
    examples = []
    count = 0
    for pred in pred_relations:
        ok = bool(pred.provenance_present or pred.evidence or pred.provenance)
        if ok:
            count += 1
            if len(examples) < 12:
                examples.append(asdict(pred))
    return safe_div(count, len(pred_relations)), examples


def compute_OC(pred_relations: list[EvalRelation], dataset: str = "") -> tuple[float, list[dict]]:
    """Compute Ontology Conformance using relation labels and malformed labels."""
    if not pred_relations:
        return 0.0, []
    examples = []
    ok_count = 0
    for pred in pred_relations:
        rel = str(pred.relation or "").strip().upper()
        head_n = normalize_text(pred.head)
        tail_n = normalize_text(pred.tail)
        rel_ok = rel in ALLOWED_XQUALITY_RELATIONS if dataset == "xquality" else bool(rel)
        malformed = bool(re.match(r"^(concept_|ont_rel_|cand_)", head_n) or re.match(r"^(concept_|ont_rel_|cand_)", tail_n))
        too_generic = head_n in GENERIC_BAD or tail_n in GENERIC_BAD
        ok = rel_ok and bool(head_n) and bool(tail_n) and not malformed and not too_generic
        if ok:
            ok_count += 1
            if len(examples) < 12:
                examples.append(asdict(pred))
    return safe_div(ok_count, len(pred_relations)), examples


def compute_CV(pred_relations: list[EvalRelation]) -> tuple[float, list[dict]]:
    """Compute simple constraint violation rate."""
    if not pred_relations:
        return 0.0, []
    examples = []
    violation_count = 0
    for pred in pred_relations:
        head_n = normalize_text(pred.head)
        tail_n = normalize_text(pred.tail)
        violated = False
        if not head_n or not tail_n:
            violated = True
        elif head_n == tail_n:
            violated = True
        elif head_n in GENERIC_BAD or tail_n in GENERIC_BAD:
            violated = True
        elif re.match(r"^(concept_|ont_rel_|cand_)", head_n) or re.match(r"^(concept_|ont_rel_|cand_)", tail_n):
            violated = True
        if violated:
            violation_count += 1
            if len(examples) < 12:
                examples.append(asdict(pred))
    return safe_div(violation_count, len(pred_relations)), examples


def compute_DVS(str_value: float, cr_value: float, oc_value: float, cv_value: float, profile: EvaluationProfile) -> tuple[float, dict]:
    """Compute binary Document-level Validation Success."""
    success = (
        str_value >= profile.dvs_min_str
        and cr_value <= profile.dvs_max_cr
        and oc_value >= profile.dvs_min_oc
        and cv_value <= profile.dvs_max_cv
    )
    return float(success), {
        "supported_ratio": str_value,
        "contradiction_rate": cr_value,
        "ontology_conformance": oc_value,
        "constraint_violations": cv_value,
        "success": success,
    }


def evaluate_validation(pred: EvaluationArtifact, gold: EvaluationArtifact, profile: EvaluationProfile) -> dict:
    """Compute validation-oriented metrics over all documents."""
    all_rows = []
    for doc in pred.documents or gold.documents:
        doc_id = doc.document_id
        pred_relations = pred.relations_by_doc.get(doc_id, [])
        if not pred_relations and len(pred.relations_by_doc) == 1:
            pred_relations = next(iter(pred.relations_by_doc.values()))
        gold_relations = gold.relations_by_doc.get(doc_id, [])
        if not gold_relations and len(gold.relations_by_doc) == 1:
            gold_relations = next(iter(gold.relations_by_doc.values()))

        str_value, str_examples = compute_STR(pred_relations, gold_relations, profile)
        cr_value, cr_examples = compute_CR(pred_relations, gold_relations, profile)
        pc_value, pc_examples = compute_PC(pred_relations)
        oc_value, oc_examples = compute_OC(pred_relations, dataset=pred.dataset)
        cv_value, cv_examples = compute_CV(pred_relations)
        dvs_value, dvs_details = compute_DVS(str_value, cr_value, oc_value, cv_value, profile)

        all_rows.append(
            {
                "document_id": doc_id,
                "STR": str_value,
                "CR": cr_value,
                "PC": pc_value,
                "OC": oc_value,
                "CV": cv_value,
                "DVS": dvs_value,
                "DVS_details": dvs_details,
                "examples": {
                    "STR": str_examples,
                    "CR": cr_examples,
                    "PC": pc_examples,
                    "OC": oc_examples,
                    "CV": cv_examples,
                },
            }
        )

    summary = {}
    for key in ["STR", "CR", "PC", "OC", "CV", "DVS"]:
        summary[key] = safe_div(sum(row[key] for row in all_rows), len(all_rows)) if all_rows else 0.0
    return {"summary": summary, "per_document": all_rows}

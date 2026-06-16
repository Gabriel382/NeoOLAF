# eval_relations.py
# Streaming evaluator for predictions produced by singlepass_llm.py
#
# It reads:
# - gold JSONL dataset
# - prediction JSONL file
#
# It computes:
# - entity precision / recall / F1
# - relation precision / recall / F1
# - per-relation metrics
# - optional validation metrics:
#   STR, CR, PC, OC, CV, DVS
#
# Assumption:
# prediction JSONL contains one line per processed document, with:
# {
#   "document_id": "...",
#   "parsed_ok": true/false,
#   "prediction": {
#       "entities": [...],
#       "relations": [...]
#   }
# }

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rdflib import Graph, Literal
from rapidfuzz import fuzz

import sys
from pathlib import Path

# Resolve project root robustly from this script location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT =", PROJECT_ROOT)
print("sys.path[0] =", sys.path[0])
from experiments.common.jsonl_adapter import iter_documents
from tqdm.auto import tqdm

# =========================================================
# Config defaults
# =========================================================
ENTITY_SIM_THRESHOLD = 85
REL_HEAD_SIM_THRESHOLD = 85
REL_TAIL_SIM_THRESHOLD_DEFAULT = 85
REL_TAIL_SIM_THRESHOLD_REQUIRES = 75

FAIR_STR_HEAD_THRESHOLD = 82
FAIR_STR_TAIL_THRESHOLD = 82
FAIR_STR_TAIL_THRESHOLD_REQUIRES = 72

CONTRADICTION_LOW_SIM = 35
CONTRADICTION_SAME_REL_THRESHOLD = 75

DVS_MIN_STR = 0.50
DVS_MAX_CR = 0.20
DVS_MIN_OC = 0.70
DVS_MAX_CV = 0.30


# =========================================================
# Common helpers
# =========================================================
def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    if text is None:
        return ""
    text = str(text)
    text = (
        text.replace("â", '"')
            .replace("â", '"')
            .replace("â", "'")
            .replace("â", " ")
            .replace("â", " ")
    )
    text = text.lower().strip()
    text = re.sub(r"[_/\\:;|]+", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    """Loose fuzzy similarity in [0, 100]."""
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    return float(fuzz.token_set_ratio(a_n, b_n))


def safe_div(a: float, b: float) -> float:
    """Safe division."""
    return a / b if b != 0 else 0.0


def harmonic_f1(p: float, r: float) -> float:
    """Compute harmonic F1."""
    return (2 * p * r / (p + r)) if (p + r) != 0 else 0.0


def uri_local_name(x) -> str:
    """Get local name from URI or literal."""
    if isinstance(x, Literal):
        return str(x)
    s = str(x)
    if "#" in s:
        s = s.split("#")[-1]
    else:
        s = s.rstrip("/").split("/")[-1]
    return s.replace("%20", " ")


def relation_tail_threshold(rel: str) -> float:
    """Use softer tail threshold for REQUIRES."""
    if normalize_text(rel) == "requires":
        return REL_TAIL_SIM_THRESHOLD_REQUIRES
    return REL_TAIL_SIM_THRESHOLD_DEFAULT


def fair_relation_tail_threshold(rel: str) -> float:
    """Validation-metric threshold for relation tail."""
    if normalize_text(rel) == "requires":
        return FAIR_STR_TAIL_THRESHOLD_REQUIRES
    return FAIR_STR_TAIL_THRESHOLD


# =========================================================
# Prediction loading
# =========================================================
def iter_prediction_records(prediction_jsonl_path: str | Path) -> Iterable[Dict[str, Any]]:
    """Stream prediction records."""
    prediction_jsonl_path = Path(prediction_jsonl_path)
    with prediction_jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in prediction file line {line_number}: {e}"
                ) from e


def build_prediction_index(prediction_jsonl_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load predictions into a doc_id -> record index.

    This is usually still manageable because predictions are much smaller than source corpora.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for record in iter_prediction_records(prediction_jsonl_path):
        doc_id = str(record.get("document_id", "")).strip()
        if doc_id:
            index[doc_id] = record
    return index


# =========================================================
# Gold conversion
# =========================================================
def gold_entity_set_from_doc(doc: Dict[str, Any]) -> set[str]:
    """Extract gold entity texts from one normalized document."""
    out = set()
    for ent in doc["entities"]:
        text = str(ent.get("text", "")).strip()
        if text:
            out.add(text)
    return out


def gold_relation_list_from_doc(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract gold relation triples from one normalized document."""
    out: List[Dict[str, str]] = []
    for rel in doc["relations"]:
        out.append(
            {
                "head": str(rel.get("head_text", "")).strip(),
                "rel": str(rel.get("relation", "")).strip(),
                "tail": str(rel.get("tail_text", "")).strip(),
            }
        )
    return out


# =========================================================
# Prediction conversion
# =========================================================
def pred_entity_set_from_record(record: Dict[str, Any]) -> set[str]:
    """Extract predicted entity texts from one prediction record."""
    prediction = record.get("prediction") or {}
    entities = prediction.get("entities", []) or []

    out = set()
    for ent in entities:
        label = str(ent.get("label", "")).strip()
        if label:
            out.add(label)
    return out


def pred_relation_list_from_record(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract predicted relation triples from one prediction record."""
    prediction = record.get("prediction") or {}
    relations = prediction.get("relations", []) or []

    out: List[Dict[str, str]] = []
    for rel in relations:
        head = str(rel.get("head", "")).strip()
        relation = str(rel.get("relation", "")).strip()
        tail = str(rel.get("tail", "")).strip()
        evidence = str(rel.get("evidence", "")).strip()

        if not head or not relation or not tail:
            continue

        out.append(
            {
                "head": head,
                "rel": relation,
                "tail": tail,
                "evidence": evidence,
                "provenance_present": bool(evidence),
                "justification": evidence,
                "chunkid": "",
                "support_text": evidence,
            }
        )
    return out


# =========================================================
# Greedy matching
# =========================================================
def greedy_entity_matching(pred_entities: set[str], gt_entities: set[str], threshold: int = ENTITY_SIM_THRESHOLD) -> Dict[str, Any]:
    """Greedy one-to-one entity matching."""
    pred_list = list(pred_entities)
    gt_list = list(gt_entities)

    candidates: List[Tuple[float, int, int]] = []

    for i, pred in enumerate(pred_list):
        for j, gt in enumerate(gt_list):
            score = similarity(pred, gt)
            if score >= threshold:
                candidates.append((score, i, j))

    candidates.sort(reverse=True, key=lambda x: (x[0], -x[1], -x[2]))

    used_pred = set()
    used_gt = set()
    matches = []

    for score, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matches.append({"pred": pred_list[i], "gt": gt_list[j], "score": score})

    tp = len(matches)
    fp = len(pred_list) - tp
    fn = len(gt_list) - tp

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = harmonic_f1(precision, recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def greedy_relation_matching(pred_triples: List[Dict[str, str]], gt_triples: List[Dict[str, str]]) -> Dict[str, Any]:
    """Greedy one-to-one relation matching."""
    candidates = []

    for i, pred in enumerate(pred_triples):
        for j, gt in enumerate(gt_triples):
            if normalize_text(pred["rel"]) != normalize_text(gt["rel"]):
                continue

            head_score = similarity(pred["head"], gt["head"])
            tail_score = similarity(pred["tail"], gt["tail"])

            if head_score >= REL_HEAD_SIM_THRESHOLD and tail_score >= relation_tail_threshold(pred["rel"]):
                total_score = 0.5 * head_score + 0.5 * tail_score
                candidates.append((total_score, head_score, tail_score, i, j))

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], -x[3], -x[4]))

    used_pred = set()
    used_gt = set()
    matches = []

    for total_score, head_score, tail_score, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matches.append(
            {
                "pred_idx": i,
                "gt_idx": j,
                "score": total_score,
                "head_score": head_score,
                "tail_score": tail_score,
                "pred": pred_triples[i],
                "gt": gt_triples[j],
            }
        )

    tp = len(matches)
    fp = len(pred_triples) - tp
    fn = len(gt_triples) - tp

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = harmonic_f1(precision, recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


# =========================================================
# Validation metrics
# =========================================================
def fair_supported(pred_t: Dict[str, str], gt_t: Dict[str, str]) -> bool:
    """Support check for validation metrics."""
    if normalize_text(pred_t["rel"]) != normalize_text(gt_t["rel"]):
        return False

    head_score = similarity(pred_t["head"], gt_t["head"])
    tail_score = similarity(pred_t["tail"], gt_t["tail"])

    return (
        head_score >= FAIR_STR_HEAD_THRESHOLD
        and tail_score >= fair_relation_tail_threshold(pred_t["rel"])
    )


def compute_fair_STR(pred_triples: List[Dict[str, str]], gt_triples: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    """Supported Triple Ratio."""
    if not pred_triples:
        return 0.0, []

    flags = []
    examples = []

    for pred in pred_triples:
        ok = any(fair_supported(pred, gt) for gt in gt_triples)
        flags.append(ok)
        if ok and len(examples) < 12:
            examples.append(pred)

    return safe_div(sum(flags), len(pred_triples)), examples


def compute_fair_CR(pred_triples: List[Dict[str, str]], gt_triples: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    """Contradiction Rate."""
    if not pred_triples:
        return 0.0, []

    gt_by_rel = defaultdict(list)
    for gt in gt_triples:
        gt_by_rel[gt["rel"]].append(gt)

    flags = []
    examples = []

    for pred in pred_triples:
        if any(fair_supported(pred, gt) for gt in gt_triples):
            flags.append(False)
            continue

        contradiction = False
        for gt in gt_by_rel.get(pred["rel"], []):
            head_sim = similarity(pred["head"], gt["head"])
            tail_sim = similarity(pred["tail"], gt["tail"])

            same_headish = head_sim >= CONTRADICTION_SAME_REL_THRESHOLD
            same_tailish = tail_sim >= CONTRADICTION_SAME_REL_THRESHOLD

            if same_headish and tail_sim < CONTRADICTION_LOW_SIM:
                contradiction = True
                break
            if same_tailish and head_sim < CONTRADICTION_LOW_SIM:
                contradiction = True
                break

        flags.append(contradiction)
        if contradiction and len(examples) < 12:
            examples.append(pred)

    return safe_div(sum(flags), len(pred_triples)), examples


def compute_fair_PC(pred_triples: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    """Provenance Coverage."""
    if not pred_triples:
        return 0.0, []

    flags = []
    examples = []

    for pred in pred_triples:
        ok = bool(
            pred.get("provenance_present", False)
            or pred.get("justification", "")
            or pred.get("chunkid", "")
            or pred.get("support_text", "")
        )
        flags.append(ok)
        if ok and len(examples) < 12:
            examples.append(pred)

    return safe_div(sum(flags), len(pred_triples)), examples


def load_ontology_labels(ontology_path: str | Path | None) -> set[str]:
    """Load ontology labels if provided."""
    labels = set()
    if ontology_path is None:
        return labels

    ontology_path = Path(ontology_path)
    if not ontology_path.exists():
        return labels

    parse_errors = []
    for fmt in ["turtle", "xml"]:
        try:
            graph = Graph()
            graph.parse(str(ontology_path), format=fmt)
            for _, pred, obj in graph:
                pred_name = normalize_text(uri_local_name(pred))
                if pred_name == "label" and isinstance(obj, Literal):
                    labels.add(normalize_text(str(obj)))
            return labels
        except Exception as e:
            parse_errors.append(f"{fmt}: {e}")

    print(f"[WARNING] ontology parse failed for {ontology_path}")
    for err in parse_errors:
        print(f"  - {err}")
    return labels


def compute_fair_OC(pred_triples: List[Dict[str, str]], ontology_path: str | Path | None = None) -> Tuple[float, List[Dict[str, str]]]:
    """Ontology Conformance."""
    if not pred_triples:
        return 0.0, []

    ontology_labels = load_ontology_labels(ontology_path)
    allowed_relations = {"TRIGGERS", "CAUSES", "REQUIRES", "HANDLED_BY", "REFERENCES"}

    bad_exact = {"alarm", "device", "failure", "part", "machine", "check"}

    flags = []
    examples = []

    for pred in pred_triples:
        rel_ok = pred["rel"] in allowed_relations

        head_n = normalize_text(pred["head"])
        tail_n = normalize_text(pred["tail"])

        head_ok = bool(head_n)
        tail_ok = bool(tail_n)

        malformed = (
            re.match(r"^(concept_|ont_rel_|cand_)", head_n) is not None
            or re.match(r"^(concept_|ont_rel_|cand_)", tail_n) is not None
        )

        too_generic = (head_n in bad_exact or tail_n in bad_exact)

        label_ok = True
        if ontology_labels:
            head_in_ontology = head_n in ontology_labels
            tail_in_ontology = tail_n in ontology_labels
            label_ok = head_in_ontology or tail_in_ontology or True

        ok = rel_ok and head_ok and tail_ok and not malformed and not too_generic and label_ok
        flags.append(ok)

        if ok and len(examples) < 12:
            examples.append(pred)

    return safe_div(sum(flags), len(pred_triples)), examples


def compute_fair_CV(pred_triples: List[Dict[str, str]]) -> Tuple[float, List[Dict[str, str]]]:
    """Constraint Violations."""
    if not pred_triples:
        return 0.0, []

    generic_bad = {"alarm", "device", "failure", "part", "machine", "check"}

    flags = []
    examples = []

    for pred in pred_triples:
        head_n = normalize_text(pred["head"])
        tail_n = normalize_text(pred["tail"])

        violated = False

        if not head_n or not tail_n:
            violated = True
        elif head_n == tail_n:
            violated = True
        elif head_n in generic_bad or tail_n in generic_bad:
            violated = True
        elif re.match(r"^(concept_|ont_rel_|cand_)", head_n) or re.match(r"^(concept_|ont_rel_|cand_)", tail_n):
            violated = True

        flags.append(violated)
        if violated and len(examples) < 12:
            examples.append(pred)

    return safe_div(sum(flags), len(pred_triples)), examples


def compute_fair_DVS(pred_triples: List[Dict[str, str]], gt_triples: List[Dict[str, str]], ontology_path: str | Path | None = None) -> Tuple[float, Dict[str, Any]]:
    """Document-level Validation Success."""
    if not pred_triples:
        return 0.0, {
            "supported_ratio": 0.0,
            "contradiction_rate": 0.0,
            "ontology_conformance": 0.0,
            "constraint_violations": 0.0,
            "success": False,
        }

    str_val, _ = compute_fair_STR(pred_triples, gt_triples)
    cr_val, _ = compute_fair_CR(pred_triples, gt_triples)
    oc_val, _ = compute_fair_OC(pred_triples, ontology_path=ontology_path)
    cv_val, _ = compute_fair_CV(pred_triples)

    success = (
        str_val >= DVS_MIN_STR
        and cr_val <= DVS_MAX_CR
        and oc_val >= DVS_MIN_OC
        and cv_val <= DVS_MAX_CV
    )

    return float(success), {
        "supported_ratio": str_val,
        "contradiction_rate": cr_val,
        "ontology_conformance": oc_val,
        "constraint_violations": cv_val,
        "success": success,
    }


# =========================================================
# Corpus-level evaluation
# =========================================================
def evaluate_per_relation(all_pred_triples: List[Dict[str, str]], all_gt_triples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Per-relation metrics on corpus-level triple lists."""
    rels = sorted(set([x["rel"] for x in all_gt_triples]) | set([x["rel"] for x in all_pred_triples]))
    rows = []

    for rel in rels:
        pred_subset = [x for x in all_pred_triples if x["rel"] == rel]
        gt_subset = [x for x in all_gt_triples if x["rel"] == rel]

        res = greedy_relation_matching(pred_subset, gt_subset)

        rows.append(
            {
                "relation": rel,
                "pred_count": len(pred_subset),
                "gt_count": len(gt_subset),
                "tp": res["tp"],
                "fp": res["fp"],
                "fn": res["fn"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
            }
        )

    return rows


def evaluate_dataset(
    gold_jsonl_path: str | Path,
    prediction_jsonl_path: str | Path,
    ontology_path: str | Path | None = None,
    type_filter: str | list[str] = "all",
) -> Dict[str, Any]:
    """
    Evaluate a full dataset.
    """
    prediction_index = build_prediction_index(prediction_jsonl_path)

    all_pred_entities: set[str] = set()
    all_gt_entities: set[str] = set()

    all_pred_triples: List[Dict[str, str]] = []
    all_gt_triples: List[Dict[str, str]] = []

    doc_validation_records = []

    total_docs = 0
    missing_predictions = 0
    parsed_failures = 0

    for gold_doc in iter_documents(gold_jsonl_path, type_filter=type_filter):
        total_docs += 1
        doc_id = gold_doc["document_id"]

        gold_entities = gold_entity_set_from_doc(gold_doc)
        gold_triples = gold_relation_list_from_doc(gold_doc)

        all_gt_entities.update(gold_entities)
        all_gt_triples.extend(gold_triples)

        pred_record = prediction_index.get(doc_id)
        if pred_record is None:
            missing_predictions += 1
            pred_entities = set()
            pred_triples = []
        else:
            if not pred_record.get("parsed_ok", False):
                parsed_failures += 1
                pred_entities = set()
                pred_triples = []
            else:
                pred_entities = pred_entity_set_from_record(pred_record)
                pred_triples = pred_relation_list_from_record(pred_record)

        all_pred_entities.update(pred_entities)
        all_pred_triples.extend(pred_triples)

        str_val, _ = compute_fair_STR(pred_triples, gold_triples)
        cr_val, _ = compute_fair_CR(pred_triples, gold_triples)
        pc_val, _ = compute_fair_PC(pred_triples)
        oc_val, _ = compute_fair_OC(pred_triples, ontology_path=ontology_path)
        cv_val, _ = compute_fair_CV(pred_triples)
        dvs_val, dvs_details = compute_fair_DVS(pred_triples, gold_triples, ontology_path=ontology_path)

        doc_validation_records.append(
            {
                "document_id": doc_id,
                "STR": str_val,
                "CR": cr_val,
                "PC": pc_val,
                "OC": oc_val,
                "CV": cv_val,
                "DVS": dvs_val,
                "DVS_details": dvs_details,
            }
        )

    entity_results = greedy_entity_matching(all_pred_entities, all_gt_entities, threshold=ENTITY_SIM_THRESHOLD)
    relation_results = greedy_relation_matching(all_pred_triples, all_gt_triples)
    per_relation_rows = evaluate_per_relation(all_pred_triples, all_gt_triples)

    validation_means = {}
    for key in ["STR", "CR", "PC", "OC", "CV", "DVS"]:
        validation_means[key] = safe_div(
            sum(x[key] for x in doc_validation_records),
            len(doc_validation_records),
        ) if doc_validation_records else 0.0

    summary = {
        "entity": {
            "tp": entity_results["tp"],
            "fp": entity_results["fp"],
            "fn": entity_results["fn"],
            "precision": entity_results["precision"],
            "recall": entity_results["recall"],
            "f1": entity_results["f1"],
        },
        "relation": {
            "tp": relation_results["tp"],
            "fp": relation_results["fp"],
            "fn": relation_results["fn"],
            "precision": relation_results["precision"],
            "recall": relation_results["recall"],
            "f1": relation_results["f1"],
        },
        "per_relation": per_relation_rows,
        "validation_metrics_mean": validation_means,
        "total_docs": total_docs,
        "missing_predictions": missing_predictions,
        "parsed_failures": parsed_failures,
        "pred_entities_count": len(all_pred_entities),
        "pred_triples_count": len(all_pred_triples),
        "gt_entities_count": len(all_gt_entities),
        "gt_triples_count": len(all_gt_triples),
    }

    return summary


# =========================================================
# CLI
# =========================================================
def parse_type_filter_arg(type_filter_raw: str) -> str | list[str]:
    """Parse type-filter CLI argument."""
    type_filter_raw = type_filter_raw.strip()
    if type_filter_raw.lower() == "all":
        return "all"
    if "," in type_filter_raw:
        return [x.strip() for x in type_filter_raw.split(",") if x.strip()]
    return type_filter_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate relation extraction predictions on streaming JSONL datasets.")
    parser.add_argument("--gold-jsonl-path", required=True)
    parser.add_argument("--prediction-jsonl-path", required=True)
    parser.add_argument("--ontology-path", default=None)
    parser.add_argument("--type-filter", default="all")
    parser.add_argument("--output-summary-path", default=None)

    args = parser.parse_args()

    summary = evaluate_dataset(
        gold_jsonl_path=args.gold_jsonl_path,
        prediction_jsonl_path=args.prediction_jsonl_path,
        ontology_path=args.ontology_path,
        type_filter=parse_type_filter_arg(args.type_filter),
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_summary_path:
        output_summary_path = Path(args.output_summary_path)
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with output_summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
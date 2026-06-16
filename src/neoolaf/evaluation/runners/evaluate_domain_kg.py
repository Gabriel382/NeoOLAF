from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from rapidfuzz import fuzz
from rdflib import Graph, Literal


# ============================================================
# CONFIG SCHEMAS
# ============================================================

@dataclass(slots=True)
class MatchingConfig:
    entity_threshold: float = 88.0
    relation_head_threshold: float = 75.0
    relation_tail_threshold_default: float = 72.0
    relation_tail_threshold_by_relation: dict[str, float] = field(
        default_factory=lambda: {"REQUIRES": 60.0}
    )

    alarm_grounding_threshold: float = 65.0
    tail_grounding_threshold: float = 52.0
    trigger_cause_grounding_threshold: float = 52.0

    fair_str_head_threshold: float = 82.0
    fair_str_tail_threshold_default: float = 82.0
    fair_str_tail_threshold_by_relation: dict[str, float] = field(
        default_factory=lambda: {"REQUIRES": 72.0}
    )

    contradiction_low_sim: float = 35.0
    contradiction_same_relation_threshold: float = 75.0

    dvs_min_str: float = 0.50
    dvs_max_cr: float = 0.20
    dvs_min_oc: float = 0.70
    dvs_max_cv: float = 0.30


@dataclass(slots=True)
class RelationInferenceRule:
    relation: str
    keywords: list[str]


@dataclass(slots=True)
class DomainEvaluationConfig:
    name: str

    relation_schema: list[str]

    gold_head_field: str
    gold_relation_field: str
    gold_tail_field: str

    gold_id_field: str | None = None
    gold_category_field: str | None = None
    gold_type_field: str | None = None

    # Optional “anchor” logic. In XQuality this is Alarm No.
    anchor_field: str | None = None

    # Relation orientation roles.
    # Example:
    # TRIGGERS = cause_to_anchor
    # CAUSES = anchor_to_tail
    relation_roles: dict[str, str] = field(default_factory=dict)

    entity_alias_map: dict[str, str | None] = field(default_factory=dict)
    anchor_alias_map: dict[str, str | None] = field(default_factory=dict)

    bad_entity_labels: set[str] = field(default_factory=set)
    generic_gold_entity_blocklist: set[str] = field(default_factory=set)

    relation_inference_rules: list[RelationInferenceRule] = field(default_factory=list)
    role_inference_rules: dict[str, list[str]] = field(default_factory=dict)

    matching: MatchingConfig = field(default_factory=MatchingConfig)

    use_gt_guided_extraction_canonicalization: bool = True
    use_raw_fair_validation: bool = True
    use_ontology_for_entity_eval: bool = True


@dataclass(slots=True)
class MethodOutputConfig:
    name: str

    # For now, this generic runner supports JSON triple outputs.
    # Later: ttl, jsonl, singlepass_json, etc.
    format: str = "json_triples"

    subject_path: list[str] = field(default_factory=lambda: ["subject", "label"])
    predicate_path: list[str] = field(default_factory=lambda: ["predicate", "label"])
    object_path: list[str] = field(default_factory=lambda: ["object", "label"])

    evidence_path: list[str] | None = field(default_factory=lambda: ["justification"])
    chunk_id_path: list[str] | None = field(default_factory=lambda: ["chunk_id"])
    confidence_path: list[str] | None = field(default_factory=lambda: ["confidence"])
    provenance_path: list[str] | None = field(default_factory=lambda: ["provenance"])

    triples_key: str = "triples"

    local_label: str = "local"
    inferred_label: str = "inferred"


@dataclass(slots=True)
class EvaluationInput:
    local_json_path: str | Path | None = None
    inferred_json_path: str | Path | None = None
    ontology_local_path: str | Path | None = None
    ontology_inferred_path: str | Path | None = None
    gold_path: str | Path | None = None
    output_dir: str | Path = "outputs/evaluation"


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text)
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("—", " ")
        .replace("–", " ")
    )
    text = text.lower().strip()
    text = re.sub(r"[_/\\:;|]+", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str | None, b: str | None) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    return float(fuzz.token_set_ratio(a_n, b_n))


def token_overlap_score(a: str | None, b: str | None) -> float:
    a_toks = set(normalize_text(a).split())
    b_toks = set(normalize_text(b).split())
    if not a_toks or not b_toks:
        return 0.0
    return 100.0 * len(a_toks & b_toks) / max(1, len(a_toks | b_toks))


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def harmonic_f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def get_nested(data: dict[str, Any], path: list[str] | None, default: Any = None) -> Any:
    if not path:
        return default

    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)

    return current


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def uri_local_name(x: Any) -> str:
    if isinstance(x, Literal):
        return str(x)
    s = str(x)
    if "#" in s:
        s = s.split("#")[-1]
    else:
        s = s.rstrip("/").split("/")[-1]
    return s.replace("%20", " ")


# ============================================================
# DOMAIN-SPECIFIC BUT CONFIGURABLE HELPERS
# ============================================================

def apply_alias(label: str | None, alias_map: dict[str, str | None]) -> str | None:
    if label is None:
        return None
    n = normalize_text(label)
    return alias_map.get(n, label)


def is_bad_entity_label(label: str | None, domain: DomainEvaluationConfig) -> bool:
    n = normalize_text(label)

    if not n:
        return True

    if n in domain.bad_entity_labels:
        return True

    if re.match(r"^(concept_|ont_rel_|cand_[ser]_|cand_e_|cand_s_|cand_r_)", n):
        return True

    if re.match(r"^(concept|ont rel|cand e|cand s|cand r)\s+\d+", n):
        return True

    if re.match(r"^rcc8[a-z0-9_]*$", n):
        return True

    return False


def entity_similarity(a: str | None, b: str | None, domain: DomainEvaluationConfig) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    if not a_n or not b_n:
        return 0.0

    if a_n == b_n:
        return 100.0

    a_tokens = a_n.split()
    b_tokens = b_n.split()

    if not a_tokens or not b_tokens:
        return 0.0

    if len(a_tokens) == 1 and a_tokens[0] in domain.bad_entity_labels and a_n != b_n:
        return 0.0

    if len(b_tokens) == 1 and b_tokens[0] in domain.bad_entity_labels and a_n != b_n:
        return 0.0

    if min(len(a_tokens), len(b_tokens)) == 1 and max(len(a_tokens), len(b_tokens)) >= 3:
        return 0.0

    score = float(fuzz.token_sort_ratio(a_n, b_n))

    longer = max(len(a_tokens), len(b_tokens))
    shorter = min(len(a_tokens), len(b_tokens))

    if longer >= 2:
        score *= shorter / longer

    return score


def relation_tail_threshold(relation: str, domain: DomainEvaluationConfig) -> float:
    relation = relation.upper()
    return domain.matching.relation_tail_threshold_by_relation.get(
        relation,
        domain.matching.relation_tail_threshold_default,
    )


def fair_relation_tail_threshold(relation: str, domain: DomainEvaluationConfig) -> float:
    relation = relation.upper()
    return domain.matching.fair_str_tail_threshold_by_relation.get(
        relation,
        domain.matching.fair_str_tail_threshold_default,
    )


# ============================================================
# GOLD LOADING
# ============================================================

def load_domain_gold(
    gold_path: str | Path,
    domain: DomainEvaluationConfig,
) -> dict[str, Any]:
    raw_rows = load_json(gold_path)

    gt_triples: list[dict[str, Any]] = []
    gt_entities: set[str] = set()

    anchor_to_label: dict[str, str] = {}
    anchor_labels: set[str] = set()
    anchor_rel_to_tails: dict[tuple[str, str], list[str]] = defaultdict(list)
    triggers_by_anchor: dict[str, list[str]] = defaultdict(list)

    for row in raw_rows:
        head = str(row.get(domain.gold_head_field, "")).strip()
        rel = str(row.get(domain.gold_relation_field, "")).strip().upper()
        tail = str(row.get(domain.gold_tail_field, "")).strip()
        anchor_id = str(row.get(domain.anchor_field, "")).strip() if domain.anchor_field else ""

        if head:
            gt_entities.add(head)
        if tail:
            gt_entities.add(tail)

        role = domain.relation_roles.get(rel)

        # General anchor logic:
        # cause_to_anchor means tail is the anchor label.
        # anchor_to_tail means head is the anchor label.
        if anchor_id:
            if role == "cause_to_anchor" and tail:
                anchor_to_label[anchor_id] = tail
            elif role in {"anchor_to_tail", "anchor_to_effect", "anchor_to_intervention", "anchor_to_responsible", "anchor_to_reference"} and head:
                anchor_to_label[anchor_id] = head

        if role == "cause_to_anchor":
            anchor_labels.add(tail)
            triggers_by_anchor[tail].append(head)
        elif role in {"anchor_to_tail", "anchor_to_effect", "anchor_to_intervention", "anchor_to_responsible", "anchor_to_reference"}:
            anchor_labels.add(head)
            anchor_rel_to_tails[(head, rel)].append(tail)

        gt_triples.append(
            {
                "head": head,
                "rel": rel,
                "tail": tail,
                "anchor_id": anchor_id,
                "raw": row,
            }
        )

    gt_entities_for_eval = {
        e for e in gt_entities
        if normalize_text(e) not in domain.generic_gold_entity_blocklist
    }

    return {
        "raw": raw_rows,
        "triples": gt_triples,
        "entities": gt_entities,
        "entities_for_eval": gt_entities_for_eval,
        "anchor_to_label": anchor_to_label,
        "anchor_labels": sorted(anchor_labels),
        "anchor_rel_to_tails": anchor_rel_to_tails,
        "triggers_by_anchor": triggers_by_anchor,
    }


# ============================================================
# METHOD OUTPUT LOADING
# ============================================================

def load_method_json_triples(
    json_path: str | Path,
    method: MethodOutputConfig,
) -> list[dict[str, Any]]:
    data = load_json(json_path)
    return data.get(method.triples_key, [])


def extract_raw_fields(
    triple: dict[str, Any],
    method: MethodOutputConfig,
) -> dict[str, Any]:
    subject = str(get_nested(triple, method.subject_path, "") or "").strip()
    predicate = str(get_nested(triple, method.predicate_path, "") or "").strip()
    object_ = str(get_nested(triple, method.object_path, "") or "").strip()

    evidence = str(get_nested(triple, method.evidence_path, "") or "").strip()
    chunk_id = str(get_nested(triple, method.chunk_id_path, "") or "").strip()
    confidence = get_nested(triple, method.confidence_path, None)
    provenance = get_nested(triple, method.provenance_path, None)

    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "evidence": evidence,
        "chunk_id": chunk_id,
        "confidence": confidence,
        "provenance": provenance,
        "raw": triple,
    }


def extract_entities_from_method_triples(
    triples: list[dict[str, Any]],
    method: MethodOutputConfig,
    domain: DomainEvaluationConfig,
) -> set[str]:
    entities: set[str] = set()

    for triple in triples:
        fields = extract_raw_fields(triple, method)

        subject = apply_alias(fields["subject"], domain.entity_alias_map)
        object_ = apply_alias(fields["object"], domain.entity_alias_map)

        if subject and not is_bad_entity_label(subject, domain):
            entities.add(subject)

        if object_ and not is_bad_entity_label(object_, domain):
            entities.add(object_)

    return entities


# ============================================================
# CONFIGURABLE RELATION INFERENCE
# ============================================================

def infer_relation_type(
    fields: dict[str, Any],
    domain: DomainEvaluationConfig,
) -> str | None:
    full = " ".join(
        normalize_text(x)
        for x in [
            fields.get("subject"),
            fields.get("predicate"),
            fields.get("object"),
            fields.get("evidence"),
        ]
        if x
    )

    for rule in domain.relation_inference_rules:
        for keyword in rule.keywords:
            if normalize_text(keyword) in full:
                return rule.relation

    return None


def extract_roles_from_texts(
    texts: list[str | None],
    domain: DomainEvaluationConfig,
) -> list[str]:
    full = " ".join(normalize_text(t) for t in texts if t)

    roles = []

    for role, keywords in domain.role_inference_rules.items():
        for keyword in keywords:
            if normalize_text(keyword) in full:
                roles.append(role)
                break

    return list(dict.fromkeys(roles))


def candidate_anchor_labels_from_texts(
    texts: list[str | None],
    gold: dict[str, Any],
    domain: DomainEvaluationConfig,
    threshold: float | None = None,
) -> list[str]:
    threshold = threshold or domain.matching.alarm_grounding_threshold

    out: list[str] = []

    for txt in texts:
        if not txt:
            continue

        alias = apply_alias(txt, domain.anchor_alias_map)
        if alias is not None:
            out.append(alias)

    for anchor in gold["anchor_labels"]:
        scores = [similarity(txt, anchor) for txt in texts if txt]
        best_score = max(scores) if scores else 0.0

        if best_score >= threshold:
            out.append(anchor)

    return list(dict.fromkeys(out))


# ============================================================
# RELAXED GT-GUIDED CANONICALIZATION
# ============================================================

def dedup_extraction_triples(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()

    for t in triples:
        key = (
            normalize_text(t["head"]),
            t["rel"],
            normalize_text(t["tail"]),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(t)

    return out


def canonicalize_triple_to_domain_gold(
    triple: dict[str, Any],
    method: MethodOutputConfig,
    domain: DomainEvaluationConfig,
    gold: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = extract_raw_fields(triple, method)

    subject = apply_alias(fields["subject"], domain.entity_alias_map)
    predicate = fields["predicate"]
    object_ = apply_alias(fields["object"], domain.entity_alias_map)
    evidence = fields["evidence"]

    texts = [subject, predicate, object_, evidence]

    rel = infer_relation_type(fields, domain)
    if rel is None:
        return []

    role = domain.relation_roles.get(rel)
    anchor_candidates = candidate_anchor_labels_from_texts(texts, gold, domain)

    results: list[dict[str, Any]] = []

    # cause_to_anchor, e.g. XQuality TRIGGERS
    if role == "cause_to_anchor":
        for anchor_label in anchor_candidates:
            possible_causes = gold["triggers_by_anchor"].get(anchor_label, [])

            for gt_cause in possible_causes:
                scores = [similarity(txt, gt_cause) for txt in texts if txt]
                overlap_scores = [token_overlap_score(txt, gt_cause) for txt in texts if txt]
                best_score = max(scores + overlap_scores) if scores or overlap_scores else 0.0

                if best_score >= domain.matching.trigger_cause_grounding_threshold:
                    results.append(
                        {
                            "head": gt_cause,
                            "rel": rel,
                            "tail": anchor_label,
                            "raw_subject": subject,
                            "raw_predicate": predicate,
                            "raw_object": object_,
                            "justification": evidence,
                            "chunkid": fields["chunk_id"],
                            "confidence": fields["confidence"],
                            "provenance_present": bool(
                                evidence or fields["chunk_id"] or fields["provenance"]
                            ),
                            "raw": triple,
                        }
                    )

        return dedup_extraction_triples(results)

    if not anchor_candidates:
        return []

    # role expansion, e.g. XQuality HANDLED_BY
    if rel in domain.role_inference_rules:
        roles = extract_roles_from_texts(texts, domain)

        for anchor_label in anchor_candidates:
            gt_tails = set(gold["anchor_rel_to_tails"].get((anchor_label, rel), []))

            for role_label in roles:
                if role_label in gt_tails:
                    results.append(
                        {
                            "head": anchor_label,
                            "rel": rel,
                            "tail": role_label,
                            "raw_subject": subject,
                            "raw_predicate": predicate,
                            "raw_object": object_,
                            "justification": evidence,
                            "chunkid": fields["chunk_id"],
                            "confidence": fields["confidence"],
                            "provenance_present": bool(
                                evidence or fields["chunk_id"] or fields["provenance"]
                            ),
                            "raw": triple,
                        }
                    )

        return dedup_extraction_triples(results)

    # anchor_to_tail relations
    for anchor_label in anchor_candidates:
        possible_tails = gold["anchor_rel_to_tails"].get((anchor_label, rel), [])

        for gt_tail in possible_tails:
            scores = [similarity(txt, gt_tail) for txt in texts if txt]
            overlap_scores = [token_overlap_score(txt, gt_tail) for txt in texts if txt]
            best_score = max(scores + overlap_scores) if scores or overlap_scores else 0.0

            if best_score >= domain.matching.tail_grounding_threshold:
                results.append(
                    {
                        "head": anchor_label,
                        "rel": rel,
                        "tail": gt_tail,
                        "raw_subject": subject,
                        "raw_predicate": predicate,
                        "raw_object": object_,
                        "justification": evidence,
                        "chunkid": fields["chunk_id"],
                        "confidence": fields["confidence"],
                        "provenance_present": bool(
                            evidence or fields["chunk_id"] or fields["provenance"]
                        ),
                        "raw": triple,
                    }
                )

    return dedup_extraction_triples(results)


def canonicalize_triples_to_domain_gold(
    triples: list[dict[str, Any]],
    method: MethodOutputConfig,
    domain: DomainEvaluationConfig,
    gold: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for triple in triples:
        out.extend(
            canonicalize_triple_to_domain_gold(
                triple=triple,
                method=method,
                domain=domain,
                gold=gold,
            )
        )

    return dedup_extraction_triples(out)


# ============================================================
# FAIR RAW VALIDATION CONVERSION
# ============================================================

def dedup_raw_triples(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()

    for t in triples:
        key = (
            normalize_text(t["head"]),
            normalize_text(t["rel"]),
            normalize_text(t["tail"]),
            normalize_text(t.get("support_text", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(t)

    return out


def raw_method_triples_to_fair_triples(
    json_path: str | Path,
    method: MethodOutputConfig,
    domain: DomainEvaluationConfig,
    gold: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_triples = load_method_json_triples(json_path, method)
    out: list[dict[str, Any]] = []

    for triple in raw_triples:
        fields = extract_raw_fields(triple, method)

        rel = infer_relation_type(fields, domain)
        if rel is None:
            continue

        subject = fields["subject"]
        predicate = fields["predicate"]
        object_ = fields["object"]
        evidence = fields["evidence"]

        texts = [subject, predicate, object_, evidence]
        anchor_candidates = candidate_anchor_labels_from_texts(
            texts=texts,
            gold=gold,
            domain=domain,
            threshold=68,
        )

        base = {
            "source_method": method.name,
            "raw_subject": subject,
            "raw_predicate": predicate,
            "raw_object": object_,
            "justification": evidence,
            "chunkid": fields["chunk_id"],
            "confidence": fields["confidence"],
            "provenance_present": bool(
                evidence or fields["chunk_id"] or fields["provenance"]
            ),
            "support_text": f"{predicate} {evidence}".strip(),
            "raw": triple,
        }

        role = domain.relation_roles.get(rel)

        if not anchor_candidates:
            out.append(
                {
                    "head": subject,
                    "rel": rel,
                    "tail": object_,
                    **base,
                }
            )
            continue

        if role == "cause_to_anchor":
            for anchor in anchor_candidates[:2]:
                out.append(
                    {
                        "head": subject if subject else predicate,
                        "rel": rel,
                        "tail": anchor,
                        **base,
                    }
                )

        elif rel in domain.role_inference_rules:
            roles = extract_roles_from_texts(texts, domain)
            if not roles and object_:
                roles = [object_]

            for anchor in anchor_candidates[:3]:
                for role_label in roles:
                    out.append(
                        {
                            "head": anchor,
                            "rel": rel,
                            "tail": role_label,
                            **base,
                        }
                    )

        else:
            for anchor in anchor_candidates[:3]:
                out.append(
                    {
                        "head": anchor,
                        "rel": rel,
                        "tail": object_ if object_ else predicate,
                        **base,
                    }
                )

    return dedup_raw_triples(out), raw_triples


# ============================================================
# MATCHING
# ============================================================

def greedy_entity_matching(
    pred_entities: set[str],
    gt_entities: set[str],
    domain: DomainEvaluationConfig,
) -> dict[str, Any]:
    pred_list = list(pred_entities)
    gt_list = list(gt_entities)

    candidates = []

    for i, pred in enumerate(pred_list):
        if not normalize_text(pred):
            continue

        for j, gt in enumerate(gt_list):
            score = entity_similarity(pred, gt, domain)
            if score >= domain.matching.entity_threshold:
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

    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": harmonic_f1(p, r),
        "matches": matches,
        "unmatched_pred": [pred_list[i] for i in range(len(pred_list)) if i not in used_pred],
        "unmatched_gold": [gt_list[j] for j in range(len(gt_list)) if j not in used_gt],
    }


def greedy_relation_matching(
    pred_triples: list[dict[str, Any]],
    gt_triples: list[dict[str, Any]],
    domain: DomainEvaluationConfig,
    fair: bool = False,
) -> dict[str, Any]:
    candidates = []

    for i, pred in enumerate(pred_triples):
        for j, gt in enumerate(gt_triples):
            if normalize_text(pred["rel"]) != normalize_text(gt["rel"]):
                continue

            head_score = similarity(pred["head"], gt["head"])
            tail_score = similarity(pred["tail"], gt["tail"])

            if fair:
                head_threshold = domain.matching.fair_str_head_threshold
                tail_threshold = fair_relation_tail_threshold(pred["rel"], domain)
            else:
                head_threshold = domain.matching.relation_head_threshold
                tail_threshold = relation_tail_threshold(pred["rel"], domain)

            if head_score >= head_threshold and tail_score >= tail_threshold:
                total = 0.5 * head_score + 0.5 * tail_score
                candidates.append((total, head_score, tail_score, i, j))

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2], -x[3], -x[4]))

    used_pred = set()
    used_gt = set()
    matches = []

    for total, head_score, tail_score, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue

        used_pred.add(i)
        used_gt.add(j)

        matches.append(
            {
                "pred_idx": i,
                "gt_idx": j,
                "score": total,
                "head_score": head_score,
                "tail_score": tail_score,
                "pred": pred_triples[i],
                "gt": gt_triples[j],
            }
        )

    tp = len(matches)
    fp = len(pred_triples) - tp
    fn = len(gt_triples) - tp

    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": harmonic_f1(p, r),
        "matches": matches,
        "unmatched_pred": [pred_triples[i] for i in range(len(pred_triples)) if i not in used_pred],
        "unmatched_gold": [gt_triples[j] for j in range(len(gt_triples)) if j not in used_gt],
    }


def evaluate_per_relation(
    pred_triples: list[dict[str, Any]],
    gt_triples: list[dict[str, Any]],
    domain: DomainEvaluationConfig,
    fair: bool = False,
) -> list[dict[str, Any]]:
    relations = sorted(set(x["rel"] for x in gt_triples) | set(x["rel"] for x in pred_triples))

    rows = []

    for rel in relations:
        pred_subset = [x for x in pred_triples if x["rel"] == rel]
        gt_subset = [x for x in gt_triples if x["rel"] == rel]

        res = greedy_relation_matching(pred_subset, gt_subset, domain, fair=fair)

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


# ============================================================
# VALIDATION METRICS ON CANONICALIZED TRIPLES
# ============================================================

def loose_triple_match(
    pred_t: dict[str, Any],
    gt_t: dict[str, Any],
    domain: DomainEvaluationConfig,
) -> bool:
    """Loose triple support match: same relation, similar head, similar tail."""
    if pred_t["rel"] != gt_t["rel"]:
        return False

    head_score = similarity(pred_t["head"], gt_t["head"])
    tail_score = similarity(pred_t["tail"], gt_t["tail"])

    return (
        head_score >= domain.matching.fair_str_head_threshold
        and tail_score >= fair_relation_tail_threshold(pred_t["rel"], domain)
    )


def compute_supported_triple_ratio(
    pred_triples: list[dict[str, Any]],
    gt_triples: list[dict[str, Any]],
    domain: DomainEvaluationConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """STR = proportion of predicted triples supported by at least one GT triple."""
    if not pred_triples:
        return 0.0, []

    supported_flags = []
    supported_examples = []

    for pred in pred_triples:
        supported = any(
            loose_triple_match(pred, gt, domain)
            for gt in gt_triples
        )

        supported_flags.append(supported)

        if supported and len(supported_examples) < 10:
            supported_examples.append(pred)

    return safe_div(sum(supported_flags), len(pred_triples)), supported_examples


def compute_contradiction_rate(
    pred_triples: list[dict[str, Any]],
    gt_triples: list[dict[str, Any]],
    domain: DomainEvaluationConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """
    CR = proportion of predicted triples that contradict GT.

    This reproduces the old pragmatic evaluator:
    - predicted triple is unsupported
    - but there exists a GT triple with the same normalized head + same relation
      and clearly different tail, or same normalized tail + same relation and
      clearly different head.
    """
    if not pred_triples:
        return 0.0, []

    by_head_rel = defaultdict(list)
    by_tail_rel = defaultdict(list)

    for gt in gt_triples:
        by_head_rel[(normalize_text(gt["head"]), gt["rel"])].append(gt)
        by_tail_rel[(normalize_text(gt["tail"]), gt["rel"])].append(gt)

    contradiction_flags = []
    contradictions = []

    for pred in pred_triples:
        if any(loose_triple_match(pred, gt, domain) for gt in gt_triples):
            contradiction_flags.append(False)
            continue

        pred_head_n = normalize_text(pred["head"])
        pred_tail_n = normalize_text(pred["tail"])
        rel = pred["rel"]

        contradiction = False

        for gt in by_head_rel.get((pred_head_n, rel), []):
            tail_score = similarity(pred["tail"], gt["tail"])
            if tail_score < 40:
                contradiction = True
                break

        if not contradiction:
            for gt in by_tail_rel.get((pred_tail_n, rel), []):
                head_score = similarity(pred["head"], gt["head"])
                if head_score < 40:
                    contradiction = True
                    break

        contradiction_flags.append(contradiction)

        if contradiction and len(contradictions) < 10:
            contradictions.append(pred)

    return safe_div(sum(contradiction_flags), len(pred_triples)), contradictions


def compute_provenance_coverage(
    pred_triples: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """PC = proportion of predicted triples with usable provenance."""
    if not pred_triples:
        return 0.0, []

    covered_flags = []
    covered_examples = []

    for triple in pred_triples:
        has_provenance = bool(
            triple.get("provenance_present", False)
            or triple.get("justification", "")
            or triple.get("chunkid", "")
        )

        covered_flags.append(has_provenance)

        if has_provenance and len(covered_examples) < 10:
            covered_examples.append(triple)

    return safe_div(sum(covered_flags), len(pred_triples)), covered_examples


def compute_ontology_conformance_old_style(
    pred_triples: list[dict[str, Any]],
    domain: DomainEvaluationConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """
    OC = old-style ontology conformance.

    This reproduces the previous notebook behavior:
    - relation must be in the expected relation schema
    - head and tail must be non-empty

    The old code had endpoint semantic checks but they were effectively permissive.
    """
    if not pred_triples:
        return 0.0, []

    allowed_relations = set(domain.relation_schema)

    flags = []
    examples = []

    for triple in pred_triples:
        rel_ok = triple["rel"] in allowed_relations
        head_ok = bool(normalize_text(triple["head"]))
        tail_ok = bool(normalize_text(triple["tail"]))

        ok = rel_ok and head_ok and tail_ok

        flags.append(ok)

        if ok and len(examples) < 10:
            examples.append(triple)

    return safe_div(sum(flags), len(pred_triples)), examples


def compute_constraint_violations_old_style(
    pred_triples: list[dict[str, Any]],
    domain: DomainEvaluationConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """
    CV = old-style lightweight constraint violations.

    Constraint proxies:
    - empty head or tail
    - relation outside expected schema
    - same head and tail
    - generic placeholders as endpoints
    """
    if not pred_triples:
        return 0.0, []

    allowed_relations = set(domain.relation_schema)

    generic_bad = {
        "alarm",
        "device",
        "failure",
        "check",
        "part",
        "machine",
        "operator",
    }

    flags = []
    examples = []

    for triple in pred_triples:
        head_n = normalize_text(triple["head"])
        tail_n = normalize_text(triple["tail"])
        rel = triple["rel"]

        violated = False

        if rel not in allowed_relations:
            violated = True
        elif not head_n or not tail_n:
            violated = True
        elif head_n == tail_n:
            violated = True
        elif head_n in generic_bad or tail_n in generic_bad:
            violated = True

        flags.append(violated)

        if violated and len(examples) < 10:
            examples.append(triple)

    return safe_div(sum(flags), len(pred_triples)), examples


def compute_document_level_validation_success_old_style(
    pred_triples: list[dict[str, Any]],
    gt_triples: list[dict[str, Any]],
    domain: DomainEvaluationConfig,
    min_supported_ratio: float = 0.5,
) -> tuple[float, dict[str, Any]]:
    """
    DVS = old-style document-level validation success.

    The old evaluator used:
    - STR >= 0.5
    - CR <= 0.2
    - OC >= 0.8

    It did not use CV as a blocking condition for DVS.
    """
    if not pred_triples:
        return 0.0, {
            "supported_ratio": 0.0,
            "contradiction_rate": 0.0,
            "ontology_conformance": 0.0,
            "success": False,
        }

    str_value, _ = compute_supported_triple_ratio(pred_triples, gt_triples, domain)
    cr_value, _ = compute_contradiction_rate(pred_triples, gt_triples, domain)
    oc_value, _ = compute_ontology_conformance_old_style(pred_triples, domain)

    success = (
        len(pred_triples) > 0
        and str_value >= min_supported_ratio
        and cr_value <= 0.2
        and oc_value >= 0.8
    )

    return float(success), {
        "supported_ratio": str_value,
        "contradiction_rate": cr_value,
        "ontology_conformance": oc_value,
        "success": success,
    }


def evaluate_validation_on_canonicalized_triples(
    pred_triples: list[dict[str, Any]],
    gt_triples: list[dict[str, Any]],
    domain: DomainEvaluationConfig,
) -> dict[str, Any]:
    """
    Reproduce the old validation-oriented metrics on canonicalized triples.

    Important:
    This intentionally uses the same canonicalized triples used for relaxed
    relation P/R/F1, matching the previous notebook evaluator.
    """
    pred_triples = dedup_extraction_triples(pred_triples)

    str_val, str_examples = compute_supported_triple_ratio(
        pred_triples,
        gt_triples,
        domain,
    )
    cr_val, cr_examples = compute_contradiction_rate(
        pred_triples,
        gt_triples,
        domain,
    )
    pc_val, pc_examples = compute_provenance_coverage(pred_triples)
    oc_val, oc_examples = compute_ontology_conformance_old_style(pred_triples, domain)
    cv_val, cv_examples = compute_constraint_violations_old_style(pred_triples, domain)
    dvs_val, dvs_details = compute_document_level_validation_success_old_style(
        pred_triples,
        gt_triples,
        domain,
    )

    return {
        "metrics": {
            "STR": str_val,
            "CR": cr_val,
            "PC": pc_val,
            "OC": oc_val,
            "CV": cv_val,
            "DVS": dvs_val,
        },
        "dvs_details": dvs_details,
        "validation_triples": pred_triples,
        "debug_examples": {
            "STR": str_examples,
            "CR": cr_examples,
            "PC": pc_examples,
            "OC": oc_examples,
            "CV": cv_examples,
        },
    }


# ============================================================
# ONTOLOGY SUMMARY
# ============================================================

def ontology_metrics_from_ttl(paths: list[str | Path | None]) -> dict[str, Any]:
    graph = Graph()
    available = False

    for path in paths:
        if not path:
            continue

        p = Path(path)
        if not p.exists():
            continue

        try:
            graph.parse(str(p), format="turtle")
            available = True
        except Exception:
            continue

    if not available:
        return {"available": False}

    class_count = 0
    property_count = 0
    hierarchy_link_count = 0
    axiom_count = 0
    comments = set()
    domains = set()
    ranges = set()
    classes = []
    properties = []

    for s, p, o in graph:
        p_name = normalize_text(uri_local_name(p))
        o_name = normalize_text(uri_local_name(o))

        if p_name == "type":
            if o_name in {"class", "owl class"} or o_name.endswith("class"):
                class_count += 1
                classes.append(uri_local_name(s))
            if "property" in o_name:
                property_count += 1
                properties.append(uri_local_name(s))

        if p_name == "subclassof":
            hierarchy_link_count += 1

        if p_name in {"equivalentclass", "disjointwith", "propertychainaxiom"}:
            axiom_count += 1

        if p_name in {"comment", "description"} and isinstance(o, Literal):
            comments.add(normalize_text(str(o)))

        if p_name == "domain":
            domains.add(str(s))

        if p_name == "range":
            ranges.add(str(s))

    return {
        "available": True,
        "class_count": class_count,
        "property_count": property_count,
        "hierarchy_link_count": hierarchy_link_count,
        "axiom_count": axiom_count,
        "description_coverage": safe_div(len(comments), max(1, class_count + property_count)),
        "domain_coverage": safe_div(len(domains), max(1, property_count)),
        "range_coverage": safe_div(len(ranges), max(1, property_count)),
        "duplicate_class_count": max(0, len(classes) - len(set(normalize_text(x) for x in classes))),
        "duplicate_property_count": max(0, len(properties) - len(set(normalize_text(x) for x in properties))),
        "hierarchy_depth": 0,
        "cycle_count": 0,
        "ontology_delta_size": 0,
        "promoted_concept_count": 0,
        "promoted_relation_count": 0,
        "ontology_growth_rate": 0.0,
    }


# ============================================================
# MAIN GENERAL EVALUATOR
# ============================================================

def evaluate_variant(
    variant_name: str,
    json_path: str | Path | None,
    ontology_path: str | Path | None,
    method: MethodOutputConfig,
    domain: DomainEvaluationConfig,
    gold: dict[str, Any],
) -> dict[str, Any] | None:
    if not json_path:
        return None

    json_path = Path(json_path)

    if not json_path.exists():
        return None

    raw_triples = load_method_json_triples(json_path, method)

    pred_triples = canonicalize_triples_to_domain_gold(
        triples=raw_triples,
        method=method,
        domain=domain,
        gold=gold,
    )

    pred_entities = extract_entities_from_method_triples(
        triples=raw_triples,
        method=method,
        domain=domain,
    )

    for triple in pred_triples:
        head = apply_alias(triple["head"], domain.entity_alias_map)
        tail = apply_alias(triple["tail"], domain.entity_alias_map)

        if head and not is_bad_entity_label(head, domain):
            pred_entities.add(head)

        if tail and not is_bad_entity_label(tail, domain):
            pred_entities.add(tail)

    entity_result = greedy_entity_matching(
        pred_entities=pred_entities,
        gt_entities=gold["entities_for_eval"],
        domain=domain,
    )

    relation_result = greedy_relation_matching(
        pred_triples=pred_triples,
        gt_triples=gold["triples"],
        domain=domain,
        fair=False,
    )

    per_relation = evaluate_per_relation(
        pred_triples=pred_triples,
        gt_triples=gold["triples"],
        domain=domain,
        fair=False,
    )

    return {
        "variant": variant_name,
        "entity": {
            key: entity_result[key]
            for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
        },
        "relation": {
            key: relation_result[key]
            for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
        },
        "per_relation": per_relation,
        "pred_triples_count": len(pred_triples),
        "pred_entities_count": len(pred_entities),
        "pred_triples": pred_triples,
        "pred_entities_full": sorted(pred_entities),
        "raw_triples_count": len(raw_triples),
        "matched_entities": entity_result["matches"],
        "matched_relations": relation_result["matches"],
        "unmatched_entities": entity_result["unmatched_pred"],
        "unmatched_relations": relation_result["unmatched_pred"],
    }


def merge_variant_summaries(
    summaries: list[dict[str, Any] | None],
    domain: DomainEvaluationConfig,
    gold: dict[str, Any],
) -> dict[str, Any]:
    merged_triples = []
    seen_triples = set()
    merged_entities = set()

    for summary in summaries:
        if summary is None:
            continue

        for triple in summary["pred_triples"]:
            key = (
                normalize_text(triple["head"]),
                triple["rel"],
                normalize_text(triple["tail"]),
            )

            if key in seen_triples:
                continue

            seen_triples.add(key)
            merged_triples.append(triple)

        for entity in summary.get("pred_entities_full", []):
            entity = apply_alias(entity, domain.entity_alias_map)
            if entity and not is_bad_entity_label(entity, domain):
                merged_entities.add(entity)

    entity_result = greedy_entity_matching(merged_entities, gold["entities_for_eval"], domain)
    relation_result = greedy_relation_matching(merged_triples, gold["triples"], domain, fair=False)
    per_relation = evaluate_per_relation(merged_triples, gold["triples"], domain, fair=False)

    return {
        "variant": "merged",
        "entity": {
            key: entity_result[key]
            for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
        },
        "relation": {
            key: relation_result[key]
            for key in ["tp", "fp", "fn", "precision", "recall", "f1"]
        },
        "per_relation": per_relation,
        "pred_triples_count": len(merged_triples),
        "pred_entities_count": len(merged_entities),
        "pred_triples": merged_triples,
        "pred_entities_full": sorted(merged_entities),
        "matched_entities": entity_result["matches"],
        "matched_relations": relation_result["matches"],
        "unmatched_entities": entity_result["unmatched_pred"],
        "unmatched_relations": relation_result["unmatched_pred"],
    }


def evaluate_domain_kg(
    input_data: EvaluationInput,
    method: MethodOutputConfig,
    domain: DomainEvaluationConfig,
    profile: str = "relaxed_fair",
) -> dict[str, Any]:
    if input_data.gold_path is None:
        raise ValueError("gold_path is required.")

    output_dir = Path(input_data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = load_domain_gold(input_data.gold_path, domain)

    local_summary = evaluate_variant(
        variant_name=method.local_label,
        json_path=input_data.local_json_path,
        ontology_path=input_data.ontology_local_path,
        method=method,
        domain=domain,
        gold=gold,
    )

    inferred_summary = evaluate_variant(
        variant_name=method.inferred_label,
        json_path=input_data.inferred_json_path,
        ontology_path=input_data.ontology_inferred_path,
        method=method,
        domain=domain,
        gold=gold,
    )

    merged_summary = merge_variant_summaries(
        summaries=[local_summary, inferred_summary],
        domain=domain,
        gold=gold,
    )

    validation = evaluate_validation_on_canonicalized_triples(
        pred_triples=merged_summary["pred_triples"],
        gt_triples=gold["triples"],
        domain=domain,
    )

    ontology_metrics = ontology_metrics_from_ttl(
        [input_data.ontology_local_path, input_data.ontology_inferred_path]
    )

    summary = {
        "dataset": domain.name,
        "method": method.name,
        "profile": profile,
        "evaluation_protocol": {
            "extraction": "relaxed_gt_guided_domain_canonicalization",
            "validation": "validation_on_canonicalized_triples_old_style",
        },
        "entity": merged_summary["entity"],
        "relation": merged_summary["relation"],
        "per_relation": merged_summary["per_relation"],
        "validation_metrics_mean": validation["metrics"],
        "DVS_details": validation["dvs_details"],
        "total_docs": 1,
        "missing_predictions": 0,
        "parsed_failures": 0,
        "pred_entities_count": merged_summary["pred_entities_count"],
        "gt_entities_count": len(gold["entities_for_eval"]),
        "pred_relations_count": merged_summary["pred_triples_count"],
        "gt_relations_count": len(gold["triples"]),
        "variants": {
            method.local_label: local_summary,
            method.inferred_label: inferred_summary,
            "merged": merged_summary,
        },
    }

    save_json(output_dir / "metrics.summary.json", summary)
    save_json(output_dir / "canonicalized_triples.json", merged_summary["pred_triples"])
    save_json(output_dir / "validation_triples.json", validation["validation_triples"])

    # Compatibility alias for old reporting/debug code.
    save_json(output_dir / "fair_raw_triples.json", validation["validation_triples"])

    save_json(output_dir / "matched_entities.json", merged_summary["matched_entities"])
    save_json(output_dir / "matched_relations.json", merged_summary["matched_relations"])
    save_json(output_dir / "unmatched_entities.json", merged_summary["unmatched_entities"])
    save_json(output_dir / "unmatched_relations.json", merged_summary["unmatched_relations"])
    save_json(output_dir / "debug_examples.json", validation["debug_examples"])
    save_json(output_dir / "ontology_metrics.json", ontology_metrics)
    save_json(output_dir / "errors.json", [])

    flat_df = pd.DataFrame(
        [
            {
                "method": method.name,
                "dataset": domain.name,
                "profile": profile,
                "entity_precision": summary["entity"]["precision"],
                "entity_recall": summary["entity"]["recall"],
                "entity_f1": summary["entity"]["f1"],
                "relation_precision": summary["relation"]["precision"],
                "relation_recall": summary["relation"]["recall"],
                "relation_f1": summary["relation"]["f1"],
                **validation["metrics"],
            }
        ]
    )

    flat_df.to_csv(output_dir / "metrics.flat.csv", index=False)

    pd.DataFrame(merged_summary["per_relation"]).to_csv(
        output_dir / "per_relation_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "document_id": domain.name,
                **validation["metrics"],
                "DVS_details": json.dumps(validation["dvs_details"], ensure_ascii=False),
            }
        ]
    ).to_csv(output_dir / "validation_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "variant": summary_item["variant"],
                "entity_precision": summary_item["entity"]["precision"],
                "entity_recall": summary_item["entity"]["recall"],
                "entity_f1": summary_item["entity"]["f1"],
                "relation_precision": summary_item["relation"]["precision"],
                "relation_recall": summary_item["relation"]["recall"],
                "relation_f1": summary_item["relation"]["f1"],
                "pred_entities_count": summary_item["pred_entities_count"],
                "pred_triples_count": summary_item["pred_triples_count"],
            }
            for summary_item in [
                item for item in [local_summary, inferred_summary, merged_summary]
                if item is not None
            ]
        ]
    ).to_csv(output_dir / "variant_metrics.csv", index=False)

    return summary
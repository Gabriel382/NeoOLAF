"""XQuality-specific gold loading and canonicalization helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from neoolaf.evaluation.matching.normalization import normalize_text, token_overlap_score
from neoolaf.evaluation.matching.similarity import loose_similarity
from neoolaf.evaluation.schema.artifact import EvalDocument, EvalEntity, EvalRelation
from neoolaf.evaluation.schema.config import EvaluationProfile

ENTITY_ALIAS_MAP = {
    normalize_text("ActiveEmergencyEvent"): "EMERGENCY ACTIVE",
    normalize_text("Open side guard alarm"): "SIDE GUARDS OPEN",
    normalize_text("OpenSideGuardAlarmEvent"): "SIDE GUARDS OPEN",
    normalize_text("OPEN FORCEPS"): "CHUCK OPEN",
    normalize_text("THERMOMAGNET SWITCHES. END OF CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
    normalize_text("THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
    normalize_text("THERMOMAGNET SWITCHES END OF CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
    normalize_text("ProgramRewindEvent"): "PROGRAM NOT REWOUND",
    normalize_text("Side guards open"): "SIDE GUARDS OPEN",
}

ALARM_ALIAS_MAP = {
    normalize_text("ActiveEmergencyEvent"): "EMERGENCY ACTIVE",
    normalize_text("Open side guard alarm"): "SIDE GUARDS OPEN",
    normalize_text("OpenSideGuardAlarmEvent"): "SIDE GUARDS OPEN",
    normalize_text("OPEN FORCEPS"): "CHUCK OPEN",
    normalize_text("THERMOMAGNET SWITCHES. END OF CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
    normalize_text("THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
    normalize_text("THERMOMAGNET SWITCHES END OF CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
    normalize_text("waiting for excessive feed"): "CDS: WAITING FOR EXCESSIVE FEED",
    normalize_text("alarm"): None,
    normalize_text("failure"): None,
    normalize_text("device"): None,
    normalize_text("check"): None,
    normalize_text("part"): None,
    normalize_text("machine"): None,
}

BAD_ENTITY_LABELS = {
    "integer", "string", "boolean", "decimal", "float", "double", "class",
    "objectproperty", "datatypeproperty", "thing", "property", "resource", "result",
    "process", "situation", "constraint", "observation", "feature", "platform",
    "staff", "manager", "geometry", "line", "sensor", "ontology", "product",
    "action", "cause", "feature of interest", "observable property", "temporal entity",
    "spatial object", "human process", "logistic process", "manufacturing process",
    "manufacturing facility", "manufacturing cell", "cell", "technician", "instant",
    "interval", "alarm", "device", "failure", "check", "consult", "part", "machine",
    "clamp", "operator",
}


def normalize_entity_alias(label: str | None) -> str | None:
    """Normalize known NeoOLAF/XQuality aliases to gold-like labels."""
    if label is None:
        return None
    return ENTITY_ALIAS_MAP.get(normalize_text(label), label)


def apply_alarm_alias(label: str | None) -> str | None:
    """Return a known alarm alias or None."""
    if label is None:
        return None
    return ALARM_ALIAS_MAP.get(normalize_text(label))


def is_bad_entity_label(label: str | None) -> bool:
    """Filter obvious ontology/meta artifacts from XQuality entity evaluation."""
    if label is None:
        return True
    raw = str(label).strip()
    norm = normalize_text(raw)
    if not norm:
        return True
    if re.match(r"^(concept_|ont_rel_|cand_[ser]_|cand_e_|cand_s_|cand_r_)", raw.lower()):
        return True
    if re.match(r"^rcc8[a-z0-9_]*$", raw.lower()):
        return True
    if re.match(r"^(concept|ont rel|cand e|cand s|cand r)\s+\d+", norm):
        return True
    if norm.startswith("has ") or norm.startswith("is ") or norm.startswith("interval "):
        return True
    if re.match(r"^situation\s+s?\d+$", norm):
        return True
    return norm in BAD_ENTITY_LABELS


def extract_alarm_number(text: str | None) -> str | None:
    """Extract alarm number from labels like `Alarm 1013`."""
    if not text:
        return None
    norm = normalize_text(text)
    match = re.search(r"\balarm\s+(\d{3,5})\b", norm)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{3,5})\b", norm)
    if match and "alarm" in norm:
        return match.group(1)
    return None


class XQualityGold:
    """Loaded XQuality ground truth and convenience lookup structures."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.documents = [EvalDocument(document_id="xquality", metadata={"dataset": "xquality"})]
        self.entities = self._build_entities()
        self.relations = self._build_relations()
        self.alarm_no_to_label = self._build_alarm_no_to_label()
        self.gt_alarm_labels = self._build_alarm_labels()
        self.alarm_rel_to_tails = self._build_alarm_rel_to_tails()
        self.triggers_by_alarm = self._build_triggers_by_alarm()

    def _build_entities(self) -> list[EvalEntity]:
        labels = set()
        for row in self.rows:
            node1 = str(row.get("Node 1", "")).strip()
            node2 = str(row.get("Node 2", "")).strip()
            if node1:
                labels.add(node1)
            if node2:
                labels.add(node2)
        return [EvalEntity(label=label) for label in sorted(labels)]

    def _build_relations(self) -> list[EvalRelation]:
        relations = []
        for row in self.rows:
            node1 = str(row.get("Node 1", "")).strip()
            relation = str(row.get("RELATION", "")).strip().upper()
            node2 = str(row.get("Node 2", "")).strip()
            if node1 and relation and node2:
                relations.append(EvalRelation(head=node1, relation=relation, tail=node2, raw=row))
        return relations

    def _build_alarm_no_to_label(self) -> dict[str, str]:
        mapping = {}
        for row in self.rows:
            node1 = str(row.get("Node 1", "")).strip()
            relation = str(row.get("RELATION", "")).strip().upper()
            node2 = str(row.get("Node 2", "")).strip()
            alarm_no = str(row.get("Alarm No.", "")).strip()
            if not alarm_no:
                continue
            if relation == "TRIGGERS" and node2:
                mapping[alarm_no] = node2
            elif relation in {"CAUSES", "REQUIRES", "HANDLED_BY", "REFERENCES"} and node1:
                mapping[alarm_no] = node1
        return mapping

    def _build_alarm_labels(self) -> list[str]:
        labels = set()
        for relation in self.relations:
            if relation.relation == "TRIGGERS":
                labels.add(relation.tail)
            elif relation.relation in {"CAUSES", "REQUIRES", "HANDLED_BY", "REFERENCES"}:
                labels.add(relation.head)
        return sorted(labels)

    def _build_alarm_rel_to_tails(self) -> dict[tuple[str, str], list[str]]:
        lookup = defaultdict(list)
        for relation in self.relations:
            if relation.relation in {"CAUSES", "REQUIRES", "HANDLED_BY", "REFERENCES"}:
                lookup[(relation.head, relation.relation)].append(relation.tail)
        return dict(lookup)

    def _build_triggers_by_alarm(self) -> dict[str, list[str]]:
        lookup = defaultdict(list)
        for relation in self.relations:
            if relation.relation == "TRIGGERS":
                lookup[relation.tail].append(relation.head)
        return dict(lookup)

    def canonicalize_alarm_label(self, label: str) -> str:
        """Replace `Alarm NNNN` with the canonical GT alarm label when possible."""
        alarm_no = extract_alarm_number(label)
        if alarm_no and alarm_no in self.alarm_no_to_label:
            return self.alarm_no_to_label[alarm_no]
        return label

    def candidate_alarm_labels_from_texts(self, texts: list[str], profile: EvaluationProfile) -> list[str]:
        """Return plausible GT alarm labels from free text."""
        out = []
        if profile.use_alias_maps:
            for text in texts:
                alias = apply_alarm_alias(text)
                if alias is not None:
                    out.append(alias)
        for alarm in self.gt_alarm_labels:
            scores = [loose_similarity(text, alarm) for text in texts if text]
            if scores and max(scores) >= profile.alarm_grounding_threshold:
                out.append(alarm)
        seen = set()
        return [x for x in out if not (x in seen or seen.add(x))]

    def canonicalize_neoolaf_relation(self, relation: EvalRelation, profile: EvaluationProfile) -> list[EvalRelation]:
        """GT-guided relaxed canonicalization for NeoOLAF JSON triples."""
        if not profile.gt_guided_canonicalization:
            return [relation]

        texts = [relation.head, relation.relation, relation.tail, relation.evidence or ""]
        inferred = infer_xquality_relation_type(texts)
        if inferred is None:
            return []

        alarm_candidates = self.candidate_alarm_labels_from_texts(texts, profile)
        results: list[EvalRelation] = []

        if inferred == "TRIGGERS":
            for alarm_label in alarm_candidates:
                for gt_cause in self.triggers_by_alarm.get(alarm_label, []):
                    scores = [loose_similarity(text, gt_cause) for text in texts if text]
                    overlaps = [token_overlap_score(text, gt_cause) for text in texts if text]
                    best = max(scores + overlaps) if scores or overlaps else 0.0
                    if best >= profile.trigger_cause_grounding_threshold:
                        results.append(
                            EvalRelation(
                                head=gt_cause,
                                relation="TRIGGERS",
                                tail=alarm_label,
                                evidence=relation.evidence,
                                confidence=relation.confidence,
                                provenance_present=relation.provenance_present,
                                provenance=relation.provenance,
                                raw=relation.raw,
                            )
                        )
            return dedup_relations(results)

        for alarm_label in alarm_candidates:
            for gt_tail in self.alarm_rel_to_tails.get((alarm_label, inferred), []):
                scores = [loose_similarity(text, gt_tail) for text in texts if text]
                overlaps = [token_overlap_score(text, gt_tail) for text in texts if text]
                best = max(scores + overlaps) if scores or overlaps else 0.0
                if best >= profile.tail_grounding_threshold:
                    results.append(
                        EvalRelation(
                            head=alarm_label,
                            relation=inferred,
                            tail=gt_tail,
                            evidence=relation.evidence,
                            confidence=relation.confidence,
                            provenance_present=relation.provenance_present,
                            provenance=relation.provenance,
                            raw=relation.raw,
                        )
                    )
        return dedup_relations(results)


def infer_xquality_relation_type(texts: list[str]) -> str | None:
    """Infer an XQuality relation label from subject/predicate/object/evidence text."""
    full = normalize_text(" ".join(text for text in texts if text))
    if any(k in full for k in ["operator/maintenance officer", "responsible for", "operator", "maintenance", "technician", "officer", "toolmaker", "tool setter", "adjuster", "programmer"]):
        return "HANDLED_BY"
    if any(k in full for k in ["page ", "input x", "diagram", "reference", "referencesdiagram", "reference diagram"]):
        return "REFERENCES"
    if any(k in full for k in ["immediate and controlled shutdown", "shutdown at the end of the cycle", "stop at end of cycle", "stop at end of block", "message display only", "program rewind", "opening of hardware authorization", "deactivation of hardware authorization", "cnc in emergency"]):
        return "CAUSES"
    if any(k in full for k in ["intervention", "check", "consult", "replace", "press", "move", "perform", "confirm", "release", "reset", "close the", "exit automatic mode", "set the", "make sure"]):
        return "REQUIRES"
    if any(k in full for k in ["has detected", "has had a problem", "has_had_problem", "failure", "problem", "trigger", "waiting for excessive feed", "pressure", "temperature", "coolant", "open", "not work", "not working", "causes", "alarm"]):
        return "TRIGGERS"
    return None


def dedup_relations(relations: list[EvalRelation]) -> list[EvalRelation]:
    """Deduplicate relations by normalized head/relation/tail."""
    out = []
    seen = set()
    for relation in relations:
        key = (normalize_text(relation.head), relation.relation, normalize_text(relation.tail))
        if key in seen:
            continue
        seen.add(key)
        out.append(relation)
    return out


def load_xquality_gold(path: str | Path) -> XQualityGold:
    """Load XQuality flat triplet JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected XQuality gold file to contain a JSON list: {path}")
    return XQualityGold(rows)

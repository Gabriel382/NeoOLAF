from __future__ import annotations

"""Automatic source-grounded, no-gold evaluation helpers.

This module is reference-free: it does not compare NeoOLAF triples to a
manually annotated gold file.  Instead, it checks whether each triple is
traceable to the source chunk/table and whether its relation is compatible with
field markers, ontology hints, and structured table records extracted during
preprocessing.

The implementation intentionally supports multilingual XQuality-like cases,
where the source evidence may be French while the normalized triples are in
English.  For general documents, it falls back to provenance, chunk text,
relation markers, token overlap, and ontology semantic roles.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neoolaf.core.pipeline_state import PipelineState


@dataclass
class SourceGroundingItem:
    """Per-triple source-grounding diagnosis."""

    triple_id: str
    predicate: str
    chunk_id: str
    has_provenance: bool
    relation_marker_supported: bool
    endpoint_supported: bool
    table_record_supported: bool
    structured_record_supported: bool
    source_grounded: bool
    support_score: float
    support_mode: str
    subject_label: str
    object_label: str
    evidence_preview: str
    notes: List[str] = field(default_factory=list)


@dataclass
class SourceGroundingReport:
    """Aggregate source-grounding metrics."""

    total_triples: int = 0
    triples_with_provenance: int = 0
    provenance_coverage: Optional[float] = None

    relation_marker_supported: int = 0
    relation_marker_support_rate: Optional[float] = None

    endpoint_supported: int = 0
    endpoint_support_rate: Optional[float] = None

    table_record_supported: int = 0
    table_record_support_rate: Optional[float] = None

    structured_record_supported: int = 0
    structured_record_support_rate: Optional[float] = None

    source_grounded: int = 0
    source_grounding_rate: Optional[float] = None

    average_support_score: Optional[float] = None
    unsupported_triple_ids: List[str] = field(default_factory=list)
    per_triple: List[SourceGroundingItem] = field(default_factory=list)


# Relation-specific markers.  The list intentionally mixes English and French
# because XQuality source chunks are often French while NeoOLAF labels are often
# English after normalization/translation.
RELATION_MARKERS: Dict[str, List[str]] = {
    "TRIGGERS": [
        "cause:",
        "cause ",
        "causes:",
        "trigger",
        "triggers",
        "déclenche",
        "declenche",
        "déclenchement",
        "declenchement",
        "signale",
        "signifie",
        "indique",
        "détecte",
        "detecte",
        "a détecté",
        "a detecte",
    ],
    "CAUSES": [
        "effet:",
        "effet.",
        "effect:",
        "produit",
        "entraîne",
        "entraine",
        "arrêt",
        "arret",
        "stop",
        "stopped",
        "alarm with",
        "alarme avec",
    ],
    "REQUIRES": [
        "intervention:",
        "intervention ",
        "required action",
        "action requise",
        "relâcher",
        "relacher",
        "appuyer",
        "press",
        "vérifier",
        "verifier",
        "check",
        "contrôler",
        "controler",
        "inspect",
        "remplacer",
        "replace",
        "corriger",
        "correct",
        "éteindre",
        "eteindre",
        "rallumer",
        "restart",
        "reset",
        "rétablir",
        "retablir",
        "consulter",
        "consult",
        "désactiver",
        "desactiver",
        "activer",
        "activate",
    ],
    "HANDLED_BY": [
        "chargé de",
        "charge de",
        "chargé de l’intervention",
        "charge de l'intervention",
        "charge de l intervention",
        "responsable",
        "responsible",
        "operator",
        "opérateur",
        "operateur",
        "maintenance",
        "entretien",
        "technician",
        "technicien",
        "programmeur",
        "programmer",
        "outilleur",
        "régleur",
        "regleur",
        "setter",
    ],
    "REFERENCES": [
        "page",
        "entrée",
        "entree",
        "input",
        "schéma",
        "schema",
        "diagram",
        "voir",
        "see",
        "manual",
        "manuel",
        "documentation",
        "x",
    ],
}

GENERIC_RECORD_MARKERS = [
    "alarme n",
    "alarme n°",
    "message n",
    "message n°",
    "texte:",
    "type:",
    "effet",
    "cause",
    "intervention",
    "chargé de",
    "charge de",
]

# Bilingual role/action aliases used only as a no-gold grounding aid.  They are
# not used to create new triples, only to decide whether a produced triple is
# plausible with respect to its source evidence.
ROLE_ALIASES: Dict[str, List[str]] = {
    "operator": ["operator", "opérateur", "operateur"],
    "maintenance technician": [
        "maintenance technician",
        "chargé de l'entretien",
        "charge de l'entretien",
        "chargé de l’entretien",
        "charge de l entretien",
        "entretien",
        "maintenance",
        "technician",
        "technicien",
    ],
    "programmer": ["programmer", "programmeur"],
    "tool setter": ["tool setter", "outilleur", "régleur", "regleur", "outilleur regleur", "outilleur-régleur"],
}

ACTION_ALIASES: Dict[str, List[str]] = {
    "press": ["press", "appuyer", "appuyez"],
    "release": ["release", "relâcher", "relacher"],
    "verify": ["verify", "check", "vérifier", "verifier", "contrôler", "controler"],
    "replace": ["replace", "remplacer"],
    "correct": ["correct", "corriger", "modifier"],
    "restart": ["restart", "rallumer", "redémarrer", "redemarrer", "éteindre", "eteindre"],
    "consult": ["consult", "consulter", "documentation", "manuel", "manual"],
    "disable": ["disable", "désactiver", "desactiver"],
    "activate": ["activate", "activer"],
}


def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _normalize(text: str) -> str:
    text = _strip_accents(str(text or "")).lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[_\-–—/;:,.()\[\]{}]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", _normalize(text)) if len(tok) >= 2}


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    norm_text = _normalize(text)
    return any(_normalize(marker) in norm_text for marker in markers)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    for key in ("text", "text_en", "text_fr", "label", "canonical_label"):
        value = _get_attr(obj, key, None)
        if value:
            return str(value)
    return str(obj)


def _metadata(triple: Any) -> dict:
    meta = _get_attr(triple, "metadata", {}) or {}
    return meta if isinstance(meta, dict) else {}


def _semantic_role_from_hints(hints: Iterable[str]) -> Optional[str]:
    for hint in hints or []:
        text = str(hint)
        if text.startswith("semantic_role:"):
            return text.split(":", 1)[1].strip()
    return None


def _chunk_id_of(obj: Any) -> str:
    return str(_get_attr(obj, "chunk_id", "") or "")


def _build_chunk_text_map(state: PipelineState) -> Dict[str, str]:
    """Return chunk_id -> full chunk text from all known document chunk stores."""
    doc = getattr(state, "document", None)
    if doc is None:
        return {}

    chunk_map: Dict[str, str] = {}
    for attr_name in (
        "chunks",
        "table_chunks",
        "page_chunks",
        "subsection_chunks",
        "content_blocks",
    ):
        for chunk in _get_attr(doc, attr_name, []) or []:
            cid = str(_get_attr(chunk, "chunk_id", None) or _get_attr(chunk, "block_id", None) or "")
            text = str(_get_attr(chunk, "text", "") or "")
            if cid and text and cid not in chunk_map:
                chunk_map[cid] = text
    return chunk_map


def _build_alarm_record_map(state: PipelineState) -> Dict[str, Any]:
    """Return chunk_id -> structured alarm/message record when available."""
    doc = getattr(state, "document", None)
    if doc is None:
        return {}
    records: Dict[str, Any] = {}
    for record in _get_attr(doc, "alarm_records", []) or []:
        cid = str(_get_attr(record, "chunk_id", "") or "")
        if cid:
            records[cid] = record
    return records


def _evidence_snippets(triple: Any, chunk_text_map: Optional[Dict[str, str]] = None) -> List[str]:
    snippets: List[str] = []
    for ev in _get_attr(triple, "provenance", []) or []:
        snippet = _get_attr(ev, "snippet", None)
        if snippet:
            snippets.append(str(snippet))

    # Add full chunk text when possible.  This avoids false negatives caused by
    # short provenance snippets being truncated before the responsible actor or
    # reference fields.
    if chunk_text_map:
        cid = _chunk_id_of(triple)
        full_text = chunk_text_map.get(cid)
        if full_text and full_text not in snippets:
            snippets.append(full_text)
    return snippets


def _label_similarity(label: str, text: str) -> float:
    label_norm = _normalize(label)
    text_norm = _normalize(text)
    if not label_norm or not text_norm:
        return 0.0
    if label_norm in text_norm or text_norm in label_norm:
        return 1.0
    label_tokens = _token_set(label_norm)
    text_tokens = _token_set(text_norm)
    if not label_tokens or not text_tokens:
        return 0.0
    overlap = label_tokens & text_tokens
    return len(overlap) / max(1, min(len(label_tokens), len(text_tokens)))


def _label_matches_text(label: str, text: str, threshold: float = 0.55) -> bool:
    return _label_similarity(label, text) >= threshold


def _label_matches_item(label: str, item: Any, threshold: float = 0.55) -> bool:
    texts: List[str] = []
    for key in ("text_en", "text_fr", "text", "label", "canonical_label"):
        value = _get_attr(item, key, None)
        if value:
            texts.append(str(value))
    return any(_label_matches_text(label, text, threshold=threshold) for text in texts)


def _label_matches_any_item(label: str, items: Iterable[Any], threshold: float = 0.55) -> bool:
    return any(_label_matches_item(label, item, threshold=threshold) for item in items or [])


def _label_matches_aliases(label: str, aliases: Iterable[str]) -> bool:
    label_norm = _normalize(label)
    alias_norms = [_normalize(alias) for alias in aliases]
    if any(alias and alias in label_norm for alias in alias_norms):
        return True
    return any(_label_matches_text(label_norm, alias, threshold=0.8) for alias in alias_norms)


def _role_alias_supported(label: str, source_text: str) -> bool:
    label_norm = _normalize(label)
    source_norm = _normalize(source_text)
    for canonical, aliases in ROLE_ALIASES.items():
        if _label_matches_aliases(label_norm, [canonical, *aliases]):
            return any(_normalize(alias) in source_norm for alias in aliases)
    return False


def _action_alias_supported(label: str, source_text: str) -> bool:
    label_norm = _normalize(label)
    source_norm = _normalize(source_text)
    for canonical, aliases in ACTION_ALIASES.items():
        if _label_matches_aliases(label_norm, [canonical, *aliases]):
            return any(_normalize(alias) in source_norm for alias in aliases)
    return False


def _endpoint_supported_by_text(label: str, snippets: List[str]) -> bool:
    """Approximate endpoint support using literal, token, and alias matching."""
    if not label or not snippets:
        return False

    joined = " ".join(snippets)
    if _label_matches_text(label, joined, threshold=0.62):
        return True

    # Token overlap support with a smaller threshold for long action sentences.
    label_tokens = _token_set(label)
    source_tokens = _token_set(joined)
    if label_tokens and source_tokens:
        overlap = label_tokens & source_tokens
        if len(overlap) >= max(2, min(4, len(label_tokens) // 2)):
            return True

    return _role_alias_supported(label, joined) or _action_alias_supported(label, joined)


def _endpoint_supported_by_role(triple: Any, snippets: List[str]) -> bool:
    """Use ontology hints/semantic roles when labels are translated."""
    meta = _metadata(triple)
    source_role = _semantic_role_from_hints(meta.get("source_ontology_hints", []))
    target_role = _semantic_role_from_hints(meta.get("target_ontology_hints", []))
    predicate = str(_get_attr(triple, "predicate_label", "")).upper()

    text = " ".join(snippets)
    marker_supported = _contains_any(text, RELATION_MARKERS.get(predicate, []))
    if not marker_supported:
        return False

    expected_roles = {
        "TRIGGERS": {"source": "cause", "target": {"alarm", "message"}},
        "CAUSES": {"source": {"alarm", "message"}, "target": "effect"},
        "REQUIRES": {"source": {"alarm", "message"}, "target": "intervention"},
        "HANDLED_BY": {"source": {"alarm", "message"}, "target": "responsible"},
        "REFERENCES": {"source": {"alarm", "message"}, "target": "reference"},
    }.get(predicate)

    if not expected_roles:
        return marker_supported

    def _role_matches(actual: Optional[str], expected: Any) -> bool:
        if actual is None:
            return False
        if isinstance(expected, set):
            return actual in expected
        return actual == expected

    return _role_matches(source_role, expected_roles["source"]) and _role_matches(
        target_role, expected_roles["target"]
    )


def _record_label_supported(label: str, record: Any) -> bool:
    labels = [
        _get_attr(record, "alarm_label_en", ""),
        _get_attr(record, "alarm_label_fr", ""),
        _get_attr(record, "message_label_en", ""),
        _get_attr(record, "message_label_fr", ""),
        _get_attr(record, "alarm_no", ""),
        _get_attr(record, "message_no", ""),
        _get_attr(record, "record_id", ""),
    ]
    return any(_label_matches_text(label, candidate, threshold=0.5) for candidate in labels if candidate)


def _record_items(record: Any, field_name: str) -> List[Any]:
    return list(_get_attr(record, field_name, []) or [])


def _structured_record_support(triple: Any, record: Any) -> Tuple[bool, str, List[str]]:
    """Check whether a triple is supported by the structured table record.

    This is still no-gold evaluation.  It uses the document preprocessing result
    produced from the source table, not the manually annotated gold triples.
    """
    if not record:
        return False, "none", ["no_structured_record_for_chunk"]

    predicate = str(_get_attr(triple, "predicate_label", "")).upper()
    subject = str(_get_attr(triple, "subject_label", ""))
    obj = str(_get_attr(triple, "object_label", ""))

    alarm_ok_subject = _record_label_supported(subject, record)
    alarm_ok_object = _record_label_supported(obj, record)

    if predicate == "TRIGGERS":
        cause_ok = _label_matches_any_item(subject, _record_items(record, "cause_items"), threshold=0.45)
        return cause_ok and alarm_ok_object, "structured_record:cause_to_alarm", [] if cause_ok and alarm_ok_object else ["record_cause_or_alarm_label_not_matched"]

    if predicate == "CAUSES":
        effect_ok = _label_matches_any_item(obj, _record_items(record, "effect_items"), threshold=0.45)
        return alarm_ok_subject and effect_ok, "structured_record:alarm_to_effect", [] if alarm_ok_subject and effect_ok else ["record_alarm_label_or_effect_not_matched"]

    if predicate == "REQUIRES":
        intervention_ok = _label_matches_any_item(obj, _record_items(record, "intervention_items"), threshold=0.45)
        return alarm_ok_subject and intervention_ok, "structured_record:alarm_to_intervention", [] if alarm_ok_subject and intervention_ok else ["record_alarm_label_or_intervention_not_matched"]

    if predicate == "HANDLED_BY":
        responsible_ok = _label_matches_any_item(obj, _record_items(record, "responsible_items"), threshold=0.45)
        return alarm_ok_subject and responsible_ok, "structured_record:alarm_to_responsible", [] if alarm_ok_subject and responsible_ok else ["record_alarm_label_or_responsible_not_matched"]

    if predicate == "REFERENCES":
        reference_ok = _label_matches_any_item(obj, _record_items(record, "reference_items"), threshold=0.4)
        return alarm_ok_subject and reference_ok, "structured_record:alarm_to_reference", [] if alarm_ok_subject and reference_ok else ["record_alarm_label_or_reference_not_matched"]

    return False, "none", ["predicate_not_supported_by_structured_record_checker"]


def compute_source_grounding(state: PipelineState) -> SourceGroundingReport:
    """Compute automatic, source-grounded, no-gold metrics for a state."""
    report = SourceGroundingReport()
    triples = list(getattr(state, "candidate_triples", []) or [])
    report.total_triples = len(triples)
    if not triples:
        return report

    chunk_text_map = _build_chunk_text_map(state)
    record_map = _build_alarm_record_map(state)
    support_scores: List[float] = []

    for triple in triples:
        triple_id = str(_get_attr(triple, "triple_id", ""))
        predicate = str(_get_attr(triple, "predicate_label", "")).upper()
        subject = str(_get_attr(triple, "subject_label", ""))
        obj = str(_get_attr(triple, "object_label", ""))
        chunk_id = _chunk_id_of(triple)
        snippets = _evidence_snippets(triple, chunk_text_map)
        evidence_text = " ".join(snippets)

        has_prov = bool(snippets or chunk_id)
        relation_ok = _contains_any(evidence_text, RELATION_MARKERS.get(predicate, []))
        table_ok = _contains_any(evidence_text, GENERIC_RECORD_MARKERS) or chunk_id.startswith("table_")

        record = record_map.get(chunk_id)
        structured_ok, structured_mode, structured_notes = _structured_record_support(triple, record)

        subject_text_ok = _endpoint_supported_by_text(subject, snippets)
        object_text_ok = _endpoint_supported_by_text(obj, snippets)
        role_ok = _endpoint_supported_by_role(triple, snippets)
        endpoint_ok = structured_ok or (subject_text_ok and object_text_ok) or role_ok

        # Weighted support.  Structured source-record agreement is the strongest
        # non-gold signal because it uses the table decomposition extracted from
        # the original document.
        score = 0.0
        if has_prov:
            score += 0.20
        if table_ok:
            score += 0.15
        if relation_ok:
            score += 0.20
        if endpoint_ok:
            score += 0.25
        if structured_ok:
            score += 0.20
        score = min(score, 1.0)

        grounded = has_prov and table_ok and relation_ok and endpoint_ok

        notes: List[str] = []
        if structured_ok:
            notes.append(structured_mode)
        elif role_ok and not (subject_text_ok and object_text_ok):
            notes.append("endpoint_supported_by_semantic_role_or_translation")
        notes.extend(structured_notes if not structured_ok else [])
        if not relation_ok:
            notes.append("missing_relation_marker_in_evidence")
        if not endpoint_ok:
            notes.append("endpoint_not_grounded_in_text_role_or_structured_record")

        support_mode = (
            structured_mode if structured_ok
            else "semantic_role" if role_ok
            else "literal_or_token_overlap" if (subject_text_ok and object_text_ok)
            else "unsupported"
        )

        item = SourceGroundingItem(
            triple_id=triple_id,
            predicate=predicate,
            chunk_id=chunk_id,
            has_provenance=has_prov,
            relation_marker_supported=relation_ok,
            endpoint_supported=endpoint_ok,
            table_record_supported=table_ok,
            structured_record_supported=structured_ok,
            source_grounded=grounded,
            support_score=round(score, 4),
            support_mode=support_mode,
            subject_label=subject,
            object_label=obj,
            evidence_preview=evidence_text[:320],
            notes=notes,
        )
        report.per_triple.append(item)
        support_scores.append(score)

        if has_prov:
            report.triples_with_provenance += 1
        if relation_ok:
            report.relation_marker_supported += 1
        if endpoint_ok:
            report.endpoint_supported += 1
        if table_ok:
            report.table_record_supported += 1
        if structured_ok:
            report.structured_record_supported += 1
        if grounded:
            report.source_grounded += 1
        else:
            report.unsupported_triple_ids.append(triple_id)

    total = report.total_triples
    report.provenance_coverage = report.triples_with_provenance / total
    report.relation_marker_support_rate = report.relation_marker_supported / total
    report.endpoint_support_rate = report.endpoint_supported / total
    report.table_record_support_rate = report.table_record_supported / total
    report.structured_record_support_rate = report.structured_record_supported / total
    report.source_grounding_rate = report.source_grounded / total
    report.average_support_score = sum(support_scores) / len(support_scores)
    return report


def source_grounding_to_dict(report: SourceGroundingReport) -> dict:
    """Serialize a SourceGroundingReport to a JSON-compatible dictionary."""
    return {
        "total_triples": report.total_triples,
        "provenance_coverage": report.provenance_coverage,
        "relation_marker_support_rate": report.relation_marker_support_rate,
        "endpoint_support_rate": report.endpoint_support_rate,
        "table_record_support_rate": report.table_record_support_rate,
        "structured_record_support_rate": report.structured_record_support_rate,
        "source_grounding_rate": report.source_grounding_rate,
        "average_support_score": report.average_support_score,
        "counts": {
            "triples_with_provenance": report.triples_with_provenance,
            "relation_marker_supported": report.relation_marker_supported,
            "endpoint_supported": report.endpoint_supported,
            "table_record_supported": report.table_record_supported,
            "structured_record_supported": report.structured_record_supported,
            "source_grounded": report.source_grounded,
        },
        "unsupported_triple_ids": report.unsupported_triple_ids,
        "per_triple": [item.__dict__ for item in report.per_triple],
    }

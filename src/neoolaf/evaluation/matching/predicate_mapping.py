"""Predicate mapping utilities."""

from __future__ import annotations

from neoolaf.evaluation.matching.normalization import normalize_text

XQUALITY_RELATIONS = {"TRIGGERS", "CAUSES", "REQUIRES", "HANDLED_BY", "REFERENCES"}

XQUALITY_PREDICATE_MAP = {
    "hascause": "TRIGGERS",
    "has cause": "TRIGGERS",
    "causes": "CAUSES",
    "cause": "CAUSES",
    "hasintervention": "REQUIRES",
    "has intervention": "REQUIRES",
    "requires": "REQUIRES",
    "hasresponsible": "HANDLED_BY",
    "has responsible": "HANDLED_BY",
    "responsiblefor": "HANDLED_BY",
    "responsible for": "HANDLED_BY",
    "handled by": "HANDLED_BY",
    "references": "REFERENCES",
    "reference": "REFERENCES",
    "hasreference": "REFERENCES",
    "has reference": "REFERENCES",
    "hasdiagram": "REFERENCES",
    "has diagram": "REFERENCES",
    "referencesdiagram": "REFERENCES",
    "references diagram": "REFERENCES",
}


def map_xquality_predicate(predicate: object) -> str | None:
    """Map a raw predicate label/local name to the XQuality relation set."""
    raw = str(predicate or "").strip()
    if raw.upper() in XQUALITY_RELATIONS:
        return raw.upper()
    return XQUALITY_PREDICATE_MAP.get(normalize_text(raw))


def should_invert_xquality_predicate(predicate: object) -> bool:
    """Return True when TaxoDrivenKG has Alarm -> Cause but GT has Cause -> Alarm."""
    return normalize_text(predicate) in {"hascause", "has cause"}

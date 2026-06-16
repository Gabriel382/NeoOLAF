"""Configuration schema for evaluation profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationProfile:
    """Metric thresholds and dataset-specific switches."""

    name: str
    entity_threshold: int = 85
    relation_head_threshold: int = 85
    relation_tail_threshold_default: int = 85
    relation_tail_threshold_requires: int = 75

    fair_str_head_threshold: int = 82
    fair_str_tail_threshold: int = 82
    fair_str_tail_threshold_requires: int = 72

    contradiction_low_sim: int = 35
    contradiction_same_relation_threshold: int = 75

    dvs_min_str: float = 0.50
    dvs_max_cr: float = 0.20
    dvs_min_oc: float = 0.70
    dvs_max_cv: float = 0.30

    use_alarm_number_anchoring: bool = False
    allow_relation_inversion: bool = False
    allow_causes_triggers_flexibility: bool = False
    reject_generic_entities: bool = False
    use_alias_maps: bool = False
    gt_guided_canonicalization: bool = False
    use_ontology_for_entity_eval: bool = False

    alarm_grounding_threshold: int = 65
    tail_grounding_threshold: int = 52
    trigger_cause_grounding_threshold: int = 52

    metadata: dict[str, Any] = field(default_factory=dict)

    def relation_tail_threshold(self, relation: str) -> int:
        """Return relation-specific matching threshold."""
        if str(relation).strip().upper() == "REQUIRES":
            return self.relation_tail_threshold_requires
        return self.relation_tail_threshold_default

    def fair_tail_threshold(self, relation: str) -> int:
        """Return relation-specific validation support threshold."""
        if str(relation).strip().upper() == "REQUIRES":
            return self.fair_str_tail_threshold_requires
        return self.fair_str_tail_threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

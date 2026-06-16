"""Registry of built-in evaluation profiles."""

from __future__ import annotations

from neoolaf.evaluation.schema.config import EvaluationProfile


def get_profile(name: str) -> EvaluationProfile:
    """Return a built-in profile by name."""
    normalized = name.strip().lower().replace("-", "_")

    if normalized == "general_relation_extraction":
        return EvaluationProfile(
            name="general_relation_extraction",
            entity_threshold=85,
            relation_head_threshold=85,
            relation_tail_threshold_default=85,
            relation_tail_threshold_requires=75,
            fair_str_head_threshold=82,
            fair_str_tail_threshold=82,
            fair_str_tail_threshold_requires=72,
            contradiction_low_sim=35,
            contradiction_same_relation_threshold=75,
            dvs_min_str=0.50,
            dvs_max_cr=0.20,
            dvs_min_oc=0.70,
            dvs_max_cv=0.30,
        )

    if normalized == "xquality_strict_extraction":
        return EvaluationProfile(
            name="xquality_strict_extraction",
            entity_threshold=92,
            relation_head_threshold=94,
            relation_tail_threshold_default=94,
            relation_tail_threshold_requires=92,
            use_alarm_number_anchoring=True,
            reject_generic_entities=True,
            allow_relation_inversion=False,
            allow_causes_triggers_flexibility=False,
            use_alias_maps=False,
            gt_guided_canonicalization=False,
        )

    if normalized == "xquality_loose":
        return EvaluationProfile(
            name="xquality_loose",
            entity_threshold=85,
            relation_head_threshold=80,
            relation_tail_threshold_default=80,
            relation_tail_threshold_requires=70,
            fair_str_head_threshold=80,
            fair_str_tail_threshold=80,
            fair_str_tail_threshold_requires=70,
            use_alarm_number_anchoring=True,
            reject_generic_entities=False,
            allow_relation_inversion=True,
            use_alias_maps=True,
            gt_guided_canonicalization=False,
        )

    if normalized == "xquality_relaxed_recall":
        return EvaluationProfile(
            name="xquality_relaxed_recall",
            entity_threshold=88,
            relation_head_threshold=75,
            relation_tail_threshold_default=72,
            relation_tail_threshold_requires=60,
            fair_str_head_threshold=75,
            fair_str_tail_threshold=72,
            fair_str_tail_threshold_requires=60,
            alarm_grounding_threshold=65,
            tail_grounding_threshold=52,
            trigger_cause_grounding_threshold=52,
            use_alarm_number_anchoring=True,
            use_alias_maps=True,
            gt_guided_canonicalization=True,
            use_ontology_for_entity_eval=True,
        )

    raise ValueError(
        f"Unknown evaluation profile: {name}. Available profiles: "
        "general_relation_extraction, xquality_strict_extraction, "
        "xquality_loose, xquality_relaxed_recall"
    )

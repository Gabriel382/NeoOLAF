from __future__ import annotations

from neoolaf.evaluation.runners.evaluate_domain_kg import (
    DomainEvaluationConfig,
    MatchingConfig,
    MethodOutputConfig,
    RelationInferenceRule,
    normalize_text,
)


def build_xquality_domain_config() -> DomainEvaluationConfig:
    return DomainEvaluationConfig(
        name="xquality",
        relation_schema=["TRIGGERS", "CAUSES", "REQUIRES", "HANDLED_BY", "REFERENCES"],
        gold_head_field="Node 1",
        gold_relation_field="RELATION",
        gold_tail_field="Node 2",
        anchor_field="Alarm No.",
        gold_type_field="Triplet Type",
        gold_category_field="Category",
        relation_roles={
            "TRIGGERS": "cause_to_anchor",
            "CAUSES": "anchor_to_effect",
            "REQUIRES": "anchor_to_intervention",
            "HANDLED_BY": "anchor_to_responsible",
            "REFERENCES": "anchor_to_reference",
        },
        entity_alias_map={
            normalize_text("ActiveEmergencyEvent"): "EMERGENCY ACTIVE",
            normalize_text("Open side guard alarm"): "SIDE GUARDS OPEN",
            normalize_text("OpenSideGuardAlarmEvent"): "SIDE GUARDS OPEN",
            normalize_text("OPEN FORCEPS"): "CHUCK OPEN",
            normalize_text("THERMOMAGNET SWITCHES. END OF CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
            normalize_text("THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
            normalize_text("THERMOMAGNET SWITCHES END OF CYCLE STOP"): "THERMOMAGNETIC SWITCHES END-OF-CYCLE STOP",
            normalize_text("ProgramRewindEvent"): "PROGRAM NOT REWOUND",
            normalize_text("Side guards open"): "SIDE GUARDS OPEN",
        },
        anchor_alias_map={
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
        },
        bad_entity_labels={
            "check", "alarm", "device", "failure", "machine", "part",
            "operator", "maintenance", "technician", "consult", "action",
            "integer", "string", "boolean", "decimal", "float", "double",
            "class", "objectproperty", "datatypeproperty", "thing", "property",
            "resource", "result", "process", "situation", "constraint",
            "observation", "feature", "platform", "staff", "manager",
            "geometry", "line", "sensor", "ontology", "product",
            "cause", "feature of interest", "observable property",
            "temporal entity", "spatial object", "human process",
            "logistic process", "manufacturing process", "manufacturing facility",
            "manufacturing cell", "cell", "instant", "interval",
            "clamp",
        },
        generic_gold_entity_blocklist={
            "check", "alarm", "device", "failure", "machine", "part",
            "operator", "maintenance technician", "consult", "action",
        },
        relation_inference_rules=[
            RelationInferenceRule(
                relation="HANDLED_BY",
                keywords=[
                    "operator/maintenance officer", "responsible for", "operator",
                    "maintenance", "technician", "officer", "toolmaker",
                    "tool setter", "adjuster", "programmer",
                ],
            ),
            RelationInferenceRule(
                relation="REFERENCES",
                keywords=[
                    "page ", "input x", "diagram", "reference",
                    "referencesdiagram", "reference diagram",
                ],
            ),
            RelationInferenceRule(
                relation="CAUSES",
                keywords=[
                    "immediate and controlled shutdown",
                    "immediate and controlled stop",
                    "shutdown at the end of the cycle",
                    "stop at end of cycle",
                    "stop at end of block",
                    "message display only",
                    "program rewind",
                    "opening of hardware authorization",
                    "deactivation of hardware authorization",
                    "cnc in emergency",
                ],
            ),
            RelationInferenceRule(
                relation="REQUIRES",
                keywords=[
                    "intervention", "check", "consult", "replace", "press",
                    "move", "perform", "confirm", "release", "reset",
                    "close the", "exit automatic mode", "set the", "make sure",
                ],
            ),
            RelationInferenceRule(
                relation="TRIGGERS",
                keywords=[
                    "has detected", "has had a problem", "has_had_problem",
                    "failure", "problem", "trigger", "waiting for excessive feed",
                    "pressure", "temperature", "coolant", "open", "not work",
                    "not working", "causes", "alarm",
                ],
            ),
        ],
        role_inference_rules={
            "Operator": ["operator"],
            "Maintenance Technician": ["maintenance", "technician", "officer"],
            "Programmer": ["programmer"],
            "Tool Setter": ["tool setter", "toolmaker", "adjuster"],
        },
        matching=MatchingConfig(
            entity_threshold=88,
            relation_head_threshold=75,
            relation_tail_threshold_default=72,
            relation_tail_threshold_by_relation={"REQUIRES": 60},

            alarm_grounding_threshold=65,
            tail_grounding_threshold=52,
            trigger_cause_grounding_threshold=52,

            fair_str_head_threshold=80,
            fair_str_tail_threshold_default=80,
            fair_str_tail_threshold_by_relation={"REQUIRES": 70},

            contradiction_low_sim=40,
            contradiction_same_relation_threshold=75,

            dvs_min_str=0.50,
            dvs_max_cr=0.20,
            dvs_min_oc=0.80,
            dvs_max_cv=1.00,
        ),
    )


def build_neoolaf_json_method_config() -> MethodOutputConfig:
    return MethodOutputConfig(
        name="neoolaf",
        format="json_triples",
        triples_key="triples",
        subject_path=["subject", "label"],
        predicate_path=["predicate", "label"],
        object_path=["object", "label"],
        evidence_path=["justification"],
        chunk_id_path=["chunk_id"],
        confidence_path=["confidence"],
        provenance_path=["provenance"],
        local_label="local",
        inferred_label="inferred",
    )
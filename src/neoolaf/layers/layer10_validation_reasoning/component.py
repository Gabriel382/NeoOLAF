from __future__ import annotations

# Standard library imports
from typing import Any, Dict, List, Tuple

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.candidate_triple import CandidateTriple
from neoolaf.domain.validation_reasoning import (
    ValidationIssue,
    ValidationReport,
    ReasoningReport,
)
from neoolaf.domain.axiom_schema import AxiomSchemaCandidate
from neoolaf.domain.general_axiom import GeneralAxiomCandidate


class ValidationReasoningLayer(BaseLayer):
    """
    Layer 10: validation / reasoning.

    Responsibilities:
    - validate the local ontology and local graph at document level
    - detect structural problems and compatibility issues
    - perform lightweight deterministic reasoning
    """

    name = "layer10_validation_reasoning"

    def __init__(
        self,
        max_triples: int | None = None,
        save_intermediate: bool = True,
        verbose: bool = False,
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
    ) -> None:
        """
        Initialize Layer 10.

        Args:
            max_triples:
                Optional debug limit on the number of triples considered.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.max_triples = max_triples
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 0.0)

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Validate and reason over the document-level local ontology and graph.
        """
        validation_report = self._validate_local_state(state)
        reasoning_report = self._reason_over_local_state(state)

        state.validation_report = validation_report
        state.reasoning_report = reasoning_report

        state.log(
            "[layer10_validation_reasoning] "
            f"valid={validation_report.is_valid}, "
            f"issues={len(validation_report.issues)}, "
            f"inferred_triples={len(reasoning_report.inferred_triples)}, "
            f"inferred_axioms={len(reasoning_report.inferred_general_axioms)}"
        )
        return state

    def _validate_local_state(self, state: PipelineState) -> ValidationReport:
        """
        Validate local graph and ontology-level artifacts.
        """
        issues: List[ValidationIssue] = []
        issue_counter = 0

        triples = state.candidate_triples
        if self.max_triples is not None:
            triples = triples[: self.max_triples]

        triple_iterator = triples
        if self.verbose:
            triple_iterator = tqdm(triples, desc="Layer 10 - validate triples", leave=False)

        # ---------------------------------------------------------
        # 1. Triple completeness and basic sanity
        # ---------------------------------------------------------
        seen_triple_keys = set()

        for triple in triple_iterator:
            # Missing required fields
            if not triple.subject_id or not triple.predicate_id or not triple.object_id:
                issues.append(
                    ValidationIssue(
                        issue_id=f"issue_{issue_counter:05d}",
                        issue_type="missing_triple_component",
                        severity="error",
                        message="Triple is missing subject, predicate, or object.",
                        related_ids=[triple.triple_id],
                    )
                )
                issue_counter += 1

            # Duplicate triple check
            triple_key = (
                triple.subject_id,
                triple.predicate_id,
                triple.object_id,
                triple.chunk_id,
            )
            if triple_key in seen_triple_keys:
                issues.append(
                    ValidationIssue(
                        issue_id=f"issue_{issue_counter:05d}",
                        issue_type="duplicate_triple",
                        severity="warning",
                        message="Duplicate candidate triple detected within the same chunk.",
                        related_ids=[triple.triple_id],
                    )
                )
                issue_counter += 1
            else:
                seen_triple_keys.add(triple_key)

        # ---------------------------------------------------------
        # 2. Validate concept hierarchy sanity
        # ---------------------------------------------------------
        for link in state.concept_hierarchy_links:
            if link.child_concept_id == link.parent_concept_id:
                issues.append(
                    ValidationIssue(
                        issue_id=f"issue_{issue_counter:05d}",
                        issue_type="self_subclass",
                        severity="error",
                        message="A concept hierarchy link cannot point to itself.",
                        related_ids=[link.link_id],
                    )
                )
                issue_counter += 1

        # ---------------------------------------------------------
        # 3. Validate relation hierarchy sanity
        # ---------------------------------------------------------
        for link in state.relation_hierarchy_links:
            if link.child_relation_id == link.parent_relation_id:
                issues.append(
                    ValidationIssue(
                        issue_id=f"issue_{issue_counter:05d}",
                        issue_type="self_subrelation",
                        severity="error",
                        message="A relation hierarchy link cannot point to itself.",
                        related_ids=[link.link_id],
                    )
                )
                issue_counter += 1

        # ---------------------------------------------------------
        # 4. Validate general axioms
        # ---------------------------------------------------------
        for axiom in state.general_axiom_candidates:
            if axiom.axiom_type == "description":
                if not axiom.literal_value or not str(axiom.literal_value).strip():
                    issues.append(
                        ValidationIssue(
                            issue_id=f"issue_{issue_counter:05d}",
                            issue_type="empty_description_axiom",
                            severity="warning",
                            message="Description axiom has an empty literal value.",
                            related_ids=[axiom.axiom_id],
                        )
                    )
                    issue_counter += 1

            if axiom.axiom_type in {"subclass", "relation_domain", "relation_range"}:
                if not axiom.object_label or not str(axiom.object_label).strip():
                    issues.append(
                        ValidationIssue(
                            issue_id=f"issue_{issue_counter:05d}",
                            issue_type="empty_structural_axiom_target",
                            severity="warning",
                            message="Structural axiom is missing a target label.",
                            related_ids=[axiom.axiom_id],
                        )
                    )
                    issue_counter += 1

        # ---------------------------------------------------------
        # 5. Validate relation domain/range compatibility against triples
        # ---------------------------------------------------------
        # Earlier versions compared only coarse node types such as "entity" or
        # "event" with ontology classes such as PLCAlarmOrMessage.  That created
        # false positives for XQuality because the useful information is stored in
        # ontology hints, concept_kind, parent_hint, and promoted concept links.
        relation_domain_map, relation_range_map = self._build_relation_schema_maps(state)
        concept_lookup = self._build_candidate_concept_lookup(state)

        compatibility_issues: List[ValidationIssue] = []
        for triple in triples:
            expected_domain = relation_domain_map.get(triple.predicate_label)
            expected_range = relation_range_map.get(triple.predicate_label)

            if expected_domain is not None and not self._ontology_node_compatible(
                triple=triple,
                side="subject",
                expected_schema_label=expected_domain,
                concept_lookup=concept_lookup,
            ):
                compatibility_issues.append(
                    ValidationIssue(
                        issue_id="",
                        issue_type="domain_mismatch",
                        severity="warning",
                        message=(
                            f"Triple subject appears incompatible with expected domain '{expected_domain}' "
                            f"for relation '{triple.predicate_label}'. "
                            "Compatibility was checked with ontology hints, concept kind, parent hint, and coarse type."
                        ),
                        related_ids=[triple.triple_id],
                    )
                )

            if expected_range is not None and not self._ontology_node_compatible(
                triple=triple,
                side="object",
                expected_schema_label=expected_range,
                concept_lookup=concept_lookup,
            ):
                compatibility_issues.append(
                    ValidationIssue(
                        issue_id="",
                        issue_type="range_mismatch",
                        severity="warning",
                        message=(
                            f"Triple object appears incompatible with expected range '{expected_range}' "
                            f"for relation '{triple.predicate_label}'. "
                            "Compatibility was checked with ontology hints, concept kind, parent hint, and coarse type."
                        ),
                        related_ids=[triple.triple_id],
                    )
                )

        for compatibility_issue in compatibility_issues:
            compatibility_issue.issue_id = f"issue_{issue_counter:05d}"
            issues.append(compatibility_issue)
            issue_counter += 1

        is_valid = not any(issue.severity == "error" for issue in issues)
        return ValidationReport(is_valid=is_valid, issues=issues)

    def _reason_over_local_state(self, state: PipelineState) -> ReasoningReport:
        """
        Perform lightweight deterministic reasoning.

        Current reasoning includes:
        - inferred duplicate-free graph projection
        - subclass-based inferred description propagation note
        - inferred axioms copied from validated schema structures when useful
        """
        inferred_triples: List[CandidateTriple] = []
        inferred_axiom_schemata: List[AxiomSchemaCandidate] = []
        inferred_general_axioms: List[GeneralAxiomCandidate] = []
        notes: List[str] = []

        triples = state.candidate_triples
        if self.max_triples is not None:
            triples = triples[: self.max_triples]

        # ---------------------------------------------------------
        # 1. Inferred graph = candidate graph projected into inferred graph
        #    with stable deduplication
        # ---------------------------------------------------------
        dedup = {}
        for triple in triples:
            key = (
                triple.subject_id,
                triple.predicate_id,
                triple.object_id,
                triple.chunk_id,
            )
            if key not in dedup:
                dedup[key] = triple

        inferred_triples = list(dedup.values())
        notes.append("Inferred graph initialized from validated candidate triples.")

        # ---------------------------------------------------------
        # 2. Carry forward validated schemata as inferred schema layer
        # ---------------------------------------------------------
        inferred_axiom_schemata = list(state.axiom_schema_candidates)
        if inferred_axiom_schemata:
            notes.append("Inferred axiom schemata copied from extracted reusable schemata.")

        # ---------------------------------------------------------
        # 3. Carry forward validated general axioms
        # ---------------------------------------------------------
        inferred_general_axioms = list(state.general_axiom_candidates)
        if inferred_general_axioms:
            notes.append("Inferred general axioms copied from candidate general axioms.")

        # ---------------------------------------------------------
        # 4. Lightweight subclass note
        # ---------------------------------------------------------
        if state.concept_hierarchy_links:
            notes.append("Concept hierarchy links are available for downstream ontology reasoning.")

        if state.relation_hierarchy_links:
            notes.append("Relation hierarchy links are available for downstream relation reasoning.")

        return ReasoningReport(
            inferred_triples=inferred_triples,
            inferred_axiom_schemata=inferred_axiom_schemata,
            inferred_general_axioms=inferred_general_axioms,
            notes=notes,
        )

    def _build_relation_schema_maps(self, state: PipelineState) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Build maps from relation label to expected domain/range label using Layer 8 schemata.
        """
        domain_map: Dict[str, str] = {}
        range_map: Dict[str, str] = {}

        for schema in state.axiom_schema_candidates:
            if schema.schema_type == "relation_domain":
                domain_map[schema.subject_label] = schema.object_label
            elif schema.schema_type == "relation_range":
                range_map[schema.subject_label] = schema.object_label

        return domain_map, range_map

    def _build_candidate_concept_lookup(self, state: PipelineState) -> Dict[str, dict[str, Any]]:
        """Build a lookup from Layer 3/5 candidate IDs to ontology-aware concept data."""
        lookup: Dict[str, dict[str, Any]] = {}

        for concept in state.concept_candidates:
            concept_payload = {
                "concept_id": getattr(concept, "concept_id", ""),
                "label": getattr(concept, "label", ""),
                "normalized_label": getattr(concept, "normalized_label", ""),
                "concept_kind": getattr(concept, "concept_kind", ""),
                "parent_hint": getattr(concept, "parent_hint", ""),
            }
            for candidate_id in getattr(concept, "source_candidate_ids", []) or []:
                lookup[candidate_id] = concept_payload

        # Add hierarchy parent labels when available.
        concept_id_to_parent = {
            link.child_concept_id: getattr(link, "parent_label", "")
            for link in state.concept_hierarchy_links
        }
        for concept in state.concept_candidates:
            parent_label = concept_id_to_parent.get(getattr(concept, "concept_id", ""))
            if not parent_label:
                continue
            for candidate_id in getattr(concept, "source_candidate_ids", []) or []:
                lookup.setdefault(candidate_id, {})["hierarchy_parent_label"] = parent_label

        return lookup

    def _ontology_node_compatible(
        self,
        *,
        triple: CandidateTriple,
        side: str,
        expected_schema_label: str,
        concept_lookup: Dict[str, dict[str, Any]],
    ) -> bool:
        """Return whether a triple node is compatible with an ontology class.

        The check intentionally uses ontology-aware signals before falling back to
        coarse node types.  This prevents false warnings such as entity vs
        PLCAlarmOrMessage when the node has semantic_role:alarm or parent_hint
        PLC Alarm.
        """
        expected = self._normalize_label(expected_schema_label)
        if not expected:
            return True

        node_id = triple.subject_id if side == "subject" else triple.object_id
        node_label = triple.subject_label if side == "subject" else triple.object_label
        node_type = triple.subject_type if side == "subject" else triple.object_type

        concept_payload = concept_lookup.get(node_id, {})
        metadata = getattr(triple, "metadata", {}) or {}
        hint_key = "source_ontology_hints" if side == "subject" else "target_ontology_hints"
        hints = metadata.get(hint_key, []) or []

        signals = [
            node_label,
            node_type,
            concept_payload.get("label", ""),
            concept_payload.get("normalized_label", ""),
            concept_payload.get("concept_kind", ""),
            concept_payload.get("parent_hint", ""),
            concept_payload.get("hierarchy_parent_label", ""),
            *[str(hint) for hint in hints],
        ]
        normalized_signals = {self._normalize_label(signal) for signal in signals if signal}

        # Direct/substring compatibility catches exact ontology hints and labels.
        for signal in normalized_signals:
            if not signal:
                continue
            if expected in signal or signal in expected:
                return True

        # Controlled class compatibility used by XQuality profiles, but generic
        # enough to rely on semantic roles and parent hints rather than document IDs.
        aliases = self._expected_class_aliases(expected)
        if normalized_signals.intersection(aliases):
            return True

        # Relation-specific fallback: when the document profile maps a relation to
        # a role, the metadata stores semantic_role:<role>.
        relation_role_compatibility = {
            "TRIGGERS": {"subject": {"alarmcause", "cause"}, "object": {"plcalarmormessage", "plcalarm", "plcmessage", "alarm", "message"}},
            "CAUSES": {"subject": {"plcalarmormessage", "plcalarm", "plcmessage", "alarm", "message"}, "object": {"machineeffect", "effect"}},
            "REQUIRES": {"subject": {"plcalarmormessage", "plcalarm", "plcmessage", "alarm", "message"}, "object": {"interventionaction", "intervention", "action"}},
            "HANDLED_BY": {"subject": {"plcalarmormessage", "plcalarm", "plcmessage", "alarm", "message"}, "object": {"responsibleactor", "responsible", "actor", "operator", "technician"}},
            "REFERENCES": {"subject": {"plcalarmormessage", "plcalarm", "plcmessage", "alarm", "message"}, "object": {"technicalreference", "reference"}},
        }
        expected_roles = relation_role_compatibility.get(triple.predicate_label, {}).get(side, set())
        if expected in expected_roles and normalized_signals.intersection(expected_roles):
            return True

        # Final fallback for non-profile/generic documents.
        return self._coarse_type_compatible(node_type, expected_schema_label)

    def _expected_class_aliases(self, expected: str) -> set[str]:
        """Return normalized aliases for common ontology class labels."""
        alias_map = {
            "plcalarmormessage": {"plcalarmormessage", "plcalarm", "plcmessage", "alarm", "message", "entityalarm", "entitymessage"},
            "plcalarm": {"plcalarm", "alarm", "entityalarm"},
            "plcmessage": {"plcmessage", "message", "entitymessage"},
            "alarmcause": {"alarmcause", "cause", "eventcause", "semanticrolecause"},
            "machineeffect": {"machineeffect", "effect", "eventeffect", "semanticroleeffect"},
            "interventionaction": {"interventionaction", "intervention", "action", "eventintervention", "semanticroleintervention"},
            "responsibleactor": {"responsibleactor", "responsible", "actor", "operator", "technician", "entityresponsible", "semanticroleresponsible"},
            "technicalreference": {"technicalreference", "reference", "entityreference", "semanticrolereference"},
        }
        return alias_map.get(expected, {expected})

    def _normalize_label(self, value: str | None) -> str:
        """Normalize a label/hint for permissive ontology compatibility checks."""
        import re

        text = str(value or "").lower()
        text = text.replace("semantic_role:", "semanticrole")
        text = text.replace("candidate_family:", "candidatefamily")
        text = text.replace("domain:", "domain")
        text = text.replace("range:", "range")
        text = text.replace("#", " ")
        return re.sub(r"[^a-z0-9]+", "", text)

    def _coarse_type_compatible(self, triple_node_type: str, expected_schema_label: str) -> bool:
        """Generic coarse fallback when no ontology-aware signals are available."""
        node_type = (triple_node_type or "").lower().strip()
        schema_label = (expected_schema_label or "").lower().strip()

        if not schema_label:
            return True

        if node_type == "entity":
            return any(token in schema_label for token in ["component", "resource", "machine", "object", "device", "entity", "actor", "reference", "alarm", "message"])
        if node_type == "event":
            return any(token in schema_label for token in ["event", "failure", "state", "process", "alarm", "cause", "effect", "intervention", "action"])
        if node_type == "attribute":
            return any(token in schema_label for token in ["value", "property", "attribute", "state", "measurement"])

        return True

    def _strategy(self, state: PipelineState) -> str:
        """Return the configured Layer 10 strategy, if any."""
        profile_config = getattr(state, "profile_config", None) or {}
        layers_cfg = profile_config.get("layers", {}) if isinstance(profile_config, dict) else {}
        layer_cfg = layers_cfg.get(self.name, {}) if isinstance(layers_cfg, dict) else {}
        return str(layer_cfg.get("strategy", "ontology_aware_validation_reasoning"))

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize Layer 10 outputs for debugging and reproducibility.
        """
        validation_payload = None
        if state.validation_report is not None:
            validation_payload = {
                "is_valid": state.validation_report.is_valid,
                "issues": [
                    {
                        "issue_id": issue.issue_id,
                        "issue_type": issue.issue_type,
                        "severity": issue.severity,
                        "message": issue.message,
                        "related_ids": issue.related_ids,
                    }
                    for issue in state.validation_report.issues
                ],
            }

        reasoning_payload = None
        if state.reasoning_report is not None:
            reasoning_payload = {
                "notes": state.reasoning_report.notes,
                "inferred_triples": [
                    {
                        "triple_id": triple.triple_id,
                        "subject_id": triple.subject_id,
                        "subject_label": triple.subject_label,
                        "subject_type": triple.subject_type,
                        "predicate_id": triple.predicate_id,
                        "predicate_label": triple.predicate_label,
                        "object_id": triple.object_id,
                        "object_label": triple.object_label,
                        "object_type": triple.object_type,
                        "chunk_id": triple.chunk_id,
                        "justification": triple.justification,
                        "confidence": triple.confidence,
                    }
                    for triple in state.reasoning_report.inferred_triples
                ],
                "inferred_axiom_schemata": [
                    {
                        "schema_id": schema.schema_id,
                        "schema_type": schema.schema_type,
                        "subject_id": schema.subject_id,
                        "subject_label": schema.subject_label,
                        "predicate": schema.predicate,
                        "object_id": schema.object_id,
                        "object_label": schema.object_label,
                        "confidence": schema.confidence,
                    }
                    for schema in state.reasoning_report.inferred_axiom_schemata
                ],
                "inferred_general_axioms": [
                    {
                        "axiom_id": axiom.axiom_id,
                        "axiom_type": axiom.axiom_type,
                        "subject_id": axiom.subject_id,
                        "subject_label": axiom.subject_label,
                        "predicate": axiom.predicate,
                        "object_id": axiom.object_id,
                        "object_label": axiom.object_label,
                        "literal_value": axiom.literal_value,
                        "confidence": axiom.confidence,
                    }
                    for axiom in state.reasoning_report.inferred_general_axioms
                ],
            }

        return {
            "layer": self.name,
            "strategy": self._strategy(state),
            "validation_report": validation_payload,
            "reasoning_report": reasoning_payload,
        }
from __future__ import annotations

# Standard library imports
from typing import Dict, List, Tuple

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
        relation_domain_map, relation_range_map = self._build_relation_schema_maps(state)

        for triple in triples:
            expected_domain = relation_domain_map.get(triple.predicate_label)
            expected_range = relation_range_map.get(triple.predicate_label)

            # We only check coarse type compatibility if hints exist.
            if expected_domain is not None:
                if not self._coarse_type_compatible(triple.subject_type, expected_domain):
                    issues.append(
                        ValidationIssue(
                            issue_id=f"issue_{issue_counter:05d}",
                            issue_type="domain_mismatch",
                            severity="warning",
                            message=(
                                f"Triple subject type '{triple.subject_type}' may be incompatible "
                                f"with expected domain '{expected_domain}' for relation '{triple.predicate_label}'."
                            ),
                            related_ids=[triple.triple_id],
                        )
                    )
                    issue_counter += 1

            if expected_range is not None:
                if not self._coarse_type_compatible(triple.object_type, expected_range):
                    issues.append(
                        ValidationIssue(
                            issue_id=f"issue_{issue_counter:05d}",
                            issue_type="range_mismatch",
                            severity="warning",
                            message=(
                                f"Triple object type '{triple.object_type}' may be incompatible "
                                f"with expected range '{expected_range}' for relation '{triple.predicate_label}'."
                            ),
                            related_ids=[triple.triple_id],
                        )
                    )
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

    def _coarse_type_compatible(self, triple_node_type: str, expected_schema_label: str) -> bool:
        """
        Lightweight type compatibility check.

        This is intentionally coarse for the first version:
        - entity is compatible with component/resource/object-like labels
        - event is compatible with event/failure/state-like labels
        - attribute is compatible with value/property/state-like labels
        """
        node_type = (triple_node_type or "").lower().strip()
        schema_label = (expected_schema_label or "").lower().strip()

        if not schema_label:
            return True

        if node_type == "entity":
            return any(token in schema_label for token in ["component", "resource", "machine", "object", "device", "entity"])
        if node_type == "event":
            return any(token in schema_label for token in ["event", "failure", "state", "process", "alarm"])
        if node_type == "attribute":
            return any(token in schema_label for token in ["value", "property", "attribute", "state", "measurement"])

        # Fallback: accept unknown coarse combinations
        return True

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
            "validation_report": validation_payload,
            "reasoning_report": reasoning_payload,
        }
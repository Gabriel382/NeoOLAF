from __future__ import annotations

# Standard library imports
from typing import Dict, List, Set, Tuple

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.completion import CompletionCandidate
from neoolaf.domain.candidate_triple import CandidateTriple
from neoolaf.domain.general_axiom import GeneralAxiomCandidate


class InferenceCompletionLayer(BaseLayer):
    """
    Layer 11: inference / completion.

    Responsibilities:
    - complete missing graph information after validation/reasoning
    - complete missing ontology information after validation/reasoning
    - keep completions explicit and traceable
    """

    name = "layer11_inference_completion"

    def __init__(
        self,
        max_inferred_triples: int | None = None,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize Layer 11.

        Args:
            max_inferred_triples:
                Optional debug limit on inferred triples considered for completion.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.max_inferred_triples = max_inferred_triples

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run completion after validation/reasoning.
        """
        completions: List[CompletionCandidate] = []
        completion_counter = 0

        # ---------------------------------------------------------
        # 1. Graph completion from inferred triples
        # ---------------------------------------------------------
        inferred_triples = []
        if state.reasoning_report is not None:
            inferred_triples = list(state.reasoning_report.inferred_triples)

        if self.max_inferred_triples is not None:
            inferred_triples = inferred_triples[: self.max_inferred_triples]

        existing_graph_keys = self._build_existing_graph_keys(state)

        triple_iterator = inferred_triples
        if self.verbose:
            triple_iterator = tqdm(
                inferred_triples,
                desc="Layer 11 - graph completion",
                leave=False,
            )

        for triple in triple_iterator:
            key = (
                triple.subject_id,
                triple.predicate_id,
                triple.object_id,
                triple.chunk_id,
            )

            # Only add if not already present in the local graph
            if key in existing_graph_keys:
                continue

            completed_triple = CandidateTriple(
                triple_id=f"completed_triple_{completion_counter:05d}",
                subject_id=triple.subject_id,
                subject_label=triple.subject_label,
                subject_type=triple.subject_type,
                predicate_id=triple.predicate_id,
                predicate_label=triple.predicate_label,
                object_id=triple.object_id,
                object_label=triple.object_label,
                object_type=triple.object_type,
                chunk_id=triple.chunk_id,
                justification=(
                    "Completed from inferred graph after validation/reasoning: "
                    + (triple.justification or "no additional justification")
                ),
                confidence=triple.confidence,
                provenance=triple.provenance,
            )

            completions.append(
                CompletionCandidate(
                    completion_id=f"completion_{completion_counter:05d}",
                    completion_type="graph_completion",
                    justification="Added missing graph triple from validated inferred graph.",
                    confidence=triple.confidence,
                    completed_triple=completed_triple,
                    completed_axiom=None,
                    evidence=triple.provenance,
                )
            )
            completion_counter += 1

        # ---------------------------------------------------------
        # 2. Ontology completion from inferred / general axioms
        # ---------------------------------------------------------
        inferred_axioms = []
        if state.reasoning_report is not None:
            inferred_axioms = list(state.reasoning_report.inferred_general_axioms)

        existing_axiom_keys = self._build_existing_axiom_keys(state)

        axiom_iterator = inferred_axioms
        if self.verbose:
            axiom_iterator = tqdm(
                inferred_axioms,
                desc="Layer 11 - ontology completion",
                leave=False,
            )

        for axiom in axiom_iterator:
            key = (
                axiom.axiom_type,
                axiom.subject_id,
                axiom.predicate,
                axiom.object_id,
                axiom.object_label,
                axiom.literal_value,
            )

            # Only add if not already present in the local ontology axiom set
            if key in existing_axiom_keys:
                continue

            completed_axiom = GeneralAxiomCandidate(
                axiom_id=f"completed_axiom_{completion_counter:05d}",
                axiom_type=axiom.axiom_type,
                subject_id=axiom.subject_id,
                subject_label=axiom.subject_label,
                predicate=axiom.predicate,
                object_id=axiom.object_id,
                object_label=axiom.object_label,
                literal_value=axiom.literal_value,
                justification=(
                    "Completed from inferred ontology after validation/reasoning: "
                    + (axiom.justification or "no additional justification")
                ),
                confidence=axiom.confidence,
                source_schema_ids=axiom.source_schema_ids,
                source_concept_ids=axiom.source_concept_ids,
                source_relation_ids=axiom.source_relation_ids,
                evidence=axiom.evidence,
            )

            completions.append(
                CompletionCandidate(
                    completion_id=f"completion_{completion_counter:05d}",
                    completion_type="ontology_completion",
                    justification="Added missing ontology/general axiom from validated inferred ontology.",
                    confidence=axiom.confidence,
                    completed_triple=None,
                    completed_axiom=completed_axiom,
                    evidence=axiom.evidence,
                )
            )
            completion_counter += 1

        # ---------------------------------------------------------
        # 3. Type-link completion from concept labels
        # ---------------------------------------------------------
        type_link_completions = self._build_type_link_completions(state, start_index=completion_counter)
        completions.extend(type_link_completions)

        state.completion_candidates = completions
        state.log(
            f"[layer11_inference_completion] produced "
            f"{len(state.completion_candidates)} completion candidates"
        )
        return state

    def _build_existing_graph_keys(self, state: PipelineState) -> Set[Tuple[str, str, str, str]]:
        """
        Build a set of already existing local graph triple keys.
        """
        keys: Set[Tuple[str, str, str, str]] = set()
        for triple in state.candidate_triples:
            keys.add(
                (
                    triple.subject_id,
                    triple.predicate_id,
                    triple.object_id,
                    triple.chunk_id,
                )
            )
        return keys

    def _build_existing_axiom_keys(self, state: PipelineState) -> Set[Tuple]:
        """
        Build a set of already existing local ontology/general axiom keys.
        """
        keys: Set[Tuple] = set()
        for axiom in state.general_axiom_candidates:
            keys.add(
                (
                    axiom.axiom_type,
                    axiom.subject_id,
                    axiom.predicate,
                    axiom.object_id,
                    axiom.object_label,
                    axiom.literal_value,
                )
            )
        return keys

    def _build_type_link_completions(
        self,
        state: PipelineState,
        start_index: int = 0,
    ) -> List[CompletionCandidate]:
        """
        Build lightweight type-link completions.

        Current strategy:
        if a concept candidate label matches a candidate label already present
        in the graph/candidate layer, add a descriptive completion axiom if missing.
        """
        completions: List[CompletionCandidate] = []
        counter = start_index

        existing_axiom_keys = self._build_existing_axiom_keys(state)

        # Build a concept lookup by normalized label
        concept_by_label = {}
        for concept in state.concept_candidates:
            concept_by_label[concept.label.lower().strip()] = concept

        # Check entity and event candidates against concept labels
        candidate_pools = list(state.entity_candidates) + list(state.event_candidates)

        for candidate in candidate_pools:
            normalized = candidate.canonical_label.lower().strip()
            concept = concept_by_label.get(normalized)

            if concept is None:
                continue

            literal_value = (
                f"Candidate '{candidate.canonical_label}' is aligned with concept "
                f"'{concept.label}'."
            )

            key = (
                "description",
                candidate.candidate_id,
                "rdfs:description",
                None,
                None,
                literal_value,
            )

            if key in existing_axiom_keys:
                continue

            completed_axiom = GeneralAxiomCandidate(
                axiom_id=f"completed_axiom_{counter:05d}",
                axiom_type="description",
                subject_id=candidate.candidate_id,
                subject_label=candidate.canonical_label,
                predicate="rdfs:description",
                object_id=None,
                object_label=None,
                literal_value=literal_value,
                justification="Completed missing type-link description from candidate/concept alignment.",
                confidence=0.8,
                source_schema_ids=[],
                source_concept_ids=[concept.concept_id],
                source_relation_ids=[],
                evidence=self._collect_candidate_evidence(candidate),
            )

            completions.append(
                CompletionCandidate(
                    completion_id=f"completion_{counter:05d}",
                    completion_type="type_link_completion",
                    justification="Added missing type-link style description from concept alignment.",
                    confidence=0.8,
                    completed_triple=None,
                    completed_axiom=completed_axiom,
                    evidence=self._collect_candidate_evidence(candidate),
                )
            )
            counter += 1

        return completions

    def _collect_candidate_evidence(self, candidate) -> list:
        """
        Collect evidence from all mentions of a candidate.
        """
        evidences = []
        for mention in candidate.mentions:
            evidences.extend(mention.evidence)
        return evidences

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize Layer 11 outputs for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "num_completion_candidates": len(state.completion_candidates),
            "completion_candidates": [
                {
                    "completion_id": item.completion_id,
                    "completion_type": item.completion_type,
                    "justification": item.justification,
                    "confidence": item.confidence,
                    "completed_triple": (
                        {
                            "triple_id": item.completed_triple.triple_id,
                            "subject_id": item.completed_triple.subject_id,
                            "subject_label": item.completed_triple.subject_label,
                            "subject_type": item.completed_triple.subject_type,
                            "predicate_id": item.completed_triple.predicate_id,
                            "predicate_label": item.completed_triple.predicate_label,
                            "object_id": item.completed_triple.object_id,
                            "object_label": item.completed_triple.object_label,
                            "object_type": item.completed_triple.object_type,
                            "chunk_id": item.completed_triple.chunk_id,
                            "confidence": item.completed_triple.confidence,
                        }
                        if item.completed_triple is not None else None
                    ),
                    "completed_axiom": (
                        {
                            "axiom_id": item.completed_axiom.axiom_id,
                            "axiom_type": item.completed_axiom.axiom_type,
                            "subject_id": item.completed_axiom.subject_id,
                            "subject_label": item.completed_axiom.subject_label,
                            "predicate": item.completed_axiom.predicate,
                            "object_id": item.completed_axiom.object_id,
                            "object_label": item.completed_axiom.object_label,
                            "literal_value": item.completed_axiom.literal_value,
                            "confidence": item.completed_axiom.confidence,
                        }
                        if item.completed_axiom is not None else None
                    ),
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in item.evidence
                    ],
                }
                for item in state.completion_candidates
            ],
        }
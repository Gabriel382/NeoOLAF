from __future__ import annotations

# Standard library imports
from typing import Any, Dict, List, Set, Tuple

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
    - materialize explicit candidate-to-concept links
    - enrich concepts with lightweight SKOS/RDF-style axioms
    - keep completions explicit and traceable
    """

    name = "layer11_inference_completion"

    def __init__(
        self,
        max_inferred_triples: int | None = None,
        save_intermediate: bool = True,
        verbose: bool = False,
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
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
            max_concurrency:
                Reserved for parallel completion strategies. The current
                ontology-aware strategy is deterministic and does not call LLMs.
            retry_failed_calls:
                Reserved for future LLM-based completion variants.
            retry_sleep_seconds:
                Reserved for future LLM-based completion variants.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.max_inferred_triples = max_inferred_triples
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 2.0)

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run completion after validation/reasoning.
        """
        strategy = self._get_strategy(state)
        if self.verbose:
            print(f"[NeoOLAF][Layer 11] strategy={strategy}")
            print(
                "[NeoOLAF][Layer 11] deterministic completion; "
                f"max_concurrency={self.max_concurrency}; no LLM calls."
            )

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
        if self.verbose and strategy != "ontology_aware_semantic_completion":
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

            # Only add if not already present in the local graph.
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
        if self.verbose and strategy != "ontology_aware_semantic_completion":
            axiom_iterator = tqdm(
                inferred_axioms,
                desc="Layer 11 - ontology completion",
                leave=False,
            )

        for axiom in axiom_iterator:
            key = self._axiom_key(
                axiom.axiom_type,
                axiom.subject_id,
                axiom.predicate,
                axiom.object_id,
                axiom.object_label,
                axiom.literal_value,
            )

            # Only add if not already present in the local ontology axiom set.
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
            existing_axiom_keys.add(key)
            completion_counter += 1

        # ---------------------------------------------------------
        # 3. Ontology-aware semantic completions
        # ---------------------------------------------------------
        semantic_completions = self._build_semantic_completions(
            state,
            start_index=completion_counter,
            existing_axiom_keys=existing_axiom_keys,
        )
        completions.extend(semantic_completions)

        state.completion_candidates = completions
        state.log(
            f"[layer11_inference_completion] produced "
            f"{len(state.completion_candidates)} completion candidates"
        )
        return state

    def _get_strategy(self, state: PipelineState) -> str:
        """
        Read the Layer 11 strategy from the active profile.
        """
        profile = getattr(state, "profile_config", None)
        if not isinstance(profile, dict):
            return "ontology_aware_semantic_completion"
        return (
            profile.get("layers", {})
            .get(self.name, {})
            .get("strategy", "ontology_aware_semantic_completion")
        )

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
                self._axiom_key(
                    axiom.axiom_type,
                    axiom.subject_id,
                    axiom.predicate,
                    axiom.object_id,
                    axiom.object_label,
                    axiom.literal_value,
                )
            )
        return keys

    def _build_semantic_completions(
        self,
        state: PipelineState,
        start_index: int = 0,
        existing_axiom_keys: Set[Tuple] | None = None,
    ) -> List[CompletionCandidate]:
        """
        Build ontology-aware semantic completions.

        Compared with the previous description-only completion, this strategy
        generates explicit, downstream-useful axioms:
        - rdf:type links from candidate nodes to induced concepts
        - owl:sameAs-style links from candidate nodes to induced concepts
        - skos:prefLabel labels for concepts
        - skos:altLabel labels from aliases, synonyms, and lexical variants
        - rdfs:comment descriptions when available

        The output remains deterministic and traceable to Layer 3/6 evidence.
        """
        completions: List[CompletionCandidate] = []
        counter = start_index
        existing = existing_axiom_keys if existing_axiom_keys is not None else self._build_existing_axiom_keys(state)

        candidates_by_id = self._build_candidate_lookup(state)
        concepts_by_source_candidate = self._build_concepts_by_source_candidate(state)

        for candidate_id, concept_list in concepts_by_source_candidate.items():
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None:
                continue

            for concept in concept_list:
                evidence = self._merge_evidence(
                    self._collect_candidate_evidence(candidate),
                    list(getattr(concept, "evidence", []) or []),
                )

                # Candidate -> concept typing link.
                counter = self._append_axiom_completion(
                    completions=completions,
                    counter=counter,
                    existing_axiom_keys=existing,
                    completion_type="rdf_type_completion",
                    axiom_type="type_link",
                    subject_id=candidate.candidate_id,
                    subject_label=candidate.canonical_label,
                    predicate="rdf:type",
                    object_id=concept.concept_id,
                    object_label=concept.label,
                    literal_value=None,
                    justification=(
                        "Added explicit rdf:type link between a typed candidate "
                        "and its promoted ontology concept."
                    ),
                    confidence=self._combined_confidence(candidate, concept, default=0.95),
                    source_concept_ids=[concept.concept_id],
                    evidence=evidence,
                )

                # Candidate -> concept identity / alignment link.
                counter = self._append_axiom_completion(
                    completions=completions,
                    counter=counter,
                    existing_axiom_keys=existing,
                    completion_type="candidate_concept_alignment_completion",
                    axiom_type="same_as",
                    subject_id=candidate.candidate_id,
                    subject_label=candidate.canonical_label,
                    predicate="owl:sameAs",
                    object_id=concept.concept_id,
                    object_label=concept.label,
                    literal_value=None,
                    justification=(
                        "Added explicit candidate-to-concept alignment. This keeps "
                        "the extracted node traceable while allowing ontology update."
                    ),
                    confidence=self._combined_confidence(candidate, concept, default=0.9),
                    source_concept_ids=[concept.concept_id],
                    evidence=evidence,
                )

                # Concept preferred label.
                counter = self._append_axiom_completion(
                    completions=completions,
                    counter=counter,
                    existing_axiom_keys=existing,
                    completion_type="skos_pref_label_completion",
                    axiom_type="pref_label",
                    subject_id=concept.concept_id,
                    subject_label=concept.label,
                    predicate="skos:prefLabel",
                    object_id=None,
                    object_label=None,
                    literal_value=concept.label,
                    justification="Added explicit preferred label for the promoted ontology concept.",
                    confidence=concept.confidence or 0.9,
                    source_concept_ids=[concept.concept_id],
                    evidence=evidence,
                )

                # Concept description/comment.
                description = getattr(concept, "description", None)
                if description:
                    counter = self._append_axiom_completion(
                        completions=completions,
                        counter=counter,
                        existing_axiom_keys=existing,
                        completion_type="rdfs_comment_completion",
                        axiom_type="comment",
                        subject_id=concept.concept_id,
                        subject_label=concept.label,
                        predicate="rdfs:comment",
                        object_id=None,
                        object_label=None,
                        literal_value=description,
                        justification="Added explicit textual comment for the promoted ontology concept.",
                        confidence=concept.confidence or 0.85,
                        source_concept_ids=[concept.concept_id],
                        evidence=evidence,
                    )

                # Alternative labels from candidate lexical information.
                alt_labels = self._candidate_alt_labels(candidate, concept.label)
                for alt_label in alt_labels:
                    counter = self._append_axiom_completion(
                        completions=completions,
                        counter=counter,
                        existing_axiom_keys=existing,
                        completion_type="skos_alt_label_completion",
                        axiom_type="alt_label",
                        subject_id=concept.concept_id,
                        subject_label=concept.label,
                        predicate="skos:altLabel",
                        object_id=None,
                        object_label=None,
                        literal_value=alt_label,
                        justification="Added alternative label from candidate aliases/synonyms/lexical variants.",
                        confidence=0.85,
                        source_concept_ids=[concept.concept_id],
                        evidence=evidence,
                    )

        return completions

    def _append_axiom_completion(
        self,
        *,
        completions: List[CompletionCandidate],
        counter: int,
        existing_axiom_keys: Set[Tuple],
        completion_type: str,
        axiom_type: str,
        subject_id: str,
        subject_label: str,
        predicate: str,
        object_id: str | None,
        object_label: str | None,
        literal_value: str | None,
        justification: str,
        confidence: float | None,
        source_concept_ids: List[str],
        evidence: list,
    ) -> int:
        """
        Append a completed axiom only if it is not already present.
        """
        key = self._axiom_key(
            axiom_type,
            subject_id,
            predicate,
            object_id,
            object_label,
            literal_value,
        )
        if key in existing_axiom_keys:
            return counter

        completed_axiom = GeneralAxiomCandidate(
            axiom_id=f"completed_axiom_{counter:05d}",
            axiom_type=axiom_type,
            subject_id=subject_id,
            subject_label=subject_label,
            predicate=predicate,
            object_id=object_id,
            object_label=object_label,
            literal_value=literal_value,
            justification=justification,
            confidence=confidence,
            source_schema_ids=[],
            source_concept_ids=source_concept_ids,
            source_relation_ids=[],
            evidence=evidence,
        )

        completions.append(
            CompletionCandidate(
                completion_id=f"completion_{counter:05d}",
                completion_type=completion_type,
                justification=justification,
                confidence=confidence,
                completed_triple=None,
                completed_axiom=completed_axiom,
                evidence=evidence,
            )
        )
        existing_axiom_keys.add(key)
        return counter + 1

    def _build_candidate_lookup(self, state: PipelineState) -> Dict[str, Any]:
        """
        Build a lookup for entity/event/attribute candidates by ID.
        """
        lookup: Dict[str, Any] = {}
        for candidate in (
            list(state.entity_candidates)
            + list(state.event_candidates)
            + list(state.attribute_candidates)
        ):
            lookup[candidate.candidate_id] = candidate
        return lookup

    def _build_concepts_by_source_candidate(self, state: PipelineState) -> Dict[str, list]:
        """
        Group concept candidates by their source candidate IDs.
        """
        grouped: Dict[str, list] = {}
        for concept in state.concept_candidates:
            for candidate_id in getattr(concept, "source_candidate_ids", []) or []:
                grouped.setdefault(candidate_id, []).append(concept)
        return grouped

    def _candidate_alt_labels(self, candidate: Any, preferred_label: str) -> List[str]:
        """
        Collect stable alternative labels from aliases, synonyms, variants, and mentions.
        """
        raw_labels: List[str] = []
        raw_labels.extend(getattr(candidate, "aliases", []) or [])
        raw_labels.extend(getattr(candidate, "synonyms", []) or [])
        raw_labels.extend(getattr(candidate, "lexical_variants", []) or [])
        for mention in getattr(candidate, "mentions", []) or []:
            if getattr(mention, "text", None):
                raw_labels.append(mention.text)

        preferred_norm = self._norm_label(preferred_label)
        seen: Set[str] = set()
        labels: List[str] = []
        for label in raw_labels:
            label = str(label).strip()
            norm = self._norm_label(label)
            if not label or norm == preferred_norm or norm in seen:
                continue
            seen.add(norm)
            labels.append(label)
        return labels

    def _collect_candidate_evidence(self, candidate: Any) -> list:
        """
        Collect evidence from all mentions of a candidate.
        """
        evidences = []
        for mention in getattr(candidate, "mentions", []) or []:
            evidences.extend(getattr(mention, "evidence", []) or [])
        return evidences

    def _merge_evidence(self, *evidence_lists: list) -> list:
        """
        Merge evidence lists while preserving order and avoiding duplicates.
        """
        merged = []
        seen = set()
        for evidence_list in evidence_lists:
            for ev in evidence_list:
                key = (
                    getattr(ev, "chunk_id", None),
                    getattr(ev, "chunk_start_char", None),
                    getattr(ev, "chunk_end_char", None),
                    getattr(ev, "snippet", None),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(ev)
        return merged

    def _combined_confidence(self, candidate: Any, concept: Any, default: float = 0.9) -> float:
        """
        Combine candidate and concept confidence conservatively.
        """
        values = [
            value
            for value in [
                getattr(candidate, "confidence", None),
                getattr(concept, "confidence", None),
            ]
            if value is not None
        ]
        if not values:
            return default
        return float(min(values))

    def _axiom_key(
        self,
        axiom_type: str,
        subject_id: str,
        predicate: str,
        object_id: str | None,
        object_label: str | None,
        literal_value: str | None,
    ) -> Tuple:
        """
        Stable key used to avoid duplicate completed axioms.
        """
        return (
            axiom_type,
            subject_id,
            predicate,
            object_id,
            object_label,
            literal_value,
        )

    def _norm_label(self, label: str) -> str:
        """
        Normalize labels for duplicate checks.
        """
        return " ".join(str(label).lower().strip().split())

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize Layer 11 outputs for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "strategy": self._get_strategy(state),
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
                            "source_schema_ids": item.completed_axiom.source_schema_ids,
                            "source_concept_ids": item.completed_axiom.source_concept_ids,
                            "source_relation_ids": item.completed_axiom.source_relation_ids,
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

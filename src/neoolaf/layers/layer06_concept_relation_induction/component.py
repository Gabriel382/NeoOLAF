from __future__ import annotations

# Standard library imports
import re
from typing import List

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.ontology_elements import ConceptCandidate, OntologyRelationCandidate
from neoolaf.layers.layer06_concept_relation_induction.prompt import (
    build_concept_system_prompt,
    build_relation_system_prompt,
    build_concept_user_prompt,
    build_relation_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend
from neoolaf.domain.user_guidance_policy import should_promote_confidence
from neoolaf.grounding.rag.types import GroundingRequest
from neoolaf.grounding.rag.formatting import build_grounding_context

class ConceptRelationInductionLayer(BaseLayer):
    """
    Layer 6: concept / relation induction.

    Responsibilities:
    - promote stable entity/event candidates into ontology concept candidates
    - promote stable relation candidates into ontology relation candidates
    - preserve provenance and evidence
    """

    name = "layer06_concept_relation_induction"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        max_concept_inputs: int | None = None,
        max_relation_inputs: int | None = None,
        temperature: float = 0.0,
        save_intermediate: bool = True,
        verbose: bool = False,
        rag_adapter=None,
    ) -> None:
        """
        Initialize Layer 6.

        Args:
            ollama_backend:
                LLM backend used for induction.
            max_concept_inputs:
                Optional debug limit for concept induction inputs.
            max_relation_inputs:
                Optional debug limit for relation induction inputs.
            temperature:
                Generation temperature.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_concept_inputs = max_concept_inputs
        self.max_relation_inputs = max_relation_inputs
        self.temperature = temperature
        self.rag_adapter = rag_adapter

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run concept and relation induction from the current candidate pools.
        """
        concept_inputs = state.entity_candidates + state.event_candidates
        relation_inputs = state.relation_candidates

        if self.max_concept_inputs is not None:
            concept_inputs = concept_inputs[: self.max_concept_inputs]

        if self.max_relation_inputs is not None:
            relation_inputs = relation_inputs[: self.max_relation_inputs]

        concept_candidates: List[ConceptCandidate] = []
        ontology_relation_candidates: List[OntologyRelationCandidate] = []

        # ---------------------------------------------------------
        # Concept induction
        # ---------------------------------------------------------
        concept_iterator = concept_inputs
        if self.verbose:
            concept_iterator = tqdm(concept_inputs, desc="Layer 6 - concepts", leave=False)

        concept_counter = 0
        for candidate in concept_iterator:
            payload = {
                "candidate_id": candidate.candidate_id,
                "canonical_label": candidate.canonical_label,
                "candidate_type": candidate.candidate_type,
                "ontology_hints": candidate.ontology_hints,
                "definition": candidate.definition,
                "aliases": candidate.aliases,
                "synonyms": candidate.synonyms,
                "lexical_variants": candidate.lexical_variants,
                "mentions": [m.text for m in candidate.mentions],
            }

            grounding_result = None
            grounding_context = ""

            if self.rag_adapter is not None:
                grounding_result = self.rag_adapter.ground(
                    GroundingRequest(
                        layer_name="layer06_concept_relation_induction",
                        query=candidate.canonical_label,
                        payload={
                            "candidate_type": candidate.candidate_type,
                            "canonical_label": candidate.canonical_label,
                            "ontology_hints": candidate.ontology_hints,
                        },
                        preferred_sources=["ontology", "artifacts", "wikidata", "wikipedia"],
                        top_k=5,
                    )
                )
                grounding_context = build_grounding_context(grounding_result)

            messages = [
                {"role": "system", "content": build_concept_system_prompt()},
                {"role": "user", "content": build_concept_user_prompt(
                    candidate_payload=payload,
                    seed_ontology=state.seed_ontology,
                    guidance=state.user_guidance,
                    grounding_context=grounding_context,
                )},
            ]

            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self.ollama_backend.extract_json(raw)

            if not parsed.get("promote", False):
                continue

            if not should_promote_confidence(parsed.get("confidence"), state.user_guidance):
                continue

            label = parsed["label"].strip()
            concept_candidates.append(
                ConceptCandidate(
                    concept_id=f"concept_{concept_counter:05d}",
                    label=label,
                    normalized_label=self._normalize_label(label),
                    description=parsed.get("description"),
                    concept_kind=parsed.get("concept_kind"),
                    parent_hint=parsed.get("parent_hint"),
                    source_candidate_ids=[candidate.candidate_id],
                    source_triple_ids=self._collect_triple_ids_for_candidate(state, candidate.candidate_id),
                    confidence=parsed.get("confidence"),
                    justification=parsed["justification"].strip(),
                    evidence=self._collect_candidate_evidence(candidate),
                )
            )
            concept_counter += 1

        # ---------------------------------------------------------
        # Relation induction
        # ---------------------------------------------------------
        relation_iterator = relation_inputs
        if self.verbose:
            relation_iterator = tqdm(relation_inputs, desc="Layer 6 - relations", leave=False)

        relation_counter = 0
        for candidate in relation_iterator:
            payload = {
                "candidate_id": candidate.candidate_id,
                "canonical_label": candidate.canonical_label,
                "candidate_type": candidate.candidate_type,
                "ontology_hints": candidate.ontology_hints,
                "definition": candidate.definition,
                "aliases": candidate.aliases,
                "synonyms": candidate.synonyms,
                "lexical_variants": candidate.lexical_variants,
                "mentions": [m.text for m in candidate.mentions],
            }

            grounding_result = None
            grounding_context = ""

            if self.rag_adapter is not None:
                grounding_result = self.rag_adapter.ground(
                    GroundingRequest(
                        layer_name="layer06_concept_relation_induction",
                        query=candidate.canonical_label,
                        payload={
                            "candidate_type": candidate.candidate_type,
                            "canonical_label": candidate.canonical_label,
                            "definition": candidate.definition,
                        },
                        preferred_sources=["ontology", "artifacts", "wikidata", "wikipedia"],
                        top_k=5,
                    )
                )
                grounding_context = build_grounding_context(grounding_result)

            messages = [
                {"role": "system", "content": build_relation_system_prompt()},
                {"role": "user", "content": build_relation_user_prompt(
                    candidate_payload=payload,
                    seed_ontology=state.seed_ontology,
                    guidance=state.user_guidance,
                    grounding_context=grounding_context,
                )},
            ]

            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self.ollama_backend.extract_json(raw)

            if not parsed.get("promote", False):
                continue

            if not should_promote_confidence(parsed.get("confidence"), state.user_guidance):
                continue

            label = parsed["label"].strip()
            ontology_relation_candidates.append(
                OntologyRelationCandidate(
                    relation_id=f"ont_rel_{relation_counter:05d}",
                    label=label,
                    normalized_label=self._normalize_label(label),
                    description=parsed.get("description"),
                    domain_hint=parsed.get("domain_hint"),
                    range_hint=parsed.get("range_hint"),
                    source_candidate_ids=[candidate.candidate_id],
                    source_triple_ids=self._collect_triple_ids_for_candidate(state, candidate.candidate_id),
                    confidence=parsed.get("confidence"),
                    justification=parsed["justification"].strip(),
                    evidence=self._collect_candidate_evidence(candidate),
                )
            )
            relation_counter += 1

        state.concept_candidates = concept_candidates
        state.ontology_relation_candidates = ontology_relation_candidates

        state.log(
            "[layer06_concept_relation_induction] "
            f"concepts={len(concept_candidates)}, "
            f"ontology_relations={len(ontology_relation_candidates)}"
        )
        return state

    def _normalize_label(self, text: str) -> str:
        """
        Normalize a label for grouping and comparison.
        """
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\-]", "", text)
        return text

    def _collect_candidate_evidence(self, candidate) -> list:
        """
        Collect evidence from all mentions of a candidate.
        """
        evidences = []
        for mention in candidate.mentions:
            evidences.extend(mention.evidence)
        return evidences

    def _collect_triple_ids_for_candidate(self, state: PipelineState, candidate_id: str) -> list[str]:
        """
        Collect candidate triple IDs involving a given candidate.
        """
        triple_ids = []
        for triple in state.candidate_triples:
            if triple.subject_id == candidate_id or triple.object_id == candidate_id or triple.predicate_id == candidate_id:
                triple_ids.append(triple.triple_id)
        return list(dict.fromkeys(triple_ids))

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize Layer 6 outputs for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "num_concept_candidates": len(state.concept_candidates),
            "num_ontology_relation_candidates": len(state.ontology_relation_candidates),
            "concept_candidates": [
                {
                    "concept_id": c.concept_id,
                    "label": c.label,
                    "normalized_label": c.normalized_label,
                    "description": c.description,
                    "concept_kind": c.concept_kind,
                    "parent_hint": c.parent_hint,
                    "source_candidate_ids": c.source_candidate_ids,
                    "source_triple_ids": c.source_triple_ids,
                    "confidence": c.confidence,
                    "justification": c.justification,
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in c.evidence
                    ],
                }
                for c in state.concept_candidates
            ],
            "ontology_relation_candidates": [
                {
                    "relation_id": r.relation_id,
                    "label": r.label,
                    "normalized_label": r.normalized_label,
                    "description": r.description,
                    "domain_hint": r.domain_hint,
                    "range_hint": r.range_hint,
                    "source_candidate_ids": r.source_candidate_ids,
                    "source_triple_ids": r.source_triple_ids,
                    "confidence": r.confidence,
                    "justification": r.justification,
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in r.evidence
                    ],
                }
                for r in state.ontology_relation_candidates
            ],
        }
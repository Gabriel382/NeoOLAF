from __future__ import annotations

# Standard library imports
from typing import List

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.axiom_schema import AxiomSchemaCandidate
from neoolaf.layers.layer08_axiom_schemata_extraction.prompt import (
    build_relation_schema_system_prompt,
    build_subclass_schema_system_prompt,
    build_relation_schema_user_prompt,
    build_subclass_schema_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend

# Grounding imports
from neoolaf.grounding.rag.types import GroundingRequest
from neoolaf.grounding.rag.formatting import build_grounding_context


class AxiomSchemataExtractionLayer(BaseLayer):
    """
    Layer 8: axiom schemata extraction.

    Responsibilities:
    - extract domain/range schemata from ontology relation candidates
    - extract subclass schemata from concept hierarchy links
    - optionally use SemanticRAG grounding to improve schema extraction
    """

    name = "layer08_axiom_schemata_extraction"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        max_relation_schema_inputs: int | None = None,
        max_subclass_inputs: int | None = None,
        temperature: float = 0.0,
        rag_adapter=None,
        save_intermediate: bool = True,
        verbose: bool = False,
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
    ) -> None:
        """
        Initialize Layer 8.

        Args:
            ollama_backend:
                LLM backend used for schema extraction.
            max_relation_schema_inputs:
                Optional debug limit on relation-schema inputs.
            max_subclass_inputs:
                Optional debug limit on subclass-schema inputs.
            temperature:
                Generation temperature.
            rag_adapter:
                Optional SemanticRAG adapter.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_relation_schema_inputs = max_relation_schema_inputs
        self.max_subclass_inputs = max_subclass_inputs
        self.temperature = temperature
        self.rag_adapter = rag_adapter
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 0.0)

    def _layer_config(self, state: PipelineState) -> dict:
        """Return this layer configuration from the document profile."""
        profile_config = getattr(state, "profile_config", None) or {}
        return (
            profile_config.get("layers", {})
            .get("layer08_axiom_schemata_extraction", {})
        )

    def _strategy(self, state: PipelineState) -> str:
        """Return the configured Layer 8 strategy."""
        return str(self._layer_config(state).get("strategy", "llm_axiom_schemata_extraction"))

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run axiom schemata extraction.
        """
        strategy = self._strategy(state)
        if self.verbose:
            print(f"[NeoOLAF][Layer 8] strategy={strategy}")

        if strategy == "ontology_aware_axiom_schema_generation":
            return self._run_ontology_aware_axiom_schema_generation(state)

        axiom_schemata: List[AxiomSchemaCandidate] = []

        # ---------------------------------------------------------
        # 1. Extract relation domain/range schemata
        # ---------------------------------------------------------
        relation_candidates = state.ontology_relation_candidates
        if self.max_relation_schema_inputs is not None:
            relation_candidates = relation_candidates[: self.max_relation_schema_inputs]

        relation_iterator = relation_candidates
        if self.verbose:
            relation_iterator = tqdm(
                relation_candidates,
                desc="Layer 8 - relation schemata",
                leave=False,
            )

        schema_counter = 0

        for relation in relation_iterator:
            # Collect related triples using both label and source candidate ids
            related_triples = [
                triple for triple in state.candidate_triples
                if triple.predicate_label == relation.label
                or triple.predicate_id in relation.source_candidate_ids
            ]

            payload = {
                "relation_candidate": {
                    "relation_id": relation.relation_id,
                    "label": relation.label,
                    "description": relation.description,
                    "domain_hint": relation.domain_hint,
                    "range_hint": relation.range_hint,
                    "source_candidate_ids": relation.source_candidate_ids,
                },
                "related_triples": [
                    {
                        "triple_id": triple.triple_id,
                        "subject_label": triple.subject_label,
                        "subject_type": triple.subject_type,
                        "predicate_label": triple.predicate_label,
                        "object_label": triple.object_label,
                        "object_type": triple.object_type,
                    }
                    for triple in related_triples
                ],
            }

            # ---------------------------------------------
            # Optional SemanticRAG grounding
            # ---------------------------------------------
            grounding_context = ""
            if self.rag_adapter is not None:
                grounding_result = self.rag_adapter.ground(
                    GroundingRequest(
                        layer_name="layer08_axiom_schemata_extraction",
                        query=relation.label,
                        payload=payload,
                        preferred_sources=["ontology", "artifacts", "wikidata", "wikipedia"],
                        top_k=5,
                    )
                )
                grounding_context = build_grounding_context(grounding_result)

            # ---------------------------------------------
            # LLM schema extraction
            # ---------------------------------------------
            messages = [
                {"role": "system", "content": build_relation_schema_system_prompt()},
                {
                    "role": "user",
                    "content": build_relation_schema_user_prompt(
                        payload=payload,
                        seed_ontology=state.seed_ontology,
                        grounding_context=grounding_context,
                    ),
                },
            ]

            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self.ollama_backend.extract_json(raw)

            # Domain schema
            if parsed.get("emit_domain_schema", False):
                domain_label = parsed["domain_label"].strip()
                axiom_schemata.append(
                    AxiomSchemaCandidate(
                        schema_id=f"schema_{schema_counter:05d}",
                        schema_type="relation_domain",
                        subject_id=relation.relation_id,
                        subject_label=relation.label,
                        predicate="domain",
                        object_id=f"domain_hint::{domain_label}",
                        object_label=domain_label,
                        justification=parsed["justification"].strip(),
                        confidence=parsed.get("confidence"),
                        source_relation_ids=[relation.relation_id],
                        source_concept_ids=[],
                        source_triple_ids=[triple.triple_id for triple in related_triples],
                        evidence=relation.evidence,
                    )
                )
                schema_counter += 1

            # Range schema
            if parsed.get("emit_range_schema", False):
                range_label = parsed["range_label"].strip()
                axiom_schemata.append(
                    AxiomSchemaCandidate(
                        schema_id=f"schema_{schema_counter:05d}",
                        schema_type="relation_range",
                        subject_id=relation.relation_id,
                        subject_label=relation.label,
                        predicate="range",
                        object_id=f"range_hint::{range_label}",
                        object_label=range_label,
                        justification=parsed["justification"].strip(),
                        confidence=parsed.get("confidence"),
                        source_relation_ids=[relation.relation_id],
                        source_concept_ids=[],
                        source_triple_ids=[triple.triple_id for triple in related_triples],
                        evidence=relation.evidence,
                    )
                )
                schema_counter += 1

        # ---------------------------------------------------------
        # 2. Extract subclass schemata from concept hierarchy links
        # ---------------------------------------------------------
        concept_links = state.concept_hierarchy_links
        if self.max_subclass_inputs is not None:
            concept_links = concept_links[: self.max_subclass_inputs]

        subclass_iterator = concept_links
        if self.verbose:
            subclass_iterator = tqdm(
                concept_links,
                desc="Layer 8 - subclass schemata",
                leave=False,
            )

        for link in subclass_iterator:
            payload = {
                "child_concept_id": link.child_concept_id,
                "child_label": link.child_label,
                "parent_concept_id": link.parent_concept_id,
                "parent_label": link.parent_label,
                "justification": link.justification,
            }

            # ---------------------------------------------
            # Optional SemanticRAG grounding
            # ---------------------------------------------
            grounding_context = ""
            if self.rag_adapter is not None:
                grounding_result = self.rag_adapter.ground(
                    GroundingRequest(
                        layer_name="layer08_axiom_schemata_extraction",
                        query=f"{link.child_label} {link.parent_label}",
                        payload=payload,
                        preferred_sources=["ontology", "artifacts"],
                        top_k=5,
                    )
                )
                grounding_context = build_grounding_context(grounding_result)

            # ---------------------------------------------
            # LLM subclass schema extraction
            # ---------------------------------------------
            messages = [
                {"role": "system", "content": build_subclass_schema_system_prompt()},
                {
                    "role": "user",
                    "content": build_subclass_schema_user_prompt(
                        payload=payload,
                        seed_ontology=state.seed_ontology,
                        grounding_context=grounding_context,
                    ),
                },
            ]

            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self.ollama_backend.extract_json(raw)

            if not parsed.get("emit_subclass_schema", False):
                continue

            axiom_schemata.append(
                AxiomSchemaCandidate(
                    schema_id=f"schema_{schema_counter:05d}",
                    schema_type="subclass",
                    subject_id=link.child_concept_id,
                    subject_label=link.child_label,
                    predicate="subclassOf",
                    object_id=link.parent_concept_id,
                    object_label=link.parent_label,
                    justification=parsed["justification"].strip(),
                    confidence=parsed.get("confidence"),
                    source_relation_ids=[],
                    source_concept_ids=[link.child_concept_id, link.parent_concept_id],
                    source_triple_ids=[],
                    evidence=link.evidence,
                )
            )
            schema_counter += 1

        state.axiom_schema_candidates = axiom_schemata
        state.log(
            f"[layer08_axiom_schemata_extraction] extracted "
            f"{len(state.axiom_schema_candidates)} axiom schemata"
        )
        return state


    def _run_ontology_aware_axiom_schema_generation(self, state: PipelineState) -> PipelineState:
        """Generate axiom schema candidates deterministically from Layers 6 and 7.

        This profile-driven strategy is useful for structured manuals where
        previous layers already produced ontology-aware concept candidates,
        ontology relation candidates, and hierarchy links. It avoids one LLM
        call per relation/concept and is therefore appropriate for ablation
        runs and reproducible benchmarks.
        """
        axiom_schemata: List[AxiomSchemaCandidate] = []
        schema_counter = 0

        relation_candidates = list(getattr(state, "ontology_relation_candidates", []) or [])
        if self.max_relation_schema_inputs is not None:
            relation_candidates = relation_candidates[: self.max_relation_schema_inputs]

        # Relation domain/range schemata.
        for relation in relation_candidates:
            related_triples = [
                triple
                for triple in (getattr(state, "candidate_triples", []) or [])
                if getattr(triple, "predicate_label", None) == relation.label
                or getattr(triple, "predicate_id", None) in relation.source_candidate_ids
                or getattr(triple, "triple_id", None) in relation.source_triple_ids
            ]
            source_triple_ids = sorted({triple.triple_id for triple in related_triples})

            if relation.domain_hint:
                domain_label = str(relation.domain_hint).strip()
                if domain_label:
                    axiom_schemata.append(
                        AxiomSchemaCandidate(
                            schema_id=f"schema_{schema_counter:05d}",
                            schema_type="relation_domain",
                            subject_id=relation.relation_id,
                            subject_label=relation.label,
                            predicate="domain",
                            object_id=f"ontology_class::{self._normalize_identifier(domain_label)}",
                            object_label=domain_label,
                            justification=(
                                "Deterministic ontology-aware axiom schema generation: "
                                "Layer 6 domain_hint is promoted as the relation domain."
                            ),
                            confidence=relation.confidence if relation.confidence is not None else 1.0,
                            source_relation_ids=[relation.relation_id],
                            source_concept_ids=[],
                            source_triple_ids=source_triple_ids,
                            evidence=relation.evidence,
                        )
                    )
                    schema_counter += 1

            if relation.range_hint:
                range_label = str(relation.range_hint).strip()
                if range_label:
                    axiom_schemata.append(
                        AxiomSchemaCandidate(
                            schema_id=f"schema_{schema_counter:05d}",
                            schema_type="relation_range",
                            subject_id=relation.relation_id,
                            subject_label=relation.label,
                            predicate="range",
                            object_id=f"ontology_class::{self._normalize_identifier(range_label)}",
                            object_label=range_label,
                            justification=(
                                "Deterministic ontology-aware axiom schema generation: "
                                "Layer 6 range_hint is promoted as the relation range."
                            ),
                            confidence=relation.confidence if relation.confidence is not None else 1.0,
                            source_relation_ids=[relation.relation_id],
                            source_concept_ids=[],
                            source_triple_ids=source_triple_ids,
                            evidence=relation.evidence,
                        )
                    )
                    schema_counter += 1

        concept_links = list(getattr(state, "concept_hierarchy_links", []) or [])
        if self.max_subclass_inputs is not None:
            concept_links = concept_links[: self.max_subclass_inputs]

        # Concept subclass schemata.
        for link in concept_links:
            axiom_schemata.append(
                AxiomSchemaCandidate(
                    schema_id=f"schema_{schema_counter:05d}",
                    schema_type="subclass",
                    subject_id=link.child_concept_id,
                    subject_label=link.child_label,
                    predicate="subclassOf",
                    object_id=link.parent_concept_id,
                    object_label=link.parent_label,
                    justification=(
                        "Deterministic ontology-aware axiom schema generation: "
                        "Layer 7 concept hierarchy link is promoted as a subclass schema."
                    ),
                    confidence=link.confidence if link.confidence is not None else 1.0,
                    source_relation_ids=[],
                    source_concept_ids=[link.child_concept_id, link.parent_concept_id],
                    source_triple_ids=[],
                    evidence=link.evidence,
                )
            )
            schema_counter += 1

        # Relation hierarchy schemata as subproperty candidates.
        for link in (getattr(state, "relation_hierarchy_links", []) or []):
            axiom_schemata.append(
                AxiomSchemaCandidate(
                    schema_id=f"schema_{schema_counter:05d}",
                    schema_type="subproperty",
                    subject_id=link.child_relation_id,
                    subject_label=link.child_label,
                    predicate="subPropertyOf",
                    object_id=link.parent_relation_id,
                    object_label=link.parent_label,
                    justification=(
                        "Deterministic ontology-aware axiom schema generation: "
                        "Layer 7 relation hierarchy link is promoted as a subproperty schema."
                    ),
                    confidence=link.confidence if link.confidence is not None else 1.0,
                    source_relation_ids=[link.child_relation_id, link.parent_relation_id],
                    source_concept_ids=[],
                    source_triple_ids=[],
                    evidence=link.evidence,
                )
            )
            schema_counter += 1

        state.axiom_schema_candidates = axiom_schemata
        state.log(
            f"[layer08_axiom_schemata_extraction] deterministically generated "
            f"{len(state.axiom_schema_candidates)} axiom schemata"
        )
        return state

    @staticmethod
    def _normalize_identifier(label: str) -> str:
        """Create a stable lightweight identifier suffix from a label."""
        return "_".join(
            part for part in "".join(
                char.lower() if char.isalnum() else " " for char in label
            ).split() if part
        )

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize Layer 8 outputs for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "strategy": self._strategy(state),
            "num_axiom_schema_candidates": len(state.axiom_schema_candidates),
            "axiom_schema_candidates": [
                {
                    "schema_id": schema.schema_id,
                    "schema_type": schema.schema_type,
                    "subject_id": schema.subject_id,
                    "subject_label": schema.subject_label,
                    "predicate": schema.predicate,
                    "object_id": schema.object_id,
                    "object_label": schema.object_label,
                    "justification": schema.justification,
                    "confidence": schema.confidence,
                    "source_relation_ids": schema.source_relation_ids,
                    "source_concept_ids": schema.source_concept_ids,
                    "source_triple_ids": schema.source_triple_ids,
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in schema.evidence
                    ],
                }
                for schema in state.axiom_schema_candidates
            ],
        }
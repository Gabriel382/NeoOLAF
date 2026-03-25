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

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run axiom schemata extraction.
        """
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

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize Layer 8 outputs for debugging and reproducibility.
        """
        return {
            "layer": self.name,
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
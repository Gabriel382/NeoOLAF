from __future__ import annotations

# Standard library imports
from typing import List, Any

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.general_axiom import GeneralAxiomCandidate
from neoolaf.layers.layer09_general_axiom_extraction.prompt import (
    build_axiom_system_prompt,
    build_axiom_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend


class GeneralAxiomExtractionLayer(BaseLayer):
    """
    Layer 9: general axiom extraction.

    Responsibilities:
    - transform Layer 8 schemata into candidate ontology axioms
    - attach rdfs:description axioms to concepts and ontology relations
    """

    name = "layer09_general_axiom_extraction"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        max_schema_inputs: int | None = None,
        max_description_inputs: int | None = None,
        temperature: float = 0.0,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize Layer 9.

        Args:
            ollama_backend:
                LLM backend used for general axiom extraction.
            max_schema_inputs:
                Optional debug limit on schema-based axiom extraction.
            max_description_inputs:
                Optional debug limit on description axiom extraction.
            temperature:
                Generation temperature.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_schema_inputs = max_schema_inputs
        self.max_description_inputs = max_description_inputs
        self.temperature = temperature

    def _normalize_llm_json_dict(self, parsed: Any) -> dict | None:
        """
        Normalize LLM JSON outputs so downstream code always receives a dict.

        Some models may return:
        - a dict
        - a list containing one dict
        - an empty list
        - invalid / unexpected types

        Returns:
            A dictionary if normalization succeeds, otherwise None.
        """
        # Already the expected format
        if isinstance(parsed, dict):
            return parsed

        # Sometimes the model returns a list of JSON objects instead of one object
        if isinstance(parsed, list):
            if len(parsed) == 0:
                return None

            first_item = parsed[0]
            if isinstance(first_item, dict):
                return first_item

            return None

        # Any other type is considered invalid for this layer
        return None

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run general axiom extraction from Layer 8 schemata and Layer 6 candidates.
        """
        axioms: List[GeneralAxiomCandidate] = []
        axiom_counter = 0

        # ---------------------------------------------------------
        # 1. Convert Layer 8 schemata into general axioms
        # ---------------------------------------------------------
        schema_candidates = state.axiom_schema_candidates
        if self.max_schema_inputs is not None:
            schema_candidates = schema_candidates[: self.max_schema_inputs]

        schema_iterator = schema_candidates
        if self.verbose:
            schema_iterator = tqdm(
                schema_candidates,
                desc="Layer 9 - schema axioms",
                leave=False,
            )

        for schema in schema_iterator:
            payload = {
                "schema_id": schema.schema_id,
                "schema_type": schema.schema_type,
                "subject_id": schema.subject_id,
                "subject_label": schema.subject_label,
                "predicate": schema.predicate,
                "object_id": schema.object_id,
                "object_label": schema.object_label,
                "justification": schema.justification,
            }

            messages = [
                {"role": "system", "content": build_axiom_system_prompt()},
                {"role": "user", "content": build_axiom_user_prompt(payload)},
            ]

            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self.ollama_backend.extract_json(raw)

            # Normalize parsed output so this layer always works with a dictionary.
            parsed = self._normalize_llm_json_dict(parsed)
            if parsed is None:
                continue

            # Skip outputs that explicitly say no axiom should be emitted.
            if not parsed.get("emit_axiom", False):
                continue

            axiom_type = str(parsed["axiom_type"]).strip()
            predicate = str(parsed["predicate"]).strip()
            object_label = parsed.get("object_label")
            literal_value = parsed.get("literal_value")

            axioms.append(
                GeneralAxiomCandidate(
                    axiom_id=f"axiom_{axiom_counter:05d}",
                    axiom_type=axiom_type,
                    subject_id=schema.subject_id,
                    subject_label=schema.subject_label,
                    predicate=predicate,
                    object_id=schema.object_id if object_label is not None else None,
                    object_label=object_label,
                    literal_value=literal_value,
                    justification=str(parsed["justification"]).strip(),
                    confidence=parsed.get("confidence"),
                    source_schema_ids=[schema.schema_id],
                    source_concept_ids=schema.source_concept_ids,
                    source_relation_ids=schema.source_relation_ids,
                    evidence=schema.evidence,
                )
            )
            axiom_counter += 1

        # ---------------------------------------------------------
        # 2. Add rdfs:description axioms to all concept candidates
        # ---------------------------------------------------------
        concept_candidates = state.concept_candidates
        relation_candidates = state.ontology_relation_candidates

        description_inputs = (
            [("concept", c) for c in concept_candidates]
            + [("relation", r) for r in relation_candidates]
        )

        if self.max_description_inputs is not None:
            description_inputs = description_inputs[: self.max_description_inputs]

        description_iterator = description_inputs
        if self.verbose:
            description_iterator = tqdm(
                description_inputs,
                desc="Layer 9 - descriptions",
                leave=False,
            )

        for item_type, item in description_iterator:
            # Prefer an existing description if already induced in Layer 6
            literal_value = getattr(item, "description", None)

            # Fallback generic description if none exists
            if literal_value is None or not str(literal_value).strip():
                if item_type == "concept":
                    literal_value = f"Ontology concept candidate representing {item.label}."
                else:
                    literal_value = f"Ontology relation candidate representing {item.label}."

            axioms.append(
                GeneralAxiomCandidate(
                    axiom_id=f"axiom_{axiom_counter:05d}",
                    axiom_type="description",
                    subject_id=item.concept_id if item_type == "concept" else item.relation_id,
                    subject_label=item.label,
                    predicate="rdfs:description",
                    object_id=None,
                    object_label=None,
                    literal_value=literal_value,
                    justification="Interpretability rule: all ontology entities, concepts, and relations receive rdfs:description.",
                    confidence=1.0,
                    source_schema_ids=[],
                    source_concept_ids=[item.concept_id] if item_type == "concept" else [],
                    source_relation_ids=[item.relation_id] if item_type == "relation" else [],
                    evidence=item.evidence,
                )
            )
            axiom_counter += 1

        # ---------------------------------------------------------
        # 3. Deduplicate axioms
        # ---------------------------------------------------------
        dedup = {}
        for axiom in axioms:
            key = (
                axiom.axiom_type,
                axiom.subject_id,
                axiom.predicate,
                axiom.object_id,
                axiom.object_label,
                axiom.literal_value,
            )
            if key not in dedup:
                dedup[key] = axiom

        state.general_axiom_candidates = list(dedup.values())
        state.log(
            f"[layer09_general_axiom_extraction] extracted "
            f"{len(state.general_axiom_candidates)} general axioms"
        )
        return state

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize Layer 9 outputs for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "num_general_axiom_candidates": len(state.general_axiom_candidates),
            "general_axiom_candidates": [
                {
                    "axiom_id": axiom.axiom_id,
                    "axiom_type": axiom.axiom_type,
                    "subject_id": axiom.subject_id,
                    "subject_label": axiom.subject_label,
                    "predicate": axiom.predicate,
                    "object_id": axiom.object_id,
                    "object_label": axiom.object_label,
                    "literal_value": axiom.literal_value,
                    "justification": axiom.justification,
                    "confidence": axiom.confidence,
                    "source_schema_ids": axiom.source_schema_ids,
                    "source_concept_ids": axiom.source_concept_ids,
                    "source_relation_ids": axiom.source_relation_ids,
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in axiom.evidence
                    ],
                }
                for axiom in state.general_axiom_candidates
            ],
        }
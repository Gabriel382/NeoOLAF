from __future__ import annotations

# Standard library imports
from itertools import permutations
from typing import List

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.hierarchy import ConceptHierarchyLink, RelationHierarchyLink
from neoolaf.layers.layer07_hierarchisation.prompt import (
    build_concept_hierarchy_system_prompt,
    build_relation_hierarchy_system_prompt,
    build_concept_hierarchy_user_prompt,
    build_relation_hierarchy_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend


class HierarchisationLayer(BaseLayer):
    """
    Layer 7: hierarchisation.

    Responsibilities:
    - organize concept candidates into concept hierarchies
    - organize ontology relation candidates into relation hierarchies
    """

    name = "layer07_hierarchisation"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        max_concept_pairs: int | None = None,
        max_relation_pairs: int | None = None,
        temperature: float = 0.0,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize Layer 7.

        Args:
            ollama_backend:
                LLM backend used for hierarchy decisions.
            max_concept_pairs:
                Optional debug limit on concept pairs.
            max_relation_pairs:
                Optional debug limit on relation pairs.
            temperature:
                Generation temperature.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_concept_pairs = max_concept_pairs
        self.max_relation_pairs = max_relation_pairs
        self.temperature = temperature

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run hierarchisation over promoted concept and relation candidates.
        """
        concept_links = self._build_concept_hierarchy(state)
        relation_links = self._build_relation_hierarchy(state)

        state.concept_hierarchy_links = concept_links
        state.relation_hierarchy_links = relation_links

        state.log(
            "[layer07_hierarchisation] "
            f"concept_links={len(concept_links)}, "
            f"relation_links={len(relation_links)}"
        )
        return state

    def _build_concept_hierarchy(self, state: PipelineState) -> List[ConceptHierarchyLink]:
        """
        Build concept hierarchy links by testing candidate pairs.
        """
        concepts = state.concept_candidates
        pairs = list(permutations(concepts, 2))

        if self.max_concept_pairs is not None:
            pairs = pairs[: self.max_concept_pairs]

        iterator = pairs
        if self.verbose:
            iterator = tqdm(pairs, desc="Layer 7 - concept pairs", leave=False)

        links: List[ConceptHierarchyLink] = []
        counter = 0

        for child, parent in iterator:
            # Do not compare a concept with itself
            if child.concept_id == parent.concept_id:
                continue

            child_payload = {
                "concept_id": child.concept_id,
                "label": child.label,
                "description": child.description,
                "concept_kind": child.concept_kind,
                "parent_hint": child.parent_hint,
            }
            parent_payload = {
                "concept_id": parent.concept_id,
                "label": parent.label,
                "description": parent.description,
                "concept_kind": parent.concept_kind,
                "parent_hint": parent.parent_hint,
            }

            messages = [
                {"role": "system", "content": build_concept_hierarchy_system_prompt()},
                {
                    "role": "user",
                    "content": build_concept_hierarchy_user_prompt(
                        child_payload=child_payload,
                        parent_payload=parent_payload,
                        seed_ontology=state.seed_ontology,
                    ),
                },
            ]

            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self.ollama_backend.extract_json(raw)

            if not parsed.get("is_subclass", False):
                continue

            links.append(
                ConceptHierarchyLink(
                    link_id=f"concept_h_{counter:05d}",
                    child_concept_id=child.concept_id,
                    child_label=child.label,
                    parent_concept_id=parent.concept_id,
                    parent_label=parent.label,
                    justification=parsed["justification"].strip(),
                    confidence=parsed.get("confidence"),
                    evidence=child.evidence,
                )
            )
            counter += 1

        # Deduplicate
        dedup = {}
        for link in links:
            key = (link.child_concept_id, link.parent_concept_id)
            if key not in dedup:
                dedup[key] = link

        return list(dedup.values())

    def _build_relation_hierarchy(self, state: PipelineState) -> List[RelationHierarchyLink]:
        """
        Build relation hierarchy links by testing candidate pairs.
        """
        relations = state.ontology_relation_candidates
        pairs = list(permutations(relations, 2))

        if self.max_relation_pairs is not None:
            pairs = pairs[: self.max_relation_pairs]

        iterator = pairs
        if self.verbose:
            iterator = tqdm(pairs, desc="Layer 7 - relation pairs", leave=False)

        links: List[RelationHierarchyLink] = []
        counter = 0

        for child, parent in iterator:
            if child.relation_id == parent.relation_id:
                continue

            child_payload = {
                "relation_id": child.relation_id,
                "label": child.label,
                "description": child.description,
                "domain_hint": child.domain_hint,
                "range_hint": child.range_hint,
            }
            parent_payload = {
                "relation_id": parent.relation_id,
                "label": parent.label,
                "description": parent.description,
                "domain_hint": parent.domain_hint,
                "range_hint": parent.range_hint,
            }

            messages = [
                {"role": "system", "content": build_relation_hierarchy_system_prompt()},
                {
                    "role": "user",
                    "content": build_relation_hierarchy_user_prompt(
                        child_payload=child_payload,
                        parent_payload=parent_payload,
                        seed_ontology=state.seed_ontology,
                    ),
                },
            ]

            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self.ollama_backend.extract_json(raw)

            if not parsed.get("is_subrelation", False):
                continue

            links.append(
                RelationHierarchyLink(
                    link_id=f"relation_h_{counter:05d}",
                    child_relation_id=child.relation_id,
                    child_label=child.label,
                    parent_relation_id=parent.relation_id,
                    parent_label=parent.label,
                    justification=parsed["justification"].strip(),
                    confidence=parsed.get("confidence"),
                    evidence=child.evidence,
                )
            )
            counter += 1

        # Deduplicate
        dedup = {}
        for link in links:
            key = (link.child_relation_id, link.parent_relation_id)
            if key not in dedup:
                dedup[key] = link

        return list(dedup.values())

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize hierarchy outputs for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "num_concept_hierarchy_links": len(state.concept_hierarchy_links),
            "num_relation_hierarchy_links": len(state.relation_hierarchy_links),
            "concept_hierarchy_links": [
                {
                    "link_id": link.link_id,
                    "child_concept_id": link.child_concept_id,
                    "child_label": link.child_label,
                    "parent_concept_id": link.parent_concept_id,
                    "parent_label": link.parent_label,
                    "justification": link.justification,
                    "confidence": link.confidence,
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in link.evidence
                    ],
                }
                for link in state.concept_hierarchy_links
            ],
            "relation_hierarchy_links": [
                {
                    "link_id": link.link_id,
                    "child_relation_id": link.child_relation_id,
                    "child_label": link.child_label,
                    "parent_relation_id": link.parent_relation_id,
                    "parent_label": link.parent_label,
                    "justification": link.justification,
                    "confidence": link.confidence,
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in link.evidence
                    ],
                }
                for link in state.relation_hierarchy_links
            ],
        }
from __future__ import annotations

# Standard library imports
import re
import time
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
from neoolaf.domain.user_guidance_policy import should_accept_hierarchy_confidence
from neoolaf.grounding.rag.types import GroundingRequest
from neoolaf.grounding.rag.formatting import build_grounding_context

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
        rag_adapter=None,
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
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
        self.rag_adapter = rag_adapter
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 0.0)

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run hierarchisation over promoted concept and relation candidates.

        The XQuality ablation profile uses a deterministic ontology-aware
        strategy. It avoids the old expensive all-pairs LLM loop while still
        producing hierarchy links that can be inspected and ablated. The old
        LLM pairwise strategy remains available for generic profiles.
        """
        strategy = self._get_strategy(state)

        if strategy == "ontology_aware_parent_hint_hierarchy":
            concept_links = self._build_ontology_aware_concept_hierarchy(state)
            relation_links = self._build_ontology_aware_relation_hierarchy(state)
        else:
            concept_links = self._build_concept_hierarchy(state)
            relation_links = self._build_relation_hierarchy(state)

        state.concept_hierarchy_links = concept_links
        state.relation_hierarchy_links = relation_links

        state.log(
            "[layer07_hierarchisation] "
            f"strategy={strategy}, "
            f"concept_links={len(concept_links)}, "
            f"relation_links={len(relation_links)}"
        )
        return state

    def _get_strategy(self, state: PipelineState) -> str:
        """Return the profile strategy for Layer 7."""
        profile_config = getattr(state, "profile_config", {}) or {}
        layers_cfg = profile_config.get("layers", {}) if isinstance(profile_config, dict) else {}
        layer_cfg = layers_cfg.get(self.name, {}) if isinstance(layers_cfg, dict) else {}
        return str(layer_cfg.get("strategy", "llm_pairwise_hierarchy"))

    @staticmethod
    def _slug(value: str) -> str:
        """Create a stable compact identifier fragment."""
        text = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower())
        return text.strip("_") or "unknown"

    def _build_ontology_aware_concept_hierarchy(self, state: PipelineState) -> List[ConceptHierarchyLink]:
        """
        Deterministically link each concept candidate to its ontology parent hint.

        Layer 6 creates concept candidates with `parent_hint` derived from
        ontology hints, for example PLC Alarm, Alarm Cause, Machine Effect,
        Intervention Action, Responsible Actor, and Technical Reference. This
        method materializes those hints as hierarchy links without asking the
        LLM to compare every concept pair.
        """
        links: List[ConceptHierarchyLink] = []
        seen: set[tuple[str, str]] = set()

        for concept in state.concept_candidates:
            parent_hint = (getattr(concept, "parent_hint", None) or "").strip()
            if not parent_hint:
                continue

            child_label = (getattr(concept, "label", "") or "").strip()
            if child_label and child_label.casefold() == parent_hint.casefold():
                continue

            parent_id = f"ontology_parent_{self._slug(parent_hint)}"
            key = (concept.concept_id, parent_id)
            if key in seen:
                continue
            seen.add(key)

            links.append(
                ConceptHierarchyLink(
                    link_id=f"concept_h_{len(links):05d}",
                    child_concept_id=concept.concept_id,
                    child_label=concept.label,
                    parent_concept_id=parent_id,
                    parent_label=parent_hint,
                    justification=(
                        "Deterministic ontology-aware hierarchy induction: "
                        "Layer 6 parent_hint is treated as the ontology parent "
                        "class for the promoted concept candidate."
                    ),
                    confidence=getattr(concept, "confidence", None) or 1.0,
                    evidence=getattr(concept, "evidence", []) or [],
                )
            )

        return links

    def _build_ontology_aware_relation_hierarchy(self, state: PipelineState) -> List[RelationHierarchyLink]:
        """
        Deterministically link ontology relation candidates to a generic parent.

        The current XQuality relation set is controlled and flat. We still
        materialize an explicit parent relation so downstream ontology
        serialization has a traceable relation hierarchy artifact.
        """
        links: List[RelationHierarchyLink] = []
        seen: set[tuple[str, str]] = set()
        parent_id = "ontology_relation_parent_object_property"
        parent_label = "Object Property"

        for relation in state.ontology_relation_candidates:
            child_label = (getattr(relation, "label", "") or "").strip()
            if child_label.casefold() == parent_label.casefold():
                continue

            key = (relation.relation_id, parent_id)
            if key in seen:
                continue
            seen.add(key)

            links.append(
                RelationHierarchyLink(
                    link_id=f"relation_h_{len(links):05d}",
                    child_relation_id=relation.relation_id,
                    child_label=relation.label,
                    parent_relation_id=parent_id,
                    parent_label=parent_label,
                    justification=(
                        "Deterministic ontology-aware hierarchy induction: "
                        "controlled ontology relation candidates are treated as "
                        "domain-specific object properties."
                    ),
                    confidence=getattr(relation, "confidence", None) or 1.0,
                    evidence=getattr(relation, "evidence", []) or [],
                )
            )

        return links

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

            grounding_result = None
            grounding_context = ""

            if self.rag_adapter is not None:
                grounding_result = self.rag_adapter.ground(
                    GroundingRequest(
                        layer_name="layer07_hierarchisation",
                        query=f"{child.label} {parent.label}",
                        payload={
                            "child_label": child.label,
                            "parent_label": parent.label,
                            "task": "concept_hierarchy",
                        },
                        preferred_sources=["ontology", "artifacts"],
                        top_k=5,
                    )
                )
                grounding_context = build_grounding_context(grounding_result)

            messages = [
                {"role": "system", "content": build_concept_hierarchy_system_prompt()},
                {
                    "role": "user",
                    "content": build_concept_hierarchy_user_prompt(
                        child_payload=child_payload,
                        parent_payload=parent_payload,
                        seed_ontology=state.seed_ontology,
                        grounding_context=grounding_context,
                    ),
                },
            ]

            parsed = self._call_llm_with_retries(
                state=state,
                messages=messages,
                task_label="concept_hierarchy",
            )

            if not parsed.get("is_subclass", False):
                continue

            if not should_accept_hierarchy_confidence(parsed.get("confidence"), state.user_guidance):
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

            grounding_result = None
            grounding_context = ""

            if self.rag_adapter is not None:
                grounding_result = self.rag_adapter.ground(
                    GroundingRequest(
                        layer_name="layer07_hierarchisation",
                        query=f"{child.label} {parent.label}",
                        payload={
                            "child_label": child.label,
                            "parent_label": parent.label,
                            "task": "relation_hierarchy",
                        },
                        preferred_sources=["ontology", "artifacts"],
                        top_k=5,
                    )
                )
                grounding_context = build_grounding_context(grounding_result)

            messages = [
                {"role": "system", "content": build_relation_hierarchy_system_prompt()},
                {
                    "role": "user",
                    "content": build_relation_hierarchy_user_prompt(
                        child_payload=child_payload,
                        parent_payload=parent_payload,
                        seed_ontology=state.seed_ontology,
                        grounding_context=grounding_context,
                    ),
                },
            ]

            parsed = self._call_llm_with_retries(
                state=state,
                messages=messages,
                task_label="relation_hierarchy",
            )

            if not parsed.get("is_subrelation", False):
                continue

            if not should_accept_hierarchy_confidence(parsed.get("confidence"), state.user_guidance):
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

    def _call_llm_with_retries(self, state: PipelineState, messages: list[dict], task_label: str) -> dict:
        """Call the LLM with simple retries for the generic pairwise strategy."""
        last_error: Exception | None = None
        for attempt in range(self.retry_failed_calls + 1):
            try:
                raw = self.ollama_backend.chat(
                    model=state.llm_model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return self.ollama_backend.extract_json(raw)
            except Exception as exc:  # provider/JSON failures are retried together
                last_error = exc
                if attempt >= self.retry_failed_calls:
                    break
                if self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        raise RuntimeError(f"Layer 7 {task_label} failed after retries: {last_error}") from last_error

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize hierarchy outputs for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "strategy": self._get_strategy(state),
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
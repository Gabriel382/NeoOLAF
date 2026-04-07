from __future__ import annotations

# Standard library imports
from collections import defaultdict
from typing import Dict, List, Tuple
# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.relation_assertion import CandidateRelationAssertion
from neoolaf.domain.linguistic_expression import Evidence
from neoolaf.layers.layer04_candidate_relation_extraction.prompt import (
    build_system_prompt,
    build_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend
from neoolaf.grounding.rag.types import GroundingRequest
from neoolaf.grounding.rag.formatting import build_grounding_context

class CandidateRelationExtractionLayer(BaseLayer):
    """
    Layer 4: candidate relation extraction.

    Responsibilities:
    - inspect relation candidates in context
    - identify which canonical entity/event candidates they connect
    - produce candidate relation assertions
    """

    name = "layer04_candidate_relation_extraction"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        max_relation_mentions: int | None = None,
        temperature: float = 0.0,
        save_intermediate: bool = True,
        verbose: bool = False,
        rag_adapter=None,
    ) -> None:
        """
        Initialize Layer 4.

        Args:
            ollama_backend:
                LLM backend used for relation extraction.
            max_relation_mentions:
                Optional debug limit on how many relation mentions are tested.
            temperature:
                Generation temperature.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Wheter to show logs or not.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_relation_mentions = max_relation_mentions
        self.temperature = temperature
        self.rag_adapter = rag_adapter

        
        def _call_model_with_retries(
            self,
            state: PipelineState,
            messages: List[Dict[str, str]],
            max_attempts: int = 5,
            retry_wait_seconds: float = 3.0,
        ) -> dict:
            """
            Call the LLM backend with retries for:
            - empty / missing responses
            - malformed JSON
            - transient backend errors

            Args:
                state:
                    Current pipeline state.
                messages:
                    OpenAI-style prompt messages.
                max_attempts:
                    Maximum number of tries.
                retry_wait_seconds:
                    Delay between attempts.

            Returns:
                Parsed JSON dictionary.

            Raises:
                RuntimeError:
                    If all attempts fail.
            """
            import time

            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    raw = self.ollama_backend.chat(
                        model=state.llm_model,
                        messages=messages,
                        temperature=self.temperature,
                    )

                    if raw is None or not isinstance(raw, str) or not raw.strip():
                        raise RuntimeError(
                            f"{self.name}: backend returned empty response on attempt {attempt}/{max_attempts}"
                        )

                    parsed = self.ollama_backend.extract_json(raw)

                    if not isinstance(parsed, dict):
                        raise RuntimeError(
                            f"{self.name}: parsed response is not a dictionary on attempt {attempt}/{max_attempts}"
                        )

                    return parsed

                except Exception as exc:
                    last_error = exc

                    if self.verbose:
                        print(
                            f"[NeoOLAF] {self.name} retry {attempt}/{max_attempts} failed: {exc}"
                        )

                    if attempt < max_attempts:
                        time.sleep(retry_wait_seconds)

            raise RuntimeError(
                f"{self.name}: failed after {max_attempts} attempts. Last error: {last_error}"
            )
        
    def _run(self, state: PipelineState) -> PipelineState:
        """
        Extract candidate relation assertions from relation candidates and local chunk context.
        """
        # Build chunk-level indices for entities/events and relations
        chunk_to_local_candidates = self._index_local_entity_event_candidates(state)
        relation_mentions = self._index_relation_mentions(state)

        # Optional debug limit
        if self.max_relation_mentions is not None:
            relation_mentions = relation_mentions[: self.max_relation_mentions]

        assertions: List[CandidateRelationAssertion] = []
        assertion_counter = 0

        relation_iterator = relation_mentions
        if self.verbose:
            relation_iterator = tqdm(relation_mentions, desc="Layer 4 - relation mentions", leave=False)

        for relation_mention in relation_iterator:
            chunk_id = relation_mention["chunk_id"]
            relation_candidate = relation_mention["relation_candidate"]
            relation_evidence = relation_mention["evidence"]

            # Retrieve chunk text
            chunk = self._get_chunk_by_id(state, chunk_id)
            if chunk is None:
                continue

            # Local participants available in this chunk
            local_candidates = chunk_to_local_candidates.get(chunk_id, [])

            # Need at least two local participants to form a relation
            if len(local_candidates) < 2:
                continue

            relation_payload = {
                "candidate_id": relation_candidate.candidate_id,
                "canonical_label": relation_candidate.canonical_label,
                "candidate_type": relation_candidate.candidate_type,
            }

            local_candidate_payload = [
                {
                    "candidate_id": cand["candidate"].candidate_id,
                    "canonical_label": cand["candidate"].canonical_label,
                    "candidate_type": cand["candidate"].candidate_type,
                }
                for cand in local_candidates
            ]
            grounding_result = None
            grounding_context = ""

            if self.rag_adapter is not None:
                grounding_result = self.rag_adapter.ground(
                    GroundingRequest(
                        layer_name="layer04_candidate_relation_extraction",
                        query=relation_candidate.canonical_label,
                        payload={
                            "relation_candidate": relation_candidate.canonical_label,
                            "chunk_text": chunk.text,
                            "local_candidates": local_candidate_payload,
                        },
                        preferred_sources=["ontology", "artifacts", "wikidata", "wikipedia", "web"],
                        top_k=5,
                    )
                )
                grounding_context = build_grounding_context(grounding_result)

            messages = [
                {"role": "system", "content": build_system_prompt()},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        chunk_text=chunk.text,
                        chunk_id=chunk_id,
                        relation_candidate=relation_payload,
                        local_candidates=local_candidate_payload,
                        guidance=state.user_guidance,
                        grounding_context=grounding_context,
                    ),
                },
            ]

            parsed = self._call_model_with_retries(
                state=state,
                messages=messages,
                max_attempts=5,
                retry_wait_seconds=3.0,
            )

            if not parsed.get("found", False):
                continue

            source_id = parsed["source_candidate_id"]
            target_id = parsed["target_candidate_id"]
            justification = parsed["justification"].strip()
            confidence = parsed.get("confidence")

            source_candidate = self._find_candidate_by_id(state, source_id)
            target_candidate = self._find_candidate_by_id(state, target_id)

            if source_candidate is None or target_candidate is None:
                continue

            assertions.append(
                CandidateRelationAssertion(
                    assertion_id=f"rel_assert_{assertion_counter:05d}",
                    relation_candidate_id=relation_candidate.candidate_id,
                    relation_label=relation_candidate.canonical_label,
                    source_candidate_id=source_candidate.candidate_id,
                    source_candidate_label=source_candidate.canonical_label,
                    source_candidate_type=source_candidate.candidate_type,
                    target_candidate_id=target_candidate.candidate_id,
                    target_candidate_label=target_candidate.canonical_label,
                    target_candidate_type=target_candidate.candidate_type,
                    chunk_id=chunk_id,
                    justification=justification,
                    confidence=confidence,
                    evidence=relation_evidence,
                )
            )
            assertion_counter += 1

        # Deduplicate by relation + source + target + chunk
        dedup = {}
        for item in assertions:
            key = (
                item.relation_candidate_id,
                item.source_candidate_id,
                item.target_candidate_id,
                item.chunk_id,
            )
            if key not in dedup:
                dedup[key] = item

        state.candidate_relation_assertions = list(dedup.values())
        state.log(
            f"[layer04_candidate_relation_extraction] extracted "
            f"{len(state.candidate_relation_assertions)} candidate relation assertions"
        )
        return state

    def _index_local_entity_event_candidates(self, state: PipelineState) -> Dict[str, List[Dict]]:
        """
        Build an index from chunk_id to local entity/event candidates present in that chunk.
        """
        chunk_map: Dict[str, List[Dict]] = defaultdict(list)

        all_candidates = (
            state.entity_candidates
            + state.event_candidates
        )

        for candidate in all_candidates:
            for mention in candidate.mentions:
                for ev in mention.evidence:
                    chunk_map[ev.chunk_id].append(
                        {
                            "candidate": candidate,
                            "mention": mention,
                            "evidence": ev,
                        }
                    )

        return chunk_map

    def _index_relation_mentions(self, state: PipelineState) -> List[Dict]:
        """
        Build a flat list of relation candidate mentions with chunk provenance.
        """
        relation_mentions: List[Dict] = []

        for candidate in state.relation_candidates:
            for mention in candidate.mentions:
                for ev in mention.evidence:
                    relation_mentions.append(
                        {
                            "relation_candidate": candidate,
                            "mention": mention,
                            "chunk_id": ev.chunk_id,
                            "evidence": [ev],
                        }
                    )

        return relation_mentions

    def _get_chunk_by_id(self, state: PipelineState, chunk_id: str):
        """
        Return the chunk object matching a given chunk_id.
        """
        for chunk in state.document.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def _find_candidate_by_id(self, state: PipelineState, candidate_id: str):
        """
        Find any canonical candidate by ID across all Layer 3 candidate pools.
        """
        all_candidates = (
            state.entity_candidates
            + state.relation_candidates
            + state.attribute_candidates
            + state.event_candidates
        )

        for candidate in all_candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize candidate relation assertions for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "num_candidate_relation_assertions": len(state.candidate_relation_assertions),
            "candidate_relation_assertions": [
                {
                    "assertion_id": item.assertion_id,
                    "relation_candidate_id": item.relation_candidate_id,
                    "relation_label": item.relation_label,
                    "source_candidate_id": item.source_candidate_id,
                    "source_candidate_label": item.source_candidate_label,
                    "source_candidate_type": item.source_candidate_type,
                    "target_candidate_id": item.target_candidate_id,
                    "target_candidate_label": item.target_candidate_label,
                    "target_candidate_type": item.target_candidate_type,
                    "chunk_id": item.chunk_id,
                    "justification": item.justification,
                    "confidence": item.confidence,
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
                for item in state.candidate_relation_assertions
            ],
        }
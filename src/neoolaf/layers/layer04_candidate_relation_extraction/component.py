from __future__ import annotations

# Standard library imports
from collections import defaultdict
from typing import Any, Dict, List

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.relation_assertion import CandidateRelationAssertion
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
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
    ) -> None:
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_relation_mentions = max_relation_mentions
        self.temperature = temperature
        self.rag_adapter = rag_adapter
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 0.0)
        self._relation_strategy: str = "generic_llm_relation_extraction"

    def _call_model_with_retries(
        self,
        state: PipelineState,
        messages: List[Dict[str, str]],
        max_attempts: int = 5,
        retry_wait_seconds: float = 3.0,
    ) -> dict:
        """
        Call the backend with retries for:
        - empty responses
        - malformed JSON
        - transient request failures
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
        self._relation_strategy = self._strategy(state)
        if self.verbose:
            print(f"[NeoOLAF][Layer 4] strategy={self._relation_strategy}")

        if self._is_record_aware_strategy(self._relation_strategy):
            return self._run_record_aware_ontology(state)

        chunk_to_local_candidates = self._index_local_entity_event_candidates(state)
        relation_mentions = self._index_relation_mentions(state)

        if self.max_relation_mentions is not None:
            relation_mentions = relation_mentions[: self.max_relation_mentions]

        assertions: List[CandidateRelationAssertion] = []
        assertion_counter = 0

        relation_iterator = relation_mentions
        if self.verbose:
            relation_iterator = tqdm(
                relation_mentions,
                desc="Layer 4 - relation mentions",
                leave=False,
            )

        for relation_mention in relation_iterator:
            chunk_id = relation_mention["chunk_id"]
            relation_candidate = relation_mention["relation_candidate"]
            relation_evidence = relation_mention["evidence"]

            chunk = self._get_chunk_by_id(state, chunk_id)
            if chunk is None:
                continue

            local_candidates = chunk_to_local_candidates.get(chunk_id, [])

            if len(local_candidates) < 2:
                if self.verbose:
                    print(
                        f"[NeoOLAF] {self.name}: chunk {chunk_id} skipped because only "
                        f"{len(local_candidates)} local entity/event candidates were available."
                    )
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

    def _strategy(self, state: PipelineState) -> str:
        """Return the Layer 4 strategy selected by the document profile."""
        layer_cfg = (state.profile_config or {}).get("layers", {}).get(self.name, {})
        return str(layer_cfg.get("strategy", "generic_llm_relation_extraction"))

    def _is_record_aware_strategy(self, strategy: str) -> bool:
        """Whether Layer 4 should build relation assertions deterministically per record."""
        return strategy in {
            "ontology_aware_record_relation_extraction",
            "record_aware_relation_extraction",
            "xquality_record_relation_extraction",
        }

    def _run_record_aware_ontology(self, state: PipelineState) -> PipelineState:
        """
        Build ontology-aware relation assertions from candidates grouped by record/chunk.

        This strategy is deterministic and profile-driven. It assumes that Layer 1
        extracted one record per table/chunk, Layer 2 added ontology role hints, and
        Layer 3 created candidates whose mentions preserve the original chunk_id.
        It does not call the LLM. The RAG backend can still be active at pipeline level,
        but no retrieval is needed here because the relation schema is already in the
        document profile.
        """
        profile = state.profile_config or {}
        field_to_relation = self._field_to_relation(profile)
        relation_by_label = self._relation_candidate_by_label(state)
        expr_role_by_id = {expr.expr_id: self._normalize_role(expr.label) for expr in state.linguistic_expressions}
        chunk_roles = self._index_candidates_by_chunk_and_role(state, expr_role_by_id)

        assertions: list[CandidateRelationAssertion] = []
        assertion_counter = 0

        for chunk_id in sorted(chunk_roles):
            role_map = chunk_roles[chunk_id]
            central_candidates = self._dedup_candidates(
                role_map.get("alarm", []) + role_map.get("message", []) + role_map.get("record", [])
            )
            if not central_candidates:
                continue
            central = central_candidates[0]

            for source_role, relation_label in field_to_relation.items():
                relation_candidate = relation_by_label.get(self._normalize_relation_label(relation_label))
                if relation_candidate is None:
                    if self.verbose:
                        print(
                            f"[NeoOLAF] {self.name}: relation candidate not found for {relation_label!r}; "
                            "skipping deterministic assertions for this relation."
                        )
                    continue

                target_candidates = self._dedup_candidates(role_map.get(source_role, []))
                if not target_candidates:
                    continue

                for candidate in target_candidates:
                    if source_role == "cause":
                        source_candidate = candidate
                        target_candidate = central
                    else:
                        source_candidate = central
                        target_candidate = candidate

                    evidence = self._best_evidence_for_assertion(
                        source_candidate=source_candidate,
                        target_candidate=target_candidate,
                        chunk_id=chunk_id,
                    )

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
                            justification=(
                                "Deterministic ontology-aware record relation extraction: "
                                f"semantic role '{source_role}' maps to relation "
                                f"'{relation_candidate.canonical_label}' in the document profile."
                            ),
                            confidence=1.0,
                            evidence=evidence,
                        )
                    )
                    assertion_counter += 1

        state.candidate_relation_assertions = self._dedup_assertions(assertions)
        state.log(
            f"[layer04_candidate_relation_extraction] strategy={self._relation_strategy}; "
            f"extracted {len(state.candidate_relation_assertions)} candidate relation assertions"
        )
        return state

    def _field_to_relation(self, profile: dict[str, Any]) -> dict[str, str]:
        """Load the semantic-role to relation-label mapping from the profile."""
        default_mapping = {
            "cause": "TRIGGERS",
            "effect": "CAUSES",
            "intervention": "REQUIRES",
            "responsible": "HANDLED_BY",
            "reference": "REFERENCES",
        }
        mapping = profile.get("field_to_relation", {})
        if not isinstance(mapping, dict) or not mapping:
            return default_mapping
        result: dict[str, str] = {}
        for role, relation in mapping.items():
            if role and relation:
                result[self._normalize_role(str(role))] = str(relation)
        return result or default_mapping

    def _relation_candidate_by_label(self, state: PipelineState) -> dict[str, Any]:
        """Index controlled relation candidates by normalized canonical label and aliases."""
        result: dict[str, Any] = {}
        for candidate in state.relation_candidates:
            labels = [candidate.canonical_label, candidate.normalized_label, *candidate.aliases, *candidate.synonyms]
            for label in labels:
                if label:
                    result[self._normalize_relation_label(str(label))] = candidate
        return result

    def _index_candidates_by_chunk_and_role(
        self,
        state: PipelineState,
        expr_role_by_id: dict[str, str],
    ) -> dict[str, dict[str, list[Any]]]:
        """Group entity/event candidates by chunk_id and semantic role."""
        grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        all_candidates = state.entity_candidates + state.event_candidates + state.attribute_candidates

        for candidate in all_candidates:
            candidate_roles = self._roles_from_candidate(candidate)
            for mention in candidate.mentions:
                mention_role = expr_role_by_id.get(mention.expr_id)
                roles = candidate_roles or ([mention_role] if mention_role else [])
                roles = [self._normalize_role(role) for role in roles if role]
                if not roles:
                    roles = ["unknown"]

                for ev in mention.evidence:
                    for role in roles:
                        grouped[ev.chunk_id][role].append(candidate)
        return grouped

    def _roles_from_candidate(self, candidate: Any) -> list[str]:
        """Extract semantic roles from ontology_hints such as semantic_role:cause."""
        roles: list[str] = []
        for hint in getattr(candidate, "ontology_hints", []) or []:
            text = str(hint)
            if text.startswith("semantic_role:"):
                roles.append(self._normalize_role(text.split(":", 1)[1]))
        return self._dedup_strings(roles)

    def _best_evidence_for_assertion(self, source_candidate: Any, target_candidate: Any, chunk_id: str):
        """Collect compact supporting evidence from source and target mentions in the same chunk."""
        evidence = []
        seen = set()
        for candidate in (source_candidate, target_candidate):
            for mention in getattr(candidate, "mentions", []) or []:
                for ev in getattr(mention, "evidence", []) or []:
                    if ev.chunk_id != chunk_id:
                        continue
                    key = (ev.chunk_id, ev.chunk_start_char, ev.chunk_end_char, ev.snippet[:120])
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence.append(ev)
        return evidence

    def _dedup_candidates(self, candidates: list[Any]) -> list[Any]:
        """Deduplicate candidates while preserving order."""
        result = []
        seen = set()
        for cand in candidates:
            candidate_id = getattr(cand, "candidate_id", None)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            result.append(cand)
        return result

    def _dedup_assertions(self, assertions: list[CandidateRelationAssertion]) -> list[CandidateRelationAssertion]:
        """Deduplicate relation assertions and rewrite stable IDs."""
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
        result = list(dedup.values())
        for index, item in enumerate(result):
            item.assertion_id = f"rel_assert_{index:05d}"
        return result

    def _normalize_role(self, role: str) -> str:
        """Normalize semantic role names from Layer 1/2/3."""
        text = str(role or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "alarm_label": "alarm",
            "label": "alarm",
            "message_label": "message",
            "actor": "responsible",
            "responsible_actor": "responsible",
            "technical_reference": "reference",
            "ref": "reference",
            "action": "intervention",
            "corrective_action": "intervention",
        }
        return aliases.get(text, text)

    def _normalize_relation_label(self, label: str) -> str:
        """Normalize controlled relation labels for lookup."""
        return str(label or "").strip().upper().replace(" ", "_")

    def _dedup_strings(self, values: list[str]) -> list[str]:
        """Deduplicate strings while preserving order."""
        result = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _index_local_entity_event_candidates(self, state: PipelineState) -> Dict[str, List[Dict]]:
        """
        Build an index from chunk_id to local entity/event candidates present in that chunk.
        """
        chunk_map: Dict[str, List[Dict]] = defaultdict(list)

        all_candidates = state.entity_candidates + state.event_candidates

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

    def _candidate_ontology_hints(self, state: PipelineState, candidate_id: str) -> list[str]:
        """Return ontology hints for a candidate ID for artifact readability."""
        candidate = self._find_candidate_by_id(state, candidate_id)
        if candidate is None:
            return []
        return list(getattr(candidate, "ontology_hints", []) or [])

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize candidate relation assertions for debugging and reproducibility.
        """
        return {
            "layer": self.name,
            "strategy": self._relation_strategy,
            "num_candidate_relation_assertions": len(state.candidate_relation_assertions),
            "candidate_relation_assertions": [
                {
                    "assertion_id": item.assertion_id,
                    "relation_candidate_id": item.relation_candidate_id,
                    "relation_label": item.relation_label,
                    "relation_ontology_hints": self._candidate_ontology_hints(state, item.relation_candidate_id),
                    "source_candidate_id": item.source_candidate_id,
                    "source_candidate_label": item.source_candidate_label,
                    "source_candidate_type": item.source_candidate_type,
                    "source_ontology_hints": self._candidate_ontology_hints(state, item.source_candidate_id),
                    "target_candidate_id": item.target_candidate_id,
                    "target_candidate_label": item.target_candidate_label,
                    "target_candidate_type": item.target_candidate_type,
                    "target_ontology_hints": self._candidate_ontology_hints(state, item.target_candidate_id),
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
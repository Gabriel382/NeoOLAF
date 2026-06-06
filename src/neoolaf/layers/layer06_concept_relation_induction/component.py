from __future__ import annotations

# Standard library imports
import re
from typing import Any, List

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
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
        temperature: float = 0.0,
        save_intermediate: bool = True,
        verbose: bool = False,
        rag_adapter=None,
    ) -> None:
        """
        Initialize Layer 6.

        Args:
            ollama_backend:
                LLM backend used for induction when the selected strategy needs it.
            max_concept_inputs:
                Optional debug limit for concept induction inputs.
            max_relation_inputs:
                Optional debug limit for relation induction inputs.
            max_concurrency:
                Accepted for CLI/orchestrator consistency. The XQuality ontology-aware
                strategy is deterministic and does not call the LLM.
            retry_failed_calls:
                Accepted for CLI/orchestrator consistency and future LLM variants.
            retry_sleep_seconds:
                Accepted for CLI/orchestrator consistency and future LLM variants.
            temperature:
                Generation temperature for LLM-based fallback strategy.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_concept_inputs = max_concept_inputs
        self.max_relation_inputs = max_relation_inputs
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 0.0)
        self.temperature = temperature
        self.rag_adapter = rag_adapter

    def _run(self, state: PipelineState) -> PipelineState:
        """Run concept and relation induction from the current candidate pools."""
        strategy = (state.profile_config or {}).get("layers", {}).get(
            self.name, {}
        ).get("strategy", "llm_concept_relation_induction")
        strategy = str(strategy or "llm_concept_relation_induction")

        if strategy == "ontology_aware_triple_concept_relation_induction":
            return self._run_ontology_aware_triple_concept_relation_induction(state)

        return self._run_llm_induction(state)

    # ------------------------------------------------------------------
    # Deterministic ontology-aware strategy
    # ------------------------------------------------------------------
    def _run_ontology_aware_triple_concept_relation_induction(
        self,
        state: PipelineState,
    ) -> PipelineState:
        """Promote typed candidates and controlled relations to ontology candidates.

        This strategy is designed for the XQuality path after Layers 2--5:
        - Layer 2 attaches ontology hints to each expression.
        - Layer 3 creates typed entity/event/relation candidates.
        - Layer 4 creates ontology-aware relation assertions.
        - Layer 5 materializes candidate triples.

        Layer 6 then turns all ontology-linked nodes and relations into explicit
        concept/relation candidates. It does not call the LLM. RAG can remain
        enabled at the pipeline level, but this strategy intentionally uses the
        already-grounded profile/ontology hints to keep the ablation deterministic.
        """
        concept_inputs = (
            list(state.entity_candidates or [])
            + list(state.event_candidates or [])
            + list(state.attribute_candidates or [])
        )
        relation_inputs = list(state.relation_candidates or [])

        if self.max_concept_inputs is not None:
            concept_inputs = concept_inputs[: self.max_concept_inputs]
        if self.max_relation_inputs is not None:
            relation_inputs = relation_inputs[: self.max_relation_inputs]

        concept_candidates: list[ConceptCandidate] = []
        ontology_relation_candidates: list[OntologyRelationCandidate] = []

        concept_iterator = concept_inputs
        if self.verbose:
            print(
                f"[NeoOLAF][Layer 6] deterministic ontology-aware concept induction "
                f"for {len(concept_inputs)} node candidates; no LLM calls."
            )

        seen_concepts: dict[tuple[str, str], str] = {}
        concept_counter = 0
        for candidate in concept_iterator:
            hints = self._parse_ontology_hints(getattr(candidate, "ontology_hints", []) or [])
            promote = self._hint_bool(hints, "promote_to_ontology", default=True)
            if not promote:
                continue

            label = str(getattr(candidate, "canonical_label", "") or "").strip()
            if not label:
                continue

            normalized = self._normalize_label(label)
            parent_hint = hints.get("class_label") or hints.get("semantic_role")
            concept_kind = self._concept_kind(candidate, hints)
            key = (normalized, str(parent_hint or concept_kind or ""))
            if key in seen_concepts:
                # Already represented, but preserve source candidate and triples.
                existing_id = seen_concepts[key]
                existing = next((c for c in concept_candidates if c.concept_id == existing_id), None)
                if existing is not None:
                    cid = str(getattr(candidate, "candidate_id", ""))
                    if cid and cid not in existing.source_candidate_ids:
                        existing.source_candidate_ids.append(cid)
                    for tid in self._collect_triple_ids_for_candidate(state, cid):
                        if tid not in existing.source_triple_ids:
                            existing.source_triple_ids.append(tid)
                    existing.evidence.extend(self._collect_candidate_evidence(candidate))
                continue

            description = (
                getattr(candidate, "definition", None)
                or hints.get("definition")
                or self._default_concept_description(candidate, hints)
            )
            candidate_id = str(getattr(candidate, "candidate_id", ""))
            concept = ConceptCandidate(
                concept_id=f"concept_{concept_counter:05d}",
                label=label,
                normalized_label=normalized,
                description=description,
                concept_kind=concept_kind,
                parent_hint=parent_hint,
                source_candidate_ids=[candidate_id] if candidate_id else [],
                source_triple_ids=self._collect_triple_ids_for_candidate(state, candidate_id),
                confidence=getattr(candidate, "confidence", None) or 1.0,
                justification=(
                    "Deterministic ontology-aware concept induction from Layer 3 typed "
                    "candidate and Layer 2 ontology hints."
                ),
                evidence=self._collect_candidate_evidence(candidate),
            )
            concept_candidates.append(concept)
            seen_concepts[key] = concept.concept_id
            concept_counter += 1

        if self.verbose:
            print(
                f"[NeoOLAF][Layer 6] deterministic ontology-aware relation induction "
                f"for {len(relation_inputs)} relation candidates; no LLM calls."
            )

        seen_relations: dict[tuple[str, str, str], str] = {}
        relation_counter = 0
        for candidate in relation_inputs:
            hints = self._parse_ontology_hints(getattr(candidate, "ontology_hints", []) or [])
            promote = self._hint_bool(hints, "promote_to_ontology", default=True)
            if not promote:
                continue

            label = str(
                hints.get("controlled_relation")
                or getattr(candidate, "canonical_label", "")
                or ""
            ).strip()
            if not label:
                continue

            normalized = self._normalize_label(label)
            domain_hint = hints.get("domain")
            range_hint = hints.get("range")
            key = (normalized, str(domain_hint or ""), str(range_hint or ""))
            if key in seen_relations:
                existing_id = seen_relations[key]
                existing = next((r for r in ontology_relation_candidates if r.relation_id == existing_id), None)
                if existing is not None:
                    cid = str(getattr(candidate, "candidate_id", ""))
                    if cid and cid not in existing.source_candidate_ids:
                        existing.source_candidate_ids.append(cid)
                    for tid in self._collect_triple_ids_for_candidate(state, cid):
                        if tid not in existing.source_triple_ids:
                            existing.source_triple_ids.append(tid)
                    existing.evidence.extend(self._collect_candidate_evidence(candidate))
                continue

            candidate_id = str(getattr(candidate, "candidate_id", ""))
            relation = OntologyRelationCandidate(
                relation_id=f"ont_rel_{relation_counter:05d}",
                label=label,
                normalized_label=normalized,
                description=(
                    getattr(candidate, "definition", None)
                    or hints.get("definition")
                    or self._default_relation_description(label, domain_hint, range_hint)
                ),
                domain_hint=domain_hint,
                range_hint=range_hint,
                source_candidate_ids=[candidate_id] if candidate_id else [],
                source_triple_ids=self._collect_triple_ids_for_candidate(state, candidate_id),
                confidence=getattr(candidate, "confidence", None) or 1.0,
                justification=(
                    "Deterministic ontology-aware relation induction from Layer 3 "
                    "controlled relation candidate and profile ontology hints."
                ),
                evidence=self._collect_candidate_evidence(candidate),
            )
            ontology_relation_candidates.append(relation)
            seen_relations[key] = relation.relation_id
            relation_counter += 1

        state.concept_candidates = concept_candidates
        state.ontology_relation_candidates = ontology_relation_candidates

        state.log(
            f"[{self.name}] strategy=ontology_aware_triple_concept_relation_induction; "
            f"concepts={len(concept_candidates)}, ontology_relations={len(ontology_relation_candidates)}"
        )
        return state

    # ------------------------------------------------------------------
    # Legacy LLM strategy
    # ------------------------------------------------------------------
    def _run_llm_induction(self, state: PipelineState) -> PipelineState:
        """Legacy LLM-based concept and relation induction."""
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_label(self, text: str) -> str:
        """Normalize a label for grouping and comparison."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\-]", "", text)
        return text

    def _collect_candidate_evidence(self, candidate) -> list:
        """Collect evidence from all mentions of a candidate."""
        evidences = []
        for mention in getattr(candidate, "mentions", []) or []:
            evidences.extend(getattr(mention, "evidence", []) or [])
        return evidences

    def _collect_triple_ids_for_candidate(self, state: PipelineState, candidate_id: str) -> list[str]:
        """Collect candidate triple IDs involving a given candidate."""
        triple_ids = []
        if not candidate_id:
            return triple_ids
        for triple in state.candidate_triples:
            if (
                triple.subject_id == candidate_id
                or triple.object_id == candidate_id
                or triple.predicate_id == candidate_id
            ):
                triple_ids.append(triple.triple_id)
        return list(dict.fromkeys(triple_ids))

    @staticmethod
    def _parse_ontology_hints(hints: list[str]) -> dict[str, Any]:
        """Parse Layer 2/3 ontology hint strings into a small dictionary."""
        parsed: dict[str, Any] = {"raw": list(hints or [])}
        uri_values: list[str] = []
        label_values: list[str] = []

        for hint in hints or []:
            text = str(hint).strip()
            if not text:
                continue
            lower = text.lower()
            if lower.startswith("http://") or lower.startswith("https://"):
                uri_values.append(text)
                continue
            if ":" in text:
                key, value = text.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key:
                    parsed[key] = value
            else:
                label_values.append(text)

        if uri_values:
            parsed["uri"] = uri_values[0]
        if label_values:
            # The profile puts the ontology class/relation label after the URI.
            parsed["class_label"] = label_values[-1]
        return parsed

    @staticmethod
    def _hint_bool(hints: dict[str, Any], key: str, default: bool = False) -> bool:
        value = hints.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _concept_kind(candidate: Any, hints: dict[str, Any]) -> str:
        semantic_role = str(hints.get("semantic_role") or "").strip()
        family = str(hints.get("candidate_family") or getattr(candidate, "candidate_type", "") or "").strip()
        if semantic_role and family:
            return f"{family}:{semantic_role}"
        return family or semantic_role or "concept"

    @staticmethod
    def _default_concept_description(candidate: Any, hints: dict[str, Any]) -> str:
        label = str(getattr(candidate, "canonical_label", "candidate") or "candidate")
        parent = hints.get("class_label") or hints.get("semantic_role") or "ontology concept"
        return f"Ontology-linked candidate '{label}' promoted under {parent}."

    @staticmethod
    def _default_relation_description(label: str, domain_hint: str | None, range_hint: str | None) -> str:
        if domain_hint or range_hint:
            return f"Ontology-linked relation {label} with domain {domain_hint or 'unknown'} and range {range_hint or 'unknown'}."
        return f"Ontology-linked relation {label}."

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """Serialize Layer 6 outputs for debugging and reproducibility."""
        return {
            "layer": self.name,
            "strategy": (state.profile_config or {}).get("layers", {}).get(self.name, {}).get("strategy"),
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

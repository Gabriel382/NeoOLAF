from __future__ import annotations

"""Fast, profile-guided native DocRED ablation support.

This module changes no file under ``src/neoolaf``. It orchestrates the existing
13 NeoOLAF layers, adds input-level UserGuidance metadata, uses separate
per-layer token/time budgets, parallelizes the existing Layer 4 task, and adds
more detailed diagnostics. It does not add a second relation extraction task,
source-entity anchoring, closure rules, or gold-derived facts.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import re
import shutil
import sys
import threading
import time
import traceback

import docred_native_ablation as v2

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.candidates import RelationCandidate
from neoolaf.domain.documents import Document
from neoolaf.domain.relation_assertion import CandidateRelationAssertion
from neoolaf.domain.user_guidance import (
    NegativeExample,
    RelationExample,
    TypingExample,
    UserGuidance,
)
from neoolaf.grounding.rag.formatting import build_grounding_context
from neoolaf.grounding.rag.types import GroundingRequest, RetrievedItem
from neoolaf.ontology.loader import SeedOntologyLoader
from neoolaf.profiles.profile_loader import load_document_profile

from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
from neoolaf.layers.layer01_linguistic_expression_extraction.component import LinguisticExpressionExtractionLayer
from neoolaf.layers.layer02_candidate_enrichment.component import CandidateEnrichmentLayer
from neoolaf.layers.layer03_candidate_typing_resolution.component import CandidateTypingResolutionLayer
from neoolaf.layers.layer04_candidate_relation_extraction.component import CandidateRelationExtractionLayer
from neoolaf.layers.layer04_candidate_relation_extraction.prompt import build_system_prompt as build_l4_system_prompt
from neoolaf.layers.layer04_candidate_relation_extraction.prompt import build_user_prompt as build_l4_user_prompt
from neoolaf.layers.layer05_candidate_triple_generation.component import CandidateTripleGenerationLayer
from neoolaf.layers.layer06_concept_relation_induction.component import ConceptRelationInductionLayer
from neoolaf.layers.layer07_hierarchisation.component import HierarchisationLayer
from neoolaf.layers.layer08_axiom_schemata_extraction.component import AxiomSchemataExtractionLayer
from neoolaf.layers.layer09_general_axiom_extraction.component import GeneralAxiomExtractionLayer
from neoolaf.layers.layer10_validation_reasoning.component import ValidationReasoningLayer
from neoolaf.layers.layer11_inference_completion.component import InferenceCompletionLayer
from neoolaf.layers.layer12_serialization.component import SerializationLayer

from experiments.methods.run_neoolaf import (
    OfflineWebSearchSource,
    OfflineWikipediaSource,
    OfflineWikidataSource,
    OpenAICompatibleBackend,
    load_user_guidance,
)

# Re-export notebook helpers from v2.
read_json = v2.read_json
read_jsonl = v2.read_jsonl
write_json = v2.write_json
append_jsonl = v2.append_jsonl
load_layer_states = v2.load_layer_states
state_counts = v2.state_counts
safe_name = v2.safe_name
Tee = v2.Tee
LAYER_NAMES = v2.LAYER_NAMES


def _dedup(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


class SharedCallLogger:
    """One thread-safe logger shared by all per-layer backends."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "responses").mkdir(parents=True, exist_ok=True)
        self.calls_path = self.log_dir / "llm_calls.jsonl"
        self.errors_path = self.log_dir / "llm_errors.jsonl"
        self.parse_errors_path = self.log_dir / "llm_parse_errors.jsonl"
        self.cap_errors_path = self.log_dir / "llm_response_cap_errors.jsonl"
        self.lock = threading.Lock()
        self.call_index = 0

    def next_index(self) -> int:
        with self.lock:
            self.call_index += 1
            return self.call_index


class TaggedLoggedBackend:
    """Logger and output guard around one existing OpenAI-compatible backend."""

    def __init__(
        self,
        backend: OpenAICompatibleBackend,
        logger: SharedCallLogger,
        *,
        layer_tag: str,
        response_hard_cap_chars: int | None = None,
    ) -> None:
        self.backend = backend
        self.logger = logger
        self.layer_tag = layer_tag
        self.response_hard_cap_chars = int(response_hard_cap_chars or 0) or None

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0, **_: Any) -> str:
        call_index = self.logger.next_index()
        started = time.time()
        meta = {
            "call_index": call_index,
            "layer_tag": self.layer_tag,
            "model": model,
            "temperature": temperature,
            "message_count": len(messages),
            "system_chars": sum(len(m.get("content", "")) for m in messages if m.get("role") == "system"),
            "user_chars": sum(len(m.get("content", "")) for m in messages if m.get("role") != "system"),
            "max_tokens": getattr(self.backend, "max_tokens", None),
            "request_timeout": getattr(self.backend, "timeout", None),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            response = self.backend.chat(model=model, messages=messages, temperature=temperature)
            response_path = (
                self.logger.log_dir / "responses" /
                f"{call_index:04d}_{safe_name(self.layer_tag)}.txt"
            )
            response_path.write_text(response, encoding="utf-8")

            parse_ok = True
            parse_error = None
            parsed_type = None
            try:
                parsed = OpenAICompatibleBackend.extract_json(response)
                parsed_type = type(parsed).__name__
            except Exception as exc:  # The layer may retry; log before it does.
                parse_ok = False
                parse_error = f"{type(exc).__name__}: {exc}"
                append_jsonl(self.logger.parse_errors_path, {
                    **meta,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "response_chars": len(response),
                    "response_path": str(response_path),
                    "error": parse_error,
                }, self.logger.lock)

            if self.response_hard_cap_chars and len(response) > self.response_hard_cap_chars:
                message = (
                    f"{self.layer_tag}: response length {len(response)} exceeds hard cap "
                    f"{self.response_hard_cap_chars}; retrying with the same native task."
                )
                append_jsonl(self.logger.cap_errors_path, {
                    **meta,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "response_chars": len(response),
                    "response_path": str(response_path),
                    "error": message,
                }, self.logger.lock)
                raise RuntimeError(message)

            append_jsonl(self.logger.calls_path, {
                **meta,
                "status": "ok",
                "elapsed_seconds": round(time.time() - started, 3),
                "response_chars": len(response),
                "response_path": str(response_path),
                "json_parse_ok": parse_ok,
                "parsed_type": parsed_type,
                "parse_error": parse_error,
            }, self.logger.lock)
            return response
        except Exception as exc:
            append_jsonl(self.logger.errors_path, {
                **meta,
                "status": "error",
                "elapsed_seconds": round(time.time() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }, self.logger.lock)
            raise

    @staticmethod
    def extract_json(text: str) -> Any:
        return OpenAICompatibleBackend.extract_json(text)


class PriorityOntologyRAGAdapter(v2.OntologyOnlyRAGAdapter):
    """Property-first retrieval with relation aliases and priority property IDs."""

    name = "priority_property_first_ontology_only"

    def __init__(
        self,
        seed_ontology: Any,
        log_path: str | Path,
        *,
        top_k: int = 10,
        query_expansions: dict[str, list[str]] | None = None,
        relation_aliases: dict[str, list[str]] | None = None,
        priority_property_ids: list[str] | None = None,
    ) -> None:
        super().__init__(
            seed_ontology,
            log_path=log_path,
            top_k=top_k,
            query_expansions=query_expansions,
        )
        self.relation_aliases = relation_aliases or {}
        self.priority_property_ids = [str(x).upper() for x in (priority_property_ids or [])]
        self.alias_to_ids: dict[str, list[str]] = {}
        for relation_id, aliases in self.relation_aliases.items():
            for alias in [relation_id, *list(aliases or [])]:
                key = re.sub(r"\s+", " ", str(alias).lower()).strip()
                if key:
                    self.alias_to_ids.setdefault(key, []).append(str(relation_id).upper())

    def _expanded_queries(self, query: str) -> list[str]:
        values = super()._expanded_queries(query)
        lowered = re.sub(r"\s+", " ", str(query or "").lower()).strip()
        for alias, relation_ids in self.alias_to_ids.items():
            if alias and (alias == lowered or alias in lowered or lowered in alias):
                for relation_id in relation_ids:
                    prop = self.seed_ontology.properties_by_uri.get(
                        f"http://www.wikidata.org/prop/direct/{relation_id}"
                    )
                    values.extend([relation_id, getattr(prop, "label", None)])
        return _dedup(values)

    def _items(self, query: str, top_k: int | None = None, layer_name: str | None = None):
        top_k = max(3, int(top_k or self.top_k))
        layer_name = str(layer_name or "")
        relation_focused = layer_name.endswith((
            "candidate_relation_extraction",
            "concept_relation_induction",
        ))
        query_l = str(query or "").lower()
        relation_like = relation_focused or any(
            token in query_l
            for token in [" based ", "part of", "member of", "owned", "subsidiary", "country", "located", "born", "died", "founded", "established", "performed", "released"]
        )
        property_budget = top_k - 1 if relation_focused else max(2, int(round(top_k * (0.8 if relation_like else 0.6))))
        class_budget = max(1, top_k - property_budget)

        expanded = self._expanded_queries(query)
        retriever = self.space.retriever
        if retriever is None:
            return [], expanded

        properties: list[Any] = []
        classes: list[Any] = []
        for expanded_query in expanded:
            normalized = str(expanded_query).lower().strip()
            property_id = re.fullmatch(r"p\d+", normalized)
            if property_id:
                prop = self.seed_ontology.properties_by_uri.get(
                    f"http://www.wikidata.org/prop/direct/{normalized.upper()}"
                )
                if prop is not None:
                    properties.append(prop)
            for uri in self.seed_ontology.property_uris_by_label.get(normalized, []):
                prop = self.seed_ontology.properties_by_uri.get(uri)
                if prop is not None:
                    properties.append(prop)
            for uri in self.seed_ontology.class_uris_by_label.get(normalized, []):
                cls = self.seed_ontology.classes_by_uri.get(uri)
                if cls is not None:
                    classes.append(cls)

        for expanded_query in expanded:
            properties.extend(retriever.nearest_properties(expanded_query, top_k=property_budget))
            classes.extend(retriever.nearest_classes(expanded_query, top_k=class_budget))

        # Priority properties are tie-breakers, not unconditional evidence.
        if relation_like:
            for relation_id in self.priority_property_ids:
                if relation_id.lower() in query_l:
                    prop = self.seed_ontology.properties_by_uri.get(
                        f"http://www.wikidata.org/prop/direct/{relation_id}"
                    )
                    if prop is not None:
                        properties.insert(0, prop)

        def dedup_objects(values: list[Any], budget: int) -> list[Any]:
            result: list[Any] = []
            seen: set[str] = set()
            for value in values:
                uri = str(value.uri)
                if uri in seen:
                    continue
                seen.add(uri)
                result.append(value)
                if len(result) >= budget:
                    break
            return result

        selected_properties = dedup_objects(properties, property_budget)
        selected_classes = dedup_objects(classes, class_budget)
        items = [self._property_item(prop) for prop in selected_properties]
        items.extend(self._class_item(cls) for cls in selected_classes)
        return items[:top_k], expanded


class CanonicalizingCandidateTypingResolutionLayer(CandidateTypingResolutionLayer):
    """Run native Layer 3, then normalize relation labels from its own hints.

    This deterministic step does not infer a relation. It only converts a
    relation candidate already linked by Layer 2 to its canonical ontology ID,
    which is the benchmark-allowed native-to-ontology mapping stage.
    """

    def __init__(self, *args: Any, relation_catalog_path: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        catalog = read_json(relation_catalog_path)["relations"]
        self.id_to_label = {item["relation_id"]: item["label"] for item in catalog}
        self.relation_metadata = {item["relation_id"]: item for item in catalog}

    @staticmethod
    def _relation_id_from_hints(hints: Iterable[str]) -> str | None:
        controlled: list[str] = []
        other: list[str] = []
        for hint in hints or []:
            text = str(hint)
            match = re.search(r"\bP\d+\b", text, re.IGNORECASE)
            if not match:
                continue
            relation_id = match.group(0).upper()
            if text.lower().startswith("controlled_relation:"):
                controlled.append(relation_id)
            else:
                other.append(relation_id)
        return (controlled or other or [None])[0]

    def _run(self, state: PipelineState) -> PipelineState:
        state = super()._run(state)
        diagnostics: list[dict[str, Any]] = []
        for candidate in state.relation_candidates or []:
            original = candidate.canonical_label
            relation_id = self._relation_id_from_hints(candidate.ontology_hints)
            if relation_id and relation_id in self.id_to_label:
                canonical = f"{relation_id} : {self.id_to_label[relation_id]}"
                candidate.aliases = _dedup([original, *list(candidate.aliases or [])])
                candidate.canonical_label = canonical
                candidate.normalized_label = self._normalize_label(canonical)
                metadata = self.relation_metadata.get(relation_id, {})
                candidate.ontology_hints = _dedup([
                    f"controlled_relation:{canonical}",
                    "promote_to_ontology:true",
                    metadata.get("uri"),
                    metadata.get("label"),
                    f"domain:{', '.join(metadata.get('domain_uris') or [])}" if metadata.get("domain_uris") else None,
                    f"range:{', '.join(metadata.get('range_uris') or [])}" if metadata.get("range_uris") else None,
                    *list(candidate.ontology_hints or []),
                ])
            diagnostics.append({
                "candidate_id": candidate.candidate_id,
                "original_label": original,
                "canonical_label": candidate.canonical_label,
                "has_mentions": bool(candidate.mentions),
                "relation_id": relation_id,
                "controlled_hint_present": any(
                    str(h).lower().startswith("controlled_relation:")
                    for h in candidate.ontology_hints or []
                ),
                "ontology_hints": list(candidate.ontology_hints or []),
            })
        if state.artifact_dir:
            path = Path(state.artifact_dir) / self.name / "relation_canonicalization.json"
            write_json(path, diagnostics)
        return state


class ParallelCandidateRelationExtractionLayer(CandidateRelationExtractionLayer):
    """Parallel orchestration of the existing native Layer 4 task."""

    def __init__(
        self,
        *args: Any,
        max_attempts_per_relation: int = 2,
        retry_wait_seconds: float = 1.0,
        failure_log_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_attempts_per_relation = max(1, int(max_attempts_per_relation))
        self.relation_retry_wait_seconds = float(retry_wait_seconds)
        self.failure_log_path = Path(failure_log_path) if failure_log_path else None
        self.failure_lock = threading.Lock()

    def _call_model_with_retries(self, state: PipelineState, messages, max_attempts=5, retry_wait_seconds=3.0):
        return super()._call_model_with_retries(
            state,
            messages,
            max_attempts=self.max_attempts_per_relation,
            retry_wait_seconds=self.relation_retry_wait_seconds,
        )

    def _process_relation_mention(
        self,
        *,
        state: PipelineState,
        relation_mention: dict[str, Any],
        chunk_to_local_candidates: dict[str, list[dict[str, Any]]],
    ) -> CandidateRelationAssertion | None:
        chunk_id = relation_mention["chunk_id"]
        relation_candidate = relation_mention["relation_candidate"]
        relation_evidence = relation_mention["evidence"]
        chunk = self._get_chunk_by_id(state, chunk_id)
        if chunk is None:
            return None
        local_candidates = chunk_to_local_candidates.get(chunk_id, [])
        if len(local_candidates) < 2:
            return None

        relation_payload = {
            "candidate_id": relation_candidate.candidate_id,
            "canonical_label": relation_candidate.canonical_label,
            "candidate_type": relation_candidate.candidate_type,
            "ontology_hints": list(relation_candidate.ontology_hints or []),
            "aliases": list(relation_candidate.aliases or []),
        }
        local_candidate_payload = [
            {
                "candidate_id": item["candidate"].candidate_id,
                "canonical_label": item["candidate"].canonical_label,
                "candidate_type": item["candidate"].candidate_type,
                "ontology_hints": list(item["candidate"].ontology_hints or []),
            }
            for item in local_candidates
        ]
        grounding_context = ""
        if self.rag_adapter is not None:
            grounding = self.rag_adapter.ground(GroundingRequest(
                layer_name="layer04_candidate_relation_extraction",
                query=relation_candidate.canonical_label,
                payload={
                    "relation_candidate": relation_candidate.canonical_label,
                    "chunk_text": chunk.text,
                    "local_candidates": local_candidate_payload,
                },
                preferred_sources=["ontology"],
                top_k=8,
            ))
            grounding_context = build_grounding_context(grounding)

        messages = [
            {"role": "system", "content": build_l4_system_prompt()},
            {"role": "user", "content": build_l4_user_prompt(
                chunk_text=chunk.text,
                chunk_id=chunk_id,
                relation_candidate=relation_payload,
                local_candidates=local_candidate_payload,
                guidance=state.user_guidance,
                grounding_context=grounding_context,
            )},
        ]
        parsed = self._call_model_with_retries(state, messages)
        if not parsed.get("found", False):
            return None
        source = self._find_candidate_by_id(state, parsed.get("source_candidate_id"))
        target = self._find_candidate_by_id(state, parsed.get("target_candidate_id"))
        if source is None or target is None:
            return None
        return CandidateRelationAssertion(
            assertion_id="pending",
            relation_candidate_id=relation_candidate.candidate_id,
            relation_label=relation_candidate.canonical_label,
            source_candidate_id=source.candidate_id,
            source_candidate_label=source.canonical_label,
            source_candidate_type=source.candidate_type,
            target_candidate_id=target.candidate_id,
            target_candidate_label=target.canonical_label,
            target_candidate_type=target.candidate_type,
            chunk_id=chunk_id,
            justification=str(parsed.get("justification") or "").strip(),
            confidence=parsed.get("confidence"),
            evidence=relation_evidence,
        )

    def _run(self, state: PipelineState) -> PipelineState:
        self._relation_strategy = self._strategy(state)
        if self.verbose:
            print(
                f"[NeoOLAF][Layer 4] strategy={self._relation_strategy}; "
                f"parallel_workers={self.max_concurrency}; attempts={self.max_attempts_per_relation}"
            )
        if self._is_record_aware_strategy(self._relation_strategy):
            return self._run_record_aware_ontology(state)

        chunk_to_local_candidates = self._index_local_entity_event_candidates(state)
        relation_mentions = self._index_relation_mentions(state)
        if self.max_relation_mentions is not None:
            relation_mentions = relation_mentions[: self.max_relation_mentions]

        results: dict[int, CandidateRelationAssertion] = {}
        failures: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            future_to_index = {
                executor.submit(
                    self._process_relation_mention,
                    state=state,
                    relation_mention=mention,
                    chunk_to_local_candidates=chunk_to_local_candidates,
                ): index
                for index, mention in enumerate(relation_mentions)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    assertion = future.result()
                    if assertion is not None:
                        results[index] = assertion
                except Exception as exc:
                    mention = relation_mentions[index]
                    failure = {
                        "index": index,
                        "relation_candidate_id": mention["relation_candidate"].candidate_id,
                        "relation_label": mention["relation_candidate"].canonical_label,
                        "chunk_id": mention["chunk_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    if self.failure_log_path:
                        append_jsonl(self.failure_log_path, failure, self.failure_lock)

        assertions: list[CandidateRelationAssertion] = []
        seen: set[tuple[str, str, str, str]] = set()
        for index in sorted(results):
            assertion = results[index]
            key = (
                assertion.relation_candidate_id,
                assertion.source_candidate_id,
                assertion.target_candidate_id,
                assertion.chunk_id,
            )
            if key in seen:
                continue
            seen.add(key)
            assertion.assertion_id = f"rel_assert_{len(assertions):05d}"
            assertions.append(assertion)

        state.candidate_relation_assertions = assertions
        state.log(
            f"[{self.name}] parallel native extraction; mentions={len(relation_mentions)}; "
            f"assertions={len(assertions)}; failed={len(failures)}"
        )
        return state


def merge_input_task_guidance(guidance: UserGuidance, record: dict[str, Any]) -> UserGuidance:
    """Merge non-gold per-document task metadata into fields NeoOLAF consumes."""
    task = record.get("task_guidance")
    if not isinstance(task, dict):
        return guidance

    allowed_ids = [str(x) for x in task.get("allowed_relation_ids") or []]
    specs = task.get("relation_specs") or []
    spec_lines = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        rid = spec.get("relation_id")
        label = spec.get("label")
        direction = spec.get("direction")
        if rid and label:
            spec_lines.append(f"{rid} : {label} — {direction or ''}".strip())
    guidance.priority_relations = _dedup([
        *list(guidance.priority_relations or []),
        *[
            line.split(" — ", 1)[0]
            for line in spec_lines
        ],
    ])
    if spec_lines:
        guidance.domain_focus = (
            (guidance.domain_focus or "")
            + " INPUT RELATION SPECIFICATION: "
            + "; ".join(spec_lines)
        ).strip()
    guidance.population_policy = " ".join(filter(None, [
        guidance.population_policy,
        task.get("canonical_hint_contract"),
        task.get("inference_policy"),
        task.get("relation_instance_policy"),
    ]))

    existing_examples = {
        (e.text, e.relation_label, e.source_label, e.target_label)
        for e in guidance.relation_examples
    }
    for item in task.get("relation_examples") or []:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("text") or ""),
            str(item.get("relation_label") or ""),
            str(item.get("source_label") or ""),
            str(item.get("target_label") or ""),
        )
        if all(key) and key not in existing_examples:
            # ``task_guidance.relation_examples`` may carry experiment-only
            # metadata (for example ``candidate_relation_ids``) used by the
            # compact DocRED prompt builder.  ``RelationExample`` intentionally
            # contains only the stable NeoOLAF guidance contract, so construct
            # it explicitly instead of forwarding every input key.
            guidance.relation_examples.append(
                RelationExample(
                    text=key[0],
                    relation_label=key[1],
                    source_label=key[2],
                    target_label=key[3],
                    explanation=(
                        str(item.get("explanation"))
                        if item.get("explanation") is not None
                        else None
                    ),
                )
            )
            existing_examples.add(key)
    return guidance


def guidance_to_dict(guidance: UserGuidance) -> dict[str, Any]:
    return asdict(guidance)


def _layer_cfg(profile: dict[str, Any], layer_name: str) -> dict[str, Any]:
    return dict((profile.get("layers") or {}).get(layer_name) or {})


def _make_backend(
    *,
    logger: SharedCallLogger,
    layer_tag: str,
    model_host: str,
    api_key: str,
    cfg: dict[str, Any],
    fallback_max_tokens: int,
    fallback_timeout: int,
    reasoning_effort: str,
) -> TaggedLoggedBackend:
    core = OpenAICompatibleBackend(
        backend_name="openrouter",
        host=model_host,
        api_key=api_key,
        timeout=int(cfg.get("request_timeout_seconds", fallback_timeout)),
        max_tokens=int(cfg.get("max_output_tokens", fallback_max_tokens)),
        reasoning_effort=reasoning_effort,
        exclude_reasoning=True,
    )
    return TaggedLoggedBackend(
        core,
        logger,
        layer_tag=layer_tag,
        response_hard_cap_chars=cfg.get("response_hard_cap_chars"),
    )


def choose_chunk_size(text: str, max_safe_chars: int = 24000) -> int:
    return v2.choose_chunk_size(text, max_safe_chars)


def build_document(record: dict[str, Any], source_path: str | Path) -> Document:
    return v2.build_document(record, source_path)


def build_pipeline(
    *,
    backends: dict[str, TaggedLoggedBackend],
    rag_adapter: PriorityOntologyRAGAdapter,
    profile_config: dict[str, Any],
    relation_catalog_path: str | Path,
    chunk_size: int,
    run_dir: str | Path,
    workers: int = 12,
    verbose: bool = True,
) -> Pipeline:
    workers = max(1, int(workers))
    l2_cfg = _layer_cfg(profile_config, "layer02_candidate_enrichment")
    l4_cfg = _layer_cfg(profile_config, "layer04_candidate_relation_extraction")
    retry_default = int((profile_config.get("orchestration") or {}).get("retry_failed_calls", 1))
    sleep_default = float((profile_config.get("orchestration") or {}).get("retry_sleep_seconds", 1.0))
    l2_workers = int(l2_cfg.get("max_concurrency", workers))
    l4_workers = int(l4_cfg.get("max_concurrency", min(workers, 8)))

    layers = [
        PreprocessingLayer(
            chunk_size=chunk_size,
            overlap=0,
            enable_chunking=True,
            translate=False,
            save_intermediate=True,
            verbose=verbose,
            profile_config=profile_config,
        ),
        LinguisticExpressionExtractionLayer(
            backends["layer01"],
            max_chunks=1,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_backend=rag_adapter,
            max_concurrency=1,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
            rag_enabled=False,
        ),
        CandidateEnrichmentLayer(
            backends["layer02"],
            wikipedia_source=OfflineWikipediaSource(),
            wikidata_source=OfflineWikidataSource(),
            web_search_source=OfflineWebSearchSource(),
            max_expressions=None,
            use_web_search=False,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=l2_workers,
            retry_failed_calls=int(l2_cfg.get("retry_failed_calls", retry_default)),
            retry_sleep_seconds=sleep_default,
        ),
        CanonicalizingCandidateTypingResolutionLayer(
            backends["other"],
            relation_catalog_path=relation_catalog_path,
            max_expressions=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        ParallelCandidateRelationExtractionLayer(
            backends["layer04"],
            max_relation_mentions=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=l4_workers,
            retry_failed_calls=int(l4_cfg.get("retry_failed_calls", retry_default)),
            retry_sleep_seconds=sleep_default,
            max_attempts_per_relation=int(l4_cfg.get("max_attempts_per_relation", 2)),
            retry_wait_seconds=float(l4_cfg.get("retry_wait_seconds", 1.0)),
            failure_log_path=Path(run_dir) / "run_logs/layer04_relation_errors.jsonl",
        ),
        CandidateTripleGenerationLayer(
            max_assertions=None,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
            save_intermediate=True,
            verbose=verbose,
        ),
        ConceptRelationInductionLayer(
            backends["other"],
            max_concept_inputs=None,
            max_relation_inputs=None,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
        ),
        HierarchisationLayer(
            backends["other"],
            max_concept_pairs=None,
            max_relation_pairs=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        AxiomSchemataExtractionLayer(
            backends["other"],
            max_relation_schema_inputs=None,
            max_subclass_inputs=None,
            temperature=0.0,
            rag_adapter=rag_adapter,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        GeneralAxiomExtractionLayer(
            backends["other"],
            max_schema_inputs=None,
            max_description_inputs=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        ValidationReasoningLayer(
            max_triples=None,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        InferenceCompletionLayer(
            max_inferred_triples=None,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        SerializationLayer(
            output_subdir="exports",
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
        ),
    ]
    return Pipeline(layers=layers, verbose=verbose, continue_from_last=False)


def run_native_pipeline(
    *,
    project_root: str | Path,
    input_jsonl: str | Path,
    ontology_path: str | Path,
    profile_path: str | Path,
    guidance_path: str | Path,
    relation_catalog_path: str | Path,
    relation_aliases_path: str | Path,
    run_dir: str | Path,
    model_name: str,
    api_key: str,
    host: str = "https://openrouter.ai/api/v1",
    workers: int = 12,
    max_tokens: int = 4096,
    request_timeout: int = 180,
    reasoning_effort: str = "minimal",
    verbose: bool = True,
    clean_run_dir: bool = True,
) -> PipelineState:
    project_root = Path(project_root).resolve()
    input_jsonl = Path(input_jsonl).resolve()
    ontology_path = Path(ontology_path).resolve()
    profile_path = Path(profile_path).resolve()
    guidance_path = Path(guidance_path).resolve()
    relation_catalog_path = Path(relation_catalog_path).resolve()
    relation_aliases_path = Path(relation_aliases_path).resolve()
    run_dir = Path(run_dir).resolve()
    if clean_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_jsonl)
    if len(records) != 1:
        raise ValueError(f"This notebook expects exactly one input document, found {len(records)}")
    record = records[0]
    if "entities" in record or "relations" in record:
        raise ValueError("Pipeline input must not contain gold entities or relations.")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    profile = load_document_profile(profile_path=profile_path)
    profile_dict = profile.to_state_dict()
    guidance = load_user_guidance(str(guidance_path))
    if guidance is None:
        guidance = UserGuidance()
    guidance = merge_input_task_guidance(guidance, record)
    write_json(run_dir / "input_task_guidance.json", record.get("task_guidance") or {})
    write_json(run_dir / "effective_user_guidance.json", guidance_to_dict(guidance))

    seed_ontology = SeedOntologyLoader().load(str(ontology_path))
    if len(seed_ontology.properties_by_uri) < 90:
        raise RuntimeError(
            f"Only {len(seed_ontology.properties_by_uri)} ontology properties were loaded."
        )

    chunk_size = choose_chunk_size(
        record["text"],
        int(profile.get("chunking.max_safe_chunk_chars", 24000)),
    )
    logger = SharedCallLogger(logs_dir)
    backends = {
        "layer01": _make_backend(
            logger=logger, layer_tag="layer01", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer01_linguistic_expression_extraction"),
            fallback_max_tokens=max_tokens, fallback_timeout=request_timeout,
            reasoning_effort=reasoning_effort,
        ),
        "layer02": _make_backend(
            logger=logger, layer_tag="layer02", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer02_candidate_enrichment"),
            fallback_max_tokens=768, fallback_timeout=90,
            reasoning_effort=reasoning_effort,
        ),
        "layer04": _make_backend(
            logger=logger, layer_tag="layer04", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer04_candidate_relation_extraction"),
            fallback_max_tokens=512, fallback_timeout=90,
            reasoning_effort=reasoning_effort,
        ),
        "other": _make_backend(
            logger=logger, layer_tag="other", model_host=host, api_key=api_key,
            cfg={}, fallback_max_tokens=1024, fallback_timeout=120,
            reasoning_effort=reasoning_effort,
        ),
    }
    aliases = read_json(relation_aliases_path)
    rag_adapter = PriorityOntologyRAGAdapter(
        seed_ontology,
        log_path=logs_dir / "ontology_retrieval.jsonl",
        top_k=int(profile.get("rag.top_k", 10)),
        query_expansions=profile.get("rag.query_expansions", {}) or {},
        relation_aliases=aliases,
        priority_property_ids=profile.get("rag.priority_property_ids", []) or [],
    )
    pipeline = build_pipeline(
        backends=backends,
        rag_adapter=rag_adapter,
        profile_config=profile_dict,
        relation_catalog_path=relation_catalog_path,
        chunk_size=chunk_size,
        run_dir=run_dir,
        workers=workers,
        verbose=verbose,
    )
    state = PipelineState(
        document=build_document(record, input_jsonl),
        llm_model=model_name,
        user_guidance=guidance,
        seed_ontology=seed_ontology,
        artifact_dir=str(run_dir),
        profile_name=profile.name,
        profile_config=profile_dict,
    )
    runner = Runner(
        pipeline=pipeline,
        runs_root=str(run_dir.parent),
        verbose=verbose,
        max_workers=workers,
        enable_checkpoints=True,
        save_chunk_checkpoints=False,
    )
    manifest = {
        "document_id": record["document_id"],
        "title": record.get("title"),
        "model_name": model_name,
        "profile_name": profile.name,
        "profile_path": str(profile_path),
        "guidance_path": str(guidance_path),
        "input_task_guidance_present": bool(record.get("task_guidance")),
        "ontology_path": str(ontology_path),
        "ontology_classes": len(seed_ontology.classes_by_uri),
        "ontology_properties": len(seed_ontology.properties_by_uri),
        "input_has_gold": False,
        "chunk_size": chunk_size,
        "whole_document_single_chunk_expected": len(record["text"]) <= chunk_size,
        "workers": workers,
        "per_layer_limits": {
            name: _layer_cfg(profile_dict, name)
            for name in [
                "layer01_linguistic_expression_extraction",
                "layer02_candidate_enrichment",
                "layer04_candidate_relation_extraction",
            ]
        },
        "anti_cheating": profile.get("anti_cheating", {}),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir / "run_manifest.json", manifest)

    console_log = logs_dir / "console.log"
    errors_path = logs_dir / "pipeline_errors.jsonl"
    started = time.time()
    with console_log.open("w", encoding="utf-8") as log_handle:
        tee_out = Tee(sys.stdout, log_handle)
        tee_err = Tee(sys.stderr, log_handle)
        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                final_state = runner.run(state, from_layer=0, to_layer=12, run_dir=run_dir)
        except Exception as exc:
            append_jsonl(errors_path, {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            raise

    manifest.update({
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 3),
        "final_counts": state_counts(final_state),
    })
    write_json(run_dir / "run_manifest.json", manifest)
    return final_state


def _assertion_predictions(
    state: PipelineState,
    gold: dict[str, Any],
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> list[dict[str, Any]]:
    entity_aliases = v2.gold_entity_aliases(gold)
    _, label_to_id = v2.relation_lookup(catalog_path, aliases_path)
    candidates = {
        c.candidate_id: c
        for c in [
            *(state.entity_candidates or []),
            *(state.event_candidates or []),
            *(state.attribute_candidates or []),
        ]
    }
    relations = {c.candidate_id: c for c in state.relation_candidates or []}
    rows = []
    for assertion in state.candidate_relation_assertions or []:
        src = candidates.get(assertion.source_candidate_id)
        dst = candidates.get(assertion.target_candidate_id)
        rel = relations.get(assertion.relation_candidate_id)
        relation_id = v2.map_predicate(
            assertion.relation_label,
            getattr(rel, "ontology_hints", []) if rel else [],
            label_to_id,
        )
        head = v2.align_candidate(src, entity_aliases) if src else None
        tail = v2.align_candidate(dst, entity_aliases) if dst else None
        rows.append({
            "assertion_id": assertion.assertion_id,
            "head_id": head,
            "relation_id": relation_id,
            "tail_id": tail,
            "fully_mapped": bool(head and relation_id and tail),
            "source_label": assertion.source_candidate_label,
            "predicate_label": assertion.relation_label,
            "target_label": assertion.target_candidate_label,
        })
    return rows


def write_cumulative_evaluation(
    run_dir: str | Path,
    gold: dict[str, Any],
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> list[dict[str, Any]]:
    rows = []
    for index, name, state in load_layer_states(run_dir):
        if index < 4:
            predictions = []
        elif index == 4:
            predictions = _assertion_predictions(state, gold, catalog_path, aliases_path)
        else:
            predictions = v2.native_predictions(state, gold, catalog_path, aliases_path)
        evaluation = v2.strict_evaluate(predictions, gold)
        rows.append({
            "layer_index": index,
            "layer_name": name,
            "mapped_predictions": sum(1 for p in predictions if p.get("fully_mapped")),
            **{
                key: evaluation[key]
                for key in [
                    "predicted", "gold", "true_positive", "false_positive",
                    "false_negative", "precision", "recall", "f1",
                ]
            },
        })
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json(analysis_dir / "cumulative_strict_evaluation.json", rows)
    if rows:
        with (analysis_dir / "cumulative_strict_evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def write_relation_trace_v3(
    run_dir: str | Path,
    gold: dict[str, Any],
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    states = {index: state for index, _, state in load_layer_states(run_dir)}
    final_state = states[max(states)]
    aliases = v2.gold_entity_aliases(gold)
    id_to_label, label_to_id = v2.relation_lookup(catalog_path, aliases_path)

    l1 = states.get(1, final_state)
    l2 = states.get(2, final_state)
    l3 = states.get(3, final_state)
    l4 = states.get(4, final_state)
    l5 = states.get(5, final_state)

    l1_nodes = v2.align_text_values([x.text for x in l1.linguistic_expressions or []], aliases)
    l2_nodes = v2.align_text_values([x.base_expression.text for x in l2.enriched_expressions or []], aliases)
    l3_node_candidates = [*(l3.entity_candidates or []), *(l3.event_candidates or []), *(l3.attribute_candidates or [])]
    l3_nodes = {v2.align_candidate(c, aliases) for c in l3_node_candidates}
    l3_nodes.discard(None)

    schema_relations: set[str] = set()
    mention_relations: dict[str, list[str]] = {}
    for candidate in l3.relation_candidates or []:
        rid = v2.map_predicate(candidate.canonical_label, candidate.ontology_hints, label_to_id)
        if not rid:
            continue
        schema_relations.add(rid)
        if candidate.mentions:
            mention_relations.setdefault(rid, []).append(candidate.candidate_id)

    assertion_rows = _assertion_predictions(l4, gold, catalog_path, aliases_path)
    exact_assertions = {
        (row["head_id"], row["relation_id"], row["tail_id"]): row["assertion_id"]
        for row in assertion_rows if row.get("fully_mapped")
    }
    triple_rows = v2.native_predictions(l5, gold, catalog_path, aliases_path)
    exact_triples = {
        (row["head_id"], row["relation_id"], row["tail_id"]): row["triple_id"]
        for row in triple_rows if row.get("fully_mapped")
    }

    rows = []
    for head, relation_id, tail in sorted(v2.gold_triples(gold)):
        head_l1, tail_l1 = head in l1_nodes, tail in l1_nodes
        head_l2, tail_l2 = head in l2_nodes, tail in l2_nodes
        head_l3, tail_l3 = head in l3_nodes, tail in l3_nodes
        schema_available = relation_id in schema_relations
        mention_ids = mention_relations.get(relation_id, [])
        mention_available = bool(mention_ids)
        assertion_id = exact_assertions.get((head, relation_id, tail))
        triple_id = exact_triples.get((head, relation_id, tail))
        if not head_l1 or not tail_l1:
            first_failure = "layer01_expression_extraction"
        elif not head_l2 or not tail_l2:
            first_failure = "layer02_enrichment_survival"
        elif not head_l3 or not tail_l3:
            first_failure = "layer03_endpoint_resolution"
        elif not schema_available:
            first_failure = "layer03_schema_property_unavailable"
        elif not mention_available:
            first_failure = "layer03_no_relation_mention_linked_to_property"
        elif not assertion_id:
            first_failure = "layer04_exact_endpoint_assignment"
        elif not triple_id:
            first_failure = "layer05_triple_materialization_or_mapping"
        else:
            first_failure = "survived_to_layer05"
        rows.append({
            "head_id": head,
            "relation_id": relation_id,
            "relation_label": id_to_label.get(relation_id),
            "tail_id": tail,
            "head_available_layer01": head_l1,
            "tail_available_layer01": tail_l1,
            "head_available_layer02": head_l2,
            "tail_available_layer02": tail_l2,
            "head_available_layer03": head_l3,
            "tail_available_layer03": tail_l3,
            "schema_property_available_layer03": schema_available,
            "predicate_mention_available_layer03": mention_available,
            "matching_relation_candidate_ids": "|".join(mention_ids),
            "exact_assertion_id_layer04": assertion_id,
            "exact_triple_id_layer05": triple_id,
            "first_failure": first_failure,
        })
    analysis_dir = run_dir / "analysis"
    write_json(analysis_dir / "gold_relation_trace_v3.json", rows)
    if rows:
        with (analysis_dir / "gold_relation_trace_v3.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def write_native_views(
    run_dir: str | Path,
    state: PipelineState,
    gold: dict[str, Any],
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    mapped = v2.native_predictions(state, gold, catalog_path, aliases_path)
    relations = {c.candidate_id: c for c in state.relation_candidates or []}
    lexical = []
    canonical = []
    for triple, mapped_row in zip(state.candidate_triples or [], mapped):
        rel = relations.get(triple.predicate_id)
        row = {
            "triple_id": triple.triple_id,
            "subject_label": triple.subject_label,
            "predicate_label": triple.predicate_label,
            "object_label": triple.object_label,
            "relation_candidate_id": triple.predicate_id,
            "relation_aliases": list(getattr(rel, "aliases", []) or []) if rel else [],
            "ontology_hints": list(getattr(rel, "ontology_hints", []) or []) if rel else [],
            "confidence": triple.confidence,
            "justification": triple.justification,
        }
        lexical.append(row)
        if mapped_row.get("relation_id"):
            canonical.append({**row, **{
                "head_id": mapped_row.get("head_id"),
                "relation_id": mapped_row.get("relation_id"),
                "tail_id": mapped_row.get("tail_id"),
                "fully_mapped": mapped_row.get("fully_mapped"),
            }})
    gold_set = v2.gold_triples(gold)
    not_in_gold = [
        {**row, "manual_review_required": True}
        for row in mapped
        if row.get("fully_mapped") and (row["head_id"], row["relation_id"], row["tail_id"]) not in gold_set
    ]
    files = {
        "native_lexical_triples": lexical,
        "ontology_canonical_triples": canonical,
        "strict_docred_predictions": mapped,
        "predictions_not_in_gold_manual_review": not_in_gold,
    }
    for name, rows in files.items():
        write_json(analysis_dir / f"{name}.json", rows)
        if rows:
            with (analysis_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    return files


def analyze_run(
    *,
    run_dir: str | Path,
    gold_jsonl: str | Path,
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> dict[str, Any]:
    summary = v2.analyze_run(
        run_dir=run_dir,
        gold_jsonl=gold_jsonl,
        catalog_path=catalog_path,
        aliases_path=aliases_path,
    )
    gold_rows = read_jsonl(gold_jsonl)
    gold = gold_rows[0]
    states = load_layer_states(run_dir)
    final_state = states[-1][2]
    cumulative = write_cumulative_evaluation(run_dir, gold, catalog_path, aliases_path)
    trace = write_relation_trace_v3(run_dir, gold, catalog_path, aliases_path)
    views = write_native_views(run_dir, final_state, gold, catalog_path, aliases_path)
    summary["cumulative_strict_evaluation"] = cumulative
    summary["failure_counts_v3"] = {}
    for row in trace:
        key = row["first_failure"]
        summary["failure_counts_v3"][key] = summary["failure_counts_v3"].get(key, 0) + 1
    summary["native_views"] = {key: len(value) for key, value in views.items()}
    write_json(Path(run_dir) / "analysis/analysis_summary_v3.json", summary)
    return summary

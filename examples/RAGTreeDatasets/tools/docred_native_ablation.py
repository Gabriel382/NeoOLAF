from __future__ import annotations

"""Notebook support for the native DocRED NeoOLAF layer ablation.

This module orchestrates existing NeoOLAF layers only. It does not add a second
LLM extraction task, alter src/neoolaf, expose gold entities/relations to the
pipeline, or add closure facts. Gold data is loaded only after the run for
analysis.
"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import re
import sys
import threading
import time
import traceback

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.documents import Document
from neoolaf.grounding.rag.base import RAGRequest, RAGResult
from neoolaf.grounding.rag.types import GroundingRequest, GroundingResult, RetrievedItem
from neoolaf.grounding.rag.spaces.ontology_space import OntologySpace
from neoolaf.ontology.loader import SeedOntologyLoader
from neoolaf.profiles.profile_loader import load_document_profile

from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
from neoolaf.layers.layer01_linguistic_expression_extraction.component import LinguisticExpressionExtractionLayer
from neoolaf.layers.layer02_candidate_enrichment.component import CandidateEnrichmentLayer
from neoolaf.layers.layer03_candidate_typing_resolution.component import CandidateTypingResolutionLayer
from neoolaf.layers.layer04_candidate_relation_extraction.component import CandidateRelationExtractionLayer
from neoolaf.layers.layer05_candidate_triple_generation.component import CandidateTripleGenerationLayer
from neoolaf.layers.layer06_concept_relation_induction.component import ConceptRelationInductionLayer
from neoolaf.layers.layer07_hierarchisation.component import HierarchisationLayer
from neoolaf.layers.layer08_axiom_schemata_extraction.component import AxiomSchemataExtractionLayer
from neoolaf.layers.layer09_general_axiom_extraction.component import GeneralAxiomExtractionLayer
from neoolaf.layers.layer10_validation_reasoning.component import ValidationReasoningLayer
from neoolaf.layers.layer11_inference_completion.component import InferenceCompletionLayer
from neoolaf.layers.layer12_serialization.component import SerializationLayer

from experiments.methods.run_neoolaf import (
    OpenAICompatibleBackend,
    OfflineWikipediaSource,
    OfflineWikidataSource,
    OfflineWebSearchSource,
    load_user_guidance,
)


LAYER_NAMES = [
    "layer00_preprocessing",
    "layer01_linguistic_expression_extraction",
    "layer02_candidate_enrichment",
    "layer03_candidate_typing_resolution",
    "layer04_candidate_relation_extraction",
    "layer05_candidate_triple_generation",
    "layer06_concept_relation_induction",
    "layer07_hierarchisation",
    "layer08_axiom_schemata_extraction",
    "layer09_general_axiom_extraction",
    "layer10_validation_reasoning",
    "layer11_inference_completion",
    "layer12_serialization",
]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: str | Path, value: dict[str, Any], lock: threading.Lock | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, default=str) + "\n"
    if lock is None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        return
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned[:100] or "document"


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class LoggedBackend:
    """Thread-safe logger around the existing OpenAI-compatible backend."""

    def __init__(self, backend: OpenAICompatibleBackend, log_dir: str | Path) -> None:
        self.backend = backend
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "responses").mkdir(parents=True, exist_ok=True)
        self.calls_path = self.log_dir / "llm_calls.jsonl"
        self.errors_path = self.log_dir / "llm_errors.jsonl"
        self.lock = threading.Lock()
        self.call_index = 0

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0, **_: Any) -> str:
        with self.lock:
            self.call_index += 1
            call_index = self.call_index
        started = time.time()
        meta = {
            "call_index": call_index,
            "model": model,
            "temperature": temperature,
            "message_count": len(messages),
            "system_chars": sum(len(m.get("content", "")) for m in messages if m.get("role") == "system"),
            "user_chars": sum(len(m.get("content", "")) for m in messages if m.get("role") != "system"),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            response = self.backend.chat(model=model, messages=messages, temperature=temperature)
            response_path = self.log_dir / "responses" / f"response_{call_index:04d}.txt"
            response_path.write_text(response, encoding="utf-8")
            append_jsonl(self.calls_path, {
                **meta,
                "status": "ok",
                "elapsed_seconds": round(time.time() - started, 3),
                "response_chars": len(response),
                "response_path": str(response_path),
            }, self.lock)
            return response
        except Exception as exc:
            append_jsonl(self.errors_path, {
                **meta,
                "status": "error",
                "elapsed_seconds": round(time.time() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }, self.lock)
            raise

    @staticmethod
    def extract_json(text: str) -> Any:
        return OpenAICompatibleBackend.extract_json(text)


class OntologyOnlyRAGAdapter:
    """Balanced deterministic retrieval over the supplied DocRED ontology.

    The repository's ``OntologySpace.retrieve`` currently appends classes before
    properties and then truncates the combined list to ``top_k``. With a normal
    ontology this can hide every property from relation-oriented prompts. This
    notebook adapter keeps the same NeoOLAF grounding contracts but retrieves
    classes and properties separately, guaranteeing property evidence for
    Layers 2-4 without changing ``src/neoolaf`` or adding another LLM task.
    """

    name = "balanced_ontology_only"

    def __init__(
        self,
        seed_ontology: Any,
        log_path: str | Path,
        top_k: int = 8,
        query_expansions: dict[str, list[str]] | None = None,
    ) -> None:
        self.seed_ontology = seed_ontology
        self.space = OntologySpace(seed_ontology)
        self.log_path = Path(log_path)
        self.top_k = max(2, int(top_k))
        self.query_expansions = {
            str(key).lower().strip(): [str(value) for value in values]
            for key, values in (query_expansions or {}).items()
            if key and isinstance(values, list)
        }
        self.lock = threading.Lock()

    def _expanded_queries(self, query: str) -> list[str]:
        original = str(query or "").strip()
        lowered = original.lower()
        values: list[str] = []
        # Put profile-provided ontology labels first so exact schema matches are
        # not displaced by weak lexical neighbors of the surface phrase.
        for trigger, expansions in self.query_expansions.items():
            if trigger and trigger in lowered:
                values.extend(expansions)
        values.append(original)
        return list(dict.fromkeys(value for value in values if value))

    @staticmethod
    def _class_item(cls: Any) -> RetrievedItem:
        return RetrievedItem(
            source="ontology",
            content=f"Class: {cls.label}. {cls.description or ''}".strip(),
            metadata={
                "type": "class",
                "uri": cls.uri,
                "label": cls.label,
                "alt_labels": getattr(cls, "alt_labels", []),
                "parents": cls.parent_uris,
                "children": cls.child_uris,
            },
            reference=cls.uri,
        )

    @staticmethod
    def _property_item(prop: Any) -> RetrievedItem:
        property_id = str(prop.uri).rstrip("/").split("/")[-1]
        return RetrievedItem(
            source="ontology",
            content=(
                f"Property: {property_id} : {prop.label}. {prop.description or ''} "
                f"Domain URIs: {', '.join(prop.domain_uris or []) or 'unspecified'}. "
                f"Range URIs: {', '.join(prop.range_uris or []) or 'unspecified'}."
            ).strip(),
            metadata={
                "type": "property",
                "uri": prop.uri,
                "property_id": property_id,
                "label": prop.label,
                "alt_labels": getattr(prop, "alt_labels", []),
                "domain_uris": prop.domain_uris,
                "range_uris": prop.range_uris,
                "parents": prop.parent_uris,
                "children": prop.child_uris,
            },
            reference=prop.uri,
        )

    def _items(
        self,
        query: str,
        top_k: int | None = None,
        layer_name: str | None = None,
    ) -> tuple[list[RetrievedItem], list[str]]:
        top_k = max(2, int(top_k or self.top_k))
        relation_focused = str(layer_name or "").endswith(
            ("candidate_relation_extraction", "concept_relation_induction")
        )
        property_budget = max(1, int(round(top_k * (0.75 if relation_focused else 0.5))))
        class_budget = max(1, top_k - property_budget)

        expanded = self._expanded_queries(query)
        retriever = self.space.retriever
        if retriever is None:
            return [], expanded

        classes: list[Any] = []
        properties: list[Any] = []
        # First collect exact matches for every profile expansion.
        for expanded_query in expanded:
            normalized_query = expanded_query.lower().strip()
            for uri in self.seed_ontology.property_uris_by_label.get(normalized_query, []):
                prop = self.seed_ontology.properties_by_uri.get(uri)
                if prop is not None:
                    properties.append(prop)
            for uri in self.seed_ontology.class_uris_by_label.get(normalized_query, []):
                cls = self.seed_ontology.classes_by_uri.get(uri)
                if cls is not None:
                    classes.append(cls)
        # Then fill remaining slots with fuzzy lexical neighbors.
        for expanded_query in expanded:
            properties.extend(retriever.nearest_properties(expanded_query, top_k=property_budget))
            classes.extend(retriever.nearest_classes(expanded_query, top_k=class_budget))

        def dedup(values: list[Any], budget: int) -> list[Any]:
            result = []
            seen = set()
            for value in values:
                uri = str(value.uri)
                if uri in seen:
                    continue
                seen.add(uri)
                result.append(value)
                if len(result) >= budget:
                    break
            return result

        selected_properties = dedup(properties, property_budget)
        selected_classes = dedup(classes, class_budget)
        items = [self._property_item(prop) for prop in selected_properties]
        items.extend(self._class_item(cls) for cls in selected_classes)
        return items[:top_k], expanded

    def _log(
        self,
        *,
        layer_name: str | None,
        query: str,
        expanded_queries: list[str],
        top_k: int,
        items: list[RetrievedItem],
    ) -> None:
        append_jsonl(self.log_path, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layer_name": layer_name,
            "query": query,
            "expanded_queries": expanded_queries,
            "top_k": top_k,
            "result_count": len(items),
            "result_types": [item.metadata.get("type") for item in items],
            "property_ids": [
                item.metadata.get("property_id")
                for item in items
                if item.metadata.get("type") == "property"
            ],
            "references": [item.reference for item in items],
            "labels": [item.metadata.get("label") for item in items],
        }, self.lock)

    def retrieve(self, request: RAGRequest | GroundingRequest) -> RAGResult:
        query = str(getattr(request, "query", "") or "")
        top_k = int(getattr(request, "top_k", self.top_k) or self.top_k)
        layer_name = getattr(request, "layer_name", None)
        items, expanded = self._items(query, top_k, layer_name)
        context = "\n".join(item.content for item in items)
        sources = [
            {
                "space": item.source,
                "text": item.content,
                "score": item.score,
                "reference": item.reference,
                "metadata": item.metadata,
            }
            for item in items
        ]
        self._log(
            layer_name=layer_name,
            query=query,
            expanded_queries=expanded,
            top_k=top_k,
            items=items,
        )
        return RAGResult(context=context, sources=sources, metadata={"backend": self.name})

    def ground(self, request: GroundingRequest | RAGRequest) -> GroundingResult:
        if isinstance(request, GroundingRequest):
            grounding_request = request
        else:
            payload = dict(getattr(request, "metadata", {}) or {})
            if getattr(request, "document_id", None):
                payload.setdefault("document_id", request.document_id)
            grounding_request = GroundingRequest(
                layer_name=getattr(request, "layer_name", "unknown_layer"),
                query=getattr(request, "query", ""),
                payload=payload,
                preferred_sources=list(getattr(request, "allowed_spaces", []) or ["ontology"]),
                top_k=int(getattr(request, "top_k", self.top_k) or self.top_k),
            )
        items, expanded = self._items(
            grounding_request.query,
            grounding_request.top_k,
            grounding_request.layer_name,
        )
        self._log(
            layer_name=grounding_request.layer_name,
            query=grounding_request.query,
            expanded_queries=expanded,
            top_k=grounding_request.top_k,
            items=items,
        )
        return GroundingResult(
            request=grounding_request,
            selected_sources=["ontology"] if items else [],
            retrieved_items=items,
            grounding_summary="\n".join(item.content for item in items),
            merged_context={"backend": self.name, "expanded_queries": expanded},
        )

def choose_chunk_size(text: str, max_safe_chars: int = 24000) -> int:
    """Prefer one whole-document chunk while retaining a safety ceiling."""
    length = len(text)
    if length <= max_safe_chars:
        return max(4096, length + 512)
    return max_safe_chars


def build_document(record: dict[str, Any], source_path: str | Path) -> Document:
    return Document(
        doc_id=str(record["document_id"]),
        source_path=str(source_path),
        raw_text=str(record["text"]),
    )


def build_pipeline(
    *,
    backend: LoggedBackend,
    rag_adapter: OntologyOnlyRAGAdapter,
    profile_config: dict[str, Any],
    chunk_size: int,
    workers: int = 4,
    retry_failed_calls: int = 2,
    retry_sleep_seconds: float = 2.0,
    verbose: bool = True,
) -> Pipeline:
    """Build the standard 13-layer sequence from existing NeoOLAF components."""
    workers = max(1, int(workers))
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
            backend,
            max_chunks=1,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_backend=rag_adapter,
            max_concurrency=1,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
            rag_enabled=False,
        ),
        CandidateEnrichmentLayer(
            backend,
            wikipedia_source=OfflineWikipediaSource(),
            wikidata_source=OfflineWikidataSource(),
            web_search_source=OfflineWebSearchSource(),
            max_expressions=None,
            use_web_search=False,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        CandidateTypingResolutionLayer(
            backend,
            max_expressions=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        CandidateRelationExtractionLayer(
            backend,
            max_relation_mentions=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        CandidateTripleGenerationLayer(
            max_assertions=None,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
            save_intermediate=True,
            verbose=verbose,
        ),
        ConceptRelationInductionLayer(
            backend,
            max_concept_inputs=None,
            max_relation_inputs=None,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
        ),
        HierarchisationLayer(
            backend,
            max_concept_pairs=None,
            max_relation_pairs=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        AxiomSchemataExtractionLayer(
            backend,
            max_relation_schema_inputs=None,
            max_subclass_inputs=None,
            temperature=0.0,
            rag_adapter=rag_adapter,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        GeneralAxiomExtractionLayer(
            backend,
            max_schema_inputs=None,
            max_description_inputs=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        ValidationReasoningLayer(
            max_triples=None,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        InferenceCompletionLayer(
            max_inferred_triples=None,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
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
    run_dir: str | Path,
    model_name: str,
    api_key: str,
    host: str = "https://openrouter.ai/api/v1",
    workers: int = 4,
    max_tokens: int = 8192,
    request_timeout: int = 600,
    reasoning_effort: str = "minimal",
    verbose: bool = True,
) -> PipelineState:
    """Run one document through all 13 native NeoOLAF layers."""
    project_root = Path(project_root).resolve()
    input_jsonl = Path(input_jsonl).resolve()
    ontology_path = Path(ontology_path).resolve()
    profile_path = Path(profile_path).resolve()
    guidance_path = Path(guidance_path).resolve()
    run_dir = Path(run_dir).resolve()
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
        raise ValueError("OPENROUTER_API_KEY is not set. The key is read from the environment and is never written to artifacts.")

    profile = load_document_profile(profile_path=profile_path)
    guidance = load_user_guidance(str(guidance_path))
    seed_ontology = SeedOntologyLoader().load(str(ontology_path))
    if len(seed_ontology.properties_by_uri) < 90:
        raise RuntimeError(
            f"Only {len(seed_ontology.properties_by_uri)} ontology properties were loaded. "
            "Use docred_redocred_neoolaf_compatible.ttl, which exposes all 96 rdf:Property predicates to the current loader."
        )

    chunk_size = choose_chunk_size(record["text"], int(profile.get("chunking.max_safe_chunk_chars", 24000)))
    core_backend = OpenAICompatibleBackend(
        backend_name="openrouter",
        host=host,
        api_key=api_key,
        timeout=request_timeout,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=True,
    )
    backend = LoggedBackend(core_backend, logs_dir)
    rag_adapter = OntologyOnlyRAGAdapter(
        seed_ontology,
        log_path=logs_dir / "ontology_retrieval.jsonl",
        top_k=int(profile.get("rag.top_k", 8)),
        query_expansions=profile.get("rag.query_expansions", {}) or {},
    )
    pipeline = build_pipeline(
        backend=backend,
        rag_adapter=rag_adapter,
        profile_config=profile.to_state_dict(),
        chunk_size=chunk_size,
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
        profile_config=profile.to_state_dict(),
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
        "ontology_path": str(ontology_path),
        "ontology_classes": len(seed_ontology.classes_by_uri),
        "ontology_properties": len(seed_ontology.properties_by_uri),
        "input_has_gold": False,
        "chunk_size": chunk_size,
        "whole_document_single_chunk_expected": len(record["text"]) <= chunk_size,
        "workers": workers,
        "max_tokens": max_tokens,
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


def state_counts(state: PipelineState) -> dict[str, int]:
    fields = [
        "linguistic_expressions", "enriched_expressions", "entity_candidates",
        "relation_candidates", "attribute_candidates", "event_candidates",
        "candidate_relation_assertions", "candidate_triples", "concept_candidates",
        "ontology_relation_candidates", "concept_hierarchy_links", "relation_hierarchy_links",
        "axiom_schema_candidates", "general_axiom_candidates", "completion_candidates",
    ]
    counts = {name: len(getattr(state, name, []) or []) for name in fields}
    counts["validation_issues"] = len(getattr(getattr(state, "validation_report", None), "issues", []) or [])
    counts["reasoning_inferred_triples"] = len(getattr(getattr(state, "reasoning_report", None), "inferred_triples", []) or [])
    return counts


def load_layer_states(run_dir: str | Path) -> list[tuple[int, str, PipelineState]]:
    run_dir = Path(run_dir)
    states: list[tuple[int, str, PipelineState]] = []
    for index, name in enumerate(LAYER_NAMES):
        path = run_dir / name / "state.json"
        if path.is_file():
            states.append((index, name, PipelineState.load_json(str(path))))
    return states


def write_layer_summary(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    rows: list[dict[str, Any]] = []
    for index, name, state in load_layer_states(run_dir):
        metadata_path = run_dir / name / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        rows.append({
            "layer_index": index,
            "layer_name": name,
            "elapsed_seconds": metadata.get("elapsed_seconds"),
            **state_counts(state),
        })
    path = run_dir / "analysis" / "layer_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_json(run_dir / "analysis" / "layer_summary.json", rows)
    return rows


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def gold_entity_aliases(gold: dict[str, Any]) -> dict[str, set[str]]:
    return {
        entity_id: {norm(m.get("trigger_word")) for m in payload.get("mentions", []) if m.get("trigger_word")}
        for entity_id, payload in gold.get("entities", {}).items()
    }


def candidate_aliases(candidate: Any) -> set[str]:
    values = {getattr(candidate, "canonical_label", "")}
    values.update(getattr(candidate, "aliases", []) or [])
    values.update(getattr(candidate, "synonyms", []) or [])
    values.update(getattr(candidate, "lexical_variants", []) or [])
    for mention in getattr(candidate, "mentions", []) or []:
        values.add(getattr(mention, "text", ""))
    return {norm(value) for value in values if norm(value)}


def align_candidate(candidate: Any, aliases: dict[str, set[str]]) -> str | None:
    c_aliases = candidate_aliases(candidate)
    exact = [entity_id for entity_id, gold_alias in aliases.items() if c_aliases.intersection(gold_alias)]
    if len(exact) == 1:
        return exact[0]
    # Conservative containment fallback for aliases such as "Athens metropolitan area".
    containment = []
    for entity_id, gold_alias in aliases.items():
        if any(a and b and (a in b or b in a) for a in c_aliases for b in gold_alias):
            containment.append(entity_id)
    return containment[0] if len(set(containment)) == 1 else None


def align_text_values(values: Iterable[str], aliases: dict[str, set[str]]) -> set[str]:
    """Align raw Layer 1/2 surface strings to gold IDs for diagnostics only."""
    normalized_values = {norm(value) for value in values if norm(value)}
    result: set[str] = set()
    for entity_id, gold_aliases in aliases.items():
        if normalized_values.intersection(gold_aliases):
            result.add(entity_id)
            continue
        if any(a and b and (a in b or b in a) for a in normalized_values for b in gold_aliases):
            result.add(entity_id)
    return result


def relation_lookup(catalog_path: str | Path, aliases_path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    catalog = read_json(catalog_path)["relations"]
    aliases = read_json(aliases_path)
    id_to_label = {item["relation_id"]: item["label"] for item in catalog}
    label_to_id: dict[str, str] = {}
    for relation_id, labels in aliases.items():
        label_to_id[norm(relation_id)] = relation_id
        for label in labels:
            label_to_id[norm(label)] = relation_id
            label_to_id[norm(f"{relation_id} : {label}")] = relation_id
    return id_to_label, label_to_id


def map_predicate(label: str, hints: Iterable[str], label_to_id: dict[str, str]) -> str | None:
    for value in [label, *list(hints or [])]:
        match = re.search(r"\bP\d+\b", str(value), re.IGNORECASE)
        if match:
            return match.group(0).upper()
    return label_to_id.get(norm(label))


def native_predictions(state: PipelineState, gold: dict[str, Any], catalog_path: str | Path, aliases_path: str | Path) -> list[dict[str, Any]]:
    aliases = gold_entity_aliases(gold)
    _, label_to_id = relation_lookup(catalog_path, aliases_path)
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in [
            *(state.entity_candidates or []), *(state.event_candidates or []),
            *(state.attribute_candidates or []), *(state.relation_candidates or []),
        ]
    }
    relation_by_id = {candidate.candidate_id: candidate for candidate in state.relation_candidates or []}
    rows: list[dict[str, Any]] = []
    for triple in state.candidate_triples or []:
        subject = candidate_by_id.get(triple.subject_id)
        obj = candidate_by_id.get(triple.object_id)
        relation_candidate = relation_by_id.get(triple.predicate_id)
        relation_id = map_predicate(
            triple.predicate_label,
            getattr(relation_candidate, "ontology_hints", []) if relation_candidate else [],
            label_to_id,
        )
        rows.append({
            "triple_id": triple.triple_id,
            "subject_label": triple.subject_label,
            "predicate_label": triple.predicate_label,
            "object_label": triple.object_label,
            "head_id": align_candidate(subject, aliases) if subject else None,
            "relation_id": relation_id,
            "tail_id": align_candidate(obj, aliases) if obj else None,
            "fully_mapped": bool(subject and obj and relation_id and align_candidate(subject, aliases) and align_candidate(obj, aliases)),
            "confidence": triple.confidence,
            "justification": triple.justification,
        })
    return rows


def gold_triples(gold: dict[str, Any]) -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()
    for relation_key, pairs in gold.get("relations", {}).items():
        relation_id = relation_key.split(":", 1)[0].strip()
        triples.update((head, relation_id, tail) for head, tail in pairs)
    return triples


def strict_evaluate(predictions: list[dict[str, Any]], gold: dict[str, Any]) -> dict[str, Any]:
    predicted = {
        (row["head_id"], row["relation_id"], row["tail_id"])
        for row in predictions
        if row.get("fully_mapped")
    }
    expected = gold_triples(gold)
    tp = predicted.intersection(expected)
    fp = predicted.difference(expected)
    fn = expected.difference(predicted)
    precision = len(tp) / len(predicted) if predicted else 0.0
    recall = len(tp) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted": len(predicted), "gold": len(expected), "true_positive": len(tp),
        "false_positive": len(fp), "false_negative": len(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": sorted(tp), "fp": sorted(fp), "fn": sorted(fn),
    }


def relation_candidates_by_id(state: PipelineState, label_to_id: dict[str, str]) -> set[str]:
    result = set()
    for candidate in state.relation_candidates or []:
        mapped = map_predicate(candidate.canonical_label, candidate.ontology_hints, label_to_id)
        if mapped:
            result.add(mapped)
    return result


def write_gold_trace(run_dir: str | Path, gold: dict[str, Any], catalog_path: str | Path, aliases_path: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    aliases = gold_entity_aliases(gold)
    id_to_label, label_to_id = relation_lookup(catalog_path, aliases_path)
    states = {index: state for index, _, state in load_layer_states(run_dir)}
    final_state = states[max(states)]
    rows: list[dict[str, Any]] = []

    layer1 = states.get(1, final_state)
    layer2 = states.get(2, final_state)
    layer3 = states.get(3, final_state)

    layer1_nodes = align_text_values(
        [expr.text for expr in (layer1.linguistic_expressions or [])], aliases
    )
    layer2_nodes = align_text_values(
        [item.base_expression.text for item in (layer2.enriched_expressions or [])], aliases
    )

    node_candidates = [
        *(layer3.entity_candidates or []),
        *(layer3.event_candidates or []),
        *(layer3.attribute_candidates or []),
    ]
    aligned_nodes = {align_candidate(candidate, aliases) for candidate in node_candidates}
    aligned_nodes.discard(None)
    available_predicates = relation_candidates_by_id(layer3, label_to_id)

    layer4 = states.get(4, final_state)
    layer5 = states.get(5, final_state)
    pred4 = []
    cand4 = {
        c.candidate_id: c
        for c in [
            *(layer4.entity_candidates or []),
            *(layer4.event_candidates or []),
            *(layer4.attribute_candidates or []),
        ]
    }
    rel4 = {c.candidate_id: c for c in layer4.relation_candidates or []}
    for assertion in layer4.candidate_relation_assertions or []:
        src = cand4.get(assertion.source_candidate_id)
        dst = cand4.get(assertion.target_candidate_id)
        rel = rel4.get(assertion.relation_candidate_id)
        pred4.append((
            align_candidate(src, aliases) if src else None,
            map_predicate(
                assertion.relation_label,
                getattr(rel, "ontology_hints", []) if rel else [],
                label_to_id,
            ),
            align_candidate(dst, aliases) if dst else None,
        ))
    pred5_rows = native_predictions(layer5, gold, catalog_path, aliases_path)
    pred5 = {
        (row["head_id"], row["relation_id"], row["tail_id"])
        for row in pred5_rows
        if row["fully_mapped"]
    }

    for head, relation_id, tail in sorted(gold_triples(gold)):
        head_l1 = head in layer1_nodes
        tail_l1 = tail in layer1_nodes
        head_l2 = head in layer2_nodes
        tail_l2 = tail in layer2_nodes
        head_l3 = head in aligned_nodes
        tail_l3 = tail in aligned_nodes
        predicate_found = relation_id in available_predicates
        assertion_found = (head, relation_id, tail) in pred4
        triple_found = (head, relation_id, tail) in pred5

        if not head_l1 or not tail_l1:
            first_failure = "layer01_expression_extraction"
        elif not head_l2 or not tail_l2:
            first_failure = "layer02_enrichment_survival"
        elif not head_l3 or not tail_l3:
            first_failure = "layer03_endpoint_resolution"
        elif not predicate_found:
            first_failure = "layer03_predicate_typing_or_ontology_linking"
        elif not assertion_found:
            first_failure = "layer04_endpoint_assignment"
        elif not triple_found:
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
            "predicate_available_layer03": predicate_found,
            "assertion_found_layer04": assertion_found,
            "triple_found_layer05": triple_found,
            "first_failure": first_failure,
        })

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "gold_relation_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(analysis_dir / "gold_relation_trace.json", rows)
    return rows

def analyze_run(*, run_dir: str | Path, gold_jsonl: str | Path, catalog_path: str | Path, aliases_path: str | Path) -> dict[str, Any]:
    run_dir=Path(run_dir)
    gold_rows=read_jsonl(gold_jsonl)
    if len(gold_rows)!=1:
        raise ValueError('Analysis expects exactly one gold document.')
    gold=gold_rows[0]
    layer_rows=write_layer_summary(run_dir)
    states=load_layer_states(run_dir)
    if not states:
        raise FileNotFoundError(f'No layer state artifacts found in {run_dir}')
    final_state=states[-1][2]
    predictions=native_predictions(final_state,gold,catalog_path,aliases_path)
    analysis_dir=run_dir/'analysis'; analysis_dir.mkdir(parents=True,exist_ok=True)
    write_json(analysis_dir/'native_mapped_predictions.json',predictions)
    with (analysis_dir/'native_mapped_predictions.jsonl').open('w',encoding='utf-8') as handle:
        for row in predictions: handle.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
    evaluation=strict_evaluate(predictions,gold)
    write_json(analysis_dir/'strict_docred_evaluation.json',evaluation)
    trace=write_gold_trace(run_dir,gold,catalog_path,aliases_path)
    summary={
        'document_id':gold['document_id'],
        'layer_summary':layer_rows,
        'strict_evaluation':evaluation,
        'mapped_native_triples':sum(1 for row in predictions if row.get('fully_mapped')),
        'unmapped_native_triples':sum(1 for row in predictions if not row.get('fully_mapped')),
        'failure_counts':{},
    }
    for row in trace:
        summary['failure_counts'][row['first_failure']]=summary['failure_counts'].get(row['first_failure'],0)+1
    write_json(analysis_dir/'analysis_summary.json',summary)
    return summary

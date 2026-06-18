#!/usr/bin/env python
"""Run NeoOLAF on RAGTree-style JSONL datasets.

This script is intentionally a thin benchmark wrapper around the NeoOLAF
library. It keeps one independent NeoOLAF pipeline execution per document,
while adding dataset-level conveniences used by the RAGTree comparison setup:

- RAGTree JSONL loading and type filtering;
- one large document chunk by default, for "no chunk" benchmark mode;
- user guidance loaded from JSON;
- optional few-shot examples extracted from the dataset;
- one fixed ontology per dataset;
- canonical JSONL prediction export;
- parallel document execution through --document-workers;
- existing NeoOLAF intra-document/chunk worker support through --max-workers.

The goal is not to create a new method. The goal is to run the same NeoOLAF
code/library with a benchmark profile and document-level parallelism.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# Allow this script to be called from notebooks located in sibling folders,
# e.g. ../../experiments/methods/run_neoolaf.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neoolaf.core.execution_config import ExecutionConfig
from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.documents import Document, DocumentChunk
from neoolaf.domain.user_guidance import (
    NegativeExample,
    PromotionExample,
    RelationExample,
    TypingExample,
    UserGuidance,
)
from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
from neoolaf.layers.layer01_linguistic_expression_extraction.component import (
    LinguisticExpressionExtractionLayer,
)
from neoolaf.layers.layer02_candidate_enrichment.component import CandidateEnrichmentLayer
from neoolaf.layers.layer03_candidate_typing_resolution.component import (
    CandidateTypingResolutionLayer,
)
from neoolaf.layers.layer04_candidate_relation_extraction.component import (
    CandidateRelationExtractionLayer,
)
from neoolaf.layers.layer05_candidate_triple_generation.component import (
    CandidateTripleGenerationLayer,
)
from neoolaf.layers.layer06_concept_relation_induction.component import (
    ConceptRelationInductionLayer,
)
from neoolaf.layers.layer07_hierarchisation.component import HierarchisationLayer
from neoolaf.layers.layer08_axiom_schemata_extraction.component import (
    AxiomSchemataExtractionLayer,
)
from neoolaf.layers.layer09_general_axiom_extraction.component import (
    GeneralAxiomExtractionLayer,
)
from neoolaf.layers.layer10_validation_reasoning.component import ValidationReasoningLayer
from neoolaf.layers.layer11_inference_completion.component import InferenceCompletionLayer
from neoolaf.layers.layer12_serialization.component import SerializationLayer
from neoolaf.ontology.loader import SeedOntologyLoader


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

class OpenAICompatibleBackend:
    """Small backend implementing the interface expected by NeoOLAF layers.

    NeoOLAF's current layer constructors expect an object with:
    - chat(model, messages, temperature) -> str
    - extract_json(text) -> dict/list

    This backend works with OpenAI-compatible APIs such as OpenRouter and vLLM.
    """

    def __init__(
        self,
        *,
        backend_name: str,
        host: str,
        api_key: str,
        timeout: int = 300,
        max_tokens: int = 2048,
    ) -> None:
        self.backend_name = backend_name
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _chat_url(self) -> str:
        """Normalize host into a chat completions URL."""
        host = self.host
        if host.endswith("/chat/completions"):
            return host
        if host.endswith("/v1"):
            return f"{host}/chat/completions"
        if host.endswith("/api"):
            return f"{host}/v1/chat/completions"
        return f"{host}/v1/chat/completions"

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        """Call an OpenAI-compatible chat endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            self._chat_url(),
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned by backend {self.backend_name}: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text")
        if content is None:
            raise RuntimeError(f"No content returned by backend {self.backend_name}: {data}")
        return str(content).strip()

    @staticmethod
    def extract_json(text: str) -> Any:
        """Robust JSON extractor shared by all layers."""
        if text is None:
            raise ValueError("Could not parse JSON from model output because it is None.")
        text = str(text).strip()
        if not text:
            raise ValueError("Could not parse JSON from model output because it is empty.")

        fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return json.loads(fenced.group(1))

        fenced_any = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if fenced_any:
            return json.loads(fenced_any.group(1))

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Prefer the earliest valid top-level array/object candidate.
        candidates = []
        for pattern in [r"(\[.*\])", r"(\{.*\})"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                candidates.append(match.group(1))
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError("Could not parse JSON from model output.")


# ---------------------------------------------------------------------------
# Dataset/guidance helpers
# ---------------------------------------------------------------------------

def safe_filename(value: str, max_len: int = 80) -> str:
    """Create a stable filesystem-safe identifier."""
    value = str(value or "document").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:16]
    if not cleaned:
        cleaned = "document"
    return f"{cleaned[:max_len]}_{digest}"


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
    return records


def filter_records(records: List[Dict[str, Any]], type_filter: str) -> List[Dict[str, Any]]:
    """Filter dataset rows by type/split, preserving 'all'."""
    if not type_filter or type_filter.lower() == "all":
        return records
    wanted = type_filter.lower()
    return [r for r in records if str(r.get("type") or r.get("split") or "").lower() == wanted]


def flatten_sentences(sentences: Any) -> str:
    """Flatten common sentence representations into document text."""
    if not sentences:
        return ""
    lines: List[str] = []
    for idx, sentence in enumerate(sentences):
        if isinstance(sentence, str):
            text = sentence
        elif isinstance(sentence, list):
            text = " ".join(str(tok) for tok in sentence)
        elif isinstance(sentence, dict):
            if "text" in sentence:
                text = str(sentence["text"])
            elif "tokens" in sentence:
                text = " ".join(str(tok) for tok in sentence.get("tokens") or [])
            else:
                text = json.dumps(sentence, ensure_ascii=False)
        else:
            text = str(sentence)
        lines.append(f"[{idx}] {text}")
    return "\n".join(lines)


def document_text_from_record(record: Dict[str, Any]) -> str:
    """Recover the best full-document text field from a normalized row."""
    for key in ["text", "raw_text", "document", "content", "abstract"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    sentence_text = flatten_sentences(record.get("sentences") or record.get("sents"))
    if sentence_text.strip():
        return sentence_text.strip()
    tokens = record.get("tokens")
    if isinstance(tokens, list):
        return " ".join(str(t) for t in tokens)
    return json.dumps(record, ensure_ascii=False)


def document_id_from_record(record: Dict[str, Any], index: int) -> str:
    """Recover a stable document identifier from a normalized row."""
    for key in ["document_id", "doc_id", "id", "pmid", "article_id"]:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"doc_{index:06d}"


def title_from_record(record: Dict[str, Any], doc_id: str) -> str:
    """Recover a readable document title."""
    for key in ["title", "name"]:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return doc_id


def record_to_document(record: Dict[str, Any], index: int) -> Document:
    """Convert one normalized JSONL row into NeoOLAF's Document object."""
    doc_id = document_id_from_record(record, index)
    title = title_from_record(record, doc_id)
    text = document_text_from_record(record)
    raw_text = f"{title}\n\n{text}" if title and title not in text[:200] else text

    doc = Document(
        doc_id=doc_id,
        source_path=f"{safe_filename(doc_id)}.jsonl",
        raw_text=raw_text,
    )
    doc.content_blocks = [
        {
            "type": "normalized_jsonl_document",
            "title": title,
            "document_id": doc_id,
            "text": text,
            "metadata": {
                "dataset_type": record.get("type") or record.get("split"),
                "original_keys": sorted(record.keys()),
            },
        }
    ]
    return doc


def load_user_guidance(path: Optional[str]) -> Optional[UserGuidance]:
    """Load a UserGuidance JSON file into the NeoOLAF dataclass format."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    guidance = UserGuidance(
        domain_focus=data.get("domain_focus"),
        abstraction_level=data.get("abstraction_level"),
        priority_relations=list(data.get("priority_relations") or []),
        population_policy=data.get("population_policy"),
        event_modeling_preference=data.get("event_modeling_preference"),
        ontology_depth=data.get("ontology_depth", "balanced"),
        promotion_min_confidence=float(data.get("promotion_min_confidence", 0.5)),
        hierarchy_min_confidence=float(data.get("hierarchy_min_confidence", 0.5)),
        concept_promotion_bias=float(data.get("concept_promotion_bias", 0.5)),
    )

    for item in data.get("typing_examples") or []:
        guidance.typing_examples.append(TypingExample(**item))
    for item in data.get("relation_examples") or []:
        guidance.relation_examples.append(RelationExample(**item))
    for item in data.get("promotion_examples") or []:
        guidance.promotion_examples.append(PromotionExample(**item))
    for item in data.get("negative_examples") or []:
        guidance.negative_examples.append(NegativeExample(**item))
    return guidance


def relation_items_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recover relation-like items from common dataset/prediction schemas."""
    for key in ["relations", "gold_relations", "labels", "triples"]:
        value = record.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    prediction = record.get("prediction")
    if isinstance(prediction, dict) and isinstance(prediction.get("relations"), list):
        return [x for x in prediction["relations"] if isinstance(x, dict)]
    return []


def entity_items_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recover entity-like items from common dataset/prediction schemas."""
    for key in ["entities", "mentions", "vertexSet"]:
        value = record.get(key)
        if isinstance(value, list):
            # DocRED vertexSet is a list of mention clusters.
            if key == "vertexSet":
                entities: List[Dict[str, Any]] = []
                for cluster in value:
                    if isinstance(cluster, list) and cluster:
                        first = cluster[0]
                        if isinstance(first, dict):
                            label = first.get("name") or first.get("text")
                            ent_type = first.get("type", "entity")
                            if label:
                                entities.append({"label": label, "type": ent_type})
                return entities
            return [x for x in value if isinstance(x, dict)]
    prediction = record.get("prediction")
    if isinstance(prediction, dict) and isinstance(prediction.get("entities"), list):
        return [x for x in prediction["entities"] if isinstance(x, dict)]
    return []


def add_few_shot_examples_from_dataset(
    guidance: Optional[UserGuidance],
    records: List[Dict[str, Any]],
    *,
    source_type: str,
    k: int,
) -> UserGuidance:
    """Add compact few-shot examples derived from dataset rows."""
    guidance = copy.deepcopy(guidance) if guidance is not None else UserGuidance()
    candidates = filter_records(records, source_type)
    if k is not None and k > 0:
        candidates = candidates[:k]

    for record in candidates:
        text = document_text_from_record(record)
        short_text = text[:600].replace("\n", " ")

        for entity in entity_items_from_record(record)[:20]:
            label = entity.get("label") or entity.get("name") or entity.get("text")
            ent_type = entity.get("type") or entity.get("entity_type") or "entity"
            if label:
                guidance.typing_examples.append(
                    TypingExample(
                        text=str(label),
                        expected_type=str(ent_type),
                        explanation="Few-shot entity/type example extracted from the dataset.",
                    )
                )

        for rel in relation_items_from_record(record)[:30]:
            head = rel.get("head") or rel.get("subject") or rel.get("source") or rel.get("h")
            tail = rel.get("tail") or rel.get("object") or rel.get("target") or rel.get("t")
            relation = rel.get("relation") or rel.get("label") or rel.get("predicate") or rel.get("r")
            evidence = rel.get("evidence") or rel.get("evidence_text") or short_text
            if isinstance(head, dict):
                head = head.get("label") or head.get("name") or head.get("text") or head.get("id")
            if isinstance(tail, dict):
                tail = tail.get("label") or tail.get("name") or tail.get("text") or tail.get("id")
            if head and tail and relation:
                guidance.relation_examples.append(
                    RelationExample(
                        text=str(evidence)[:600],
                        source_label=str(head),
                        relation_label=str(relation),
                        target_label=str(tail),
                        explanation="Few-shot relation example extracted from the dataset.",
                    )
                )

    return guidance


# ---------------------------------------------------------------------------
# Pipeline construction and output conversion
# ---------------------------------------------------------------------------

def build_backend(args: argparse.Namespace) -> OpenAICompatibleBackend:
    """Create a fresh backend instance for one document worker."""
    return OpenAICompatibleBackend(
        backend_name=args.backend_name,
        host=args.host,
        api_key=args.api_key,
        timeout=args.request_timeout,
        max_tokens=args.max_tokens,
    )


def build_pipeline(args: argparse.Namespace, backend: OpenAICompatibleBackend) -> Pipeline:
    """Build the full NeoOLAF pipeline using the existing library layers."""
    layers = [
        PreprocessingLayer(
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            enable_chunking=True,
            translate=False,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        LinguisticExpressionExtractionLayer(
            backend,
            max_chunks=args.max_chunks,
            temperature=args.temperature,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        CandidateEnrichmentLayer(
            backend,
            max_expressions=args.max_expressions,
            use_web_search=not args.no_web_search,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        CandidateTypingResolutionLayer(
            backend,
            max_expressions=args.max_expressions,
            temperature=args.temperature,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        CandidateRelationExtractionLayer(
            backend,
            max_relation_mentions=args.max_relation_mentions,
            temperature=args.temperature,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        CandidateTripleGenerationLayer(
            max_assertions=args.max_relation_mentions,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        ConceptRelationInductionLayer(
            backend,
            max_concept_inputs=args.max_concept_inputs,
            max_relation_inputs=args.max_relation_inputs,
            temperature=args.temperature,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        HierarchisationLayer(
            backend,
            max_concept_pairs=args.max_concept_pairs,
            max_relation_pairs=args.max_relation_pairs,
            temperature=args.temperature,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        AxiomSchemataExtractionLayer(
            backend,
            max_relation_schema_inputs=args.max_relation_schema_inputs,
            max_subclass_inputs=args.max_subclass_inputs,
            temperature=args.temperature,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        GeneralAxiomExtractionLayer(
            backend,
            max_schema_inputs=args.max_schema_inputs,
            max_description_inputs=args.max_description_inputs,
            temperature=args.temperature,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        ValidationReasoningLayer(
            max_triples=args.max_triples,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        InferenceCompletionLayer(
            max_inferred_triples=args.max_inferred_triples,
            save_intermediate=True,
            verbose=args.verbose,
        ),
        SerializationLayer(
            output_subdir="exports",
            save_intermediate=True,
            verbose=args.verbose,
        ),
    ]
    return Pipeline(layers=layers, verbose=args.verbose, continue_from_last=not args.no_resume)


def evidence_to_text(evidence_items: Iterable[Any]) -> str:
    """Convert NeoOLAF evidence objects into a compact evidence string."""
    snippets: List[str] = []
    for ev in evidence_items or []:
        snippet = getattr(ev, "snippet", None)
        if snippet:
            snippets.append(str(snippet))
    return " | ".join(dict.fromkeys(snippets))


def state_to_canonical_prediction(state: PipelineState) -> Dict[str, Any]:
    """Convert final NeoOLAF state into the canonical prediction schema."""
    entities_by_label: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for candidate in list(state.entity_candidates or []) + list(state.event_candidates or []):
        label = getattr(candidate, "canonical_label", "") or ""
        if not label.strip():
            continue
        typ = getattr(candidate, "candidate_type", "entity") or "entity"
        key = (label.strip(), str(typ).strip())
        entities_by_label[key] = {
            "label": label.strip(),
            "type": str(typ).strip(),
            "description": getattr(candidate, "definition", None) or "",
        }

    relations: List[Dict[str, Any]] = []
    for triple in state.candidate_triples or []:
        head = getattr(triple, "subject_label", "") or ""
        relation = getattr(triple, "predicate_label", "") or ""
        tail = getattr(triple, "object_label", "") or ""
        if not head.strip() or not relation.strip() or not tail.strip():
            continue
        relations.append(
            {
                "head": head.strip(),
                "relation": relation.strip(),
                "tail": tail.strip(),
                "evidence": evidence_to_text(getattr(triple, "provenance", []))
                or getattr(triple, "justification", "")
                or "",
            }
        )

    return {
        "entities": list(entities_by_label.values()),
        "relations": relations,
    }


def raw_counts_from_state(state: PipelineState, prediction: Dict[str, Any]) -> Dict[str, int]:
    """Collect compact count diagnostics for one document."""
    return {
        "linguistic_expressions": len(state.linguistic_expressions or []),
        "enriched_expressions": len(state.enriched_expressions or []),
        "entity_candidates": len(state.entity_candidates or []),
        "event_candidates": len(state.event_candidates or []),
        "attribute_candidates": len(state.attribute_candidates or []),
        "relation_candidates": len(state.relation_candidates or []),
        "candidate_relation_assertions": len(state.candidate_relation_assertions or []),
        "candidate_triples": len(state.candidate_triples or []),
        "concept_candidates": len(state.concept_candidates or []),
        "ontology_relation_candidates": len(state.ontology_relation_candidates or []),
        "axiom_schema_candidates": len(state.axiom_schema_candidates or []),
        "general_axiom_candidates": len(state.general_axiom_candidates or []),
        "completion_candidates": len(state.completion_candidates or []),
        "canonical_entities": len(prediction.get("entities") or []),
        "canonical_relations": len(prediction.get("relations") or []),
    }


def make_error_result(
    record: Dict[str, Any],
    index: int,
    *,
    method: str,
    error: Exception | str,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a canonical error row instead of crashing the full dataset run."""
    doc_id = document_id_from_record(record, index)
    return {
        "document_id": doc_id,
        "title": title_from_record(record, doc_id),
        "type": record.get("type") or record.get("split"),
        "method": method,
        "parsed_ok": False,
        "prediction": {"entities": [], "relations": []},
        "raw_counts": {"canonical_entities": 0, "canonical_relations": 0},
        "artifact_dir": artifact_dir,
        "error": str(error),
    }


def run_one_document(
    *,
    args: argparse.Namespace,
    record: Dict[str, Any],
    index: int,
    guidance: Optional[UserGuidance],
    seed_ontology: Any,
    run_stamp: str,
) -> Tuple[int, Dict[str, Any]]:
    """Execute the full NeoOLAF pipeline for one dataset document."""
    doc_id = document_id_from_record(record, index)
    safe_doc_id = safe_filename(doc_id)
    artifact_dir = str(Path(args.artifacts_root) / safe_doc_id / f"run_{run_stamp}")

    try:
        document = record_to_document(record, index)
        backend = build_backend(args)
        pipeline = build_pipeline(args, backend)

        state = PipelineState(
            document=document,
            llm_model=args.model_name,
            user_guidance=copy.deepcopy(guidance),
            seed_ontology=seed_ontology,
            artifact_dir=artifact_dir,
        )

        execution_config = ExecutionConfig(mode="document_mode")
        runner = Runner(
            pipeline=pipeline,
            runs_root=artifact_dir,
            verbose=args.verbose,
            execution_config=execution_config,
            max_workers=args.max_workers,
            enable_checkpoints=not args.no_checkpoints,
            save_chunk_checkpoints=not args.no_chunk_checkpoints,
        )

        start = time.time()
        final_state = runner.run(state)
        elapsed = time.time() - start

        prediction = state_to_canonical_prediction(final_state)
        result = {
            "document_id": doc_id,
            "title": title_from_record(record, doc_id),
            "type": record.get("type") or record.get("split"),
            "method": "neoolaf",
            "parsed_ok": True,
            "prediction": prediction,
            "raw_counts": raw_counts_from_state(final_state, prediction),
            "artifact_dir": artifact_dir,
            "runtime_seconds": elapsed,
            "llm_call_policy": "full_pipeline_document_run",
        }
        return index, result
    except Exception as exc:
        return index, make_error_result(
            record,
            index,
            method="neoolaf",
            error=exc,
            artifact_dir=artifact_dir,
        )


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeoOLAF on a RAGTree JSONL dataset.")

    parser.add_argument("--dataset-jsonl-path", required=True)
    parser.add_argument("--ontology-path", required=True)
    parser.add_argument("--output-jsonl-path", required=True)
    parser.add_argument("--backend-name", default="openrouter")
    parser.add_argument("--host", default="https://openrouter.ai/api")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--type-filter", default="all")
    parser.add_argument("--user-guidance-path", default=None)
    parser.add_argument("--few-shot-from-dataset", action="store_true")
    parser.add_argument("--few-shot-source-type", default="all")
    parser.add_argument("--few-shot-k", type=int, default=0)
    parser.add_argument("--output-format", default="canonical", choices=["canonical"])
    parser.add_argument("--artifacts-root", default="./runs/neoolaf_artifacts")

    # No-chunk benchmark mode is represented by one very large chunk.
    parser.add_argument("--chunk-size", type=int, default=10_000_000)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=1)

    # Existing intra-document limits and workers.
    parser.add_argument("--max-expressions", type=int, default=None)
    parser.add_argument("--max-relation-mentions", type=int, default=None)
    parser.add_argument("--max-concept-inputs", type=int, default=None)
    parser.add_argument("--max-relation-inputs", type=int, default=None)
    parser.add_argument("--max-concept-pairs", type=int, default=None)
    parser.add_argument("--max-relation-pairs", type=int, default=None)
    parser.add_argument("--max-relation-schema-inputs", type=int, default=None)
    parser.add_argument("--max-subclass-inputs", type=int, default=None)
    parser.add_argument("--max-schema-inputs", type=int, default=None)
    parser.add_argument("--max-description-inputs", type=int, default=None)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--max-inferred-triples", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1, help="Intra-document/chunk workers kept for compatibility.")

    # New document-level parallelism.
    parser.add_argument(
        "--document-workers",
        type=int,
        default=1,
        help="Number of documents to process in parallel. Default preserves old sequential behavior.",
    )

    # Runtime controls.
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search in enrichment for speed/reproducibility.")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--no-chunk-checkpoints", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Write canonical JSONL output atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    start = time.time()

    records_all = load_jsonl(args.dataset_jsonl_path)
    records = filter_records(records_all, args.type_filter)
    if args.max_docs is not None:
        records = records[: args.max_docs]

    if not records:
        raise SystemExit("No records selected. Check --dataset-jsonl-path and --type-filter.")

    guidance = load_user_guidance(args.user_guidance_path)
    if args.few_shot_from_dataset:
        guidance = add_few_shot_examples_from_dataset(
            guidance,
            records_all,
            source_type=args.few_shot_source_type,
            k=args.few_shot_k,
        )

    seed_ontology = SeedOntologyLoader().load(args.ontology_path)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")

    print(
        "[NeoOLAF benchmark] "
        f"documents={len(records)} document_workers={args.document_workers} "
        f"max_workers={args.max_workers} model={args.model_name}"
    )

    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    workers = max(1, int(args.document_workers or 1))

    if workers == 1:
        for idx, record in enumerate(records):
            out_idx, result = run_one_document(
                args=args,
                record=record,
                index=idx,
                guidance=guidance,
                seed_ontology=seed_ontology,
                run_stamp=run_stamp,
            )
            results[out_idx] = result
            status = "ok" if result.get("parsed_ok") else "error"
            print(f"[{out_idx + 1}/{len(records)}] {result['document_id']} {status}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    run_one_document,
                    args=args,
                    record=record,
                    index=idx,
                    guidance=guidance,
                    seed_ontology=seed_ontology,
                    run_stamp=run_stamp,
                ): idx
                for idx, record in enumerate(records)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    out_idx, result = future.result()
                except Exception as exc:
                    # This should be rare because run_one_document catches document errors.
                    out_idx = idx
                    result = make_error_result(
                        records[idx],
                        idx,
                        method="neoolaf",
                        error=exc,
                        artifact_dir=None,
                    )
                results[out_idx] = result
                completed += 1
                status = "ok" if result.get("parsed_ok") else "error"
                print(f"[{completed}/{len(records)}] {result['document_id']} {status}")

    final_rows = [row for row in results if row is not None]
    write_jsonl(args.output_jsonl_path, final_rows)

    parsed_ok = sum(1 for row in final_rows if row.get("parsed_ok"))
    total_relations = sum(len((row.get("prediction") or {}).get("relations") or []) for row in final_rows)
    elapsed = time.time() - start
    print(
        "[NeoOLAF benchmark] finished "
        f"parsed_ok={parsed_ok}/{len(final_rows)} relations={total_relations} "
        f"elapsed_seconds={elapsed:.2f} output={args.output_jsonl_path}"
    )


if __name__ == "__main__":
    main()

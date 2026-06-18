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
import traceback
from collections import Counter
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
# Progress/error helpers
# ---------------------------------------------------------------------------

class _NullProgress:
    """Tiny tqdm-compatible fallback used when tqdm is unavailable."""

    def __init__(self, total: int = 0, desc: str = "") -> None:
        self.total = total
        self.desc = desc

    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def make_progress(total: int, desc: str, *, disable: bool = False) -> Any:
    """Return a tqdm progress bar when available, otherwise a silent fallback."""
    if disable:
        return _NullProgress(total=total, desc=desc)
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, unit="doc")
    except Exception:
        return _NullProgress(total=total, desc=desc)


def progress_write(message: str, *, disable_tqdm: bool = False) -> None:
    """Write messages without breaking tqdm output when tqdm is installed."""
    if not disable_tqdm:
        try:
            from tqdm.auto import tqdm  # type: ignore

            tqdm.write(message)
            return
        except Exception:
            pass
    print(message)


def shorten_text(value: Any, limit: int = 280) -> str:
    """Compact long error messages for terminal logs."""
    text = str(value or "").replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def collect_artifact_error_files(artifact_dir: Optional[str], *, limit: int = 8) -> List[Dict[str, Any]]:
    """Collect compact previews of error-like files created by layer artifacts."""
    if not artifact_dir:
        return []
    root = Path(artifact_dir)
    if not root.exists():
        return []

    candidates: List[Path] = []
    for pattern in ["**/*error*.txt", "**/*error*.json", "**/raw_response*.txt", "**/prompt*.txt"]:
        candidates.extend(root.glob(pattern))

    # Keep newest and avoid duplicates.
    unique = sorted(set(candidates), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    previews: List[Dict[str, Any]] = []
    for path in unique[:limit]:
        preview = ""
        try:
            if path.is_file():
                preview = path.read_text(encoding="utf-8", errors="replace")[:1200]
        except Exception as exc:
            preview = f"<could not read preview: {exc}>"
        previews.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "preview": preview,
            }
        )
    return previews


def write_document_error_report(
    artifact_dir: Optional[str],
    *,
    doc_id: str,
    error: Exception,
    traceback_text: str,
) -> None:
    """Persist a detailed per-document error report in the document artifact folder."""
    if not artifact_dir:
        return
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "document_id": doc_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback_text,
        "artifact_error_files": collect_artifact_error_files(artifact_dir),
    }
    (root / "neoolaf_document_error_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "neoolaf_document_error_traceback.txt").write_text(traceback_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

def compact_backend_debug(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact, safe-to-log summary of an OpenAI-compatible response."""
    choices = data.get("choices") or []
    choice0 = choices[0] if choices else {}
    message = choice0.get("message") or {}
    usage = data.get("usage") or {}
    content = message.get("content")
    reasoning = message.get("reasoning")
    reasoning_details = message.get("reasoning_details")
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "provider": data.get("provider"),
        "finish_reason": choice0.get("finish_reason"),
        "native_finish_reason": choice0.get("native_finish_reason"),
        "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
        "content_is_none": content is None,
        "content_len": len(str(content or "")),
        "has_reasoning": bool(reasoning),
        "reasoning_len": len(str(reasoning or "")),
        "has_reasoning_details": bool(reasoning_details),
        "usage": usage,
    }


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
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = "minimal",
        exclude_reasoning: bool = True,
        dump_raw_responses: bool = False,
    ) -> None:
        self.backend_name = backend_name
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.exclude_reasoning = exclude_reasoning
        self.dump_raw_responses = dump_raw_responses
        self._call_index = 0

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

        # OpenRouter exposes reasoning controls for thinking/reasoning models.
        # This is especially important for gpt-oss providers: without this, a
        # provider can spend the output budget on reasoning and return empty
        # message.content even though the request succeeded.
        if self.backend_name.lower() == "openrouter":
            reasoning: Dict[str, Any] = {}
            if self.reasoning_effort:
                reasoning["effort"] = self.reasoning_effort
            if self.exclude_reasoning:
                reasoning["exclude"] = True
            if reasoning:
                payload["reasoning"] = reasoning

        self._call_index += 1
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
            raise RuntimeError(f"No choices returned by backend {self.backend_name}: {compact_backend_debug(data)}")

        choice0 = choices[0]
        message = choice0.get("message") or {}
        content = message.get("content")
        if content is None:
            content = choice0.get("text")

        # Some OpenAI-compatible providers return content as a list of typed
        # blocks. Normalize those into plain text before returning.
        if isinstance(content, list):
            content = "".join(
                str(block.get("text") or block.get("content") or "")
                if isinstance(block, dict)
                else str(block)
                for block in content
            )

        if content is None or not str(content).strip():
            debug = compact_backend_debug(data)
            raise RuntimeError(
                "No final message.content returned by backend "
                f"{self.backend_name}. This is not an API-key/credits error: "
                "the request returned choices, but the final assistant content was empty. "
                "For OpenRouter reasoning models, try --max-tokens 8192 "
                "--openrouter-reasoning-effort minimal --openrouter-exclude-reasoning. "
                f"Backend debug: {debug}"
            )
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
        reasoning_effort=(args.openrouter_reasoning_effort or None),
        exclude_reasoning=args.openrouter_exclude_reasoning,
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
    traceback_text: str = "",
) -> Dict[str, Any]:
    """Return a canonical error row instead of crashing the full dataset run."""
    doc_id = document_id_from_record(record, index)
    error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
    error_message = str(error)
    return {
        "document_id": doc_id,
        "title": title_from_record(record, doc_id),
        "type": record.get("type") or record.get("split"),
        "method": method,
        "parsed_ok": False,
        "prediction": {"entities": [], "relations": []},
        "raw_counts": {"canonical_entities": 0, "canonical_relations": 0},
        "artifact_dir": artifact_dir,
        "runtime_seconds": None,
        "error": error_message,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": traceback_text,
        "artifact_error_files": collect_artifact_error_files(artifact_dir),
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
        traceback_text = traceback.format_exc()
        write_document_error_report(
            artifact_dir,
            doc_id=doc_id,
            error=exc,
            traceback_text=traceback_text,
        )
        return index, make_error_result(
            record,
            index,
            method="neoolaf",
            error=exc,
            artifact_dir=artifact_dir,
            traceback_text=traceback_text,
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

    # Diagnostics/progress.
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--show-error-traceback", action="store_true", help="Print full traceback for document errors.")
    parser.add_argument("--error-log-jsonl-path", default=None, help="Optional JSONL file for document-level errors.")
    parser.add_argument("--summary-output-path", default=None, help="Optional JSON summary of the benchmark run.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first document-level error.")

    # Runtime controls.
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--openrouter-reasoning-effort",
        default="minimal",
        choices=["xhigh", "high", "medium", "low", "minimal", "none", ""],
        help="OpenRouter reasoning effort for reasoning models. Use minimal/none to avoid empty final content on gpt-oss providers.",
    )
    parser.add_argument(
        "--openrouter-exclude-reasoning",
        action="store_true",
        default=True,
        help="Request OpenRouter to exclude reasoning traces from the response. Enabled by default.",
    )
    parser.add_argument(
        "--openrouter-include-reasoning",
        dest="openrouter_exclude_reasoning",
        action="store_false",
        help="Debug option: allow reasoning traces to be returned by OpenRouter.",
    )
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


def default_error_log_path(output_jsonl_path: str | Path) -> Path:
    """Default error JSONL path derived from the prediction output path."""
    path = Path(output_jsonl_path)
    return path.with_name(path.stem + ".errors.jsonl")


def default_summary_path(output_jsonl_path: str | Path) -> Path:
    """Default run summary JSON path derived from the prediction output path."""
    path = Path(output_jsonl_path)
    return path.with_name(path.stem + ".run_summary.json")


def relation_count_from_result(result: Dict[str, Any]) -> int:
    """Return number of canonical predicted relations for one result row."""
    return len(((result.get("prediction") or {}).get("relations") or []))


def build_run_summary(
    *,
    args: argparse.Namespace,
    final_rows: List[Dict[str, Any]],
    elapsed: float,
) -> Dict[str, Any]:
    """Build a compact dataset-level run summary with error diagnostics."""
    parsed_ok = sum(1 for row in final_rows if row.get("parsed_ok"))
    error_rows = [row for row in final_rows if not row.get("parsed_ok")]
    zero_relation_docs = [
        row.get("document_id")
        for row in final_rows
        if row.get("parsed_ok") and relation_count_from_result(row) == 0
    ]
    return {
        "dataset_jsonl_path": args.dataset_jsonl_path,
        "ontology_path": args.ontology_path,
        "output_jsonl_path": args.output_jsonl_path,
        "model_name": args.model_name,
        "type_filter": args.type_filter,
        "documents": len(final_rows),
        "parsed_ok": parsed_ok,
        "failed": len(error_rows),
        "relations": sum(relation_count_from_result(row) for row in final_rows),
        "elapsed_seconds": elapsed,
        "document_workers": args.document_workers,
        "max_workers": args.max_workers,
        "error_type_counts": dict(Counter(str(row.get("error_type", "Error")) for row in error_rows)),
        "zero_relation_docs_count": len(zero_relation_docs),
        "zero_relation_doc_ids_preview": zero_relation_docs[:20],
        "errors_preview": [
            {
                "document_id": row.get("document_id"),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message") or row.get("error"),
                "artifact_dir": row.get("artifact_dir"),
                "artifact_error_files": row.get("artifact_error_files", [])[:3],
            }
            for row in error_rows[:20]
        ],
    }


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
    progress = make_progress(len(records), "NeoOLAF documents", disable=args.no_tqdm)

    def handle_result(completed_no: int, out_idx: int, result: Dict[str, Any]) -> None:
        """Store and log one document result."""
        results[out_idx] = result
        progress.update(1)
        relation_count = relation_count_from_result(result)
        runtime = result.get("runtime_seconds")
        runtime_txt = f" time={runtime:.2f}s" if isinstance(runtime, (int, float)) else ""
        if result.get("parsed_ok"):
            msg = (
                f"[{completed_no}/{len(records)}] {result['document_id']} ok "
                f"relations={relation_count}{runtime_txt}"
            )
        else:
            err_type = result.get("error_type", "Error")
            err_msg = shorten_text(result.get("error_message") or result.get("error"))
            msg = (
                f"[{completed_no}/{len(records)}] {result['document_id']} error "
                f"{err_type}: {err_msg} artifact={result.get('artifact_dir')}"
            )
        progress_write(msg, disable_tqdm=args.no_tqdm)
        if (not result.get("parsed_ok")) and args.show_error_traceback:
            progress_write(str(result.get("error_traceback") or ""), disable_tqdm=args.no_tqdm)
        if (not result.get("parsed_ok")) and args.fail_fast:
            raise RuntimeError(f"Fail-fast after document error: {result.get('document_id')} {result.get('error_message')}")

    try:
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
                handle_result(idx + 1, out_idx, result)
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
                            traceback_text=traceback.format_exc(),
                        )
                    completed += 1
                    handle_result(completed, out_idx, result)
    finally:
        progress.close()

    final_rows = [row for row in results if row is not None]
    write_jsonl(args.output_jsonl_path, final_rows)

    error_rows = [row for row in final_rows if not row.get("parsed_ok")]
    error_log_path = Path(args.error_log_jsonl_path) if args.error_log_jsonl_path else default_error_log_path(args.output_jsonl_path)
    if error_rows:
        write_jsonl(error_log_path, error_rows)

    elapsed = time.time() - start
    run_summary = build_run_summary(args=args, final_rows=final_rows, elapsed=elapsed)
    summary_path = Path(args.summary_output_path) if args.summary_output_path else default_summary_path(args.output_jsonl_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    parsed_ok = run_summary["parsed_ok"]
    total_relations = run_summary["relations"]
    print(
        "[NeoOLAF benchmark] finished "
        f"parsed_ok={parsed_ok}/{len(final_rows)} relations={total_relations} "
        f"failed={run_summary['failed']} zero_relation_docs={run_summary['zero_relation_docs_count']} "
        f"elapsed_seconds={elapsed:.2f} output={args.output_jsonl_path} summary={summary_path}"
    )
    if error_rows:
        print(f"[NeoOLAF benchmark] error_log={error_log_path}")
        print(f"[NeoOLAF benchmark] error_type_counts={run_summary['error_type_counts']}")
        for err in run_summary["errors_preview"][:5]:
            print(
                "[NeoOLAF benchmark][error-preview] "
                f"{err['document_id']} {err['error_type']}: {shorten_text(err['error_message'])} "
                f"artifact={err['artifact_dir']}"
            )


if __name__ == "__main__":
    main()

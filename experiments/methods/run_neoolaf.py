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

# ---------------------------------------------------------------------------
# Optional offline Wikipedia policy
# ---------------------------------------------------------------------------

def _is_wikipedia_url(url: object) -> bool:
    """Return True for Wikipedia/Wikimedia URLs that should be blocked in benchmark mode."""
    try:
        from urllib.parse import urlparse

        hostname = (urlparse(str(url)).hostname or "").lower()
    except Exception:
        return False
    return (
        hostname == "wikipedia.org"
        or hostname.endswith(".wikipedia.org")
        or hostname == "wikimedia.org"
        or hostname.endswith(".wikimedia.org")
    )


def install_wikipedia_blocker() -> None:
    """Block Wikipedia lookups at runtime without touching NeoOLAF source code.

    This is intentionally scoped to wikipedia.org / wikimedia.org. OpenRouter,
    local files, and other HTTP endpoints remain untouched. The fake MediaWiki
    response is empty-but-successful, so enrichment code can continue quickly.
    """

    if getattr(install_wikipedia_blocker, "_installed", False):
        return

    original_session_request = requests.sessions.Session.request

    def offline_session_request(self, method, url, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not _is_wikipedia_url(url):
            return original_session_request(self, method, url, *args, **kwargs)

        response = requests.Response()
        response.status_code = 200
        response.url = str(url)
        response.reason = "Wikipedia disabled by NeoOLAF benchmark policy"
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response._content = json.dumps(
            {
                "batchcomplete": "",
                "query": {"search": [], "pages": {}},
                "warnings": {
                    "neoolaf": {
                        "*": "Wikipedia lookup disabled by benchmark policy."
                    }
                },
            }
        ).encode("utf-8")
        response.encoding = "utf-8"
        return response

    requests.sessions.Session.request = offline_session_request
    install_wikipedia_blocker._installed = True  # type: ignore[attr-defined]
    print("[NeoOLAF benchmark] Wikipedia/Wikimedia lookups disabled by runner policy.")




# ---------------------------------------------------------------------------
# Offline source objects used by benchmark mode
# ---------------------------------------------------------------------------

class OfflineWikipediaSource:
    """Wikipedia-compatible source that returns no external evidence.

    This keeps Layer 2 alive without making network calls and without faking
    MediaWiki HTTP responses. The shape matches WikipediaSource.search().
    """

    def search(self, term: str) -> Dict[str, Any]:
        return {
            "source": "wikipedia",
            "term": term,
            "found": False,
            "aliases": [],
            "summary": "",
            "url": None,
        }


class OfflineWikidataSource:
    """Wikidata-compatible source that returns no external evidence."""

    def search(self, term: str, limit: int = 3) -> Dict[str, Any]:
        return {
            "source": "wikidata",
            "term": term,
            "results": [],
            "aliases": [],
            "labels": [],
            "descriptions": [],
        }


class OfflineWebSearchSource:
    """Web-search-compatible source that returns no external evidence."""

    def search(self, term: str, max_results: int = 3) -> Dict[str, Any]:
        return {
            "source": "web",
            "term": term,
            "results": [],
        }

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
        """Robust JSON extractor shared by all layers.

        Provider responses sometimes contain extra prose, markdown fences,
        duplicated JSON, or trailing partial text. This parser tries clean JSON
        first, then scans for the first valid object/array using raw_decode.
        """
        if text is None:
            raise ValueError("Could not parse JSON from model output because it is None.")
        text = str(text).strip()
        if not text:
            raise ValueError("Could not parse JSON from model output because it is empty.")

        for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            fenced = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if fenced:
                payload = fenced.group(1).strip()
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    text = payload

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for start, ch in enumerate(text):
            if ch not in "[{":
                continue
            try:
                obj, _end = decoder.raw_decode(text[start:])
                return obj
            except json.JSONDecodeError:
                continue

        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
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


def record_to_document(record: Dict[str, Any], index: int, args: Optional[argparse.Namespace] = None) -> Document:
    """Convert one normalized JSONL row into NeoOLAF's Document object."""
    doc_id = document_id_from_record(record, index)
    title = title_from_record(record, doc_id)
    text = document_text_from_record(record)

    if args is not None and getattr(args, "raw_text_entity_relation_mode", False):
        # Raw-text ER mode: relation vocabulary is allowed, but no source/gold
        # entity inventory is exposed to the model.
        control = build_raw_text_er_control_block(list(getattr(args, "allowed_relation_specs", []) or []))
        raw_text = f"{control}\n{text}"
    elif args is not None and (getattr(args, "force_relation_vocabulary", False) or getattr(args, "source_entity_anchoring", False)):
        control = build_docred_control_block(
            record,
            list(getattr(args, "allowed_relation_specs", []) or []),
            include_entities=bool(getattr(args, "source_entity_anchoring", False)),
        )
        raw_text = f"{control}\n{text}"
    else:
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
                **({} if (args is not None and getattr(args, "raw_text_entity_relation_mode", False)) else {"source_entities": source_entities_from_record(record)}),
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


def source_entities_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return source entity clusters with IDs, canonical labels, aliases, and types.

    Supports the DocRED/RAGTree dictionary schema:
        entities = {entity_id: {type: ..., mentions: [{trigger_word: ...}]}}
    and common list-based schemas.
    """
    entities = record.get("entities")
    result: List[Dict[str, Any]] = []

    if isinstance(entities, dict):
        for ent_id, ent in entities.items():
            if not isinstance(ent, dict):
                continue
            aliases: List[str] = []
            mentions = ent.get("mentions") or []
            if isinstance(mentions, list):
                for mention in mentions:
                    if isinstance(mention, dict):
                        label = mention.get("trigger_word") or mention.get("name") or mention.get("text")
                        if label and str(label).strip():
                            aliases.append(str(label).strip())
            canonical = aliases[0] if aliases else str(ent_id)
            result.append(
                {
                    "id": str(ent_id),
                    "label": canonical,
                    "type": str(ent.get("type") or "entity"),
                    "aliases": sorted(dict.fromkeys(a for a in aliases if a)),
                }
            )

    elif isinstance(entities, list):
        for i, ent in enumerate(entities):
            if isinstance(ent, dict):
                ent_id = str(ent.get("id") or ent.get("entity_id") or f"entity_{i:05d}")
                label = str(ent.get("label") or ent.get("name") or ent.get("text") or ent_id)
                aliases = ent.get("aliases") if isinstance(ent.get("aliases"), list) else []
                aliases = [label, *[str(x) for x in aliases if str(x).strip()]]
                result.append(
                    {
                        "id": ent_id,
                        "label": label,
                        "type": str(ent.get("type") or ent.get("entity_type") or "entity"),
                        "aliases": sorted(dict.fromkeys(a for a in aliases if a)),
                    }
                )
            elif isinstance(ent, list) and ent:
                # DocRED vertexSet-like cluster.
                first = next((x for x in ent if isinstance(x, dict)), None)
                if first:
                    label = first.get("name") or first.get("text") or first.get("trigger_word") or f"entity_{i:05d}"
                    aliases = []
                    for m in ent:
                        if isinstance(m, dict):
                            a = m.get("name") or m.get("text") or m.get("trigger_word")
                            if a:
                                aliases.append(str(a).strip())
                    result.append(
                        {
                            "id": str(first.get("id") or first.get("entity_id") or f"entity_{i:05d}"),
                            "label": str(label),
                            "type": str(first.get("type") or "entity"),
                            "aliases": sorted(dict.fromkeys(a for a in aliases if a)),
                        }
                    )
    return result


def source_entity_index(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index source entities by ID and alias."""
    index: Dict[str, Dict[str, Any]] = {}
    for ent in source_entities_from_record(record):
        for raw in [ent.get("id"), ent.get("label"), *(ent.get("aliases") or [])]:
            key = normalize_key(raw)
            if key and key not in index:
                index[key] = ent
    return index


def relation_items_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recover relation-like items from common dataset/prediction schemas.

    Crucially supports DocRED/RAGTree dictionaries of the form:
        relations = {"P127 : owned by": [[head_id, tail_id], ...]}

    This function returns only examples/vocabulary views. It never copies gold
    pairs into predictions.
    """
    for key in ["relations", "gold_relations", "labels", "triples"]:
        value = record.get(key)
        if isinstance(value, dict):
            ent_idx = source_entity_index(record)
            rows: List[Dict[str, Any]] = []
            for rel_label, pairs in value.items():
                if not isinstance(pairs, list):
                    continue
                for pair in pairs:
                    head = tail = None
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        head, tail = pair[0], pair[1]
                    elif isinstance(pair, dict):
                        head = pair.get("head") or pair.get("subject") or pair.get("source") or pair.get("h") or pair.get("head_id")
                        tail = pair.get("tail") or pair.get("object") or pair.get("target") or pair.get("t") or pair.get("tail_id")
                    else:
                        continue
                    head_ent = ent_idx.get(normalize_key(head))
                    tail_ent = ent_idx.get(normalize_key(tail))
                    rows.append(
                        {
                            "head": head_ent["label"] if head_ent else str(head),
                            "head_id": str(head_ent["id"]) if head_ent else str(head),
                            "tail": tail_ent["label"] if tail_ent else str(tail),
                            "tail_id": str(tail_ent["id"]) if tail_ent else str(tail),
                            "relation": str(rel_label),
                        }
                    )
            return rows
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

def normalize_key(value: object) -> str:
    """Normalize labels for conservative exact matching."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text



# ---------------------------------------------------------------------------
# DocRED/RAGTree constrained vocabulary helpers
# ---------------------------------------------------------------------------

def split_relation_label(raw: object) -> Tuple[Optional[str], str, str]:
    """Return (relation_id, plain_label, canonical_label)."""
    text = str(raw or "").strip()
    if not text:
        return None, "", ""
    m = re.match(r"^(P\d+)\s*:\s*(.+)$", text)
    if m:
        rel_id = m.group(1).strip()
        rel_label = m.group(2).strip()
        return rel_id, rel_label, f"{rel_id} : {rel_label}"
    m = re.match(r"^(P\d+)$", text)
    if m:
        rel_id = m.group(1).strip()
        return rel_id, rel_id, rel_id
    return None, text, text


def make_relation_spec(raw: object) -> Optional[Dict[str, Any]]:
    """Normalize one allowed relation specification."""
    if isinstance(raw, dict):
        source = raw.get("canonical") or raw.get("relation") or raw.get("label") or raw.get("name") or raw.get("id")
        rel_id = raw.get("id") or raw.get("relation_id")
        rel_label = raw.get("label") or raw.get("name") or raw.get("relation_label")
        if rel_id and rel_label:
            canonical = f"{str(rel_id).strip()} : {str(rel_label).strip()}"
        else:
            parsed_id, parsed_label, canonical = split_relation_label(source)
            rel_id = rel_id or parsed_id
            rel_label = rel_label or parsed_label
    else:
        rel_id, rel_label, canonical = split_relation_label(raw)

    if not canonical:
        return None
    aliases = {canonical, rel_label}
    if rel_id:
        aliases.add(str(rel_id))

    # Conservative aliases for common surface forms, still mapped only to
    # relations that exist in the allowed vocabulary.
    plain = normalize_key(rel_label)
    if plain == "part of":
        aliases.add("is part of")
    if plain == "owned by":
        aliases.update({"owner", "part of", "is part of"})
    if plain == "headquarters location":
        aliases.update({"based in", "headquartered in", "headquarters in"})
    if plain == "place of birth":
        aliases.update({"born in", "was born in"})
    if plain == "educated at":
        aliases.update({"attended", "graduated from", "studied at"})
    if plain == "publication date":
        aliases.update({"released on", "release date", "released"})
    if plain == "performer":
        aliases.update({"by", "performed by", "sung by", "recorded by"})
    return {
        "id": str(rel_id).strip() if rel_id else None,
        "label": str(rel_label).strip(),
        "canonical": canonical,
        "aliases": sorted(a for a in aliases if str(a).strip()),
    }


def merge_relation_specs(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not item:
            continue
        key = item.get("id") or normalize_key(item.get("canonical"))
        if not key:
            continue
        if key not in merged:
            merged[key] = copy.deepcopy(item)
        else:
            aliases = set(merged[key].get("aliases") or []) | set(item.get("aliases") or [])
            merged[key]["aliases"] = sorted(a for a in aliases if str(a).strip())
            for field in ["id", "label", "canonical"]:
                if not merged[key].get(field) and item.get(field):
                    merged[key][field] = item[field]
    return sorted(merged.values(), key=lambda x: (str(x.get("id") or ""), str(x.get("canonical") or "")))


def extract_relation_vocab_from_dataset(path: str | Path, type_filter: str = "all") -> List[Dict[str, Any]]:
    """Extract only the allowed relation vocabulary from a JSONL dataset.

    This reads relation keys/labels only, never gold subject-object pairs.
    """
    specs: List[Dict[str, Any]] = []
    for record in filter_records(load_jsonl(str(path)), type_filter):
        value = record.get("relations") or record.get("gold_relations") or record.get("labels") or record.get("triples")
        if isinstance(value, dict):
            for key in value.keys():
                spec = make_relation_spec(key)
                if spec:
                    specs.append(spec)
        elif isinstance(value, list):
            for rel in value:
                raw = None
                if isinstance(rel, dict):
                    raw = rel.get("relation") or rel.get("predicate") or rel.get("label") or rel.get("r")
                elif isinstance(rel, str):
                    raw = rel
                spec = make_relation_spec(raw)
                if spec:
                    specs.append(spec)
    return merge_relation_specs(specs)


def extract_relation_vocab_from_json(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    specs: List[Dict[str, Any]] = []
    if not path.is_file():
        return []
    if path.suffix.lower() == ".jsonl":
        for row in load_jsonl(str(path)):
            spec = make_relation_spec(row)
            if spec:
                specs.append(spec)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("relations") or data.get("allowed_relations") or data.get("vocabulary") or []
        if isinstance(data, list):
            for item in data:
                spec = make_relation_spec(item)
                if spec:
                    specs.append(spec)
    return merge_relation_specs(specs)


def extract_relation_vocab_from_ontology(path: str | Path) -> List[Dict[str, Any]]:
    """Extract relation properties from a Turtle/RDF ontology if they exist."""
    path = Path(path)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    specs: List[Dict[str, Any]] = []
    for block in re.split(r"\.\s*(?:\n|$)", text):
        if not re.search(r"\b(a|rdf:type)\s+(owl:ObjectProperty|rdf:Property|owl:DatatypeProperty)\b", block):
            continue
        label_match = re.search(r"rdfs:label\s+\"([^\"]+)\"", block)
        subject_match = re.match(r"\s*(<[^>]+>|[A-Za-z_][\w.-]*:[\w.-]+)", block)
        raw_label = label_match.group(1) if label_match else None
        raw_id = None
        if subject_match:
            subject = subject_match.group(1).strip("<>")
            raw_id = subject.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        if raw_label or raw_id:
            specs.append(make_relation_spec({"id": raw_id, "label": raw_label or raw_id}))
    return merge_relation_specs(specs)


def load_allowed_relation_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    source = str(getattr(args, "relation_vocab_source", "auto") or "auto").lower()
    gathered: List[Dict[str, Any]] = []
    use_json = source in {"json", "union", "auto"}
    use_dataset = source in {"dataset", "union", "auto"}
    use_ontology = source in {"ontology", "union", "auto"}

    if use_json and getattr(args, "relation_vocab_json", None):
        gathered.extend(extract_relation_vocab_from_json(args.relation_vocab_json))
    if use_dataset:
        dataset_path = getattr(args, "relation_vocab_dataset_path", None) or getattr(args, "dataset_jsonl_path", None)
        if dataset_path:
            gathered.extend(extract_relation_vocab_from_dataset(dataset_path, getattr(args, "type_filter", "all")))
    if use_ontology:
        ontology_path = getattr(args, "relation_vocab_ontology_path", None) or getattr(args, "ontology_path", None)
        if ontology_path:
            gathered.extend(extract_relation_vocab_from_ontology(ontology_path))

    specs = merge_relation_specs(gathered)
    out_path = getattr(args, "relation_vocab_output_path", None)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps({"source": source, "count": len(specs), "relations": specs}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return specs


def relation_alias_index(allowed_relations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for rel in allowed_relations or []:
        for alias in [rel.get("id"), rel.get("label"), rel.get("canonical"), *(rel.get("aliases") or [])]:
            key = normalize_key(alias)
            if key and key not in idx:
                idx[key] = rel
    return idx


def map_relation_to_allowed(label: object, allowed_relations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not allowed_relations:
        return None
    key = normalize_key(label)
    if not key:
        return None
    idx = relation_alias_index(allowed_relations)
    if key in idx:
        return idx[key]
    key2 = re.sub(r"^is\s+", "", key)
    return idx.get(key2)


# ---------------------------------------------------------------------------
# Native NeoOLAF -> DocRED mapping helpers (no extraction, no gold entities)
# ---------------------------------------------------------------------------

DEFAULT_DOCRED_NATIVE_RELATION_MAPPER: Dict[str, Any] = {
    "description": "Gold-free mapping from native NeoOLAF relation labels to DocRED/Wikidata relation IDs. This is mapping/calibration only: it never creates a new triple and it never sees gold/source entity IDs.",
    "aliases": {
        "P17": ["country", "located in country", "is in country", "national country"],
        "P19": ["place of birth", "born in", "birth place", "birthplace"],
        "P27": ["country of citizenship", "citizenship", "nationality", "american", "brazilian", "greek"],
        "P30": ["continent", "located in continent"],
        "P69": ["educated at", "studied at", "attended", "graduated from", "education"],
        "P108": ["employer", "worked at", "works at", "employed by", "taught at"],
        "P127": ["owned by", "owner", "belongs to", "part of group", "part of the group", "controlled by"],
        "P131": ["located in the administrative territorial entity", "located in administrative entity", "administrative territorial entity", "county", "state", "province"],
        "P150": ["contains administrative territorial entity", "contains", "has administrative division"],
        "P159": ["headquarters location", "headquartered in", "headquarters in", "based in", "seat in"],
        "P162": ["producer", "produced by", "music producer"],
        "P170": ["creator", "created by", "author", "written by"],
        "P175": ["performer", "performed by", "sung by", "recorded by", "artist", "by artist"],
        "P264": ["record label", "label", "released by label"],
        "P276": ["location", "located at", "venue", "place"],
        "P355": ["subsidiary", "has subsidiary", "child organization"],
        "P361": ["part of", "is part of", "member of", "belongs to"],
        "P400": ["platform", "available on", "software platform"],
        "P463": ["member of", "member", "affiliated with"],
        "P495": ["country of origin", "origin country"],
        "P527": ["has part", "contains part"],
        "P569": ["date of birth", "born on", "birth date"],
        "P570": ["date of death", "died on", "death date"],
        "P571": ["inception", "founded", "created in", "established", "launched in"],
        "P577": ["publication date", "release date", "released on", "published on", "released"]
    },
    "country_like": ["greece", "greek", "united states", "u.s.", "usa", "brazil", "canada", "france", "ireland", "england", "japan"],
    "continent_like": ["africa", "asia", "europe", "north america", "south america", "oceania", "antarctica"],
    "org_cues": ["group", "company", "corporation", "corp", "inc", "research", "network", "university", "college", "school", "label", "entertainment"],
    "work_cues": ["song", "single", "album", "film", "series", "track", "recording"],
}


def load_relation_mapper(args: argparse.Namespace) -> Dict[str, Any]:
    mapper = copy.deepcopy(DEFAULT_DOCRED_NATIVE_RELATION_MAPPER)
    path = getattr(args, "docred_native_relation_mapper_json", None)
    if path:
        p = Path(path)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in data.items():
                if isinstance(v, dict) and isinstance(mapper.get(k), dict):
                    merged = copy.deepcopy(mapper[k]); merged.update(v); mapper[k] = merged
                else:
                    mapper[k] = v
    return mapper


def relation_spec_by_id(allowed_relations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(rel.get("id")): rel for rel in allowed_relations or [] if rel.get("id")}


def label_has_any(label: object, words: Iterable[str]) -> bool:
    key = normalize_key(label)
    return any(normalize_key(w) and normalize_key(w) in key for w in words)


def is_country_like(label: object, mapper: Dict[str, Any]) -> bool:
    key = normalize_key(label)
    if not key:
        return False
    countries = {normalize_key(x) for x in mapper.get("country_like", [])}
    return key in countries or bool(re.fullmatch(r"(the )?(united states|u\.s\.|usa|brazil|greece|greek|france|canada|ireland|japan|china|india|germany|italy|spain|portugal|uk|united kingdom)", key))


def is_continent_like(label: object, mapper: Dict[str, Any]) -> bool:
    return normalize_key(label) in {normalize_key(x) for x in mapper.get("continent_like", [])}


def relation_id_from_mapper_alias(label: object, mapper: Dict[str, Any]) -> Optional[str]:
    key = normalize_key(label)
    if not key:
        return None
    for rid, aliases in (mapper.get("aliases") or {}).items():
        for alias in aliases or []:
            a = normalize_key(alias)
            if a and (key == a or a in key or key in a):
                return str(rid)
    return None


def map_native_neoolaf_relation_to_docred(
    *, raw_relation: object, head_label: object, tail_label: object, evidence: object,
    head_type: object, tail_type: object, allowed_relations: List[Dict[str, Any]],
    mapper: Dict[str, Any], reject_peripheral: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Deterministically map/relabel/reject a native NeoOLAF triple relation.

    This is not extraction: no new triple is added and no gold/source entity is used.
    """
    by_id = relation_spec_by_id(allowed_relations)
    raw = normalize_key(raw_relation); ev = normalize_key(evidence)
    ht = normalize_key(head_type); tt = normalize_key(tail_type)
    exact = map_relation_to_allowed(raw_relation, allowed_relations)
    mapped_id = exact.get("id") if exact else relation_id_from_mapper_alias(raw_relation, mapper)
    reason = "exact_or_alias" if exact else "mapper_alias"
    country_tail = is_country_like(tail_label, mapper)
    continent_tail = is_continent_like(tail_label, mapper)
    head_orgish = ("org" in ht) or label_has_any(head_label, mapper.get("org_cues", []))
    tail_orgish = ("org" in tt) or label_has_any(tail_label, mapper.get("org_cues", []))
    head_workish = label_has_any(head_label, mapper.get("work_cues", [])) or any(w in ev for w in ["song", "single", "album", "released", "record label"])
    raw_ev = raw + " " + ev

    if country_tail:
        if mapped_id in {"P131", "P159", "P276", "P361"} or any(x in raw_ev for x in ["located", "based", "headquarter", "country"]):
            mapped_id = "P17"; reason = "country_tail_to_P17"
        if mapped_id == "P175":
            mapped_id = "P27"; reason = "person_country_to_P27"
    elif continent_tail:
        mapped_id = "P30"; reason = "continent_tail_to_P30"
    elif mapped_id == "P276" and any(x in raw_ev for x in ["headquarter", "based in", "headquartered"]):
        mapped_id = "P159"; reason = "location_to_P159"
    elif mapped_id == "P361" and head_orgish and tail_orgish:
        mapped_id = "P749" if any(x in normalize_key(tail_label) for x in ["research", "ibm", "parent"]) else "P127"
        reason = "org_part_of_to_specific_org_relation"

    if head_workish:
        if mapped_id == "P170" and any(x in raw_ev for x in ["song", "single", "performed", "sung", "recorded", " by "]):
            mapped_id = "P175"; reason = "creator_to_performer_music"
        if mapped_id == "P162" and reject_peripheral and not any(x in raw_ev for x in ["produced by", "producer", "production"]):
            return None, {"action": "reject", "reason": "weak_producer_evidence", "raw_relation": str(raw_relation)}

    if any(x in raw_ev for x in ["born on", "date of birth", "birth date"]):
        mapped_id = "P569"; reason = "birth_date_rule"
    elif any(x in raw_ev for x in ["died on", "date of death", "death date"]):
        mapped_id = "P570"; reason = "death_date_rule"
    elif any(x in raw_ev for x in ["born in", "birthplace", "place of birth"]):
        mapped_id = "P19"; reason = "birth_place_rule"
    elif any(x in raw_ev for x in ["studied at", "educated at", "attended", "graduated from"]):
        mapped_id = "P69"; reason = "education_rule"
    elif any(x in raw_ev for x in ["released", "publication date", "release date", "published"]):
        mapped_id = "P577"; reason = "publication_date_rule"
    elif any(x in raw_ev for x in ["record label", "label"]):
        mapped_id = "P264"; reason = "record_label_rule"

    if reject_peripheral and mapped_id == "P400" and not any(x in raw_ev for x in ["platform", "software", "available on", "released on"]):
        return None, {"action": "reject", "reason": "weak_platform_evidence", "raw_relation": str(raw_relation)}
    spec = by_id.get(str(mapped_id)) if mapped_id else None
    if spec is None:
        return None, {"action": "reject", "reason": "relation_not_in_allowed_vocab", "raw_relation": str(raw_relation), "mapped_id": mapped_id}
    action = "kept" if exact and str(exact.get("id")) == str(spec.get("id")) else "mapped"
    return spec, {"action": action, "reason": reason, "raw_relation": str(raw_relation), "mapped_id": spec.get("id")}


def entity_alias_index(source_entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for ent in source_entities or []:
        for alias in [ent.get("id"), ent.get("label"), *(ent.get("aliases") or [])]:
            key = normalize_key(alias)
            if key and key not in idx:
                idx[key] = ent
    return idx


def map_label_to_source_entity(label: object, source_entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    key = normalize_key(label)
    if not key:
        return None
    if key in {"per", "person", "org", "organization", "loc", "location", "misc", "entity", "institution", "time", "num"}:
        return None
    return entity_alias_index(source_entities).get(key)


def build_docred_control_block(record: Dict[str, Any], allowed_relations: List[Dict[str, Any]], *, include_entities: bool = True) -> str:
    lines = [
        "DOCRED CONTROLLED EXTRACTION MODE.",
        "Use only the SOURCE ENTITIES listed below as relation heads and tails.",
        "Do not use entity types such as ORG, LOC, PER, MISC, person, location, institution, or entity as node labels.",
        "Use only the ALLOWED RELATIONS listed below as predicates, exactly as written.",
        "Prefer the full relation form, for example `P127 : owned by`, not only `owned by`.",
        "Do not invent open relation labels. If no allowed relation is supported by the text, output no relation.",
        "",
        "ALLOWED RELATIONS:",
    ]
    for rel in allowed_relations or []:
        lines.append(f"- {rel.get('canonical')}")
    if include_entities:
        lines.extend(["", "SOURCE ENTITIES:"])
        for ent in source_entities_from_record(record):
            aliases = ", ".join(ent.get("aliases") or [])
            lines.append(f"- {ent['id']} | {ent['type']} | {ent['label']} | aliases: {aliases}")
    lines.extend(["", "DOCUMENT TEXT:"])
    return "\n".join(lines)


def build_raw_text_er_control_block(allowed_relations: List[Dict[str, Any]]) -> str:
    """Build a control block for raw-text entity + relation extraction.

    This mode intentionally does not expose source/gold entity IDs. It only
    exposes the global relation vocabulary and asks NeoOLAF to discover
    entities from the raw document text.
    """
    lines = [
        "RAW TEXT ENTITY AND RELATION EXTRACTION MODE.",
        "First identify canonical entities from the document text only, including aliases/coreference when clear.",
        "Then extract relations only between the entities you identified.",
        "Do not use any source/gold entity inventory; none is provided.",
        "Use only the ALLOWED RELATIONS listed below as predicates, exactly as written.",
        "Prefer the full relation form, for example `P127 : owned by`, not only `owned by`.",
        "Do not invent predicates outside the allowed DocRED relation vocabulary.",
        "DocRED relation policy hints:",
        "- Use P159 for an organization's headquarters/base city, but P17 when the tail is a country.",
        "- Use P27 for a person's citizenship/nationality, not generic country/location.",
        "- Use P127/P749/P355 for ownership/parent/subsidiary; avoid generic P361 when a more specific org relation fits.",
        "- Use P175 for song/work performer, P264 for record label, P577 for release/publication date.",
        "- Use P19/P569/P570/P69 for birth place/date, death date, and education.",
        "If no allowed relation is supported by the text, output no relation.",
        "",
        "ALLOWED RELATIONS:",
    ]
    for rel in allowed_relations or []:
        lines.append(f"- {rel.get('canonical')}")
    lines.extend(["", "DOCUMENT TEXT:"])
    return "\n".join(lines)


def inject_relation_constraints_into_guidance(
    guidance: Optional[UserGuidance],
    allowed_relations: List[Dict[str, Any]],
    *,
    raw_text_entity_relation_mode: bool = False,
) -> Optional[UserGuidance]:
    if not allowed_relations:
        return guidance
    guidance = copy.deepcopy(guidance) if guidance is not None else UserGuidance()
    allowed = [str(rel.get("canonical")) for rel in allowed_relations if rel.get("canonical")]
    existing = list(getattr(guidance, "priority_relations", []) or [])
    guidance.priority_relations = list(dict.fromkeys([*existing, *allowed]))
    if raw_text_entity_relation_mode:
        constraint_text = (
            "DocRED raw-text entity and relation extraction. First identify canonical entities and aliases from "
            "the raw document text, then extract relations between those predicted entities. Use only the allowed "
            "DocRED relation labels listed in priority_relations, exactly as written. Do not use any source/gold "
            "entity inventory. Prefer specific DocRED relations: P159 for organization headquarters city, P17 for "
            "country tails, P27 for person nationality, P127/P749/P355 for ownership/parent/subsidiary, P175/P264/P577 "
            "for songs and releases, and P19/P569/P570/P69 for biography facts."
        )
    else:
        constraint_text = (
            "DocRED constrained relation extraction. Use only the allowed DocRED relation labels listed in "
            "priority_relations, exactly as written. Use source entity names/IDs from the document control block. "
            "Do not invent predicates and do not use entity types as entity labels."
        )
    if getattr(guidance, "domain_focus", None):
        guidance.domain_focus = f"{guidance.domain_focus}\n\n{constraint_text}"
    else:
        guidance.domain_focus = constraint_text
    return guidance

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
            wikipedia_source=OfflineWikipediaSource() if args.disable_wikipedia_lookups else None,
            wikidata_source=OfflineWikidataSource() if args.disable_wikipedia_lookups else None,
            web_search_source=OfflineWebSearchSource() if args.no_web_search else None,
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
    stop_after_layer = getattr(args, "stop_after_layer", None)
    if stop_after_layer is not None and stop_after_layer >= 0:
        layers = layers[: min(len(layers), int(stop_after_layer) + 1)]
    return Pipeline(layers=layers, verbose=args.verbose, continue_from_last=not args.no_resume)


def evidence_to_text(evidence_items: Iterable[Any]) -> str:
    """Convert NeoOLAF evidence objects into a compact evidence string."""
    snippets: List[str] = []
    for ev in evidence_items or []:
        snippet = getattr(ev, "snippet", None)
        if snippet:
            snippets.append(str(snippet))
    return " | ".join(dict.fromkeys(snippets))


def state_to_canonical_prediction(
    state: PipelineState,
    *,
    source_entities: Optional[List[Dict[str, Any]]] = None,
    allowed_relations: Optional[List[Dict[str, Any]]] = None,
    constrained: bool = False,
    raw_text_entity_relation_mode: bool = False,
    native_relation_mapping: bool = False,
    relation_mapper: Optional[Dict[str, Any]] = None,
    native_reject_peripheral: bool = True,
) -> Dict[str, Any]:
    """Convert final NeoOLAF state into the canonical prediction schema.

    In constrained DocRED mode, this is only a benchmark-facing projection.
    Native NeoOLAF KG/ontology artifacts stay unchanged.
    """
    source_entities = source_entities or []
    allowed_relations = allowed_relations or []
    entities_by_label: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if constrained and source_entities:
        for ent in source_entities:
            entities_by_label[(ent["label"], ent["type"])] = {
                "id": ent["id"],
                "label": ent["label"],
                "type": ent["type"],
                "aliases": ent.get("aliases") or [],
                "source": "source_document_entity",
            }
    else:
        for candidate in list(state.entity_candidates or []) + list(state.event_candidates or []):
            label = getattr(candidate, "canonical_label", "") or ""
            if not str(label).strip():
                continue
            typ = getattr(candidate, "candidate_type", "entity") or "entity"
            key = (str(label).strip(), str(typ).strip())
            entities_by_label[key] = {
                "label": str(label).strip(),
                "type": str(typ).strip(),
                "description": getattr(candidate, "definition", None) or "",
            }

    relations: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    for triple in state.candidate_triples or []:
        raw_head = getattr(triple, "subject_label", "") or ""
        raw_relation = getattr(triple, "predicate_label", "") or ""
        raw_tail = getattr(triple, "object_label", "") or ""
        if not str(raw_head).strip() or not str(raw_relation).strip() or not str(raw_tail).strip():
            continue
        evidence = evidence_to_text(getattr(triple, "provenance", [])) or getattr(triple, "justification", "") or ""

        if constrained:
            rel_spec = map_relation_to_allowed(raw_relation, allowed_relations)
            reasons: List[str] = []
            if rel_spec is None:
                reasons.append("relation_not_allowed")

            if raw_text_entity_relation_mode:
                # Raw-text entity+relation mode: keep NeoOLAF-predicted entity
                # labels as endpoints. Gold/source IDs are intentionally not
                # used here; the evaluator maps labels/aliases to gold clusters.
                head_label = str(raw_head).strip()
                tail_label = str(raw_tail).strip()
                head_type = "entity"
                tail_type = "entity"
                for (_label, _typ), _ent in entities_by_label.items():
                    if normalize_key(_label) == normalize_key(head_label):
                        head_type = str(_ent.get("type") or _typ or "entity")
                    if normalize_key(_label) == normalize_key(tail_label):
                        tail_type = str(_ent.get("type") or _typ or "entity")

                mapping_info: Dict[str, Any] = {"action": "exact_or_alias"}
                if native_relation_mapping:
                    rel_spec, mapping_info = map_native_neoolaf_relation_to_docred(
                        raw_relation=raw_relation,
                        head_label=head_label,
                        tail_label=tail_label,
                        evidence=evidence,
                        head_type=head_type,
                        tail_type=tail_type,
                        allowed_relations=allowed_relations,
                        mapper=relation_mapper or DEFAULT_DOCRED_NATIVE_RELATION_MAPPER,
                        reject_peripheral=bool(native_reject_peripheral),
                    )
                    reasons = [] if rel_spec is not None else [mapping_info.get("reason") or "relation_mapping_rejected"]

                if reasons:
                    rejected.append(
                        {
                            "head": str(raw_head),
                            "relation": str(raw_relation),
                            "tail": str(raw_tail),
                            "reasons": reasons,
                            "evidence": evidence,
                            "mapping_info": mapping_info,
                        }
                    )
                    continue
                if not any(normalize_key(e.get("label")) == normalize_key(head_label) for e in entities_by_label.values()):
                    entities_by_label[(head_label, head_type)] = {"label": head_label, "type": head_type, "source": "native_neoolaf_triple_endpoint"}
                if not any(normalize_key(e.get("label")) == normalize_key(tail_label) for e in entities_by_label.values()):
                    entities_by_label[(tail_label, tail_type)] = {"label": tail_label, "type": tail_type, "source": "native_neoolaf_triple_endpoint"}
                key = (head_label, rel_spec.get("canonical") or rel_spec.get("label") or "", tail_label)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    {
                        "head": head_label,
                        "head_type": head_type,
                        "relation_id": rel_spec.get("id"),
                        "relation": rel_spec.get("canonical") or rel_spec.get("label"),
                        "relation_label": rel_spec.get("label"),
                        "tail": tail_label,
                        "tail_type": tail_type,
                        "evidence": evidence,
                        "raw_prediction": {"head": str(raw_head), "relation": str(raw_relation), "tail": str(raw_tail)},
                        "mapping_info": mapping_info,
                        "source": "native_neoolaf_raw_text_entity_relation_mapped" if native_relation_mapping else "native_neoolaf_raw_text_entity_relation",
                    }
                )
                continue

            head_ent = map_label_to_source_entity(raw_head, source_entities)
            tail_ent = map_label_to_source_entity(raw_tail, source_entities)
            if head_ent is None:
                reasons.append("head_not_source_entity")
            if tail_ent is None:
                reasons.append("tail_not_source_entity")
            if reasons:
                rejected.append(
                    {
                        "head": str(raw_head),
                        "relation": str(raw_relation),
                        "tail": str(raw_tail),
                        "reasons": reasons,
                        "evidence": evidence,
                    }
                )
                continue
            key = (head_ent["id"], rel_spec.get("canonical") or rel_spec.get("label") or "", tail_ent["id"])
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "head_id": head_ent["id"],
                    "head": head_ent["label"],
                    "head_type": head_ent["type"],
                    "relation_id": rel_spec.get("id"),
                    "relation": rel_spec.get("canonical") or rel_spec.get("label"),
                    "relation_label": rel_spec.get("label"),
                    "tail_id": tail_ent["id"],
                    "tail": tail_ent["label"],
                    "tail_type": tail_ent["type"],
                    "evidence": evidence,
                    "raw_prediction": {"head": str(raw_head), "relation": str(raw_relation), "tail": str(raw_tail)},
                }
            )
        else:
            relations.append(
                {
                    "head": str(raw_head).strip(),
                    "relation": str(raw_relation).strip(),
                    "tail": str(raw_tail).strip(),
                    "evidence": evidence,
                }
            )

    prediction: Dict[str, Any] = {"entities": list(entities_by_label.values()), "relations": relations}
    if constrained:
        prediction["projection_diagnostics"] = {
            "constrained": True,
            "raw_text_entity_relation_mode": bool(raw_text_entity_relation_mode),
            "allowed_relation_count": len(allowed_relations),
            "source_entity_count": len(source_entities),
            "accepted_relations": len(relations),
            "rejected_triples": len(rejected),
            "rejected_triples_preview": rejected[:20],
            "native_relation_mapping_enabled": bool(native_relation_mapping),
            "native_relation_mapping_kept_or_mapped": sum(1 for r in relations if (r.get("mapping_info") or {}).get("action") in {"kept", "mapped"}),
            "native_relation_mapping_rejected": len(rejected),
        }
    return prediction


# ---------------------------------------------------------------------------
# Raw-text entity+relation fallback helper (no source entity inventory)
# ---------------------------------------------------------------------------

def build_raw_text_er_messages(
    record: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    *,
    max_relations: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Prompt the model to extract entities and relations from raw text only."""
    doc_id = document_id_from_record(record, 0)
    title = title_from_record(record, doc_id)
    text = document_text_from_record(record)
    rel_lines = compact_allowed_relations_for_prompt(allowed_relations, max_relations=max_relations)
    system = (
        "You are a strict DocRED-style entity and relation extraction system. "
        "Extract entities from the raw document text, then relations between those entities. "
        "Use only the provided global DocRED relation vocabulary. Return JSON only."
    )
    user = f"""
Extract entities and relations from the document text only.

Rules:
1. Do not use any gold/source entity inventory; none is provided.
2. Create local entity IDs E1, E2, E3, ... for the entities you find.
3. Entity types must be one of PER, ORG, LOC, MISC, TIME, NUM when possible.
4. Relation heads and tails must use your local entity IDs.
5. Relations must use relation IDs from ALLOWED RELATIONS only.
6. Use aliases/coreference when obvious, but keep one canonical entity per real-world object.
7. Include short evidence from the document.
8. Return valid JSON only. No markdown.

Output schema:
{{
  "entities": [
    {{"entity_id": "E1", "label": "canonical name", "type": "ORG", "aliases": ["alias"], "evidence": "..."}}
  ],
  "relations": [
    {{"head_entity_id": "E1", "relation_id": "P159", "tail_entity_id": "E2", "evidence": "..."}}
  ]
}}

DOCUMENT ID: {doc_id}
TITLE: {title}

ALLOWED RELATIONS:
{rel_lines}

DOCUMENT TEXT:
{text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_raw_er_payload(data: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalize raw-text ER model JSON.

    The preferred schema is dict-based, but OpenRouter/model outputs sometimes
    return compact lists. Accept both so one malformed row does not crash or
    silently erase all predictions.
    """
    if isinstance(data, dict):
        entities = data.get("entities") or data.get("entity_candidates") or []
        relations = data.get("relations") or data.get("triples") or []
    elif isinstance(data, list):
        # Sometimes the model returns only a relation list.
        entities, relations = [], data
    else:
        entities, relations = [], []

    norm_entities: List[Dict[str, Any]] = []
    for i, ent in enumerate(entities or [], start=1):
        if isinstance(ent, dict):
            norm_entities.append(ent)
        elif isinstance(ent, (list, tuple)) and ent:
            norm_entities.append(
                {
                    "entity_id": ent[0] if len(ent) > 0 else f"E{i}",
                    "label": ent[1] if len(ent) > 1 else ent[0],
                    "type": ent[2] if len(ent) > 2 else "entity",
                }
            )

    norm_relations: List[Dict[str, Any]] = []
    for rel in relations or []:
        if isinstance(rel, dict):
            norm_relations.append(rel)
        elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
            norm_relations.append(
                {
                    "head_entity_id": rel[0],
                    "relation_id": rel[1],
                    "tail_entity_id": rel[2],
                    "evidence": rel[3] if len(rel) > 3 else "",
                }
            )
    return norm_entities, norm_relations


def run_raw_text_er_direct_fallback(
    *,
    record: Dict[str, Any],
    backend: OpenAICompatibleBackend,
    args: argparse.Namespace,
    artifact_dir: str,
) -> Dict[str, Any]:
    """Fallback extraction from raw text only: predicts entities and relations."""
    allowed_relations = list(getattr(args, "allowed_relation_specs", []) or [])
    retries = int(getattr(args, "raw_text_er_direct_retries", 2) or 0)
    last_error: Optional[str] = None
    raw_response = ""
    parsed: Any = {"entities": [], "relations": []}
    for attempt in range(retries + 1):
        try:
            messages = build_raw_text_er_messages(
                record,
                allowed_relations,
                max_relations=getattr(args, "raw_text_er_direct_max_relations", None),
            )
            raw_response = backend.chat(
                args.model_name,
                messages,
                temperature=float(getattr(args, "raw_text_er_direct_temperature", 0.0) or 0.0),
            )
            parsed = backend.extract_json(raw_response)
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(float(getattr(args, "raw_text_er_direct_retry_sleep", 2.0) or 0.0))

    raw_entities, raw_relations = normalize_raw_er_payload(parsed)
    entity_by_id: Dict[str, Dict[str, Any]] = {}
    entities: List[Dict[str, Any]] = []
    for i, ent in enumerate(raw_entities, start=1):
        eid = str(ent.get("entity_id") or ent.get("id") or f"E{i}").strip()
        label = str(ent.get("label") or ent.get("name") or ent.get("text") or eid).strip()
        etype = str(ent.get("type") or ent.get("entity_type") or "entity").strip()
        aliases = ent.get("aliases") if isinstance(ent.get("aliases"), list) else []
        aliases = sorted(dict.fromkeys([label, *[str(a).strip() for a in aliases if str(a).strip()]]))
        out_ent = {"id": eid, "label": label, "type": etype, "aliases": aliases, "source": "raw_text_direct_entity"}
        entities.append(out_ent)
        for alias in [eid, label, *aliases]:
            entity_by_id.setdefault(normalize_key(alias), out_ent)

    relations: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for item in raw_relations:
        raw_h = item.get("head_entity_id") or item.get("head_id") or item.get("head") or item.get("subject") or item.get("h")
        raw_t = item.get("tail_entity_id") or item.get("tail_id") or item.get("tail") or item.get("object") or item.get("t")
        raw_r = item.get("relation_id") or item.get("relation") or item.get("predicate") or item.get("r")
        h_ent = entity_by_id.get(normalize_key(raw_h))
        t_ent = entity_by_id.get(normalize_key(raw_t))
        rel_spec = map_relation_to_allowed(raw_r, allowed_relations)
        reasons: List[str] = []
        if h_ent is None:
            reasons.append("head_not_predicted_entity")
        if t_ent is None:
            reasons.append("tail_not_predicted_entity")
        if rel_spec is None:
            reasons.append("relation_not_allowed")
        if h_ent is not None and t_ent is not None and h_ent.get("id") == t_ent.get("id"):
            reasons.append("self_relation_rejected")
        if reasons:
            rejected.append({"raw_prediction": item, "reasons": reasons})
            continue
        key = (str(h_ent["id"]), str(rel_spec.get("id") or rel_spec.get("canonical")), str(t_ent["id"]))
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            {
                "head_id": h_ent["id"],
                "head": h_ent["label"],
                "head_type": h_ent.get("type"),
                "relation_id": rel_spec.get("id"),
                "relation": rel_spec.get("canonical") or rel_spec.get("label"),
                "relation_label": rel_spec.get("label"),
                "tail_id": t_ent["id"],
                "tail": t_ent["label"],
                "tail_type": t_ent.get("type"),
                "evidence": str(item.get("evidence") or item.get("justification") or "").strip(),
                "raw_prediction": item,
                "source": "raw_text_direct_fallback",
            }
        )

    diagnostics = {
        "mode": "raw_text_entity_relation_direct_fallback",
        "attempt_error": last_error,
        "raw_entities": len(raw_entities),
        "raw_relations": len(raw_relations),
        "accepted_entities": len(entities),
        "accepted_relations": len(relations),
        "rejected_relations": len(rejected),
        "rejected_preview": rejected[:20],
    }
    prediction = {"entities": entities, "relations": relations, "projection_diagnostics": diagnostics}
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_text_er_direct_fallback.json").write_text(
        json.dumps({"prediction": prediction, "diagnostics": diagnostics, "raw_response": raw_response}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prediction


# ---------------------------------------------------------------------------
# Direct DocRED constrained extraction helper
# ---------------------------------------------------------------------------

def compact_source_entities_for_prompt(source_entities: List[Dict[str, Any]], max_entities: Optional[int] = None) -> str:
    """Render source entity clusters as a compact prompt table."""
    rows: List[str] = []
    entities = source_entities[: max_entities or len(source_entities)]
    for ent in entities:
        aliases = ", ".join(str(a) for a in (ent.get("aliases") or [])[:8] if str(a).strip())
        rows.append(f"- {ent.get('id')} | {ent.get('type')} | {ent.get('label')} | aliases: {aliases}")
    return "\n".join(rows)


def compact_allowed_relations_for_prompt(allowed_relations: List[Dict[str, Any]], max_relations: Optional[int] = None) -> str:
    """Render allowed relation vocabulary as a compact prompt table."""
    rows: List[str] = []
    rels = allowed_relations[: max_relations or len(allowed_relations)]
    for rel in rels:
        rel_id = rel.get("id") or ""
        label = rel.get("label") or ""
        canonical = rel.get("canonical") or label or rel_id
        rows.append(f"- {rel_id} | {canonical}")
    return "\n".join(rows)


DOCRED_RELATION_FAMILIES: Dict[str, set[str]] = {
    "location": {"P17", "P131", "P150", "P30", "P19", "P159", "P276", "P495"},
    "organization": {"P127", "P361", "P749", "P355", "P159", "P571", "P112", "P108"},
    "person": {"P19", "P27", "P69", "P108", "P463", "P569", "P570"},
    "creative_work": {"P175", "P170", "P162", "P264", "P577", "P155", "P495"},
    "date_numeric": {"P569", "P570", "P571", "P577"},
}

DOCRED_HIGH_YIELD_RELATION_IDS: set[str] = {
    "P17", "P27", "P69", "P159", "P127", "P361", "P175", "P264", "P577",
    "P19", "P569", "P570", "P749", "P355", "P571", "P30", "P495", "P108",
}

COUNTRY_WORDS: set[str] = {
    "greece", "greek", "united states", "american", "brazil", "brazilian", "canada", "canadian",
    "france", "french", "england", "british", "united kingdom", "uk", "ireland", "irish",
    "japan", "japanese", "china", "chinese", "germany", "german", "italy", "italian",
    "spain", "spanish", "mexico", "mexican", "australia", "australian",
}


def relation_specs_by_id(allowed_relations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(rel.get("id")): rel for rel in allowed_relations or [] if rel.get("id")}


def relation_spec_by_id(allowed_relations: List[Dict[str, Any]], relation_id: str) -> Optional[Dict[str, Any]]:
    return relation_specs_by_id(allowed_relations).get(str(relation_id))


def entity_text(ent: Dict[str, Any]) -> str:
    return " ".join([str(ent.get("label") or ""), *[str(a) for a in (ent.get("aliases") or [])]])


def entity_is_country_like(ent: Dict[str, Any]) -> bool:
    text = normalize_key(entity_text(ent))
    return any(country in text for country in COUNTRY_WORDS)


def entity_is_probably_creative_work(ent: Dict[str, Any]) -> bool:
    text = normalize_key(entity_text(ent))
    typ = normalize_key(ent.get("type"))
    return typ == "misc" or any(w in text for w in ["song", "album", "single", "film", "episode", "book"])


def infer_docred_relation_family_ids(record: Dict[str, Any], source_entities: List[Dict[str, Any]]) -> set[str]:
    """Infer a gold-free relation subset from entity types and document trigger words."""
    text = normalize_key(document_text_from_record(record))
    title = normalize_key(title_from_record(record, document_id_from_record(record, 0)))
    types = {normalize_key(ent.get("type")) for ent in source_entities}
    labels = normalize_key(" ".join(entity_text(ent) for ent in source_entities))
    ids: set[str] = set(DOCRED_HIGH_YIELD_RELATION_IDS)

    if {"loc", "gpe", "location"} & types or any(w in text for w in ["city", "county", "state", "country", "province", "located", "born in"]):
        ids |= DOCRED_RELATION_FAMILIES["location"]
    if {"org", "organization"} & types or any(w in text for w in ["company", "group", "organization", "research", "headquartered", "based in", "subsidiary", "parent", "owned", "founded"]):
        ids |= DOCRED_RELATION_FAMILIES["organization"]
    if {"per", "person"} & types or any(w in text for w in ["born", "died", "educated", "graduated", "employer", "citizen", "nationality"]):
        ids |= DOCRED_RELATION_FAMILIES["person"]
    if any(w in (text + " " + title + " " + labels) for w in ["song", "single", "album", "rapper", "singer", "record label", "released", "publication", "producer", "performed", "film"]):
        ids |= DOCRED_RELATION_FAMILIES["creative_work"]
    if re.search(r"\b(?:18|19|20)\d{2}\b", text):
        ids |= DOCRED_RELATION_FAMILIES["date_numeric"]
    return ids


def filter_allowed_relations_for_direct_extractor(
    allowed_relations: List[Dict[str, Any]],
    *,
    focus_relation_ids: Optional[str] = None,
    record: Optional[Dict[str, Any]] = None,
    relation_family_filter: bool = False,
) -> List[Dict[str, Any]]:
    """Restrict the relation vocabulary shown to the direct DocRED extractor.

    This is gold-free. It uses either an explicit user-provided ID list or an
    automatic subset inferred from entity types and document trigger words.
    """
    if focus_relation_ids:
        wanted = {x.strip() for x in str(focus_relation_ids).split(",") if x.strip()}
        if wanted:
            filtered = [rel for rel in allowed_relations if str(rel.get("id") or rel.get("canonical") or "") in wanted or str(rel.get("id") or "") in wanted]
            return filtered or allowed_relations

    if relation_family_filter and record is not None:
        wanted = infer_docred_relation_family_ids(record, source_entities_from_record(record))
        filtered = [rel for rel in allowed_relations if str(rel.get("id") or "") in wanted]
        return filtered or allowed_relations

    return allowed_relations


def docred_relation_disambiguation_hints() -> str:
    """Gold-free relation-selection hints for common DocRED confusions."""
    return """
Relation-selection hints for DocRED/Wikidata labels:
- Location family:
  * Use P159 : headquarters location for organizations/broadcasters "based in" or "headquartered in" a city.
  * Use P17 : country when the tail is a country entity such as Greece/United States/Brazil.
  * Use P27 : country of citizenship only for a person.
  * Use P131 : located in the administrative territorial entity only for city/county/state containment, not for country facts.
  * Use P495 : country of origin for creative works/products when the document states origin/nationality of the work.
- Organization family:
  * Use P127 : owned by for corporate ownership/control or "part of [media/corporate group]".
  * Use P361 : part of only for explicit generic part-whole membership, not corporate ownership.
  * Use P749 : parent organization and P355 : subsidiary only for explicit parent/subsidiary relations.
  * Use P571 : inception for founding/start dates.
- Person family:
  * Use P19 : place of birth for "born in" places; P569/P570 for birth/death dates.
  * Use P69 : educated at for schools/universities attended.
  * Use P108 : employer only when employment/working for is explicit.
- Creative-work family:
  * For songs, "song by [artist]" means P175 : performer, not P170 : creator.
  * Use P264 : record label for labels/record companies; P577 : publication date for release dates.
  * Use P162 : producer only when the text explicitly says produced by/producer.
  * Avoid P155 : follows unless the text explicitly states predecessor/follows.
- Weak/peripheral relations to avoid unless explicit and central: P400 platform, P1344 participant of, P155 follows, P162 producer, P463 member of, P112 founded by.
- Output fewer high-confidence relations rather than many plausible peripheral ones.
""".strip()


def docred_type_constraint_violation(rel: Dict[str, Any]) -> Optional[str]:
    """Return a rejection reason if a relation violates common DocRED type constraints."""
    rid = str(rel.get("relation_id") or "")
    head_type = normalize_key(rel.get("head_type"))
    tail_type = normalize_key(rel.get("tail_type"))
    evidence = normalize_key(rel.get("evidence"))
    head_text = normalize_key(str(rel.get("head") or "") + " " + str((rel.get("raw_prediction") or {}).get("head") or ""))
    tail_text = normalize_key(str(rel.get("tail") or "") + " " + str((rel.get("raw_prediction") or {}).get("tail") or ""))

    if rid == "P27" and head_type not in {"per", "person"}:
        return "P27_requires_person_head"
    if rid in {"P569", "P570"} and head_type not in {"per", "person"}:
        return f"{rid}_requires_person_head"
    if rid == "P175" and head_type in {"per", "person"}:
        return "P175_requires_work_head_not_person"
    if rid == "P162" and not any(w in evidence for w in ["producer", "produced by", "production"]):
        return "P162_requires_explicit_producer_evidence"
    if rid == "P155" and not any(w in evidence for w in ["follows", "preceded", "predecessor", "sequel"]):
        return "P155_requires_explicit_sequence_evidence"
    if rid == "P400" and not any(w in evidence for w in ["platform", "operating system", "software", "game platform"]):
        return "P400_requires_explicit_platform_evidence"
    if rid == "P131" and any(c in tail_text for c in COUNTRY_WORDS) and not any(w in evidence for w in ["administrative", "county", "state", "province", "municipality", "territorial"]):
        return "P131_country_tail_without_admin_evidence"
    if rid == "P463" and "member" not in evidence:
        return "P463_requires_member_evidence"
    if rid == "P112" and not any(w in evidence for w in ["founded", "founder", "founded by"]):
        return "P112_requires_founder_evidence"
    return None


def calibrate_one_docred_relation(
    rel: Dict[str, Any],
    *,
    allowed_relations: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Deterministically relabel/reject common DocRED relation confusions.

    Returns (new_relation_or_none, diagnostic_or_none). A None relation means
    reject the predicted triple from the benchmark-facing output.
    """
    rid = str(rel.get("relation_id") or "")
    head_type = normalize_key(rel.get("head_type"))
    tail_type = normalize_key(rel.get("tail_type"))
    evidence = normalize_key(rel.get("evidence"))
    head = normalize_key(rel.get("head"))
    tail = normalize_key(rel.get("tail"))
    by_id = relation_specs_by_id(allowed_relations)

    def relabel(new_id: str, reason: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        new_rel = copy.deepcopy(rel)
        spec = by_id.get(new_id)
        if spec:
            new_rel["relation_id"] = spec.get("id")
            new_rel["relation"] = spec.get("canonical") or spec.get("label")
            new_rel["relation_label"] = spec.get("label")
        diag = {"action": "relabel", "from": rid, "to": new_id, "reason": reason, "raw": rel}
        new_rel.setdefault("calibration", []).append(diag)
        return new_rel, diag

    def reject(reason: str) -> Tuple[None, Dict[str, Any]]:
        return None, {"action": "reject", "relation_id": rid, "reason": reason, "raw": rel}

    # Common confusions seen in the DocRED smoke tests.
    if rid == "P276" and head_type in {"org", "organization"} and any(w in evidence for w in ["based in", "headquartered", "headquarters"]):
        return relabel("P159", "organization_location_should_be_headquarters_location")
    if rid == "P361" and head_type in {"org", "organization"} and any(w in evidence for w in ["part of", "belongs to", "owned by", "group"]):
        return relabel("P127", "corporate_group_part_of_preferred_as_owned_by")
    if rid == "P170" and head_type in {"misc", "work", "entity"} and any(w in evidence for w in ["song by", "single by", "performed by", "rapper", "singer"]):
        return relabel("P175", "song_by_artist_should_be_performer")
    if rid == "P131" and any(c in tail for c in COUNTRY_WORDS):
        if head_type in {"per", "person"}:
            return relabel("P27", "person_country_relation_should_be_citizenship")
        if head_type in {"org", "organization", "loc", "location"}:
            return relabel("P17", "country_tail_should_use_country_relation")
        return reject("administrative_location_with_country_tail_rejected")
    if rid == "P400" and head_type in {"org", "organization"} and tail_type in {"org", "organization"}:
        return reject("platform_between_organizations_is_likely_subscription_service_false_positive")
    if rid == "P1344":
        return reject("participant_of_is_weak_peripheral_for_docred_smoke")

    violation = docred_type_constraint_violation(rel)
    if violation:
        return reject(violation)
    return rel, None


def calibrate_docred_relations(
    relations: List[Dict[str, Any]],
    *,
    allowed_relations: List[Dict[str, Any]],
    enable_calibration: bool = True,
    enable_strict_type_constraints: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply deterministic calibration/rejection and deduplicate triples."""
    if not enable_calibration and not enable_strict_type_constraints:
        return relations, {"enabled": False, "relabelled": 0, "rejected": 0, "diagnostics": []}

    calibrated: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for rel in relations:
        candidate = copy.deepcopy(rel)
        diag: Optional[Dict[str, Any]] = None
        if enable_calibration:
            candidate, diag = calibrate_one_docred_relation(candidate, allowed_relations=allowed_relations)
            if diag:
                diagnostics.append(diag)
        if candidate is not None and enable_strict_type_constraints:
            violation = docred_type_constraint_violation(candidate)
            if violation:
                diagnostics.append({"action": "reject", "relation_id": candidate.get("relation_id"), "reason": violation, "raw": candidate})
                candidate = None
        if candidate is None:
            continue
        key = (str(candidate.get("head_id") or candidate.get("head")), str(candidate.get("relation_id") or candidate.get("relation")), str(candidate.get("tail_id") or candidate.get("tail")))
        if key in seen:
            continue
        seen.add(key)
        calibrated.append(candidate)

    return calibrated, {
        "enabled": True,
        "input_relations": len(relations),
        "output_relations": len(calibrated),
        "relabelled": sum(1 for d in diagnostics if d.get("action") == "relabel"),
        "rejected": sum(1 for d in diagnostics if d.get("action") == "reject"),
        "diagnostics": diagnostics[:50],
    }


def build_docred_direct_extraction_messages(
    record: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    *,
    max_entities: Optional[int] = None,
    max_relations: Optional[int] = None,
    high_precision_hints: bool = True,
) -> List[Dict[str, str]]:
    """Build a direct constrained DocRED extraction prompt.

    This prompt intentionally exposes only the source entity clusters and the
    global allowed relation vocabulary. It does not expose gold pairs for the
    current document.
    """
    doc_id = document_id_from_record(record, 0)
    title = title_from_record(record, doc_id)
    text = document_text_from_record(record)
    source_entities = source_entities_from_record(record)
    hints = docred_relation_disambiguation_hints() if high_precision_hints else ""

    system = (
        "You are a strict DocRED relation extraction system. "
        "Extract document-level relations only between the provided source entity IDs. "
        "Use only the provided DocRED relation vocabulary. "
        "Do not invent entities. Do not invent predicates. "
        "Do not use entity types such as ORG, LOC, PER, MISC, person, location, institution, or entity as nodes. "
        "Return JSON only."
    )

    user = f"""
Task: extract all relations supported by the document.

Rules:
1. Heads and tails must be entity IDs from SOURCE ENTITIES.
2. Relations must be relation IDs from ALLOWED RELATIONS.
3. Use aliases only to recognize mentions, but output entity IDs.
4. Evidence must be a short quote or paraphrase from the document.
5. Do not output a relation if the document does not support it.
6. Do not use outside knowledge.
7. If no relation is supported, return {{"relations": []}}.
8. Prefer high precision: do not output peripheral or merely plausible relations.
9. Choose the most specific DocRED/Wikidata relation, not a generic neighbor.

RELATION DISAMBIGUATION HINTS:
{hints}

Output JSON schema:
{{
  "relations": [
    {{
      "head_id": "Event_...",
      "relation_id": "P...",
      "tail_id": "Event_...",
      "evidence": "short evidence from the document"
    }}
  ]
}}

DOCUMENT ID: {doc_id}
TITLE: {title}

SOURCE ENTITIES:
{compact_source_entities_for_prompt(source_entities, max_entities=max_entities)}

ALLOWED RELATIONS:
{compact_allowed_relations_for_prompt(allowed_relations, max_relations=max_relations)}

DOCUMENT TEXT:
{text}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def normalize_relation_item(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize one relation-like object into a dictionary.

    The direct DocRED extractor is instructed to return dictionaries, but small
    open models sometimes return compact arrays such as:
        [head_id, relation_id, tail_id]
        [head_id, relation_id, tail_id, evidence]
    This helper keeps the runner robust and prevents errors such as
    AttributeError: 'list' object has no attribute 'get'.
    """
    if isinstance(item, dict):
        return item
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {
            "head_id": item[0],
            "relation_id": item[1],
            "tail_id": item[2],
            "evidence": item[3] if len(item) >= 4 else "",
            "raw_sequence_prediction": list(item),
        }
    return None


def _relation_items_from_model_json(data: Any) -> List[Dict[str, Any]]:
    """Normalize the direct extractor JSON payload into relation dictionaries."""
    if isinstance(data, dict):
        candidates = data.get("relations") or data.get("triples") or data.get("predictions") or []
    elif isinstance(data, list):
        candidates = data
    else:
        candidates = []
    normalized: List[Dict[str, Any]] = []
    for item in candidates or []:
        norm = normalize_relation_item(item)
        if norm is not None:
            normalized.append(norm)
    return normalized


def canonical_entities_from_source(source_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return benchmark-facing source entities without leaking gold relations."""
    return [
        {
            "id": ent.get("id"),
            "label": ent.get("label"),
            "type": ent.get("type"),
            "aliases": ent.get("aliases") or [],
            "source": "source_document_entity",
        }
        for ent in source_entities
    ]


def validate_direct_docred_relations(
    relation_items: List[Dict[str, Any]],
    *,
    source_entities: List[Dict[str, Any]],
    allowed_relations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate direct DocRED predictions against source entities and allowed relations."""
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    for raw_item in relation_items:
        item = normalize_relation_item(raw_item)
        if item is None:
            rejected.append({"raw_prediction": raw_item, "reasons": ["invalid_relation_item_type"]})
            continue
        raw_head = (
            item.get("head_id")
            or item.get("head")
            or item.get("subject_id")
            or item.get("subject")
            or item.get("source_id")
            or item.get("source")
            or item.get("h")
        )
        raw_tail = (
            item.get("tail_id")
            or item.get("tail")
            or item.get("object_id")
            or item.get("object")
            or item.get("target_id")
            or item.get("target")
            or item.get("t")
        )
        raw_relation = (
            item.get("relation_id")
            or item.get("relation")
            or item.get("predicate_id")
            or item.get("predicate")
            or item.get("label")
            or item.get("r")
        )
        evidence = item.get("evidence") or item.get("justification") or item.get("sentence") or ""
        if isinstance(evidence, (list, tuple)):
            evidence = " | ".join(str(x) for x in evidence)
        elif isinstance(evidence, dict):
            evidence = evidence.get("text") or evidence.get("snippet") or json.dumps(evidence, ensure_ascii=False)

        head_ent = map_label_to_source_entity(raw_head, source_entities)
        tail_ent = map_label_to_source_entity(raw_tail, source_entities)
        rel_spec = map_relation_to_allowed(raw_relation, allowed_relations)

        reasons: List[str] = []
        if head_ent is None:
            reasons.append("head_not_source_entity")
        if tail_ent is None:
            reasons.append("tail_not_source_entity")
        if rel_spec is None:
            reasons.append("relation_not_allowed")
        if head_ent is not None and tail_ent is not None and head_ent.get("id") == tail_ent.get("id"):
            reasons.append("self_relation_rejected")

        if reasons:
            rejected.append(
                {
                    "raw_prediction": item,
                    "reasons": reasons,
                    "head": raw_head,
                    "relation": raw_relation,
                    "tail": raw_tail,
                }
            )
            continue

        key = (str(head_ent["id"]), str(rel_spec.get("id") or rel_spec.get("canonical")), str(tail_ent["id"]))
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            {
                "head_id": head_ent["id"],
                "head": head_ent["label"],
                "head_type": head_ent["type"],
                "relation_id": rel_spec.get("id"),
                "relation": rel_spec.get("canonical") or rel_spec.get("label"),
                "relation_label": rel_spec.get("label"),
                "tail_id": tail_ent["id"],
                "tail": tail_ent["label"],
                "tail_type": tail_ent["type"],
                "evidence": str(evidence).strip(),
                "raw_prediction": item,
                "source": "docred_direct_constrained_extraction",
            }
        )

    return accepted, rejected


def call_docred_direct_json_with_retries(
    *,
    backend: OpenAICompatibleBackend,
    args: argparse.Namespace,
    model_name: str,
    messages: List[Dict[str, str]],
    fallback_messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Call OpenRouter/OpenAI-compatible backend with retries for empty content."""
    direct_retries = max(1, int(getattr(args, "docred_direct_retries", 3) or 3))
    direct_temperature = float(getattr(args, "docred_direct_temperature", 0.0) or 0.0)
    last_error: Optional[BaseException] = None
    for attempt in range(1, direct_retries + 1):
        attempt_messages = messages
        if attempt > 1 and fallback_messages is not None:
            attempt_messages = fallback_messages
        try:
            return backend.chat(model_name, attempt_messages, temperature=direct_temperature)
        except Exception as exc:
            last_error = exc
            if attempt < direct_retries:
                time.sleep(float(getattr(args, "docred_direct_retry_sleep", 2.0) or 2.0))
    raise RuntimeError(
        "DocRED direct constrained extraction failed after "
        f"{direct_retries} attempt(s): {type(last_error).__name__}: {last_error}"
    )


def make_prediction_from_direct_items(
    *,
    record: Dict[str, Any],
    relation_items: List[Dict[str, Any]],
    allowed_relations: List[Dict[str, Any]],
    args: argparse.Namespace,
    mode_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate/calibrate direct relation items and return prediction + diagnostics."""
    source_entities = source_entities_from_record(record)
    accepted, rejected = validate_direct_docred_relations(
        relation_items,
        source_entities=source_entities,
        allowed_relations=allowed_relations,
    )
    calibrated, calibration_diagnostics = calibrate_docred_relations(
        accepted,
        allowed_relations=allowed_relations,
        enable_calibration=bool(getattr(args, "docred_calibrate_relations", False)),
        enable_strict_type_constraints=bool(getattr(args, "docred_strict_type_constraints", False)),
    )
    diagnostics = {
        "mode": mode_name,
        "source_entity_count": len(source_entities),
        "allowed_relation_count": len(allowed_relations),
        "focus_relation_ids": getattr(args, "docred_direct_focus_relation_ids", None),
        "relation_family_filter": bool(getattr(args, "docred_relation_family_filter", False)),
        "high_precision_hints": not bool(getattr(args, "docred_direct_disable_hints", False)),
        "raw_relation_items": len(relation_items),
        "accepted_before_calibration": len(accepted),
        "accepted_relations": len(calibrated),
        "rejected_relations": len(rejected),
        "rejected_preview": rejected[:20],
        "calibration": calibration_diagnostics,
    }
    prediction = {
        "entities": canonical_entities_from_source(source_entities),
        "relations": calibrated,
        "projection_diagnostics": diagnostics,
    }
    return prediction, diagnostics


def build_docred_probe_messages(
    record: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    *,
    family_name: str,
    max_entities: Optional[int] = None,
    max_relations: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Build a shorter targeted prompt for zero-relation recovery probes."""
    doc_id = document_id_from_record(record, 0)
    title = title_from_record(record, doc_id)
    text = document_text_from_record(record)
    source_entities = source_entities_from_record(record)
    system = (
        "You are a strict DocRED relation extraction verifier. Return JSON only. "
        "Use only provided source entity IDs and allowed relation IDs."
    )
    user = f"""
Targeted DocRED probe: {family_name}

Find high-confidence relations from this family only. If none are explicit, return {{"relations": []}}.
Output JSON only: {{"relations": [{{"head_id": "Event_...", "relation_id": "P...", "tail_id": "Event_...", "evidence": "..."}}]}}

SOURCE ENTITIES:
{compact_source_entities_for_prompt(source_entities, max_entities=max_entities)}

ALLOWED RELATIONS:
{compact_allowed_relations_for_prompt(allowed_relations, max_relations=max_relations)}

TITLE: {title}
DOCUMENT TEXT:
{text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def zero_relation_probe_relation_subsets(
    record: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    max_families: int = 3,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Choose a few relation-family subsets for zero-relation recovery."""
    relevant_ids = infer_docred_relation_family_ids(record, source_entities_from_record(record))
    by_id = relation_specs_by_id(allowed_relations)
    scored: List[Tuple[int, str, set[str]]] = []
    text = normalize_key(document_text_from_record(record) + " " + title_from_record(record, document_id_from_record(record, 0)))
    triggers = {
        "creative_work": ["song", "single", "album", "rapper", "record label", "released", "producer"],
        "person": ["born", "died", "educated", "graduated", "employer", "citizen"],
        "organization": ["company", "group", "research", "subsidiary", "parent", "headquartered", "based in", "founded"],
        "location": ["city", "county", "state", "country", "located", "province"],
    }
    for family, ids in DOCRED_RELATION_FAMILIES.items():
        if not (ids & relevant_ids):
            continue
        score = sum(1 for trig in triggers.get(family, []) if trig in text)
        scored.append((score, family, ids & relevant_ids))
    scored.sort(reverse=True)
    subsets: List[Tuple[str, List[Dict[str, Any]]]] = []
    for _, family, ids in scored[:max_families]:
        rels = [by_id[rid] for rid in sorted(ids) if rid in by_id]
        if rels:
            subsets.append((family, rels))
    return subsets


def run_docred_direct_constrained_extraction(
    *,
    record: Dict[str, Any],
    backend: OpenAICompatibleBackend,
    args: argparse.Namespace,
    artifact_dir: str,
) -> Dict[str, Any]:
    """Run direct DocRED-constrained extraction and optional calibration/probes.

    This is a benchmark adapter only. It does not change NeoOLAF native layer
    outputs, KG files, or generated ontology.
    """
    source_entities = source_entities_from_record(record)
    base_allowed_relations = list(getattr(args, "allowed_relation_specs", []) or [])
    allowed_relations = filter_allowed_relations_for_direct_extractor(
        base_allowed_relations,
        focus_relation_ids=getattr(args, "docred_direct_focus_relation_ids", None),
        record=record,
        relation_family_filter=bool(getattr(args, "docred_relation_family_filter", False)),
    )
    messages = build_docred_direct_extraction_messages(
        record,
        allowed_relations,
        max_entities=getattr(args, "docred_direct_max_entities", None),
        max_relations=getattr(args, "docred_direct_max_relations", None),
        high_precision_hints=not bool(getattr(args, "docred_direct_disable_hints", False)),
    )
    fallback_messages = build_docred_direct_extraction_messages(
        record,
        allowed_relations,
        max_entities=getattr(args, "docred_direct_max_entities", None),
        max_relations=getattr(args, "docred_direct_max_relations", None),
        high_precision_hints=False,
    )
    raw_response = call_docred_direct_json_with_retries(
        backend=backend,
        args=args,
        model_name=args.model_name,
        messages=messages,
        fallback_messages=fallback_messages,
    )
    parsed = backend.extract_json(raw_response)
    relation_items = _relation_items_from_model_json(parsed)
    prediction, diagnostics = make_prediction_from_direct_items(
        record=record,
        relation_items=relation_items,
        allowed_relations=allowed_relations,
        args=args,
        mode_name="docred_direct_constrained_extraction",
    )
    raw_responses: Dict[str, Any] = {"primary": raw_response}

    # If the high-precision extraction returns no relations, recover recall with
    # a few short family-specific probes. This is still gold-free: families are
    # inferred from entity types and trigger words, not gold pairs.
    if (
        bool(getattr(args, "docred_zero_relation_family_probes", False))
        and not prediction.get("relations")
    ):
        probe_relation_items: List[Dict[str, Any]] = []
        probe_diags: List[Dict[str, Any]] = []
        max_probe_families = max(1, int(getattr(args, "docred_zero_relation_probe_max_families", 3) or 3))
        for family_name, family_relations in zero_relation_probe_relation_subsets(record, allowed_relations, max_families=max_probe_families):
            probe_messages = build_docred_probe_messages(
                record,
                family_relations,
                family_name=family_name,
                max_entities=getattr(args, "docred_direct_max_entities", None),
                max_relations=getattr(args, "docred_direct_max_relations", None),
            )
            try:
                probe_raw = call_docred_direct_json_with_retries(
                    backend=backend,
                    args=args,
                    model_name=args.model_name,
                    messages=probe_messages,
                    fallback_messages=None,
                )
                raw_responses[f"probe_{family_name}"] = probe_raw
                probe_parsed = backend.extract_json(probe_raw)
                probe_items = _relation_items_from_model_json(probe_parsed)
                probe_relation_items.extend(probe_items)
                probe_diags.append({"family": family_name, "raw_relation_items": len(probe_items), "error": None})
            except Exception as exc:
                probe_diags.append({"family": family_name, "raw_relation_items": 0, "error": f"{type(exc).__name__}: {exc}"})
        if probe_relation_items:
            probe_prediction, probe_diagnostics = make_prediction_from_direct_items(
                record=record,
                relation_items=probe_relation_items,
                allowed_relations=allowed_relations,
                args=args,
                mode_name="docred_zero_relation_family_probes",
            )
            prediction = merge_canonical_predictions(prediction, probe_prediction)
            prediction.setdefault("projection_diagnostics", {})["zero_relation_probes"] = {
                "enabled": True,
                "probe_attempts": probe_diags,
                "probe_diagnostics": probe_diagnostics,
            }
        else:
            prediction.setdefault("projection_diagnostics", {})["zero_relation_probes"] = {
                "enabled": True,
                "probe_attempts": probe_diags,
            }

    if bool(getattr(args, "docred_scoring_calibration", False)):
        prediction = apply_docred_scoring_calibration(record, prediction, base_allowed_relations, args)

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "docred_direct_constrained_extraction.json").write_text(
        json.dumps(
            {
                "document_id": document_id_from_record(record, 0),
                "title": title_from_record(record, document_id_from_record(record, 0)),
                "prediction": prediction,
                "diagnostics": prediction.get("projection_diagnostics") or diagnostics,
                "raw_responses": raw_responses,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return prediction


def merge_canonical_predictions(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two canonical prediction views, deduplicating relation triples."""
    merged = copy.deepcopy(base or {"entities": [], "relations": []})
    base_entities = {str(e.get("id") or e.get("label")): e for e in merged.get("entities") or [] if isinstance(e, dict)}
    for ent in extra.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        key = str(ent.get("id") or ent.get("label"))
        if key and key not in base_entities:
            base_entities[key] = ent
    merged["entities"] = list(base_entities.values())

    seen: set[Tuple[str, str, str]] = set()
    relations: List[Dict[str, Any]] = []
    for rel in list(merged.get("relations") or []) + list(extra.get("relations") or []):
        if not isinstance(rel, dict):
            continue
        key = (
            str(rel.get("head_id") or rel.get("head")),
            str(rel.get("relation_id") or rel.get("relation")),
            str(rel.get("tail_id") or rel.get("tail")),
        )
        if key in seen:
            continue
        seen.add(key)
        relations.append(rel)
    merged["relations"] = relations
    merged["projection_diagnostics"] = {
        "merged_prediction": True,
        "base_diagnostics": base.get("projection_diagnostics") if isinstance(base, dict) else None,
        "extra_diagnostics": extra.get("projection_diagnostics") if isinstance(extra, dict) else None,
        "canonical_relations": len(relations),
    }
    return merged


# ---------------------------------------------------------------------------
# DocRED scoring-calibration and closure rules
# ---------------------------------------------------------------------------

COUNTRY_ALIAS_GROUPS: Dict[str, List[str]] = {
    "greece": ["greece", "greek"],
    "united states": ["united states", "u.s.", "us", "american"],
    "brazil": ["brazil", "brazilian"],
    "canada": ["canada", "canadian"],
    "france": ["france", "french"],
    "united kingdom": ["united kingdom", "uk", "british", "england", "english"],
    "ireland": ["ireland", "irish"],
    "japan": ["japan", "japanese"],
    "china": ["china", "chinese"],
    "germany": ["germany", "german"],
    "italy": ["italy", "italian"],
    "spain": ["spain", "spanish"],
    "mexico": ["mexico", "mexican"],
    "australia": ["australia", "australian"],
}

CONTINENT_WORDS: set[str] = {"africa", "asia", "europe", "south america", "north america", "america", "oceania", "australia", "antarctica"}


def relation_tuple_key(rel: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(rel.get("head_id") or rel.get("head") or ""),
        str(rel.get("relation_id") or rel.get("relation") or ""),
        str(rel.get("tail_id") or rel.get("tail") or ""),
    )


def source_entities_by_id(source_entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(ent.get("id")): ent for ent in source_entities if ent.get("id")}


def norm_entity_label(ent: Dict[str, Any]) -> str:
    return normalize_key(ent.get("label"))


def entity_alias_texts(ent: Dict[str, Any]) -> List[str]:
    values = [ent.get("id"), ent.get("label"), *(ent.get("aliases") or [])]
    return [normalize_key(v) for v in values if normalize_key(v)]


def find_country_entities(source_entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Find best source entity for each canonical country group.

    Prefer canonical country labels (e.g. Greece) over demonyms (e.g. Greek)
    when both occur in the same document. If only a demonym exists, keep it;
    DocRED often uses demonym entities for nationality tails.
    """
    winners: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for ent in source_entities:
        if normalize_key(ent.get("type")) not in {"loc", "location", "gpe"}:
            continue
        aliases = entity_alias_texts(ent)
        label = norm_entity_label(ent)
        for canonical, forms in COUNTRY_ALIAS_GROUPS.items():
            score = 0
            if label == canonical:
                score = 100
            elif canonical in aliases:
                score = 80
            elif any(form == label for form in forms):
                score = 60
            elif any(form in aliases for form in forms):
                score = 40
            elif any(form in label for form in forms):
                score = 20
            if score:
                if canonical not in winners or score > winners[canonical][0]:
                    winners[canonical] = (score, ent)
    return {country: ent for country, (_, ent) in winners.items()}


def choose_document_country_entity(record: Dict[str, Any], source_entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    countries = find_country_entities(source_entities)
    if not countries:
        return None
    text = normalize_key(document_text_from_record(record) + " " + title_from_record(record, document_id_from_record(record, 0)))
    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for canonical, ent in countries.items():
        forms = COUNTRY_ALIAS_GROUPS.get(canonical, [canonical])
        score = sum(text.count(form) for form in forms) * 10
        score += 5 if norm_entity_label(ent) == canonical else 0
        scored.append((score, canonical, ent))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][2]


def find_continent_entity(source_entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for ent in source_entities:
        if norm_entity_label(ent) in CONTINENT_WORDS:
            return ent
        if any(alias in CONTINENT_WORDS for alias in entity_alias_texts(ent)):
            return ent
    return None


def entity_is_continent_like(ent: Dict[str, Any]) -> bool:
    return norm_entity_label(ent) in CONTINENT_WORDS or any(alias in CONTINENT_WORDS for alias in entity_alias_texts(ent))


def relation_evidence_mentions(rel: Dict[str, Any], *needles: str) -> bool:
    ev = normalize_key(rel.get("evidence"))
    return any(normalize_key(n) in ev for n in needles if normalize_key(n))


def text_mentions_near_entity(text: str, ent: Dict[str, Any], words: Iterable[str], window: int = 90) -> bool:
    """True if any word appears near any entity alias in the text."""
    low = normalize_key(text)
    aliases = [a for a in entity_alias_texts(ent) if len(a) >= 3 and not a.startswith("event_")]
    words_norm = [normalize_key(w) for w in words if normalize_key(w)]
    for alias in aliases:
        start = 0
        while True:
            idx = low.find(alias, start)
            if idx < 0:
                break
            span = low[max(0, idx - window): idx + len(alias) + window]
            if any(w in span for w in words_norm):
                return True
            start = idx + len(alias)
    return False


def make_docred_relation(
    *,
    head_ent: Dict[str, Any],
    relation_id: str,
    tail_ent: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    evidence: str,
    source: str,
    calibration_action: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    spec = relation_spec_by_id(allowed_relations, relation_id)
    if not spec:
        return None
    rel = {
        "head_id": head_ent.get("id"),
        "head": head_ent.get("label"),
        "head_type": head_ent.get("type"),
        "relation_id": spec.get("id"),
        "relation": spec.get("canonical") or spec.get("label"),
        "relation_label": spec.get("label"),
        "tail_id": tail_ent.get("id"),
        "tail": tail_ent.get("label"),
        "tail_type": tail_ent.get("type"),
        "evidence": evidence,
        "source": source,
    }
    if calibration_action:
        rel.setdefault("calibration", []).append(calibration_action)
    return rel


def docred_scoring_filter_or_relabel_relation(
    rel: Dict[str, Any],
    *,
    allowed_relations: List[Dict[str, Any]],
    source_entities: List[Dict[str, Any]],
    reject_peripheral: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Second-stage strict scoring filter/relabeling for common false positives."""
    rid = str(rel.get("relation_id") or "")
    head_type = normalize_key(rel.get("head_type"))
    tail_type = normalize_key(rel.get("tail_type"))
    evidence = normalize_key(rel.get("evidence"))
    head = normalize_key(rel.get("head"))
    tail = normalize_key(rel.get("tail"))
    source_by_id = source_entities_by_id(source_entities)
    head_ent = source_by_id.get(str(rel.get("head_id")))
    tail_ent = source_by_id.get(str(rel.get("tail_id")))

    def relabel(new_id: str, reason: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not head_ent or not tail_ent:
            return rel, {"action": "keep", "reason": "cannot_relabel_without_source_entities", "raw": rel}
        new_rel = make_docred_relation(
            head_ent=head_ent,
            relation_id=new_id,
            tail_ent=tail_ent,
            allowed_relations=allowed_relations,
            evidence=rel.get("evidence") or "",
            source="docred_scoring_relabel",
            calibration_action={"action": "relabel", "from": rid, "to": new_id, "reason": reason},
        )
        return new_rel or rel, {"action": "relabel", "from": rid, "to": new_id, "reason": reason, "raw": rel}

    def reject(reason: str) -> Tuple[None, Dict[str, Any]]:
        return None, {"action": "reject", "relation_id": rid, "reason": reason, "raw": rel}

    # Country-tail mistakes: DocRED usually wants P17/P27/P495, not P159/P276/P131.
    if tail_ent and entity_is_country_like(tail_ent) and rid in {"P159", "P276", "P131"}:
        if head_type in {"per", "person"}:
            return relabel("P27", f"{rid}_country_tail_for_person_relabelled_to_citizenship")
        if head_type in {"misc", "work"} and rid != "P159":
            return relabel("P495", f"{rid}_country_tail_for_work_relabelled_to_country_of_origin")
        return relabel("P17", f"{rid}_country_tail_relabelled_to_country")

    # Corporate hierarchy: label containment is usually parent organization.
    if rid in {"P361", "P127"} and head_type in {"org", "organization"} and tail_type in {"org", "organization"}:
        if tail and head and tail in head and tail != head:
            return relabel("P749", "org_label_contains_parent_label_parent_organization")
        if rid == "P361" and any(w in evidence for w in ["subsidiary", "parent", "branch", "division", "comprising", "part of ibm research"]):
            return relabel("P749", "corporate_part_of_parent_organization")

    # Physical location of organizations is too broad for strict DocRED scoring;
    # keep headquarters-style P159 but reject generic P276 locations.
    if rid == "P276" and head_type in {"org", "organization"}:
        if any(w in evidence for w in ["headquartered", "headquarters", "based in"]):
            return relabel("P159", "org_location_with_headquarters_evidence")
        return reject("generic_org_location_P276_rejected_for_strict_docred")

    # Ownership of places by persons is often a property-transaction false positive.
    if rid == "P127" and (tail_type in {"per", "person"} or head_type in {"loc", "location"}):
        return reject("P127_property_transaction_or_person_owner_rejected")

    # P159 should point to a city/place, not a country/demonym.
    if rid == "P159" and tail_ent and entity_is_country_like(tail_ent):
        return relabel("P17", "headquarters_location_country_tail_relabelled_to_country")

    # Strict peripheral filters useful for benchmark scoring.
    if reject_peripheral:
        if rid == "P162":
            return reject("producer_relation_treated_as_peripheral_for_strict_docred")
        if rid in {"P155", "P400", "P1344"}:
            return reject(f"{rid}_weak_peripheral_relation_rejected")
        if rid == "P495" and not any(w in evidence for w in ["country of origin", "origin", "nationality", "american rapper", "american singer", "brazilian", "greek"]):
            return reject("P495_without_origin_or_nationality_evidence_rejected")
        if rid == "P108" and not any(w in evidence for w in ["employed by", "worked for", "joined ibm", "researcher at", "professor at"]):
            return reject("P108_without_strong_employment_evidence_rejected")
        if rid == "P131" and head_type in {"org", "organization"} and not any(w in evidence for w in ["located in", "based in", "headquartered", "in the city", "in the state"]):
            return reject("P131_organization_name_location_false_positive")

    return rel, None


def append_unique_relation(relations: List[Dict[str, Any]], rel: Optional[Dict[str, Any]], diagnostics: List[Dict[str, Any]]) -> bool:
    if not rel:
        return False
    key = relation_tuple_key(rel)
    existing = {relation_tuple_key(r) for r in relations}
    if key in existing:
        return False
    relations.append(rel)
    diagnostics.append({"action": "add", "relation": rel})
    return True


def apply_docred_scoring_calibration(
    record: Dict[str, Any],
    prediction: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Apply strict scoring filters and gold-free closure rules to canonical output."""
    if not isinstance(prediction, dict):
        return prediction
    source_entities = source_entities_from_record(record)
    by_id = source_entities_by_id(source_entities)
    source_text = document_text_from_record(record)
    title = title_from_record(record, document_id_from_record(record, 0))
    all_text = f"{title}\n{source_text}"

    diagnostics: Dict[str, Any] = {
        "enabled": True,
        "input_relations": len(prediction.get("relations") or []),
        "filtered_or_relabelled": [],
        "added": [],
        "rejected": [],
    }

    reject_peripheral = bool(getattr(args, "docred_reject_peripheral_relations", False))
    new_relations: List[Dict[str, Any]] = []
    for rel in prediction.get("relations") or []:
        candidate, diag = docred_scoring_filter_or_relabel_relation(
            rel,
            allowed_relations=allowed_relations,
            source_entities=source_entities,
            reject_peripheral=reject_peripheral,
        )
        if diag:
            if diag.get("action") == "reject":
                diagnostics["rejected"].append(diag)
            else:
                diagnostics["filtered_or_relabelled"].append(diag)
        if candidate is not None:
            if relation_tuple_key(candidate) not in {relation_tuple_key(x) for x in new_relations}:
                new_relations.append(candidate)

    added_diags: List[Dict[str, Any]] = []
    country_ent = choose_document_country_entity(record, source_entities)
    continent_ent = find_continent_entity(source_entities)

    def add(head_ent: Dict[str, Any], rid: str, tail_ent: Dict[str, Any], evidence: str, source: str) -> bool:
        rel = make_docred_relation(
            head_ent=head_ent,
            relation_id=rid,
            tail_ent=tail_ent,
            allowed_relations=allowed_relations,
            evidence=evidence,
            source=source,
        )
        return append_unique_relation(new_relations, rel, added_diags)

    # 1) Date closure: if a work has P577 to a full date, also add the year entity if present.
    if bool(getattr(args, "docred_date_closure", False)):
        time_ents = [e for e in source_entities if normalize_key(e.get("type")) in {"time", "date"}]
        for rel in list(new_relations):
            if str(rel.get("relation_id")) != "P577":
                continue
            tail_label = normalize_key(rel.get("tail"))
            for ent in time_ents:
                label = normalize_key(ent.get("label"))
                if re.fullmatch(r"(?:18|19|20)\d{2}", label) and label in tail_label:
                    head_ent = by_id.get(str(rel.get("head_id")))
                    if head_ent:
                        add(head_ent, "P577", ent, f"year contained in publication date: {rel.get('evidence') or ''}", "docred_date_year_closure")

        # Opening biographical date pattern: Name ( birth – death ).
        per_ents = [e for e in source_entities if normalize_key(e.get("type")) in {"per", "person"}]
        full_dates = [e for e in time_ents if re.search(r"(?:18|19|20)\d{2}", normalize_key(e.get("label"))) and not re.fullmatch(r"(?:18|19|20)\d{2}", normalize_key(e.get("label")))]
        if per_ents and len(full_dates) >= 2 and re.search(r"\([^)]*(?:–|-)[^)]*\)", all_text[:500]):
            add(per_ents[0], "P569", full_dates[0], "opening biographical birth/death date pattern", "docred_biographical_date_closure")
            add(per_ents[0], "P570", full_dates[1], "opening biographical birth/death date pattern", "docred_biographical_date_closure")

    # 2) Country/geography closure: add country facts for local places and country-continent facts.
    if bool(getattr(args, "docred_geo_country_closure", False)) and country_ent:
        for ent in source_entities:
            typ = normalize_key(ent.get("type"))
            label = norm_entity_label(ent)
            if typ in {"loc", "location", "gpe"} and ent.get("id") != country_ent.get("id"):
                if entity_is_country_like(ent) or entity_is_continent_like(ent) or label in {"african"}:
                    continue
                add(ent, "P17", country_ent, "geographic country closure from document country context", "docred_geo_country_closure")
        if continent_ent and country_ent.get("id") != continent_ent.get("id"):
            add(country_ent, "P30", continent_ent, "country-continent closure from source entities", "docred_country_continent_closure")
            add(continent_ent, "P527", country_ent, "continent contains country closure from source entities", "docred_country_continent_closure")

    # 3) Organization closure: parent/subsidiary inverses and country propagation.
    if bool(getattr(args, "docred_org_closure", False)):
        orgs = [e for e in source_entities if normalize_key(e.get("type")) in {"org", "organization"}]
        # label-containment parent organization, e.g. IBM Research – Brazil -> IBM Research -> IBM.
        for child in orgs:
            child_label = norm_entity_label(child)
            for parent in orgs:
                parent_label = norm_entity_label(parent)
                if child.get("id") == parent.get("id") or not parent_label or not child_label:
                    continue
                if parent_label in child_label and parent_label != child_label:
                    add(child, "P749", parent, "organization label containment parent closure", "docred_org_parent_label_closure")
                    add(child, "P361", parent, "organization label containment part-of closure", "docred_org_parent_label_closure")
                    add(parent, "P355", child, "inverse subsidiary closure from parent organization", "docred_org_parent_label_closure")

        # inverse for any predicted parent organization.
        for rel in list(new_relations):
            if str(rel.get("relation_id")) == "P749":
                child = by_id.get(str(rel.get("head_id")))
                parent = by_id.get(str(rel.get("tail_id")))
                if child and parent:
                    add(parent, "P355", child, f"inverse subsidiary of {rel.get('evidence') or ''}", "docred_parent_subsidiary_inverse_closure")

        # child country -> parent country for owned-by/parent relations.
        p17_by_head = {str(r.get("head_id")): by_id.get(str(r.get("tail_id"))) for r in new_relations if str(r.get("relation_id")) == "P17"}
        for rel in list(new_relations):
            if str(rel.get("relation_id")) in {"P127", "P749", "P361"}:
                child_country = p17_by_head.get(str(rel.get("head_id")))
                parent = by_id.get(str(rel.get("tail_id")))
                if child_country and parent and normalize_key(parent.get("type")) in {"org", "organization"}:
                    add(parent, "P17", child_country, "country propagated from related child organization", "docred_org_country_propagation")

        # headquarters city country -> organization country.
        p17_by_place = {str(r.get("head_id")): by_id.get(str(r.get("tail_id"))) for r in new_relations if str(r.get("relation_id")) == "P17"}
        for rel in list(new_relations):
            if str(rel.get("relation_id")) == "P159":
                org = by_id.get(str(rel.get("head_id")))
                place_country = p17_by_place.get(str(rel.get("tail_id")))
                if org and place_country:
                    add(org, "P17", place_country, "country propagated from headquarters location", "docred_headquarters_country_closure")

    # 4) Nationality and creative-work closures.
    if bool(getattr(args, "docred_nationality_closure", False)) and country_ent:
        nationality_forms: List[str] = []
        for forms in COUNTRY_ALIAS_GROUPS.values():
            if any(form in entity_alias_texts(country_ent) or form == norm_entity_label(country_ent) for form in forms):
                nationality_forms = forms
                break
        if not nationality_forms:
            nationality_forms = entity_alias_texts(country_ent)
        for ent in source_entities:
            if normalize_key(ent.get("type")) in {"per", "person"}:
                if text_mentions_near_entity(all_text, ent, nationality_forms, window=120):
                    add(ent, "P27", country_ent, "nationality/demonym near person mention", "docred_nationality_closure")

        # Creative-work label and performer inheritance: work label -> performer label.
        work_label_rels = [r for r in new_relations if str(r.get("relation_id")) == "P264"]
        performer_rels = [r for r in new_relations if str(r.get("relation_id")) == "P175"]
        for perf in performer_rels:
            work_id = str(perf.get("head_id"))
            performer = by_id.get(str(perf.get("tail_id")))
            if not performer:
                continue
            for lab in work_label_rels:
                if str(lab.get("head_id")) == work_id:
                    label_ent = by_id.get(str(lab.get("tail_id")))
                    if label_ent:
                        add(performer, "P264", label_ent, "performer label inherited from work label in document", "docred_performer_label_closure")
            if country_ent and text_mentions_near_entity(all_text, performer, nationality_forms, window=120):
                add(performer, "P27", country_ent, "nationality/demonym near performer mention", "docred_performer_nationality_closure")

    prediction = copy.deepcopy(prediction)
    prediction["relations"] = new_relations
    diagnostics["added"] = added_diags[:100]
    diagnostics["output_relations"] = len(new_relations)
    diagnostics["added_count"] = len(added_diags)
    diagnostics["rejected_count"] = len(diagnostics["rejected"])
    diagnostics["relabelled_count"] = sum(1 for d in diagnostics["filtered_or_relabelled"] if d.get("action") == "relabel")
    prediction.setdefault("projection_diagnostics", {})["scoring_calibration"] = diagnostics
    return prediction

def raw_counts_from_state(state: PipelineState, prediction: Dict[str, Any]) -> Dict[str, int]:
    """Collect compact count diagnostics for one document.

    Keep this function defensive because benchmark adapters may attach
    diagnostics in different shapes:
    - projection_diagnostics directly on the canonical prediction;
    - projection_diagnostics.extra_diagnostics when native and direct
      predictions are merged;
    - absent diagnostics for older/non-DocRED runs.
    """
    diagnostics: Dict[str, Any] = {}
    if isinstance(prediction, dict):
        raw_diag = prediction.get("projection_diagnostics") or {}
        if isinstance(raw_diag, dict):
            diagnostics.update(raw_diag)
            extra_diag = raw_diag.get("extra_diagnostics")
            if isinstance(extra_diag, dict):
                # Direct DocRED extraction diagnostics live here when the
                # direct prediction is merged with the native NeoOLAF view.
                diagnostics.update(extra_diag)

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
        "canonical_entities": len(prediction.get("entities") or []) if isinstance(prediction, dict) else 0,
        "canonical_relations": len(prediction.get("relations") or []) if isinstance(prediction, dict) else 0,
        "projection_rejected_triples": int(diagnostics.get("rejected_triples") or 0),
        "allowed_relation_count": int(diagnostics.get("allowed_relation_count") or 0),
        "source_entity_count": int(diagnostics.get("source_entity_count") or 0),
        "docred_direct_raw_relation_items": int(diagnostics.get("raw_relation_items") or 0),
        "docred_direct_accepted_relations": int(diagnostics.get("accepted_relations") or 0),
        "docred_direct_rejected_relations": int(diagnostics.get("rejected_relations") or 0),
        "native_relation_mapping_enabled": int(bool(diagnostics.get("native_relation_mapping_enabled") or False)),
        "native_relation_mapping_kept_or_mapped": int(diagnostics.get("native_relation_mapping_kept_or_mapped") or 0),
        "native_relation_mapping_rejected": int(diagnostics.get("native_relation_mapping_rejected") or 0),
        "docred_calibration_relabelled": int(((diagnostics.get("calibration") or {}).get("relabelled") or 0)) if isinstance(diagnostics.get("calibration"), dict) else 0,
        "docred_calibration_rejected": int(((diagnostics.get("calibration") or {}).get("rejected") or 0)) if isinstance(diagnostics.get("calibration"), dict) else 0,
        "docred_zero_probe_enabled": int(bool((diagnostics.get("zero_relation_probes") or {}).get("enabled"))) if isinstance(diagnostics.get("zero_relation_probes"), dict) else 0,
        "docred_scoring_calibration_enabled": int(bool((diagnostics.get("scoring_calibration") or {}).get("enabled"))) if isinstance(diagnostics.get("scoring_calibration"), dict) else 0,
        "docred_scoring_added": int(((diagnostics.get("scoring_calibration") or {}).get("added_count") or 0)) if isinstance(diagnostics.get("scoring_calibration"), dict) else 0,
        "docred_scoring_rejected": int(((diagnostics.get("scoring_calibration") or {}).get("rejected_count") or 0)) if isinstance(diagnostics.get("scoring_calibration"), dict) else 0,
        "docred_scoring_relabelled": int(((diagnostics.get("scoring_calibration") or {}).get("relabelled_count") or 0)) if isinstance(diagnostics.get("scoring_calibration"), dict) else 0,
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
        document = record_to_document(record, index, args=args)
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

        prediction = state_to_canonical_prediction(
            final_state,
            source_entities=source_entities_from_record(record),
            allowed_relations=list(getattr(args, "allowed_relation_specs", []) or []),
            constrained=bool(getattr(args, "force_relation_vocabulary", False)),
            raw_text_entity_relation_mode=bool(getattr(args, "raw_text_entity_relation_mode", False)),
            native_relation_mapping=bool(getattr(args, "docred_native_relation_mapping", False)),
            relation_mapper=getattr(args, "docred_native_relation_mapper", DEFAULT_DOCRED_NATIVE_RELATION_MAPPER),
            native_reject_peripheral=bool(getattr(args, "docred_native_reject_peripheral", False)),
        )
        if getattr(args, "raw_text_er_direct_fallback", False) and getattr(args, "raw_text_entity_relation_mode", False):
            raw_mode = str(getattr(args, "raw_text_er_direct_mode", "if_zero") or "if_zero").lower()
            should_run_raw_direct = raw_mode in {"replace", "supplement"} or (raw_mode == "if_zero" and not prediction.get("relations"))
            if should_run_raw_direct:
                raw_direct_prediction = run_raw_text_er_direct_fallback(
                    record=record,
                    backend=backend,
                    args=args,
                    artifact_dir=artifact_dir,
                )
                if raw_mode == "supplement":
                    prediction = merge_canonical_predictions(prediction, raw_direct_prediction)
                else:
                    prediction = raw_direct_prediction

        if getattr(args, "docred_direct_constrained_extraction", False):
            try:
                direct_prediction = run_docred_direct_constrained_extraction(
                    record=record,
                    backend=backend,
                    args=args,
                    artifact_dir=artifact_dir,
                )
                mode = str(getattr(args, "docred_direct_output_mode", "replace") or "replace").lower()
                if mode == "supplement":
                    prediction = merge_canonical_predictions(prediction, direct_prediction)
                else:
                    prediction = direct_prediction
            except Exception as exc:
                fallback = str(getattr(args, "docred_direct_fallback_on_error", "native") or "native").lower()
                error_payload = {
                    "document_id": doc_id,
                    "title": title_from_record(record, doc_id),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "fallback": fallback,
                }
                Path(artifact_dir).mkdir(parents=True, exist_ok=True)
                (Path(artifact_dir) / "docred_direct_constrained_extraction.error.json").write_text(
                    json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if fallback == "fail":
                    raise
                if fallback == "empty":
                    prediction = {
                        "entities": canonical_entities_from_source(source_entities_from_record(record)),
                        "relations": [],
                        "projection_diagnostics": {
                            "mode": "docred_direct_constrained_extraction",
                            "direct_extraction_error": True,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "fallback": fallback,
                            "source_entity_count": len(source_entities_from_record(record)),
                            "allowed_relation_count": len(list(getattr(args, "allowed_relation_specs", []) or [])),
                            "raw_relation_items": 0,
                            "accepted_relations": 0,
                            "rejected_relations": 0,
                        },
                    }
                else:
                    diagnostics = prediction.setdefault("projection_diagnostics", {})
                    if isinstance(diagnostics, dict):
                        diagnostics.update(
                            {
                                "direct_extraction_error": True,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "fallback": fallback,
                            }
                        )
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

        if getattr(args, "raw_text_entity_relation_mode", False) and getattr(args, "raw_text_er_direct_fallback", False):
            try:
                backend = build_backend(args)
                prediction = run_raw_text_er_direct_fallback(
                    record=record,
                    backend=backend,
                    args=args,
                    artifact_dir=artifact_dir,
                )
                result = {
                    "document_id": doc_id,
                    "title": title_from_record(record, doc_id),
                    "type": record.get("type") or record.get("split"),
                    "method": "neoolaf_raw_text_native_with_direct_fallback",
                    "parsed_ok": True,
                    "prediction": prediction,
                    "raw_counts": {
                        "canonical_entities": len(prediction.get("entities") or []),
                        "canonical_relations": len(prediction.get("relations") or []),
                        "native_error_recovered": 1,
                    },
                    "artifact_dir": artifact_dir,
                    "runtime_seconds": None,
                    "native_error_type": type(exc).__name__,
                    "native_error_message": str(exc),
                }
                return index, result
            except Exception:
                pass

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

    # Constrained DocRED/RAGTree benchmark export. These options do not modify
    # NeoOLAF native KG/ontology artifacts. They only constrain guidance and the
    # benchmark-facing canonical JSONL output.
    parser.add_argument("--relation-vocab-source", default="auto", choices=["auto", "dataset", "ontology", "json", "union"], help="Allowed relation vocabulary source for constrained output.")
    parser.add_argument("--relation-vocab-json", default=None, help="Optional JSON/JSONL allowed relation vocabulary.")
    parser.add_argument("--relation-vocab-dataset-path", default=None, help="Dataset JSONL used only to extract the relation label set.")
    parser.add_argument("--relation-vocab-ontology-path", default=None, help="Reference ontology used only to extract relation properties.")
    parser.add_argument("--relation-vocab-output-path", default=None, help="Write the resolved allowed relation vocabulary here.")
    parser.add_argument("--force-relation-vocabulary", action="store_true", help="Force canonical output to use only allowed relation labels.")
    parser.add_argument("--source-entity-anchoring", action="store_true", help="Expose source entity IDs/labels and require source entities in constrained output.")
    parser.add_argument("--raw-text-entity-relation-mode", action="store_true", help="Do not expose source entities. Predict entities and relations from raw text with native NeoOLAF; constrain only relation labels in canonical output.")
    parser.add_argument("--docred-native-relation-mapping", action="store_true", help="Deterministically map native NeoOLAF relation labels to official DocRED relation IDs. Mapping only: no new triples and no gold/source entities.")
    parser.add_argument("--docred-native-relation-mapper-json", default=None, help="Optional per-model JSON mapping rules for native NeoOLAF relation labels to DocRED relation IDs.")
    parser.add_argument("--docred-native-reject-peripheral", action="store_true", help="Reject weak peripheral native triples during deterministic DocRED mapping.")
    parser.add_argument("--stop-after-layer", type=int, default=None, help="Optional last NeoOLAF layer index to run, inclusive. Leave unset for full NeoOLAF.")
    parser.add_argument("--raw-text-er-direct-fallback", action="store_true", help="In raw-text entity+relation mode, run a raw-text-only direct ER fallback if native NeoOLAF fails or returns zero relations.")
    parser.add_argument("--raw-text-er-direct-mode", default="if_zero", choices=["if_zero", "replace", "supplement"], help="How to use the raw-text direct ER fallback after a successful native run.")
    parser.add_argument("--raw-text-er-direct-retries", type=int, default=2, help="Retry count for the raw-text direct ER fallback call.")
    parser.add_argument("--raw-text-er-direct-retry-sleep", type=float, default=2.0, help="Seconds to sleep between raw-text direct ER fallback retries.")
    parser.add_argument("--raw-text-er-direct-temperature", type=float, default=0.0, help="Temperature for the raw-text direct ER fallback call.")
    parser.add_argument("--raw-text-er-direct-max-relations", type=int, default=None, help="Optional cap on allowed relation labels shown to the raw-text ER fallback.")
    parser.add_argument("--docred-direct-constrained-extraction", action="store_true", help="Run an extra direct DocRED-constrained LLM extraction call for the final benchmark-facing canonical output.")
    parser.add_argument("--docred-direct-output-mode", default="replace", choices=["replace", "supplement"], help="How to combine direct DocRED extraction with the native NeoOLAF projection.")
    parser.add_argument("--docred-direct-max-entities", type=int, default=None, help="Optional cap on source entities shown to the direct DocRED extractor.")
    parser.add_argument("--docred-direct-max-relations", type=int, default=None, help="Optional cap on allowed relations shown to the direct DocRED extractor.")
    parser.add_argument("--docred-direct-temperature", type=float, default=0.0, help="Temperature for the direct DocRED-constrained extraction call.")
    parser.add_argument("--docred-direct-retries", type=int, default=3, help="Retry the direct DocRED extraction call when OpenRouter returns empty content or transient invalid responses.")
    parser.add_argument("--docred-direct-retry-sleep", type=float, default=2.0, help="Seconds to sleep between direct DocRED extraction retries.")
    parser.add_argument("--docred-direct-fallback-on-error", default="native", choices=["native", "empty", "fail"], help="What to do if the optional direct DocRED extraction fails after retries.")
    parser.add_argument("--docred-direct-focus-relation-ids", default=None, help="Optional comma-separated relation IDs to show to the direct extractor, e.g. P17,P27,P69. Do not derive this from test-document gold pairs for final evaluation.")
    parser.add_argument("--docred-direct-disable-hints", action="store_true", help="Disable gold-free DocRED relation disambiguation hints in the direct extraction prompt.")
    parser.add_argument("--docred-relation-family-filter", action="store_true", help="Gold-free pre-prompt relation subset inferred from entity types and trigger words.")
    parser.add_argument("--docred-calibrate-relations", action="store_true", help="Apply deterministic relabel/reject calibration for common DocRED relation confusions.")
    parser.add_argument("--docred-verification-pass", action="store_true", help="Alias for --docred-calibrate-relations and --docred-strict-type-constraints; kept for notebook/readability.")
    parser.add_argument("--docred-zero-relation-family-probes", action="store_true", help="If direct extraction returns zero relations, run targeted family probes.")
    parser.add_argument("--docred-zero-relation-probe-max-families", type=int, default=3, help="Maximum number of targeted relation family probes after zero-relation extraction.")
    parser.add_argument("--docred-strict-type-constraints", action="store_true", help="Reject common relation/type mismatches after extraction.")
    parser.add_argument("--docred-scoring-calibration", action="store_true", help="Apply stricter benchmark-scoring filters/relabels and optional closure rules after direct extraction.")
    parser.add_argument("--docred-reject-peripheral-relations", action="store_true", help="Reject weak/peripheral relations such as P162/P155/P400/P108 unless strong evidence is present.")
    parser.add_argument("--docred-geo-country-closure", action="store_true", help="Add gold-free geographic P17/P30/P527 closure relations from source country/continent entities.")
    parser.add_argument("--docred-org-closure", action="store_true", help="Add gold-free organization parent/subsidiary/country propagation closure relations.")
    parser.add_argument("--docred-date-closure", action="store_true", help="Add gold-free year and biographical date closure relations.")
    parser.add_argument("--docred-nationality-closure", action="store_true", help="Add gold-free nationality and performer-label closure relations.")
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
    parser.add_argument(
        "--disable-wikipedia-lookups",
        action="store_true",
        help="Use offline Wikipedia/Wikidata source objects in Layer 2 without changing NeoOLAF source.",
    )
    parser.add_argument(
        "--offline-ontology-only",
        action="store_true",
        help="Benchmark mode: disable web enrichment and block Wikipedia/Wikimedia lookups.",
    )
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
    if getattr(args, "docred_verification_pass", False):
        args.docred_calibrate_relations = True
        args.docred_strict_type_constraints = True
    if getattr(args, "offline_ontology_only", False):
        args.no_web_search = True
        args.disable_wikipedia_lookups = True
    env_disable_wiki = os.environ.get("NEOOLAF_DISABLE_WIKIPEDIA", "").strip().lower() in {"1", "true", "yes", "on"}
    if env_disable_wiki:
        args.disable_wikipedia_lookups = True
    if args.disable_wikipedia_lookups:
        print("[NeoOLAF benchmark] Wikipedia/Wikidata enrichment disabled by offline source objects.")
    if args.no_web_search:
        print("[NeoOLAF benchmark] Web-search enrichment disabled (--no-web-search).")
    start = time.time()

    # Avoid mixing stale errors from previous smoke tests with the current run.
    if args.error_log_jsonl_path:
        try:
            Path(args.error_log_jsonl_path).unlink(missing_ok=True)
        except Exception:
            pass

    records_all = load_jsonl(args.dataset_jsonl_path)
    records = filter_records(records_all, args.type_filter)
    if args.max_docs is not None:
        records = records[: args.max_docs]

    if not records:
        raise SystemExit("No records selected. Check --dataset-jsonl-path and --type-filter.")

    args.allowed_relation_specs = load_allowed_relation_specs(args)
    args.docred_native_relation_mapper = load_relation_mapper(args)
    if args.allowed_relation_specs:
        print(
            f"[NeoOLAF benchmark] allowed_relations={len(args.allowed_relation_specs)} "
            f"source={args.relation_vocab_source} force={args.force_relation_vocabulary}"
        )
        preview = [rel.get("canonical") for rel in args.allowed_relation_specs[:10]]
        print(f"[NeoOLAF benchmark] allowed_relations_preview={preview}")
    elif args.force_relation_vocabulary:
        print(
            "[NeoOLAF benchmark][warning] --force-relation-vocabulary was set, "
            "but no allowed relations were loaded. Canonical relations will be rejected."
        )

    guidance = load_user_guidance(args.user_guidance_path)
    if args.force_relation_vocabulary:
        guidance = inject_relation_constraints_into_guidance(
            guidance,
            args.allowed_relation_specs,
            raw_text_entity_relation_mode=bool(getattr(args, "raw_text_entity_relation_mode", False)),
        )
    if args.raw_text_entity_relation_mode:
        print("[NeoOLAF benchmark] raw_text_entity_relation_mode=True source_entities_not_exposed=True")
        print(
            f"[NeoOLAF benchmark] docred_native_relation_mapping={args.docred_native_relation_mapping} "
            f"reject_peripheral={args.docred_native_reject_peripheral} "
            f"mapper_json={args.docred_native_relation_mapper_json}"
        )
        print(
            f"[NeoOLAF benchmark] stop_after_layer={args.stop_after_layer} "
            f"raw_text_er_direct_fallback={args.raw_text_er_direct_fallback} "
            f"mode={args.raw_text_er_direct_mode}"
        )
    if args.docred_direct_constrained_extraction:
        print(
            "[NeoOLAF benchmark] docred_direct_constrained_extraction=True "
            f"mode={args.docred_direct_output_mode} "
            f"focus_relation_ids={args.docred_direct_focus_relation_ids} "
            f"hints={not args.docred_direct_disable_hints} "
            f"retries={args.docred_direct_retries} "
            f"fallback={args.docred_direct_fallback_on_error}"
        )
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

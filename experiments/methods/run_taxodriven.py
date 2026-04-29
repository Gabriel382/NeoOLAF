from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm.auto import tqdm
from rdflib import Graph, Literal

# ============================================================
# Resolve project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXODRIVEN_DIR = PROJECT_ROOT / "examples" / "approaches" / "TaxoDrivenKG"
COMMON_DIR = PROJECT_ROOT / "experiments" / "common"

if str(TAXODRIVEN_DIR) not in sys.path:
    sys.path.insert(0, str(TAXODRIVEN_DIR))

if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

# ============================================================
# Local imports
# ============================================================
from jsonl_adapter import count_documents, iter_documents  # type: ignore
from taxonomy import OntologyRetriever  # type: ignore
from backends.openai_compatible import OpenAICompatibleBackend  # type: ignore
from backends.ollama_backend import OllamaBackend  # type: ignore


# ============================================================
# Backend
# ============================================================
def build_backend(
    backend_name: str,
    host: str,
    api_key: str = "dummy",
    referer: str = "",
    title: str = "TaxoDrivenKG-JSONL",
):
    """Build the requested backend."""
    backend_name = backend_name.strip().lower()

    if backend_name == "ollama":
        return OllamaBackend(host=host)

    extra_headers = None
    if backend_name == "openrouter":
        extra_headers = {}
        if referer:
            extra_headers["HTTP-Referer"] = referer
        if title:
            extra_headers["X-Title"] = title

    return OpenAICompatibleBackend(
        base_url=host,
        api_key=api_key,
        extra_headers=extra_headers,
    )


# ============================================================
# Small helpers
# ============================================================
def normalize_text_field(value: Any) -> str:
    """Convert a possibly missing field into a clean string."""
    if value is None:
        return ""
    return str(value).strip()


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSON object as one JSONL line."""
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_processed_ids(output_jsonl_path: Path) -> Set[str]:
    """Load already processed ids from output JSONL for resume mode."""
    processed: Set[str] = set()

    if not output_jsonl_path.exists():
        return processed

    with output_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                doc_id = str(row.get("document_id", "")).strip()
                if doc_id:
                    processed.add(doc_id)
            except Exception:
                continue

    return processed


def safe_document_id(doc: Dict[str, Any], fallback_index: int) -> str:
    """Return a robust document id."""
    doc_id = normalize_text_field(doc.get("document_id"))
    if doc_id:
        return doc_id

    title = normalize_text_field(doc.get("title"))
    if title:
        return f"doc_{fallback_index:07d}_{title}"

    return f"doc_{fallback_index:07d}"


def build_document_text(doc: Dict[str, Any]) -> str:
    """
    Build document text from common fields.

    Priority:
    1. text
    2. sentences
    3. tokens
    """
    text = normalize_text_field(doc.get("text"))
    if text:
        return text

    sentences = doc.get("sentences")
    if isinstance(sentences, list) and sentences:
        merged = "\n".join(str(x).strip() for x in sentences if str(x).strip())
        if merged.strip():
            return merged

    tokens = doc.get("tokens")
    if isinstance(tokens, list) and tokens:
        rebuilt_sentences: List[str] = []
        for sent in tokens:
            if isinstance(sent, list):
                rebuilt = " ".join(str(tok) for tok in sent).strip()
                if rebuilt:
                    rebuilt_sentences.append(rebuilt)

        if rebuilt_sentences:
            return "\n".join(rebuilt_sentences)

    return ""


def parse_manual_candidates(manual_candidates: str) -> List[str]:
    """Parse comma-separated manual candidates."""
    if not manual_candidates.strip():
        return []
    return [x.strip() for x in manual_candidates.split(",") if x.strip()]


# ============================================================
# Ontology helpers
# ============================================================
def summarize_ontology(ontology_path: Path) -> Dict[str, Any]:
    """Load ontology and return a lightweight summary for debug."""
    summary: Dict[str, Any] = {
        "path": str(ontology_path),
        "exists": ontology_path.exists(),
        "triples": 0,
        "labels_preview": [],
        "comments_preview": [],
        "parse_ok": False,
        "parse_error": None,
    }

    if not ontology_path.exists():
        summary["parse_error"] = "Ontology path does not exist."
        return summary

    graph = Graph()
    parse_errors = []

    for fmt in ["turtle", "xml"]:
        try:
            graph.parse(str(ontology_path), format=fmt)
            summary["parse_ok"] = True
            break
        except Exception as e:
            parse_errors.append(f"{fmt}: {e}")

    if not summary["parse_ok"]:
        summary["parse_error"] = " | ".join(parse_errors)
        return summary

    summary["triples"] = len(graph)

    labels: List[str] = []
    comments: List[str] = []

    for _, pred, obj in graph:
        pred_s = str(pred).lower()
        if "label" in pred_s and isinstance(obj, Literal):
            labels.append(str(obj))
        if "comment" in pred_s and isinstance(obj, Literal):
            comments.append(str(obj))

    summary["labels_preview"] = labels[:20]
    summary["comments_preview"] = comments[:10]
    return summary


def normalize_retrieved_nodes(raw_retrieved_nodes: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize ontology retrieval output into a dict format:
    {
      label: {"uri": ..., "label": ..., "type": ..., "score": ...}
    }
    """
    if isinstance(raw_retrieved_nodes, dict):
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in raw_retrieved_nodes.items():
            label = str(key).strip()
            if not label:
                continue

            if isinstance(value, dict):
                normalized[label] = {
                    "uri": str(value.get("uri", "")).strip(),
                    "label": str(value.get("label", label)).strip() or label,
                    "type": str(value.get("type", "unknown")).strip(),
                    "score": float(value.get("score", 1.0)),
                }
            else:
                normalized[label] = {
                    "uri": "",
                    "label": label,
                    "type": "unknown",
                    "score": 1.0,
                }
        return normalized

    if isinstance(raw_retrieved_nodes, list):
        normalized = {}
        for idx, item in enumerate(raw_retrieved_nodes):
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip() or f"candidate_{idx}"
                normalized[label] = {
                    "uri": str(item.get("uri", "")).strip(),
                    "label": label,
                    "type": str(item.get("type", "unknown")).strip(),
                    "score": float(item.get("score", 1.0)),
                }
            else:
                label = str(item).strip()
                if label:
                    normalized[label] = {
                        "uri": "",
                        "label": label,
                        "type": "unknown",
                        "score": 1.0,
                    }
        return normalized

    raise TypeError(
        f"Unsupported retrieved_nodes type: {type(raw_retrieved_nodes).__name__}"
    )


# ============================================================
# Prompt helpers
# ============================================================
def format_candidates_for_prompt(retrieved_nodes: Dict[str, Dict[str, Any]]) -> str:
    """Format ontology candidates for the prompt."""
    if not retrieved_nodes:
        return ""

    return ", ".join(retrieved_nodes.keys())


def format_output_records_for_example(example_doc: Dict[str, Any], max_records: int = 40) -> str:
    """
    Build TaxoDrivenKG-style example output from one dataset entry.

    We use gold entities and gold relations from the JSONL entry itself.
    """
    entities = example_doc.get("entities", {})
    relations = example_doc.get("relations", {})

    if not isinstance(entities, dict):
        entities = {}
    if not isinstance(relations, dict):
        relations = {}

    records: List[str] = []
    added_entities: Set[str] = set()

    # Convert entities dict -> entity records.
    for entity_id, entity_info in entities.items():
        if not isinstance(entity_info, dict):
            continue

        ent_type = normalize_text_field(entity_info.get("type")) or "UNKNOWN"
        mentions = entity_info.get("mentions", [])

        if isinstance(mentions, list) and mentions:
            mention = mentions[0]
            if isinstance(mention, dict):
                trigger = normalize_text_field(mention.get("trigger_word"))
                if trigger and trigger not in added_entities:
                    records.append(
                        f'("entity"<|>{trigger}<|>{ent_type}<|>{ent_type} entity)'
                    )
                    added_entities.add(trigger)

    # Build id -> surface label map.
    id_to_label: Dict[str, str] = {}
    for entity_id, entity_info in entities.items():
        if not isinstance(entity_info, dict):
            continue
        mentions = entity_info.get("mentions", [])
        if isinstance(mentions, list) and mentions:
            mention = mentions[0]
            if isinstance(mention, dict):
                trigger = normalize_text_field(mention.get("trigger_word"))
                if trigger:
                    id_to_label[str(entity_id)] = trigger

    # Convert relation dict -> relationship records.
    for rel_name, pairs in relations.items():
        rel_label = normalize_text_field(rel_name)
        if not rel_label:
            continue

        if isinstance(pairs, list):
            for pair in pairs:
                if not (isinstance(pair, list) or isinstance(pair, tuple)) or len(pair) != 2:
                    continue

                src_id = str(pair[0])
                tgt_id = str(pair[1])

                src_label = id_to_label.get(src_id, "")
                tgt_label = id_to_label.get(tgt_id, "")

                if src_label and tgt_label:
                    records.append(
                        f'("relationship"<|>{src_label}<|>{tgt_label}<|>{rel_label})'
                    )

    return "##".join(records[:max_records])


def build_few_shot_examples_text(
    examples: List[Dict[str, Any]],
    max_chars_per_example: int = 1500,
) -> str:
    """
    Build the '-Examples-' prompt block from dataset entries.
    """
    blocks: List[str] = []

    for ex in examples:
        ex_text = build_document_text(ex)
        if not ex_text:
            continue

        ex_text = ex_text[:max_chars_per_example]
        ex_output = format_output_records_for_example(ex)

        # Skip empty examples.
        if not ex_output.strip():
            continue

        block = (
            "Example:\n"
            f"Text: {ex_text}\n"
            "Output:\n"
            f"{ex_output}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_taxodriven_prompt(
    text: str,
    retrieved_nodes: Dict[str, Dict[str, Any]],
    few_shot_examples_text: str,
) -> str:
    """
    Build the TaxoDrivenKG prompt text.
    """
    candidates_text = format_candidates_for_prompt(retrieved_nodes)

    prompt = f"""
-Goal-
Given a text document and a list of potential entities from a domain taxonomy, identify all entities and relationships that are explicitly supported by the text.

-Steps-
1. Extract all named or domain-relevant entities in the text. For each entity, output:
   - entity_name
   - entity_type
   - entity_description

2. Extract all explicit relationships between the extracted entities. For each relationship, output:
   - source_entity
   - target_entity
   - relationship_type

3. Return the answer in English as a single list of records separated by **##**.

Use exactly these formats:
("entity"<|><entity_name><|><entity_type><|><entity_description>)
("relationship"<|><source_entity><|><target_entity><|><relationship_type>)

Potential entity candidates from the ontology: {candidates_text}

######################
-Examples-
{few_shot_examples_text}
######################
-Real Data-
######################
Text: {text}
######################
Output:
""".strip()

    return prompt


# ============================================================
# Backend calling and parsing
# ============================================================
def call_backend_chat(
    backend: Any,
    backend_name: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """
    Unified backend call for the three supported backends.
    """
    backend_name = backend_name.lower()

    if backend_name == "ollama":
        response = backend.chat(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if isinstance(response, str):
            return response
        return str(response)

    # OpenAI-compatible wrappers for vllm/openrouter.
    response = backend.chat(
        messages=messages,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if isinstance(response, str):
        return response

    return str(response)


def parse_taxodriven_output(content: str) -> Dict[str, Any]:
    """
    Parse TaxoDrivenKG string output into structured entities/relationships.

    Expected records:
    ("entity"<|>name<|>type<|>description)
    ("relationship"<|>source<|>target<|>relation)
    """
    content = content.strip()

    if not content:
        return {"entities": [], "relationships": []}

    parts = [p.strip() for p in content.split("##") if p.strip()]

    entities: List[Dict[str, str]] = []
    relationships: List[Dict[str, str]] = []

    for part in parts:
        part = part.strip()

        if not part.startswith("(") or not part.endswith(")"):
            # Skip incomplete/truncated records.
            continue

        inner = part[1:-1]
        fields = inner.split("<|>")

        if not fields:
            continue

        record_type = fields[0].strip().strip('"').strip("'").lower()

        if record_type == "entity" and len(fields) >= 4:
            entities.append(
                {
                    "name": fields[1].strip(),
                    "label": fields[2].strip(),
                    "description": fields[3].strip(),
                }
            )

        elif record_type == "relationship" and len(fields) >= 4:
            relationships.append(
                {
                    "source": fields[1].strip(),
                    "target": fields[2].strip(),
                    "relation": fields[3].strip(),
                }
            )

    return {"entities": entities, "relationships": relationships}


# ============================================================
# Few-shot selection from the dataset itself
# ============================================================
def select_few_shot_examples_from_dataset(
    dataset_jsonl_path: Path,
    current_document_id: str,
    few_shot_source_type: str,
    few_shot_k: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Select few-shot examples from the same dataset JSONL, line by line.

    Rules:
    - never use the current document as an example
    - if few_shot_source_type == "all", accept all types
    - otherwise match exact type
    - keep only entries that seem to have entities or relations
    """
    if few_shot_k <= 0:
        return []

    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0

    type_filter = few_shot_source_type

    for doc in iter_documents(dataset_jsonl_path, type_filter=type_filter):
        document_id = normalize_text_field(doc.get("document_id"))
        if document_id == current_document_id:
            continue

        entities = doc.get("entities", {})
        relations = doc.get("relations", {})

        has_entities = isinstance(entities, dict) and len(entities) > 0
        has_relations = isinstance(relations, dict) and len(relations) > 0

        if not (has_entities or has_relations):
            continue

        seen += 1

        if len(reservoir) < few_shot_k:
            reservoir.append(doc)
        else:
            j = rng.randint(1, seen)
            if j <= few_shot_k:
                reservoir[j - 1] = doc

    return reservoir


# ============================================================
# Debug helpers
# ============================================================
def maybe_print_debug_header(
    doc_idx: int,
    document_id: str,
    title: str,
    doc_type: str,
    full_text: str,
) -> None:
    """Pretty debug header."""
    print("\n" + "=" * 80)
    print(f"[DEBUG] DOC #{doc_idx}")
    print(f"[DEBUG] document_id: {document_id}")
    print(f"[DEBUG] title      : {title}")
    print(f"[DEBUG] type       : {doc_type}")
    print(f"[DEBUG] chars      : {len(full_text)}")
    print(f"[DEBUG] text[:800] :\n{full_text[:800]}")
    print("=" * 80)


# ============================================================
# Per-document run
# ============================================================
def run_one_document(
    doc_idx: int,
    doc: Dict[str, Any],
    backend: Any,
    backend_name: str,
    model_name: str,
    chunker_text_max_tokens: int,
    retriever: Optional[OntologyRetriever],
    max_taxonomy_hits: int,
    dataset_jsonl_path: Path,
    few_shot_source_type: str,
    few_shot_k: int,
    debug: bool = False,
    debug_show_prompt: bool = False,
    debug_show_output: bool = False,
    manual_candidates: Optional[List[str]] = None,
    generation_max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Run TaxoDrivenKG extraction for one document."""
    document_id = normalize_text_field(doc.get("document_id"))
    doc_type = normalize_text_field(doc.get("type"))
    title = normalize_text_field(doc.get("title"))
    full_text = build_document_text(doc)

    if not full_text:
        return {
            "document_id": document_id,
            "title": title,
            "type": doc_type,
            "status": "skipped_empty_text",
            "num_chars": 0,
            "num_chunks": 0,
            "outputs": {},
            "prompts": {},
            "conversations": {},
            "retrieved_nodes": {},
            "few_shot_examples": [],
        }

    if debug:
        maybe_print_debug_header(doc_idx, document_id, title, doc_type, full_text)

    # Build few-shot examples once per document.
    few_shot_examples = select_few_shot_examples_from_dataset(
        dataset_jsonl_path=dataset_jsonl_path,
        current_document_id=document_id,
        few_shot_source_type=few_shot_source_type,
        few_shot_k=few_shot_k,
        seed=42,
    )
    few_shot_examples_text = build_few_shot_examples_text(few_shot_examples)

    # Chunk text.
    # Keep simple, consistent with TaxoDrivenKG spirit.
    # We use one chunk if small enough, otherwise rough char-based split fallback.
    # Since original TextChunker from TaxoDrivenKG is not used here, we avoid
    # hidden few-shot dependencies there.
    text_chunks: List[str]
    start_idxs: List[int]

    if len(full_text) <= chunker_text_max_tokens * 6:
        text_chunks = [full_text]
        start_idxs = [0]
    else:
        approx_chunk_chars = chunker_text_max_tokens * 6
        text_chunks = []
        start_idxs = []
        start = 0
        while start < len(full_text):
            end = min(len(full_text), start + approx_chunk_chars)
            text_chunks.append(full_text[start:end])
            start_idxs.append(start)
            start = end

    outputs: Dict[str, Any] = {}
    prompts: Dict[str, str] = {}
    conversations: Dict[str, Any] = {}
    retrieved_by_chunk: Dict[str, Any] = {}

    for chunk_i, (text, start_idx) in enumerate(zip(text_chunks, start_idxs), start=1):
        end_idx = start_idx + len(text)
        span_key = str((start_idx, end_idx))

        if manual_candidates is not None and len(manual_candidates) > 0:
            raw_retrieved_nodes = [
                {
                    "uri": "",
                    "label": cand,
                    "type": "manual",
                    "score": 1.0,
                }
                for cand in manual_candidates
            ]
        else:
            if retriever is None:
                raise ValueError("Retriever is None and no manual candidates were provided.")
            raw_retrieved_nodes = retriever.retrieve(text, max_hits=max_taxonomy_hits)

        retrieved_nodes = normalize_retrieved_nodes(raw_retrieved_nodes)
        retrieved_by_chunk[span_key] = retrieved_nodes

        prompt = build_taxodriven_prompt(
            text=text,
            retrieved_nodes=retrieved_nodes,
            few_shot_examples_text=few_shot_examples_text,
        )
        prompts[span_key] = prompt

        messages = [{"role": "user", "content": prompt}]
        assistant_content = call_backend_chat(
            backend=backend,
            backend_name=backend_name,
            model_name=model_name,
            messages=messages,
            max_tokens=generation_max_tokens,
            temperature=0.0,
        )

        parsed_output = parse_taxodriven_output(assistant_content)
        outputs[span_key] = parsed_output
        conversations[span_key] = messages + [{"role": "assistant", "content": assistant_content}]

        if debug:
            print(f"\n[DEBUG] chunk {chunk_i}/{len(text_chunks)} span={span_key}")
            print(f"[DEBUG] chunk text[:500]:\n{text[:500]}")
            print(f"[DEBUG] retrieved_nodes count: {len(retrieved_nodes)}")
            for k, v in list(retrieved_nodes.items())[:20]:
                print(f"   - {k}: {v}")

        if debug and debug_show_prompt:
            print("\n[DEBUG] PROMPT")
            print("-" * 60)
            print(prompt)
            print("-" * 60)

        if debug and debug_show_output:
            print("\n[DEBUG] ASSISTANT CONTENT")
            print("-" * 60)
            print(assistant_content)
            print("-" * 60)
            print("\n[DEBUG] PARSED OUTPUT")
            print("-" * 60)
            print(parsed_output)
            print("-" * 60)

    return {
        "document_id": document_id,
        "title": title,
        "type": doc_type,
        "status": "ok",
        "num_chars": len(full_text),
        "num_chunks": len(text_chunks),
        "outputs": outputs,
        "prompts": prompts,
        "conversations": conversations,
        "retrieved_nodes": retrieved_by_chunk,
        "few_shot_examples": [
            {
                "document_id": normalize_text_field(ex.get("document_id")),
                "title": normalize_text_field(ex.get("title")),
                "type": normalize_text_field(ex.get("type")),
            }
            for ex in few_shot_examples
        ],
    }


# ============================================================
# Dataset runner
# ============================================================
def run_taxodriven_dataset(
    dataset_jsonl_path: str | Path,
    ontology_path: str | Path,
    output_jsonl_path: str | Path,
    backend_name: str,
    host: str,
    model_name: str,
    api_key: str = "dummy",
    referer: str = "",
    title: str = "TaxoDrivenKG-JSONL",
    type_filter: str = "all",
    chunk_tokens: int = 600,
    max_taxonomy_hits: int = 40,
    resume: bool = True,
    debug: bool = False,
    debug_max_docs: int = 3,
    debug_show_prompt: bool = False,
    debug_show_output: bool = False,
    manual_candidates: Optional[List[str]] = None,
    few_shot_source_type: str = "all",
    few_shot_k: int = 3,
    generation_max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Run TaxoDrivenKG line by line on a JSONL dataset."""
    dataset_jsonl_path = Path(dataset_jsonl_path)
    ontology_path = Path(ontology_path)
    output_jsonl_path = Path(output_jsonl_path)

    if not dataset_jsonl_path.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {dataset_jsonl_path}")

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

    ensure_parent_dir(output_jsonl_path)

    if debug:
        ontology_summary = summarize_ontology(ontology_path)
        print("\n" + "=" * 80)
        print("[DEBUG] ONTOLOGY SUMMARY")
        print(json.dumps(ontology_summary, indent=2, ensure_ascii=False))
        print("=" * 80)

    processed_ids: Set[str] = set()
    if resume:
        processed_ids = load_processed_ids(output_jsonl_path)

    backend = build_backend(
        backend_name=backend_name,
        host=host,
        api_key=api_key,
        referer=referer,
        title=title,
    )

    retriever = None if manual_candidates else OntologyRetriever(str(ontology_path))

    total_docs = count_documents(dataset_jsonl_path, type_filter=type_filter)
    docs_iter = iter_documents(dataset_jsonl_path, type_filter=type_filter)

    start_time = dt.datetime.now()
    seen = 0
    done = 0
    skipped_resume = 0
    failed = 0

    for row_idx, doc in enumerate(
        tqdm(docs_iter, total=total_docs, desc="TaxoDrivenKG docs", unit="doc"),
        start=1,
    ):
        seen += 1
        document_id = safe_document_id(doc, row_idx)

        if document_id in processed_ids:
            skipped_resume += 1
            continue

        try:
            doc = dict(doc)
            doc["document_id"] = document_id

            local_debug = debug and done < debug_max_docs

            result_row = run_one_document(
                doc_idx=row_idx,
                doc=doc,
                backend=backend,
                backend_name=backend_name,
                model_name=model_name,
                chunker_text_max_tokens=chunk_tokens,
                retriever=retriever,
                max_taxonomy_hits=max_taxonomy_hits,
                dataset_jsonl_path=dataset_jsonl_path,
                few_shot_source_type=few_shot_source_type,
                few_shot_k=few_shot_k,
                debug=local_debug,
                debug_show_prompt=debug_show_prompt,
                debug_show_output=debug_show_output,
                manual_candidates=manual_candidates,
                generation_max_tokens=generation_max_tokens,
            )

            append_jsonl(output_jsonl_path, result_row)
            done += 1

        except KeyboardInterrupt:
            raise

        except Exception as e:
            failed += 1
            error_row = {
                "document_id": document_id,
                "title": normalize_text_field(doc.get("title")),
                "type": normalize_text_field(doc.get("type")),
                "status": "error",
                "error": str(e),
            }
            append_jsonl(output_jsonl_path, error_row)

            if debug:
                print(f"\n[DEBUG] ERROR on document {document_id}: {e}")

    end_time = dt.datetime.now()

    summary = {
        "dataset_jsonl_path": str(dataset_jsonl_path),
        "ontology_path": str(ontology_path),
        "output_jsonl_path": str(output_jsonl_path),
        "backend_name": backend_name,
        "host": host,
        "model_name": model_name,
        "type_filter": type_filter,
        "chunk_tokens": chunk_tokens,
        "max_taxonomy_hits": max_taxonomy_hits,
        "few_shot_source_type": few_shot_source_type,
        "few_shot_k": few_shot_k,
        "generation_max_tokens": generation_max_tokens,
        "resume": resume,
        "seen": seen,
        "done": done,
        "skipped_resume": skipped_resume,
        "failed": failed,
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
    }

    return summary


# ============================================================
# CLI
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run TaxoDrivenKG on a JSONL dataset line-by-line."
    )

    parser.add_argument("--dataset-jsonl-path", required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--ontology-path", required=True, help="Path to ontology file.")
    parser.add_argument("--output-jsonl-path", required=True, help="Path to output JSONL predictions.")

    parser.add_argument(
        "--backend-name",
        required=True,
        choices=["ollama", "vllm", "openrouter"],
        help="Backend type.",
    )
    parser.add_argument("--host", required=True, help="Backend host URL.")
    parser.add_argument("--api-key", default="dummy", help="API key for OpenAI-compatible backends.")
    parser.add_argument("--model-name", required=True, help="Model name exposed by backend.")

    parser.add_argument("--type-filter", default="all", help='Filter by "type". Use "all" for no filter.')
    parser.add_argument("--chunk-tokens", type=int, default=600, help="Approx chunk size in tokens.")
    parser.add_argument("--max-taxonomy-hits", type=int, default=40, help="Max ontology retrieval hits.")
    parser.add_argument("--referer", default="", help="Optional HTTP-Referer for OpenRouter.")
    parser.add_argument("--title", default="TaxoDrivenKG-JSONL", help="Optional X-Title for OpenRouter.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume mode.")

    parser.add_argument("--few-shot-source-type", default="all", help='Type used to sample few-shot examples from the same dataset. Use "all" for all types.')
    parser.add_argument("--few-shot-k", type=int, default=3, help="Number of few-shot examples to sample from the dataset.")
    parser.add_argument("--generation-max-tokens", type=int, default=4096, help="Max generation tokens for the model.")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--debug-max-docs", type=int, default=3, help="How many processed docs to debug.")
    parser.add_argument("--debug-show-prompt", action="store_true", help="Print prompts in debug mode.")
    parser.add_argument("--debug-show-output", action="store_true", help="Print raw outputs in debug mode.")
    parser.add_argument(
        "--manual-candidates",
        default="",
        help="Comma-separated manual ontology candidates to bypass retrieval for debugging.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_argparser()
    args = parser.parse_args()

    manual_candidates = parse_manual_candidates(args.manual_candidates)

    summary = run_taxodriven_dataset(
        dataset_jsonl_path=args.dataset_jsonl_path,
        ontology_path=args.ontology_path,
        output_jsonl_path=args.output_jsonl_path,
        backend_name=args.backend_name,
        host=args.host,
        model_name=args.model_name,
        api_key=args.api_key,
        referer=args.referer,
        title=args.title,
        type_filter=args.type_filter,
        chunk_tokens=args.chunk_tokens,
        max_taxonomy_hits=args.max_taxonomy_hits,
        resume=not args.no_resume,
        debug=args.debug,
        debug_max_docs=args.debug_max_docs,
        debug_show_prompt=args.debug_show_prompt,
        debug_show_output=args.debug_show_output,
        manual_candidates=manual_candidates if manual_candidates else None,
        few_shot_source_type=args.few_shot_source_type,
        few_shot_k=args.few_shot_k,
        generation_max_tokens=args.generation_max_tokens,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
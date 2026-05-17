# diagnose_taxodrivenkg.py
# ------------------------------------------------------------
# Full diagnosis script for TaxoDrivenKG benchmark outputs.
#
# It checks:
# 1. Dataset/gold annotation structure.
# 2. Whether TaxoDrivenKG prediction JSONL exists and is valid.
# 3. Whether predictions are truly empty.
# 4. Whether assistant responses are empty.
# 5. Whether prompts have empty few-shot examples.
# 6. Whether ontology candidates look suspicious.
# 7. Whether prediction format is compatible with eval_relations.py.
# 8. Whether document IDs match between gold and prediction files.
# 9. Whether the backend returns a non-empty answer on a tiny test.
# 10. Optional: run TaxoDrivenKG before diagnosing.
#
# Usage example:
#
# python diagnose_taxodrivenkg.py \
#   --dataset-jsonl-path "../../../ragtree/data/preprocessed/docred_causal.jsonl" \
#   --ontology-path "../../../ragtree/data/ontology/DocREDOntology/ontology.ttl" \
#   --output-jsonl-path "./runs/taxodrive_docred_predictions.jsonl" \
#   --run-taxodriven-path "../../experiments/methods/run_taxodriven.py" \
#   --backend-name vllm \
#   --host "http://localhost:8000/v1" \
#   --api-key "dummy" \
#   --model-name "$MODEL_NAME" \
#   --type-filter dev \
#   --few-shot-source-type dev \
#   --few-shot-k 3 \
#   --backend-smoke-test
#
# Add --run-method if you want this script to launch run_taxodriven.py first.
# Add --force-rerun to delete/overwrite behavior depending on your runner.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Basic helpers
# ============================================================

def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    """Stream JSONL records from a file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}, line {line_number}: {e}") from e


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Write a JSON report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def shorten(text: str, max_len: int = 500) -> str:
    """Shorten long text for printing/reporting."""
    text = str(text or "")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [TRUNCATED]"


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary values."""
    cur: Any = d

    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)

    return cur if cur is not None else default


def normalize_type_filter(type_filter: str) -> str | List[str]:
    """Parse type filter into all/string/list."""
    type_filter = str(type_filter).strip()

    if type_filter.lower() == "all":
        return "all"

    if "," in type_filter:
        return [x.strip() for x in type_filter.split(",") if x.strip()]

    return type_filter


def doc_matches_type(doc: Dict[str, Any], type_filter: str | List[str]) -> bool:
    """Return whether a document matches the requested type filter."""
    if type_filter == "all":
        return True

    doc_type = str(doc.get("type", "")).strip()

    if isinstance(type_filter, list):
        return doc_type in type_filter

    return doc_type == str(type_filter)


# ============================================================
# Gold dataset inspection
# ============================================================

def extract_gold_entity_labels(doc: Dict[str, Any]) -> List[str]:
    """
    Extract gold entity surface labels from either:
    - original DocRED-like dict format
    - normalized list format
    """
    entities = doc.get("entities", {})
    labels: List[str] = []

    # Normalized format:
    # "entities": [{"text": "...", "type": "..."}]
    if isinstance(entities, list):
        for ent in entities:
            if not isinstance(ent, dict):
                continue

            label = (
                ent.get("text")
                or ent.get("label")
                or ent.get("name")
                or ent.get("trigger_word")
            )

            if label:
                labels.append(str(label).strip())

    # Original DocRED-like format:
    # "entities": {"Event_x": {"type": "...", "mentions": [{"trigger_word": "..."}]}}
    elif isinstance(entities, dict):
        for _, ent_info in entities.items():
            if not isinstance(ent_info, dict):
                continue

            mentions = ent_info.get("mentions", [])

            if isinstance(mentions, list) and mentions:
                mention = mentions[0]

                if isinstance(mention, dict):
                    label = mention.get("trigger_word")

                    if label:
                        labels.append(str(label).strip())

    return [x for x in labels if x]


def extract_gold_relations(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract gold relation triples from either:
    - original DocRED-like dict format
    - normalized list format
    """
    relations = doc.get("relations", {})
    entities = doc.get("entities", {})
    triples: List[Dict[str, str]] = []

    # Normalized format:
    # "relations": [{"head_text": "...", "relation": "...", "tail_text": "..."}]
    if isinstance(relations, list):
        for rel in relations:
            if not isinstance(rel, dict):
                continue

            head = rel.get("head_text") or rel.get("head")
            tail = rel.get("tail_text") or rel.get("tail")
            relation = rel.get("relation") or rel.get("rel")

            if head and tail and relation:
                triples.append(
                    {
                        "head": str(head).strip(),
                        "relation": str(relation).strip(),
                        "tail": str(tail).strip(),
                    }
                )

    # Original DocRED-like format:
    # "relations": {"P17 : country": [["Event_a", "Event_b"]]}
    elif isinstance(relations, dict):
        id_to_label: Dict[str, str] = {}

        if isinstance(entities, dict):
            for entity_id, ent_info in entities.items():
                if not isinstance(ent_info, dict):
                    continue

                mentions = ent_info.get("mentions", [])

                if isinstance(mentions, list) and mentions:
                    mention = mentions[0]

                    if isinstance(mention, dict):
                        label = mention.get("trigger_word")

                        if label:
                            id_to_label[str(entity_id)] = str(label).strip()

        for rel_label, pairs in relations.items():
            if not isinstance(pairs, list):
                continue

            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue

                head = id_to_label.get(str(pair[0]), "")
                tail = id_to_label.get(str(pair[1]), "")

                if head and tail:
                    triples.append(
                        {
                            "head": head,
                            "relation": str(rel_label).strip(),
                            "tail": tail,
                        }
                    )

    return triples


def inspect_gold_dataset(dataset_jsonl_path: str | Path, type_filter_raw: str) -> Dict[str, Any]:
    """Inspect gold dataset structure and annotation counts."""
    type_filter = normalize_type_filter(type_filter_raw)

    total_lines = 0
    selected_docs = 0

    docs_with_entities = 0
    docs_with_relations = 0

    total_entities = 0
    total_relations = 0

    entity_structure_counter = Counter()
    relation_structure_counter = Counter()
    doc_type_counter = Counter()
    relation_label_counter = Counter()

    sample_docs: List[Dict[str, Any]] = []
    selected_doc_ids: List[str] = []

    for doc in read_jsonl(dataset_jsonl_path):
        total_lines += 1

        doc_type = str(doc.get("type", "")).strip()
        doc_type_counter[doc_type] += 1

        if not doc_matches_type(doc, type_filter):
            continue

        selected_docs += 1

        doc_id = str(doc.get("document_id", "")).strip()
        if doc_id:
            selected_doc_ids.append(doc_id)

        entities = doc.get("entities", {})
        relations = doc.get("relations", {})

        entity_structure_counter[type(entities).__name__] += 1
        relation_structure_counter[type(relations).__name__] += 1

        ent_labels = extract_gold_entity_labels(doc)
        rel_triples = extract_gold_relations(doc)

        if ent_labels:
            docs_with_entities += 1

        if rel_triples:
            docs_with_relations += 1

        total_entities += len(ent_labels)
        total_relations += len(rel_triples)

        for rel in rel_triples:
            relation_label_counter[rel["relation"]] += 1

        if len(sample_docs) < 3:
            sample_docs.append(
                {
                    "document_id": doc_id,
                    "title": doc.get("title", ""),
                    "type": doc.get("type", ""),
                    "text_preview": shorten(doc.get("text", ""), 300),
                    "entity_examples": ent_labels[:10],
                    "relation_examples": rel_triples[:10],
                }
            )

    return {
        "total_jsonl_lines": total_lines,
        "selected_docs_after_type_filter": selected_docs,
        "doc_type_counts": dict(doc_type_counter),
        "entity_structure_counts": dict(entity_structure_counter),
        "relation_structure_counts": dict(relation_structure_counter),
        "docs_with_entities": docs_with_entities,
        "docs_with_relations": docs_with_relations,
        "total_gold_entities": total_entities,
        "total_gold_relations": total_relations,
        "top_relation_labels": relation_label_counter.most_common(30),
        "sample_docs": sample_docs,
        "selected_doc_ids": selected_doc_ids,
    }


# ============================================================
# Prediction inspection
# ============================================================

def get_taxodriven_chunk_outputs(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Get TaxoDrivenKG chunk outputs safely."""
    outputs = row.get("outputs", {})

    if isinstance(outputs, dict):
        return outputs

    return {}


def get_assistant_messages(row: Dict[str, Any]) -> List[str]:
    """Extract assistant messages from TaxoDrivenKG conversations."""
    assistant_messages: List[str] = []
    conversations = row.get("conversations", {})

    if not isinstance(conversations, dict):
        return assistant_messages

    for _, conv in conversations.items():
        if not isinstance(conv, list):
            continue

        for msg in conv:
            if not isinstance(msg, dict):
                continue

            if msg.get("role") == "assistant":
                assistant_messages.append(str(msg.get("content", "")))

    return assistant_messages


def get_prompt_texts(row: Dict[str, Any]) -> List[str]:
    """Extract prompt texts from TaxoDrivenKG row."""
    prompts = row.get("prompts", {})
    out: List[str] = []

    if isinstance(prompts, dict):
        for _, prompt in prompts.items():
            out.append(str(prompt or ""))

    return out


def extract_examples_block(prompt: str) -> str:
    """Extract the -Examples- block from a TaxoDrivenKG prompt."""
    prompt = str(prompt or "")

    # Flexible regex to capture text between -Examples- and -Real Data-
    match = re.search(
        r"-Examples-\s*(.*?)\s*#+\s*-Real Data-",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return ""


def extract_candidate_line(prompt: str) -> str:
    """Extract ontology candidate line from prompt."""
    prompt = str(prompt or "")

    patterns = [
        r"Potential entity candidates from the ontology:\s*(.*)",
        r"Ontology hints:\s*(.*)",
        r"Potential.*?ontology.*?:\s*(.*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)

        if match:
            return match.group(1).strip()

    return ""


def split_candidates(candidate_line: str) -> List[str]:
    """Split candidate line into candidate strings."""
    if not candidate_line:
        return []

    # Remove markdown-ish clutter.
    candidate_line = candidate_line.strip().strip("-").strip()

    # Split by comma or semicolon.
    parts = re.split(r"[,;]", candidate_line)

    return [x.strip() for x in parts if x.strip()]


def row_has_evaluator_format(row: Dict[str, Any]) -> bool:
    """Check whether row is directly compatible with eval_relations.py."""
    if "parsed_ok" not in row:
        return False

    prediction = row.get("prediction")

    if not isinstance(prediction, dict):
        return False

    return isinstance(prediction.get("entities", []), list) and isinstance(
        prediction.get("relations", []), list
    )


def inspect_taxodriven_predictions(prediction_jsonl_path: str | Path) -> Dict[str, Any]:
    """Inspect TaxoDrivenKG raw prediction file."""
    prediction_jsonl_path = Path(prediction_jsonl_path)

    if not prediction_jsonl_path.exists():
        return {
            "exists": False,
            "error": f"Prediction file does not exist: {prediction_jsonl_path}",
        }

    total_rows = 0
    status_counter = Counter()

    total_chunks = 0
    chunks_with_entities = 0
    chunks_with_relationships = 0

    docs_with_entities = 0
    docs_with_relationships = 0

    total_entities = 0
    total_relationships = 0

    total_assistant_messages = 0
    empty_assistant_messages = 0
    nonempty_assistant_messages = 0

    rows_with_evaluator_format = 0
    rows_with_taxodriven_format = 0

    prompts_total = 0
    prompts_with_empty_examples = 0
    prompts_with_nonempty_examples = 0
    prompts_with_no_candidate_line = 0

    candidate_counter = Counter()
    candidates_per_prompt: List[int] = []

    prediction_doc_ids: List[str] = []

    sample_empty_docs: List[Dict[str, Any]] = []
    sample_nonempty_docs: List[Dict[str, Any]] = []
    sample_prompts: List[Dict[str, Any]] = []

    for row in read_jsonl(prediction_jsonl_path):
        total_rows += 1

        doc_id = str(row.get("document_id", "")).strip()
        if doc_id:
            prediction_doc_ids.append(doc_id)

        status = str(row.get("status", row.get("parsed_ok", "missing_status"))).strip()
        status_counter[status] += 1

        if row_has_evaluator_format(row):
            rows_with_evaluator_format += 1

        if "outputs" in row:
            rows_with_taxodriven_format += 1

        row_entity_count = 0
        row_relationship_count = 0

        # TaxoDrivenKG raw format.
        outputs = get_taxodriven_chunk_outputs(row)

        for _, chunk in outputs.items():
            total_chunks += 1

            if not isinstance(chunk, dict):
                continue

            ents = chunk.get("entities", []) or []
            rels = chunk.get("relationships", []) or []

            row_entity_count += len(ents)
            row_relationship_count += len(rels)

            if ents:
                chunks_with_entities += 1

            if rels:
                chunks_with_relationships += 1

        # Evaluator-compatible canonical format.
        prediction = row.get("prediction", {})
        if isinstance(prediction, dict):
            ents = prediction.get("entities", []) or []
            rels = prediction.get("relations", []) or []

            row_entity_count += len(ents)
            row_relationship_count += len(rels)

        if row_entity_count > 0:
            docs_with_entities += 1

        if row_relationship_count > 0:
            docs_with_relationships += 1

        total_entities += row_entity_count
        total_relationships += row_relationship_count

        assistant_messages = get_assistant_messages(row)
        total_assistant_messages += len(assistant_messages)

        row_empty_assistant = 0
        row_nonempty_assistant = 0

        for msg in assistant_messages:
            if msg.strip():
                nonempty_assistant_messages += 1
                row_nonempty_assistant += 1
            else:
                empty_assistant_messages += 1
                row_empty_assistant += 1

        prompts = get_prompt_texts(row)

        for prompt in prompts:
            prompts_total += 1

            examples_block = extract_examples_block(prompt)

            if examples_block.strip():
                prompts_with_nonempty_examples += 1
            else:
                prompts_with_empty_examples += 1

            candidate_line = extract_candidate_line(prompt)
            candidates = split_candidates(candidate_line)

            if not candidate_line:
                prompts_with_no_candidate_line += 1

            candidates_per_prompt.append(len(candidates))

            for candidate in candidates:
                candidate_counter[candidate] += 1

            if len(sample_prompts) < 3:
                sample_prompts.append(
                    {
                        "document_id": doc_id,
                        "candidate_line": candidate_line,
                        "num_candidates": len(candidates),
                        "examples_block_is_empty": not bool(examples_block.strip()),
                        "examples_preview": shorten(examples_block, 500),
                        "prompt_preview": shorten(prompt, 1000),
                    }
                )

        if row_entity_count == 0 and row_relationship_count == 0 and len(sample_empty_docs) < 5:
            sample_empty_docs.append(
                {
                    "document_id": doc_id,
                    "title": row.get("title", ""),
                    "status": row.get("status", ""),
                    "num_chunks": row.get("num_chunks", None),
                    "assistant_messages_empty": row_empty_assistant,
                    "assistant_messages_nonempty": row_nonempty_assistant,
                    "first_assistant_preview": shorten(assistant_messages[0] if assistant_messages else "", 500),
                    "first_prompt_preview": shorten(prompts[0] if prompts else "", 1000),
                }
            )

        if row_entity_count > 0 and len(sample_nonempty_docs) < 5:
            sample_nonempty_docs.append(
                {
                    "document_id": doc_id,
                    "title": row.get("title", ""),
                    "status": row.get("status", ""),
                    "entity_count": row_entity_count,
                    "relationship_count": row_relationship_count,
                    "outputs_preview": shorten(json.dumps(outputs, ensure_ascii=False), 1000),
                    "first_assistant_preview": shorten(assistant_messages[0] if assistant_messages else "", 500),
                }
            )

    average_candidates = (
        sum(candidates_per_prompt) / len(candidates_per_prompt)
        if candidates_per_prompt
        else 0.0
    )

    return {
        "exists": True,
        "path": str(prediction_jsonl_path),
        "total_prediction_rows": total_rows,
        "status_counts": dict(status_counter),
        "rows_with_evaluator_format": rows_with_evaluator_format,
        "rows_with_taxodriven_raw_format": rows_with_taxodriven_format,
        "total_chunks": total_chunks,
        "chunks_with_entities": chunks_with_entities,
        "chunks_with_relationships": chunks_with_relationships,
        "docs_with_entities": docs_with_entities,
        "docs_with_relationships": docs_with_relationships,
        "total_predicted_entities": total_entities,
        "total_predicted_relationships": total_relationships,
        "total_assistant_messages": total_assistant_messages,
        "empty_assistant_messages": empty_assistant_messages,
        "nonempty_assistant_messages": nonempty_assistant_messages,
        "prompts_total": prompts_total,
        "prompts_with_empty_examples": prompts_with_empty_examples,
        "prompts_with_nonempty_examples": prompts_with_nonempty_examples,
        "prompts_with_no_candidate_line": prompts_with_no_candidate_line,
        "average_candidates_per_prompt": average_candidates,
        "top_ontology_candidates": candidate_counter.most_common(30),
        "sample_empty_docs": sample_empty_docs,
        "sample_nonempty_docs": sample_nonempty_docs,
        "sample_prompts": sample_prompts,
        "prediction_doc_ids": prediction_doc_ids,
    }


# ============================================================
# Cross-check gold vs predictions
# ============================================================

def compare_gold_and_predictions(
    gold_report: Dict[str, Any],
    prediction_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare document IDs between selected gold docs and prediction rows."""
    gold_doc_ids = set(gold_report.get("selected_doc_ids", []))
    pred_doc_ids = set(prediction_report.get("prediction_doc_ids", []))

    missing_predictions = sorted(gold_doc_ids - pred_doc_ids)[:50]
    extra_predictions = sorted(pred_doc_ids - gold_doc_ids)[:50]

    return {
        "selected_gold_doc_count": len(gold_doc_ids),
        "prediction_doc_count": len(pred_doc_ids),
        "matched_doc_count": len(gold_doc_ids & pred_doc_ids),
        "missing_prediction_count": len(gold_doc_ids - pred_doc_ids),
        "extra_prediction_count": len(pred_doc_ids - gold_doc_ids),
        "missing_prediction_examples": missing_predictions,
        "extra_prediction_examples": extra_predictions,
    }


# ============================================================
# Backend smoke test
# ============================================================

def openai_compatible_chat_completion(
    host: str,
    api_key: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Minimal OpenAI-compatible /chat/completions call using stdlib only.

    This works for vLLM OpenAI-compatible endpoints.
    """
    host = host.rstrip("/")
    url = f"{host}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    started = time.time()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            elapsed = time.time() - started

            parsed = json.loads(raw)

            content = ""
            try:
                content = parsed["choices"][0]["message"]["content"]
            except Exception:
                content = ""

            return {
                "ok": True,
                "url": url,
                "elapsed_seconds": elapsed,
                "content": content,
                "raw_response_preview": shorten(raw, 2000),
            }

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "url": url,
            "error_type": "HTTPError",
            "status_code": e.code,
            "error": shorten(error_body, 2000),
        }

    except Exception as e:
        return {
            "ok": False,
            "url": url,
            "error_type": type(e).__name__,
            "error": str(e),
        }


def backend_smoke_test(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a tiny backend test independent of TaxoDrivenKG."""
    prompt = textwrap.dedent(
        """
        Extract entities and relationships from this text.

        Text:
        Skai TV is based in Piraeus and is part of the Skai Group.

        Return only records in this exact format:
        ("entity"<|><entity_name><|><entity_type><|><entity_description>)
        ##
        ("relationship"<|><source_entity><|><target_entity><|><relationship_type>)
        """
    ).strip()

    if args.backend_name.lower() not in {"vllm", "openai_compatible", "openai-compatible"}:
        return {
            "skipped": True,
            "reason": f"Smoke test currently implemented only for OpenAI-compatible HTTP backends. backend_name={args.backend_name}",
        }

    result = openai_compatible_chat_completion(
        host=args.host,
        api_key=args.api_key,
        model_name=args.model_name,
        prompt=prompt,
        max_tokens=256,
        temperature=0.0,
        timeout=args.backend_timeout,
    )

    result["content_is_empty"] = not bool(str(result.get("content", "")).strip())

    return result


# ============================================================
# Optional TaxoDrivenKG launch
# ============================================================

def run_taxodriven_method(args: argparse.Namespace) -> Dict[str, Any]:
    """Optionally launch run_taxodriven.py with the same parameters."""
    run_path = Path(args.run_taxodriven_path)

    if not run_path.exists():
        return {
            "ok": False,
            "error": f"run_taxodriven.py not found: {run_path}",
        }

    command = [
        sys.executable,
        str(run_path),
        "--dataset-jsonl-path",
        str(args.dataset_jsonl_path),
        "--ontology-path",
        str(args.ontology_path),
        "--output-jsonl-path",
        str(args.output_jsonl_path),
        "--backend-name",
        str(args.backend_name),
        "--host",
        str(args.host),
        "--api-key",
        str(args.api_key),
        "--model-name",
        str(args.model_name),
        "--type-filter",
        str(args.type_filter),
        "--few-shot-source-type",
        str(args.few_shot_source_type),
        "--few-shot-k",
        str(args.few_shot_k),
    ]

    # Optional arguments if your runner supports them.
    if args.chunk_tokens is not None:
        command.extend(["--chunk-tokens", str(args.chunk_tokens)])

    if args.max_taxonomy_hits is not None:
        command.extend(["--max-taxonomy-hits", str(args.max_taxonomy_hits)])

    if args.generation_max_tokens is not None:
        command.extend(["--generation-max-tokens", str(args.generation_max_tokens)])

    if args.no_resume:
        command.append("--no-resume")

    if args.debug:
        command.append("--debug")

    if args.debug_show_prompt:
        command.append("--debug-show-prompt")

    if args.debug_show_output:
        command.append("--debug-show-output")

    print("\n[RUNNING TaxoDrivenKG]")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in command))

    started = time.time()

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    elapsed = time.time() - started

    return {
        "ok": process.returncode == 0,
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
        "stdout_tail": process.stdout[-5000:],
        "stderr_tail": process.stderr[-5000:],
    }


# ============================================================
# Diagnosis conclusions
# ============================================================

def build_conclusions(report: Dict[str, Any]) -> List[str]:
    """Build human-readable diagnosis conclusions."""
    conclusions: List[str] = []

    gold = report.get("gold_dataset", {})
    pred = report.get("taxodriven_predictions", {})
    cross = report.get("gold_prediction_alignment", {})
    backend = report.get("backend_smoke_test", {})

    # Gold sanity.
    if gold.get("selected_docs_after_type_filter", 0) == 0:
        conclusions.append(
            "CRITICAL: The selected type filter matches zero gold documents. Check --type-filter."
        )

    if gold.get("total_gold_entities", 0) == 0:
        conclusions.append(
            "CRITICAL: No gold entities were detected in the selected dataset subset. The dataset adapter/format may be incompatible."
        )

    if gold.get("total_gold_relations", 0) == 0:
        conclusions.append(
            "WARNING: No gold relations were detected in the selected dataset subset. Relation evaluation will be meaningless."
        )

    # Prediction file.
    if not pred.get("exists", False):
        conclusions.append(
            "CRITICAL: Prediction file does not exist. TaxoDrivenKG did not run or wrote to another path."
        )
        return conclusions

    if pred.get("total_prediction_rows", 0) == 0:
        conclusions.append(
            "CRITICAL: Prediction file exists but contains zero JSONL rows."
        )

    # Format.
    if pred.get("rows_with_taxodriven_raw_format", 0) > 0 and pred.get("rows_with_evaluator_format", 0) == 0:
        conclusions.append(
            "IMPORTANT: The prediction file is in raw TaxoDrivenKG format, not eval_relations.py format. You need canonicalization before evaluation."
        )

    # Empty predictions.
    if pred.get("total_predicted_entities", 0) == 0 and pred.get("total_predicted_relationships", 0) == 0:
        conclusions.append(
            "CRITICAL: TaxoDrivenKG produced zero predicted entities and zero predicted relationships. Canonicalization will not fix this; generation/prompt/backend must be checked first."
        )

    elif pred.get("total_predicted_entities", 0) > 0 and pred.get("rows_with_evaluator_format", 0) == 0:
        conclusions.append(
            "GOOD NEWS: TaxoDrivenKG produced some predictions, but eval_relations.py cannot read them yet. Build a canonicalizer."
        )

    # Assistant content.
    total_assistant = pred.get("total_assistant_messages", 0)
    empty_assistant = pred.get("empty_assistant_messages", 0)
    nonempty_assistant = pred.get("nonempty_assistant_messages", 0)

    if total_assistant > 0 and empty_assistant == total_assistant:
        conclusions.append(
            "CRITICAL: All assistant messages are empty. This strongly suggests a backend response issue, max_tokens issue, or prompt/model refusal/format issue."
        )
    elif empty_assistant > 0 and nonempty_assistant > 0:
        conclusions.append(
            "WARNING: Some assistant messages are empty and some are non-empty. Inspect sample_empty_docs and backend stability."
        )

    # Few-shot examples.
    prompts_total = pred.get("prompts_total", 0)
    empty_examples = pred.get("prompts_with_empty_examples", 0)

    if prompts_total > 0 and empty_examples == prompts_total:
        conclusions.append(
            "IMPORTANT: All TaxoDrivenKG prompts have empty few-shot examples. This likely hurts tuple-format extraction, especially for DocRED."
        )

    # Candidate quality.
    avg_candidates = pred.get("average_candidates_per_prompt", 0.0)
    if prompts_total > 0 and avg_candidates == 0:
        conclusions.append(
            "WARNING: Prompts appear to have no ontology candidates/hints."
        )

    top_candidates = [x[0].lower() for x in pred.get("top_ontology_candidates", [])]
    suspicious_generic = {"work", "country", "member of", "part of", "child"}
    if any(x in suspicious_generic for x in top_candidates):
        conclusions.append(
            "WARNING: Top ontology candidates include generic labels/properties such as work/country/member of/part of/child. They may be misleading if treated as entity candidates."
        )

    # Gold/pred ID alignment.
    if cross.get("selected_gold_doc_count", 0) > 0 and cross.get("matched_doc_count", 0) == 0:
        conclusions.append(
            "CRITICAL: No document IDs match between gold and predictions. Evaluation will produce all false negatives."
        )
    elif cross.get("missing_prediction_count", 0) > 0:
        conclusions.append(
            f"WARNING: {cross.get('missing_prediction_count')} selected gold documents have no prediction row."
        )

    # Backend smoke test.
    if backend:
        if backend.get("skipped"):
            conclusions.append(
                f"INFO: Backend smoke test skipped: {backend.get('reason')}"
            )
        elif not backend.get("ok"):
            conclusions.append(
                "CRITICAL: Backend smoke test failed. Fix backend connectivity/model name before debugging TaxoDrivenKG."
            )
        elif backend.get("content_is_empty"):
            conclusions.append(
                "CRITICAL: Backend smoke test returned empty content. This indicates the backend/model call itself may be the root issue."
            )
        else:
            conclusions.append(
                "GOOD: Backend smoke test returned non-empty content. The empty TaxoDrivenKG output is more likely due to prompt/few-shot/parser logic."
            )

    if not conclusions:
        conclusions.append(
            "No critical issue detected by static diagnosis. Inspect samples in the JSON report."
        )

    return conclusions


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose TaxoDrivenKG outputs before evaluation."
    )

    parser.add_argument("--dataset-jsonl-path", required=True)
    parser.add_argument("--ontology-path", required=True)
    parser.add_argument("--output-jsonl-path", required=True)

    parser.add_argument("--run-taxodriven-path", default="../../experiments/methods/run_taxodriven.py")

    parser.add_argument("--backend-name", default="vllm")
    parser.add_argument("--host", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model-name", required=True)

    parser.add_argument("--type-filter", default="all")
    parser.add_argument("--few-shot-source-type", default="all")
    parser.add_argument("--few-shot-k", type=int, default=3)

    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--max-taxonomy-hits", type=int, default=None)
    parser.add_argument("--generation-max-tokens", type=int, default=None)

    parser.add_argument("--run-method", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-show-prompt", action="store_true")
    parser.add_argument("--debug-show-output", action="store_true")

    parser.add_argument("--backend-smoke-test", action="store_true")
    parser.add_argument("--backend-timeout", type=int, default=60)

    parser.add_argument("--report-path", default=None)

    return parser.parse_args()


def main() -> None:
    """Main diagnosis flow."""
    args = parse_args()

    dataset_path = Path(args.dataset_jsonl_path)
    ontology_path = Path(args.ontology_path)
    output_path = Path(args.output_jsonl_path)

    print("\n========== TaxoDrivenKG Diagnosis ==========")
    print(f"Dataset JSONL: {dataset_path.resolve()}")
    print(f"Ontology path: {ontology_path.resolve()}")
    print(f"Prediction JSONL: {output_path.resolve()}")
    print(f"Type filter: {args.type_filter}")
    print(f"Model: {args.model_name}")
    print("============================================\n")

    report: Dict[str, Any] = {
        "inputs": {
            "dataset_jsonl_path": str(dataset_path),
            "ontology_path": str(ontology_path),
            "output_jsonl_path": str(output_path),
            "run_taxodriven_path": str(args.run_taxodriven_path),
            "backend_name": args.backend_name,
            "host": args.host,
            "model_name": args.model_name,
            "type_filter": args.type_filter,
            "few_shot_source_type": args.few_shot_source_type,
            "few_shot_k": args.few_shot_k,
        }
    }

    # ------------------------------------------------------------
    # Check input files.
    # ------------------------------------------------------------
    file_checks = {
        "dataset_exists": dataset_path.exists(),
        "ontology_exists": ontology_path.exists(),
        "prediction_exists_before_run": output_path.exists(),
        "run_taxodriven_exists": Path(args.run_taxodriven_path).exists(),
    }
    report["file_checks"] = file_checks

    if not dataset_path.exists():
        print(f"[CRITICAL] Dataset file not found: {dataset_path}")
        report["conclusions"] = ["CRITICAL: Dataset file not found."]
        finalize_report(args, report)
        return

    if not ontology_path.exists():
        print(f"[WARNING] Ontology file not found: {ontology_path}")

    # ------------------------------------------------------------
    # Optional method run.
    # ------------------------------------------------------------
    if args.run_method:
        run_report = run_taxodriven_method(args)
        report["taxodriven_run"] = run_report

        if not run_report.get("ok"):
            print("[CRITICAL] TaxoDrivenKG run failed.")
            print("\n--- STDOUT tail ---")
            print(run_report.get("stdout_tail", ""))
            print("\n--- STDERR tail ---")
            print(run_report.get("stderr_tail", ""))

    # ------------------------------------------------------------
    # Inspect gold dataset.
    # ------------------------------------------------------------
    print("[1/5] Inspecting gold dataset...")
    gold_report = inspect_gold_dataset(dataset_path, args.type_filter)
    report["gold_dataset"] = gold_report

    print(f"  Selected docs: {gold_report['selected_docs_after_type_filter']}")
    print(f"  Gold entities: {gold_report['total_gold_entities']}")
    print(f"  Gold relations: {gold_report['total_gold_relations']}")

    # ------------------------------------------------------------
    # Inspect TaxoDrivenKG predictions.
    # ------------------------------------------------------------
    print("\n[2/5] Inspecting TaxoDrivenKG prediction file...")
    pred_report = inspect_taxodriven_predictions(output_path)
    report["taxodriven_predictions"] = pred_report

    if not pred_report.get("exists"):
        print(f"  Prediction file missing: {pred_report.get('error')}")
    else:
        print(f"  Prediction rows: {pred_report['total_prediction_rows']}")
        print(f"  Predicted entities: {pred_report['total_predicted_entities']}")
        print(f"  Predicted relationships: {pred_report['total_predicted_relationships']}")
        print(f"  Empty assistant messages: {pred_report['empty_assistant_messages']}")
        print(f"  Non-empty assistant messages: {pred_report['nonempty_assistant_messages']}")
        print(f"  Prompts with empty examples: {pred_report['prompts_with_empty_examples']} / {pred_report['prompts_total']}")
        print(f"  Rows directly compatible with eval_relations.py: {pred_report['rows_with_evaluator_format']}")

    # ------------------------------------------------------------
    # Gold/prediction alignment.
    # ------------------------------------------------------------
    print("\n[3/5] Checking gold/prediction document ID alignment...")
    alignment_report = compare_gold_and_predictions(gold_report, pred_report)
    report["gold_prediction_alignment"] = alignment_report

    print(f"  Matched docs: {alignment_report['matched_doc_count']}")
    print(f"  Missing predictions: {alignment_report['missing_prediction_count']}")
    print(f"  Extra predictions: {alignment_report['extra_prediction_count']}")

    # ------------------------------------------------------------
    # Optional backend smoke test.
    # ------------------------------------------------------------
    if args.backend_smoke_test:
        print("\n[4/5] Running backend smoke test...")
        smoke_report = backend_smoke_test(args)
        report["backend_smoke_test"] = smoke_report

        if smoke_report.get("skipped"):
            print(f"  Skipped: {smoke_report.get('reason')}")
        elif smoke_report.get("ok"):
            print(f"  Backend OK: yes")
            print(f"  Content empty: {smoke_report.get('content_is_empty')}")
            print(f"  Content preview: {shorten(smoke_report.get('content', ''), 500)}")
        else:
            print(f"  Backend OK: no")
            print(f"  Error: {smoke_report.get('error')}")
    else:
        print("\n[4/5] Backend smoke test skipped. Add --backend-smoke-test to enable.")

    # ------------------------------------------------------------
    # Conclusions.
    # ------------------------------------------------------------
    print("\n[5/5] Building conclusions...")
    conclusions = build_conclusions(report)
    report["conclusions"] = conclusions

    print("\n========== Conclusions ==========")
    for i, conclusion in enumerate(conclusions, start=1):
        print(f"{i}. {conclusion}")

    print("\n========== Useful samples ==========")

    sample_empty_docs = safe_get(report, "taxodriven_predictions", "sample_empty_docs", default=[])
    if sample_empty_docs:
        print("\nSample empty TaxoDrivenKG docs:")
        for sample in sample_empty_docs[:2]:
            print(json.dumps(sample, indent=2, ensure_ascii=False))

    sample_prompts = safe_get(report, "taxodriven_predictions", "sample_prompts", default=[])
    if sample_prompts:
        print("\nSample prompt diagnostics:")
        for sample in sample_prompts[:2]:
            print(json.dumps(sample, indent=2, ensure_ascii=False))

    finalize_report(args, report)


def finalize_report(args: argparse.Namespace, report: Dict[str, Any]) -> None:
    """Save final report if requested or default path."""
    if args.report_path:
        report_path = Path(args.report_path)
    else:
        output_path = Path(args.output_jsonl_path)
        report_path = output_path.with_suffix(".diagnosis.json")

    write_json(report_path, report)
    print(f"\nDiagnosis report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
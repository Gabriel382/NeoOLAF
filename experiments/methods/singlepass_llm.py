# singlepass_llm.py
# Streaming SinglePass baseline for generic JSONL datasets.
#
# For each document:
# - reads one line from the dataset
# - optionally injects a truncated ontology
# - calls one backend once
# - writes one prediction line to an output JSONL
#
# Supported backends:
# - openrouter
# - vllm / openai-compatible
# - ollama

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

import sys
from pathlib import Path

# Resolve project root robustly from this script location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT =", PROJECT_ROOT)
print("sys.path[0] =", sys.path[0])

from experiments.common.jsonl_adapter import iter_documents
from tqdm.auto import tqdm

# =========================================================
# Basic file helpers
# =========================================================
def load_text_file(path: str | Path) -> str:
    """Load a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text_file(path: str | Path, text: str) -> None:
    """Save text to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    """Append one record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_processed_ids(output_jsonl_path: str | Path) -> set[str]:
    """
    Load already processed document ids to support resume.
    """
    output_jsonl_path = Path(output_jsonl_path)
    processed = set()

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


# =========================================================
# Ontology helpers
# =========================================================
def load_ontology_snippet(ontology_path: str | Path, max_chars: int = 20000) -> str:
    """
    Load a truncated ontology snippet for prompt injection.
    """
    raw = load_text_file(ontology_path)
    raw = raw.strip()
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars]


# =========================================================
# Prompt building
# =========================================================
def build_prompt(document: Dict[str, Any], ontology_text: Optional[str]) -> str:
    """
    Build the prompt for one document.

    Output schema is strict JSON:
    {
      "entities": [{"id": "...", "label": "...", "type": "..."}],
      "relations": [{"head": "...", "relation": "...", "tail": "...", "evidence": "..."}]
    }
    """
    ontology_block = ""
    if ontology_text:
        ontology_block = f"""
SEED ONTOLOGY (reference only)
------------------------------
{ontology_text}
"""

    prompt = f"""
You are an expert information extraction system.

Your task is to extract a small document-level knowledge graph from the document below.

Rules:
1. Return ONLY valid JSON.
2. Do not wrap the JSON in markdown.
3. Keep entities concise and textual.
4. Use only information supported by the document text.
5. Keep relation names exactly as found in the gold-style schema when possible.
6. If unsure, prefer fewer relations over hallucinated ones.
7. Evidence must be a short text span or short sentence fragment from the document.
8. Output format must be:

{{
  "entities": [
    {{"id": "E1", "label": "entity text", "type": "entity type"}}
  ],
  "relations": [
    {{"head": "entity text", "relation": "relation label", "tail": "entity text", "evidence": "short evidence"}}
  ]
}}

DOCUMENT METADATA
-----------------
document_id: {document["document_id"]}
title: {document["title"]}
type: {document["type"]}

DOCUMENT TEXT
-------------
{document["text"]}

{ontology_block}
Now return only the JSON object.
""".strip()

    return prompt


# =========================================================
# Backend calls
# =========================================================
def call_openai_compatible(
    host: str,
    api_key: str,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Call an OpenAI-compatible chat completions endpoint.
    """
    url = host.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_ollama(
    host: str,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: Optional[int] = None,
) -> str:
    """
    Call Ollama chat endpoint.
    """
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    return data.get("message", {}).get("content", "")


def call_backend_generate(
    backend_name: str,
    host: str,
    api_key: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    referer: str = "",
    title: str = "",
    timeout: Optional[int] = None,
) -> str:
    """
    Dispatch one generation call to the selected backend.
    """
    messages = [
        {"role": "system", "content": "You are a precise knowledge graph extraction assistant."},
        {"role": "user", "content": prompt},
    ]

    backend_name = backend_name.lower().strip()

    if backend_name in {"vllm", "openai"}:
        return call_openai_compatible(
            host=host,
            api_key=api_key,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=None,
            timeout=timeout,
        )

    if backend_name == "openrouter":
        extra_headers: Dict[str, str] = {}
        if referer:
            extra_headers["HTTP-Referer"] = referer
        if title:
            extra_headers["X-Title"] = title

        return call_openai_compatible(
            host=host,
            api_key=api_key,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )

    if backend_name == "ollama":
        return call_ollama(
            host=host,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    raise ValueError(f"Unsupported backend_name: {backend_name}")


# =========================================================
# Parsing helpers
# =========================================================
def strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences if present."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    return text.strip()


def try_parse_full_json(text: str) -> Dict[str, Any]:
    """Try direct JSON parsing."""
    return json.loads(text)


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first top-level JSON object from a string.
    Useful if the model adds extra text around JSON.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)

    raise ValueError("No complete JSON object found in model response.")


def parse_model_output(raw_response: str) -> Dict[str, Any]:
    """
    Parse the model response into JSON.
    """
    cleaned = strip_code_fences(raw_response)

    try:
        return try_parse_full_json(cleaned)
    except Exception:
        return extract_first_json_object(cleaned)


# =========================================================
# Main runner
# =========================================================
def run_singlepass_dataset(
    dataset_jsonl_path: str | Path,
    ontology_path: str | Path | None,
    output_jsonl_path: str | Path,
    backend_name: str,
    host: str,
    api_key: str,
    model_name: str,
    type_filter: str | list[str] = "all",
    use_ontology: bool = True,
    ontology_max_chars: int = 20000,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    referer: str = "",
    title: str = "SinglePass-Run",
    timeout: Optional[int] = None,
    save_raw_dir: str | Path | None = None,
    save_prompt_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Run the single-pass baseline on a full JSONL dataset in streaming mode.
    """
    dataset_jsonl_path = Path(dataset_jsonl_path)
    output_jsonl_path = Path(output_jsonl_path)

    processed_ids = load_processed_ids(output_jsonl_path)

    ontology_text: Optional[str] = None
    if use_ontology and ontology_path is not None:
        ontology_text = load_ontology_snippet(ontology_path, max_chars=ontology_max_chars)

    total_seen = 0
    total_done = 0
    total_skipped = 0
    total_errors = 0

    start_time = time.time()

    docs_iter = iter_documents(dataset_jsonl_path, type_filter=type_filter)

    for doc in tqdm(docs_iter, desc="SinglePass docs", unit="doc"):
        total_seen += 1
        document_id = doc["document_id"]

        if document_id in processed_ids:
            total_skipped += 1
            continue

        prompt = build_prompt(doc, ontology_text=ontology_text)

        raw_response = ""
        parsed_ok = False
        parsed_json: Optional[Dict[str, Any]] = None
        parsing_error: Optional[str] = None
        runtime_error: Optional[str] = None

        try:
            raw_response = call_backend_generate(
                backend_name=backend_name,
                host=host,
                api_key=api_key,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                referer=referer,
                title=title,
                timeout=timeout,
            )

            try:
                parsed_json = parse_model_output(raw_response)
                parsed_ok = True
            except Exception as e:
                parsing_error = str(e)

        except Exception as e:
            runtime_error = str(e)

        record = {
            "document_id": document_id,
            "title": doc["title"],
            "type": doc["type"],
            "dataset_path": str(dataset_jsonl_path),
            "backend_name": backend_name,
            "model_name": model_name,
            "parsed_ok": parsed_ok,
            "parsing_error": parsing_error,
            "runtime_error": runtime_error,
            "prediction": parsed_json,
        }

        append_jsonl(output_jsonl_path, record)

        if save_raw_dir is not None:
            raw_dir = Path(save_raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)
            save_text_file(raw_dir / f"{document_id}.txt", raw_response)

        if save_prompt_dir is not None:
            prompt_dir = Path(save_prompt_dir)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            save_text_file(prompt_dir / f"{document_id}.txt", prompt)

        if parsed_ok:
            total_done += 1
        else:
            total_errors += 1

    elapsed = time.time() - start_time

    return {
        "dataset_jsonl_path": str(dataset_jsonl_path),
        "output_jsonl_path": str(output_jsonl_path),
        "total_seen": total_seen,
        "total_done": total_done,
        "total_skipped": total_skipped,
        "total_errors": total_errors,
        "elapsed_seconds": elapsed,
    }


# =========================================================
# CLI
# =========================================================
def parse_type_filter_arg(type_filter_raw: str) -> str | list[str]:
    """
    Accept:
    - all
    - dev
    - dev,test
    """
    type_filter_raw = type_filter_raw.strip()
    if type_filter_raw.lower() == "all":
        return "all"
    if "," in type_filter_raw:
        return [x.strip() for x in type_filter_raw.split(",") if x.strip()]
    return type_filter_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming SinglePass LLM baseline on JSONL datasets.")
    parser.add_argument("--dataset-jsonl-path", required=True, help="Path to the input dataset JSONL.")
    parser.add_argument("--ontology-path", required=False, default=None, help="Path to the ontology file.")
    parser.add_argument("--output-jsonl-path", required=True, help="Path to the output predictions JSONL.")

    parser.add_argument("--backend-name", required=True, choices=["openrouter", "vllm", "openai", "ollama"])
    parser.add_argument("--host", required=True, help="Backend host URL.")
    parser.add_argument("--api-key", default="", help="API key if needed.")
    parser.add_argument("--model-name", required=True, help="Model name.")

    parser.add_argument("--type-filter", default="all", help='One type, comma-separated list, or "all".')
    parser.add_argument("--use-ontology", action="store_true", help="Whether to inject ontology text into prompt.")
    parser.add_argument("--ontology-max-chars", type=int, default=20000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--referer", default="")
    parser.add_argument("--title", default="SinglePass-Run")
    parser.add_argument("--timeout", type=int, default=None)

    parser.add_argument("--save-raw-dir", default=None)
    parser.add_argument("--save-prompt-dir", default=None)

    args = parser.parse_args()

    summary = run_singlepass_dataset(
        dataset_jsonl_path=args.dataset_jsonl_path,
        ontology_path=args.ontology_path,
        output_jsonl_path=args.output_jsonl_path,
        backend_name=args.backend_name,
        host=args.host,
        api_key=args.api_key,
        model_name=args.model_name,
        type_filter=parse_type_filter_arg(args.type_filter),
        use_ontology=args.use_ontology,
        ontology_max_chars=args.ontology_max_chars,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        referer=args.referer,
        title=args.title,
        timeout=args.timeout,
        save_raw_dir=args.save_raw_dir,
        save_prompt_dir=args.save_prompt_dir,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
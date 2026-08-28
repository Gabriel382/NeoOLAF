from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

# This module lives under olaf_baseline/src.
BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
VENDOR = BASE / "vendor" / "olaf"

for p in (SRC, VENDOR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset_io import positive_gold_relation_count, strip_gold
from olaf_lite_pipeline import run_olaf_lite_document


def _safe_key(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120]


def run_docred_worker(
    row: dict[str, Any],
    run_root: str,
    model_name: str,
    reasoning_effort: str,
    spacy_model_name: str = "en_core_web_sm",
) -> dict[str, Any]:
    """Process-safe one-document OLAF worker.

    Each process creates its own spaCy model + OLAF Pipeline.
    No mutable OLAF state is shared between documents.
    """
    document_id = str(row.get("document_id") or row.get("title") or "unknown")
    safe = _safe_key(document_id)

    run_dir = Path(run_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / f"{safe}.json"
    debug_path = run_dir / f"{safe}_llm_debug.jsonl"

    # Resume safety for repeated local tests.
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["resumed"] = True
        return payload

    clean = strip_gold(row)
    forbidden = {"entities", "relations", "pred_relations", "ontology_links"} & set(clean)
    if forbidden:
        raise RuntimeError(f"Gold leakage guard failed for {document_id}: {forbidden}")

    started = perf_counter()
    result = run_olaf_lite_document(
        clean["text"],
        spacy_model_name=spacy_model_name,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        debug_log_path=str(debug_path),
    )
    worker_wall = perf_counter() - started

    relations_with_both_endpoints = sum(
        1
        for r in result.relations
        if r.get("source") is not None and r.get("target") is not None
    )

    payload = {
        "dataset": "docred",
        "document_id": document_id,
        "title": row.get("title"),
        "type": row.get("type"),
        "gold_relation_count_posthoc_only": positive_gold_relation_count(row),
        "pipeline_elapsed_seconds": result.elapsed_seconds,
        "worker_wall_seconds": worker_wall,
        "concept_count": len(result.concepts),
        "relation_count": len(result.relations),
        "relations_with_both_endpoints": relations_with_both_endpoints,
        "concepts": result.concepts,
        "relations": result.relations,
        "resumed": False,
    }

    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(result_path)
    return payload

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
VENDOR = BASE / "vendor" / "olaf"
for p in (SRC, VENDOR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset_io import positive_gold_relation_count, strip_gold
from olaf_lite_pipeline import run_olaf_lite_document


def safe_key(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    return "".join(out)[:150]


def _read_usage(debug_path: Path) -> dict[str, Any]:
    prompt = completion = total = calls = 0
    usage_missing_calls = 0

    if debug_path.is_file():
        for line in debug_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            calls += 1
            usage = row.get("usage") or {}
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")

            if not isinstance(pt, (int, float)) or not isinstance(ct, (int, float)):
                usage_missing_calls += 1
                continue

            prompt += int(pt)
            completion += int(ct)
            total += int(tt) if isinstance(tt, (int, float)) else int(pt) + int(ct)

    return {
        "llm_calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "usage_missing_calls": usage_missing_calls,
        "usage_complete": calls > 0 and usage_missing_calls == 0,
    }


def result_path_for(run_root: Path, dataset: str, document_id: str) -> Path:
    return run_root / dataset / f"{safe_key(document_id)}.json"


def run_one(
    dataset: str,
    row: dict[str, Any],
    run_root: str,
    model_name: str,
    reasoning_effort: str,
    spacy_model_name: str,
    input_usd_per_m: float,
    output_usd_per_m: float,
) -> dict[str, Any]:
    run_root_p = Path(run_root)
    document_id = str(row.get("document_id") or row.get("title") or "unknown")
    dataset_dir = run_root_p / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_path_for(run_root_p, dataset, document_id)
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["resumed"] = True
        return payload

    debug_path = dataset_dir / f"{safe_key(document_id)}_llm_debug.jsonl"
    if debug_path.exists():
        debug_path.unlink()

    clean = strip_gold(row)
    forbidden = {"entities", "relations", "pred_relations", "ontology_links"} & set(clean)
    if forbidden:
        raise RuntimeError(f"Gold leakage guard failed: {forbidden}")

    t0 = perf_counter()
    result = run_olaf_lite_document(
        clean["text"],
        spacy_model_name=spacy_model_name,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        debug_log_path=str(debug_path),
    )
    worker_wall = perf_counter() - t0

    usage = _read_usage(debug_path)
    estimated_cost = (
        usage["prompt_tokens"] / 1_000_000 * input_usd_per_m
        + usage["completion_tokens"] / 1_000_000 * output_usd_per_m
    )

    payload = {
        "dataset": dataset,
        "document_id": document_id,
        "title": row.get("title"),
        "type": row.get("type"),
        "gold_relation_count_posthoc_only": positive_gold_relation_count(row),
        "pipeline_elapsed_seconds": result.elapsed_seconds,
        "worker_wall_seconds": worker_wall,
        "concept_count": len(result.concepts),
        "relation_count": len(result.relations),
        "relations_with_both_endpoints": sum(
            1 for r in result.relations
            if r.get("source") is not None and r.get("target") is not None
        ),
        "concepts": result.concepts,
        "relations": result.relations,
        **usage,
        "estimated_cost_usd": estimated_cost,
        "resumed": False,
        "imported_from_previous_smoke": False,
    }

    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(result_path)
    return payload

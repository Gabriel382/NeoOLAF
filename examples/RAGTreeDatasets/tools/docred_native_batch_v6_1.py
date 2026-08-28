from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import statistics
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import docred_native_ablation_v5 as v5


BATCH_VERSION = "v6.1-batch-native-v5.1-analysis-recovery"
GOLD_FIELDS = {"entities", "relations", "labels", "vertexSet"}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            yield value


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_id(record: dict[str, Any], fallback_index: int | None = None) -> str:
    value = (
        record.get("document_id")
        or record.get("doc_id")
        or record.get("id")
        or record.get("title")
    )
    if value is None:
        if fallback_index is None:
            raise KeyError("Document record has no document_id, doc_id, id, or title")
        value = f"document_{fallback_index:06d}"
    return str(value)


def safe_slug(value: str, *, max_length: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        cleaned = "document"
    suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    prefix_limit = max(8, max_length - len(suffix) - 1)
    return f"{cleaned[:prefix_limit]}_{suffix}"


def strip_gold(record: dict[str, Any], task_guidance: dict[str, Any]) -> dict[str, Any]:
    # Keep the original raw-text/document structure, but never expose benchmark
    # entities or relations to the native NeoOLAF pipeline.
    result = {key: value for key, value in record.items() if key not in GOLD_FIELDS}
    result["document_id"] = document_id(record)
    result["task_guidance"] = task_guidance
    if "text" not in result or not str(result.get("text") or "").strip():
        sentences = result.get("sentences") or []
        if isinstance(sentences, list):
            result["text"] = " ".join(str(sentence) for sentence in sentences)
    if not str(result.get("text") or "").strip():
        raise ValueError(f"Document {result['document_id']} has no usable text")
    return result


def record_matches_type(record: dict[str, Any], type_filter: str | None) -> bool:
    if type_filter is None or str(type_filter).strip().lower() in {"", "all", "*"}:
        return True
    expected = str(type_filter).strip().lower()
    actual = str(record.get("type") or record.get("split") or "").strip().lower()
    return actual == expected


@dataclass(frozen=True)
class PreparedDocument:
    selection_index: int
    source_index: int
    document_id: str
    title: str | None
    slug: str
    input_jsonl: str
    gold_jsonl: str
    run_dir: str
    input_sha256: str
    gold_sha256: str


@dataclass(frozen=True)
class BatchRunConfig:
    project_root: str
    ontology_path: str
    profile_path: str
    guidance_path: str
    relation_catalog_path: str
    relation_aliases_path: str
    model_name: str
    host: str = "https://openrouter.ai/api/v1"
    document_workers: int = 4
    layer_workers: int = 16
    reasoning_effort: str = "minimal"
    max_tokens: int = 4096
    request_timeout: int = 120
    resume_completed: bool = True
    retry_failed_documents: bool = True
    document_attempts: int = 2
    retry_backoff_seconds: float = 8.0
    launch_stagger_seconds: float = 0.75
    verbose_documents: bool = False
    progress_every: int = 1


def prepare_documents(
    *,
    dataset_jsonl: str | Path,
    task_guidance_path: str | Path,
    batch_root: str | Path,
    run_all_documents: bool,
    smoke_document_limit: int = 5,
    type_filter: str | None = None,
    start_index: int = 0,
) -> list[PreparedDocument]:
    dataset_jsonl = Path(dataset_jsonl).resolve()
    task_guidance_path = Path(task_guidance_path).resolve()
    batch_root = Path(batch_root).resolve()
    selected_root = batch_root / "selected_documents"
    selected_root.mkdir(parents=True, exist_ok=True)
    task_guidance = read_json(task_guidance_path)

    selected: list[PreparedDocument] = []
    aggregate_input = batch_root / "selected_input_no_gold.jsonl"
    aggregate_gold = batch_root / "selected_gold.jsonl"
    aggregate_input.parent.mkdir(parents=True, exist_ok=True)

    input_handle = aggregate_input.open("w", encoding="utf-8")
    gold_handle = aggregate_gold.open("w", encoding="utf-8")
    try:
        matched_index = 0
        for source_index, record in enumerate(iter_jsonl(dataset_jsonl)):
            if not record_matches_type(record, type_filter):
                continue
            if matched_index < start_index:
                matched_index += 1
                continue
            if not run_all_documents and len(selected) >= smoke_document_limit:
                break

            doc_id = document_id(record, source_index)
            slug = safe_slug(doc_id)
            doc_root = selected_root / f"{len(selected):06d}_{slug}"
            doc_root.mkdir(parents=True, exist_ok=True)
            input_record = strip_gold(record, task_guidance)
            input_path = doc_root / "input.jsonl"
            gold_path = doc_root / "gold.jsonl"
            input_line = json.dumps(input_record, ensure_ascii=False, separators=(",", ":"))
            gold_line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            input_path.write_text(input_line + "\n", encoding="utf-8")
            gold_path.write_text(gold_line + "\n", encoding="utf-8")
            input_handle.write(input_line + "\n")
            gold_handle.write(gold_line + "\n")

            selected.append(PreparedDocument(
                selection_index=len(selected),
                source_index=source_index,
                document_id=doc_id,
                title=str(record.get("title")) if record.get("title") is not None else None,
                slug=slug,
                input_jsonl=str(input_path),
                gold_jsonl=str(gold_path),
                run_dir=str(batch_root / "document_runs" / f"{len(selected):06d}_{slug}"),
                input_sha256=sha256_bytes((input_line + "\n").encode("utf-8")),
                gold_sha256=sha256_bytes((gold_line + "\n").encode("utf-8")),
            ))
            matched_index += 1
    finally:
        input_handle.close()
        gold_handle.close()

    if not selected:
        raise ValueError(
            f"No documents selected from {dataset_jsonl}; type_filter={type_filter!r}, "
            f"start_index={start_index}"
        )

    selection_manifest = {
        "batch_version": BATCH_VERSION,
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_sha256": sha256_file(dataset_jsonl),
        "task_guidance_path": str(task_guidance_path),
        "task_guidance_sha256": sha256_file(task_guidance_path),
        "run_all_documents": run_all_documents,
        "smoke_document_limit": smoke_document_limit,
        "type_filter": type_filter,
        "start_index": start_index,
        "selected_count": len(selected),
        "documents": [asdict(item) for item in selected],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(batch_root / "selection_manifest.json", selection_manifest)
    return selected


def _scientific_fingerprint(job: dict[str, Any], config: BatchRunConfig) -> str:
    payload = {
        "batch_version": BATCH_VERSION,
        "input_sha256": job["input_sha256"],
        "gold_sha256": job["gold_sha256"],
        "model_name": config.model_name,
        "host": config.host,
        "reasoning_effort": config.reasoning_effort,
        "layer_workers": config.layer_workers,
        "max_tokens": config.max_tokens,
        "request_timeout": config.request_timeout,
        "ontology_sha256": sha256_file(config.ontology_path),
        "profile_sha256": sha256_file(config.profile_path),
        "guidance_sha256": sha256_file(config.guidance_path),
        "relation_catalog_sha256": sha256_file(config.relation_catalog_path),
        "relation_aliases_sha256": sha256_file(config.relation_aliases_path),
    }
    return sha256_bytes(_stable_json(payload).encode("utf-8"))


def _set_metrics(predicted: set[str], expected: set[str]) -> dict[str, Any]:
    tp = predicted & expected
    fp = predicted - expected
    fn = expected - predicted
    precision = len(tp) / len(predicted) if predicted else 0.0
    recall = len(tp) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted": len(predicted),
        "gold": len(expected),
        "true_positive": len(tp),
        "false_positive": len(fp),
        "false_negative": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": sorted(tp),
        "fp": sorted(fp),
        "fn": sorted(fn),
    }


def _entity_metrics(run_dir: Path, gold: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    states = {index: state for index, _, state in v5.load_layer_states(run_dir)}
    final_state = states[max(states)]
    candidate_values = [
        *(final_state.entity_candidates or []),
        *(final_state.event_candidates or []),
        *(final_state.attribute_candidates or []),
    ]
    predicted_entities: set[str] = set()
    for candidate in candidate_values:
        projection = v5.project_candidate_to_gold(candidate, gold)
        entity_id = projection.get("entity_id")
        if entity_id:
            predicted_entities.add(str(entity_id))
    gold_entities = set(str(key) for key in (gold.get("entities") or {}).keys())

    predicted_endpoints: set[str] = set()
    for row in predictions:
        if not row.get("fully_mapped"):
            continue
        predicted_endpoints.add(str(row["head_id"]))
        predicted_endpoints.add(str(row["tail_id"]))
    gold_endpoints: set[str] = set()
    for _, pairs in (gold.get("relations") or {}).items():
        for head, tail in pairs:
            gold_endpoints.add(str(head))
            gold_endpoints.add(str(tail))
    return {
        "entity_inventory": _set_metrics(predicted_entities, gold_entities),
        "relation_endpoint_inventory": _set_metrics(predicted_endpoints, gold_endpoints),
    }


def _is_transient_exception(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "429", "rate limit", "too many requests", "timeout", "timed out",
        "connection reset", "connection aborted", "502", "503", "504",
        "server error", "temporarily unavailable",
    ]
    return any(marker in text for marker in markers)


def _completed_pipeline_manifest(
    run_dir: Path,
    job: dict[str, Any],
    config: BatchRunConfig,
) -> dict[str, Any] | None:
    """Return a compatible completed pipeline manifest, if one exists.

    This permits evaluation-only recovery after a post-pipeline analysis failure
    without deleting artifacts or paying for the LLM calls a second time.
    """
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = read_json(path)
    except Exception:
        return None
    if not manifest.get("completed_at"):
        return None
    if str(manifest.get("document_id")) != str(job.get("document_id")):
        return None
    if str(manifest.get("model_name")) != str(config.model_name):
        return None
    # The v5.1 run manifest records the exact scientific resource paths. Resolve
    # them when possible, but tolerate copied repositories with the same basename.
    for manifest_key, configured in [
        ("profile_path", config.profile_path),
        ("guidance_path", config.guidance_path),
        ("ontology_path", config.ontology_path),
    ]:
        recorded = str(manifest.get(manifest_key) or "")
        if recorded and Path(recorded).name != Path(configured).name:
            return None
    return manifest


def _completed_result_from_saved_pipeline(
    *,
    job: dict[str, Any],
    config: BatchRunConfig,
    run_dir: Path,
    fingerprint: str,
    manifest: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """Rebuild evaluation from saved Layer 0--12 artifacts only."""
    summary = v5.analyze_run(
        run_dir=run_dir,
        gold_jsonl=job["gold_jsonl"],
        catalog_path=config.relation_catalog_path,
        aliases_path=config.relation_aliases_path,
    )
    strict = summary["strict_evaluation_v5"]
    predictions = summary["strict_docred_predictions_v5"]
    gold = read_jsonl(job["gold_jsonl"])[0]
    entities = _entity_metrics(run_dir, gold, predictions)
    return {
        "status": "completed",
        "batch_version": BATCH_VERSION,
        "scientific_fingerprint": fingerprint,
        "selection_index": job["selection_index"],
        "source_index": job["source_index"],
        "document_id": job["document_id"],
        "title": job.get("title"),
        "slug": job["slug"],
        "run_dir": str(run_dir),
        "attempt": 0,
        "wall_seconds": round(time.time() - started, 3),
        "pipeline_seconds": manifest.get("elapsed_seconds"),
        "pipeline_reused": True,
        "analysis_recovered_from_saved_artifacts": True,
        "relation_metrics": strict,
        "entity_metrics": entities,
        "predictions": predictions,
        "cumulative_evaluation": summary.get("cumulative_strict_evaluation_v5", []),
        "failure_counts": summary.get("failure_counts_v5", {}),
        "final_counts": manifest.get("final_counts", {}),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _worker_run_document(job: dict[str, Any], config_dict: dict[str, Any], api_key: str) -> dict[str, Any]:
    config = BatchRunConfig(**config_dict)
    run_dir = Path(job["run_dir"]).resolve()
    result_path = run_dir / "document_result.json"
    failure_path = run_dir / "document_failure.json"
    fingerprint = _scientific_fingerprint(job, config)

    if config.launch_stagger_seconds > 0:
        time.sleep((int(job["selection_index"]) % max(1, config.document_workers)) * config.launch_stagger_seconds)

    if config.resume_completed and result_path.is_file():
        previous = read_json(result_path)
        if previous.get("status") == "completed" and previous.get("scientific_fingerprint") == fingerprint:
            previous["status"] = "skipped_completed"
            previous["resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return previous

    attempts = max(1, int(config.document_attempts if config.retry_failed_documents else 1))
    started = time.time()
    last_exc: BaseException | None = None

    # A previous v6 invocation may have completed all NeoOLAF layers and failed
    # only while writing evaluation CSV files. Recover analysis first and never
    # delete or rerun a compatible completed pipeline merely because evaluation
    # failed. This protects both artifacts and OpenRouter credits.
    manifest = _completed_pipeline_manifest(run_dir, job, config)
    if config.resume_completed and manifest is not None:
        try:
            result = _completed_result_from_saved_pipeline(
                job=job, config=config, run_dir=run_dir, fingerprint=fingerprint,
                manifest=manifest, started=started,
            )
            write_json(result_path, result)
            if failure_path.exists():
                failure_path.unlink()
            return result
        except BaseException as exc:
            failure = {
                "status": "failed_analysis_recovery",
                "failure_stage": "analysis_only",
                "pipeline_completed": True,
                "pipeline_seconds": manifest.get("elapsed_seconds"),
                "batch_version": BATCH_VERSION,
                "scientific_fingerprint": fingerprint,
                "selection_index": job["selection_index"],
                "source_index": job["source_index"],
                "document_id": job["document_id"],
                "title": job.get("title"),
                "slug": job["slug"],
                "run_dir": str(run_dir),
                "attempt": 0,
                "attempts_allowed": attempts,
                "transient": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_json(failure_path, failure)
            return failure

    for attempt in range(1, attempts + 1):
        try:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            v5.run_native_pipeline(
                project_root=config.project_root,
                input_jsonl=job["input_jsonl"],
                ontology_path=config.ontology_path,
                profile_path=config.profile_path,
                guidance_path=config.guidance_path,
                relation_catalog_path=config.relation_catalog_path,
                relation_aliases_path=config.relation_aliases_path,
                run_dir=run_dir,
                model_name=config.model_name,
                api_key=api_key,
                host=config.host,
                workers=config.layer_workers,
                max_tokens=config.max_tokens,
                request_timeout=config.request_timeout,
                reasoning_effort=config.reasoning_effort,
                verbose=config.verbose_documents,
                clean_run_dir=False,
            )
            summary = v5.analyze_run(
                run_dir=run_dir,
                gold_jsonl=job["gold_jsonl"],
                catalog_path=config.relation_catalog_path,
                aliases_path=config.relation_aliases_path,
            )
            strict = summary["strict_evaluation_v5"]
            predictions = summary["strict_docred_predictions_v5"]
            gold = read_jsonl(job["gold_jsonl"])[0]
            entities = _entity_metrics(run_dir, gold, predictions)
            manifest = read_json(run_dir / "run_manifest.json")
            result = {
                "status": "completed",
                "batch_version": BATCH_VERSION,
                "scientific_fingerprint": fingerprint,
                "selection_index": job["selection_index"],
                "source_index": job["source_index"],
                "document_id": job["document_id"],
                "title": job.get("title"),
                "slug": job["slug"],
                "run_dir": str(run_dir),
                "attempt": attempt,
                "wall_seconds": round(time.time() - started, 3),
                "pipeline_seconds": manifest.get("elapsed_seconds"),
                "relation_metrics": strict,
                "entity_metrics": entities,
                "predictions": predictions,
                "cumulative_evaluation": summary.get("cumulative_strict_evaluation_v5", []),
                "failure_counts": summary.get("failure_counts_v5", {}),
                "final_counts": manifest.get("final_counts", {}),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            write_json(result_path, result)
            if failure_path.exists():
                failure_path.unlink()
            return result
        except BaseException as exc:  # persist every document-level failure
            last_exc = exc
            transient = _is_transient_exception(exc)
            failure = {
                "status": "failed",
                "batch_version": BATCH_VERSION,
                "scientific_fingerprint": fingerprint,
                "selection_index": job["selection_index"],
                "source_index": job["source_index"],
                "document_id": job["document_id"],
                "title": job.get("title"),
                "slug": job["slug"],
                "run_dir": str(run_dir),
                "attempt": attempt,
                "attempts_allowed": attempts,
                "transient": transient,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(failure_path, failure)
            if attempt < attempts and transient:
                time.sleep(config.retry_backoff_seconds * attempt)
                continue
            return failure

    assert last_exc is not None
    raise last_exc


def run_documents_parallel(
    *,
    documents: Sequence[PreparedDocument],
    config: BatchRunConfig,
    api_key: str,
    batch_root: str | Path,
) -> list[dict[str, Any]]:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    if not 1 <= int(config.document_workers) <= 5:
        raise ValueError("document_workers must be between 1 and 5 for this notebook")
    if int(config.layer_workers) < 1:
        raise ValueError("layer_workers must be positive")

    batch_root = Path(batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    events_path = batch_root / "batch_events.jsonl"
    # Begin a new event stream for this invocation. Per-document results remain resumable.
    events_path.write_text("", encoding="utf-8")
    config_dict = asdict(config)
    jobs = [asdict(item) for item in documents]
    results: list[dict[str, Any]] = []
    started = time.time()

    # spawn is required for safe Windows/Jupyter document isolation. Each child
    # has independent stdout redirection, NeoOLAF state, and layer thread pools.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=config.document_workers, mp_context=context) as pool:
        future_to_job = {
            pool.submit(_worker_run_document, job, config_dict, api_key): job
            for job in jobs
        }
        completed = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            try:
                result = future.result()
            except BaseException as exc:
                result = {
                    "status": "failed_parent_process",
                    "selection_index": job["selection_index"],
                    "source_index": job["source_index"],
                    "document_id": job["document_id"],
                    "title": job.get("title"),
                    "slug": job["slug"],
                    "run_dir": job["run_dir"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            event = {
                "completed": completed,
                "total": len(jobs),
                "status": result.get("status"),
                "document_id": result.get("document_id"),
                "pipeline_seconds": result.get("pipeline_seconds"),
                "pipeline_reused": result.get("pipeline_reused", False),
                "attempt": result.get("attempt"),
                "transient": result.get("transient"),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
                "elapsed_batch_seconds": round(time.time() - started, 3),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_jsonl(events_path, event)
            progress_every = max(1, int(config.progress_every))
            if completed == 1 or completed == len(jobs) or completed % progress_every == 0 or str(event["status"]).startswith("failed"):
                pipeline_text = (
                    f"{event['pipeline_seconds']}s"
                    if event.get("pipeline_seconds") is not None else "n/a"
                )
                suffix = " | reused saved pipeline" if event.get("pipeline_reused") else ""
                if str(event.get("status", "")).startswith("failed"):
                    suffix += (
                        f" | {event.get('error_type')}: {event.get('error')}"
                        f" | transient={event.get('transient')}"
                    )
                print(
                    f"[{completed}/{len(jobs)}] {event['status']}: "
                    f"{event['document_id']} | pipeline={pipeline_text}{suffix}"
                )

    results.sort(key=lambda item: int(item.get("selection_index", 10**12)))
    write_json(batch_root / "document_results.json", results)
    return results


def _aggregate_count_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    predicted = sum(int(row.get("predicted", 0)) for row in rows)
    gold = sum(int(row.get("gold", 0)) for row in rows)
    tp = sum(int(row.get("true_positive", 0)) for row in rows)
    fp = sum(int(row.get("false_positive", 0)) for row in rows)
    fn = sum(int(row.get("false_negative", 0)) for row in rows)
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted": predicted,
        "gold": gold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_batch_results(
    *,
    results: Sequence[dict[str, Any]],
    batch_root: str | Path,
    relation_catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    batch_root = Path(batch_root).resolve()
    analysis_root = batch_root / "aggregate_analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    completed = [row for row in results if row.get("status") in {"completed", "skipped_completed"}]
    failed = [row for row in results if row.get("status") not in {"completed", "skipped_completed"}]

    relation_rows = [row["relation_metrics"] for row in completed]
    entity_rows = [row["entity_metrics"]["entity_inventory"] for row in completed]
    endpoint_rows = [row["entity_metrics"]["relation_endpoint_inventory"] for row in completed]

    micro_rel = _aggregate_count_metrics(relation_rows)
    micro_ent = _aggregate_count_metrics(entity_rows)
    micro_end = _aggregate_count_metrics(endpoint_rows)
    macro_rel = {
        "precision": _mean(row.get("precision", 0.0) for row in relation_rows),
        "recall": _mean(row.get("recall", 0.0) for row in relation_rows),
        "f1": _mean(row.get("f1", 0.0) for row in relation_rows),
    }
    macro_ent = {
        "precision": _mean(row.get("precision", 0.0) for row in entity_rows),
        "recall": _mean(row.get("recall", 0.0) for row in entity_rows),
        "f1": _mean(row.get("f1", 0.0) for row in entity_rows),
    }

    per_document: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    relation_counts: dict[str, dict[str, int]] = {}
    failure_counts: dict[str, int] = {}
    for result in completed:
        rel = result["relation_metrics"]
        ent = result["entity_metrics"]["entity_inventory"]
        endpoint = result["entity_metrics"]["relation_endpoint_inventory"]
        per_document.append({
            "selection_index": result.get("selection_index"),
            "source_index": result.get("source_index"),
            "document_id": result.get("document_id"),
            "title": result.get("title"),
            "status": result.get("status"),
            "pipeline_seconds": result.get("pipeline_seconds"),
            "wall_seconds": result.get("wall_seconds"),
            "relation_predicted": rel.get("predicted"),
            "relation_gold": rel.get("gold"),
            "relation_tp": rel.get("true_positive"),
            "relation_fp": rel.get("false_positive"),
            "relation_fn": rel.get("false_negative"),
            "relation_precision": rel.get("precision"),
            "relation_recall": rel.get("recall"),
            "relation_f1": rel.get("f1"),
            "entity_precision": ent.get("precision"),
            "entity_recall": ent.get("recall"),
            "entity_f1": ent.get("f1"),
            "endpoint_precision": endpoint.get("precision"),
            "endpoint_recall": endpoint.get("recall"),
            "endpoint_f1": endpoint.get("f1"),
            "run_dir": result.get("run_dir"),
        })
        for prediction in result.get("predictions", []):
            prediction_rows.append({
                "document_id": result.get("document_id"),
                "title": result.get("title"),
                **prediction,
            })
        for bucket in ("tp", "fp", "fn"):
            for triple in rel.get(bucket, []):
                if len(triple) != 3:
                    continue
                relation_id = str(triple[1])
                counts = relation_counts.setdefault(relation_id, {"tp": 0, "fp": 0, "fn": 0})
                counts[bucket] += 1
        for reason, count in (result.get("failure_counts") or {}).items():
            failure_counts[str(reason)] = failure_counts.get(str(reason), 0) + int(count)

    labels: dict[str, str] = {}
    if relation_catalog_path and Path(relation_catalog_path).is_file():
        catalog = read_json(relation_catalog_path)
        for row in catalog.get("properties", []):
            relation_id = str(row.get("property_id") or row.get("id") or "")
            if relation_id:
                labels[relation_id] = str(row.get("label") or "")
    per_relation: list[dict[str, Any]] = []
    for relation_id, counts in sorted(relation_counts.items()):
        predicted = counts["tp"] + counts["fp"]
        gold = counts["tp"] + counts["fn"]
        precision = counts["tp"] / predicted if predicted else 0.0
        recall = counts["tp"] / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_relation.append({
            "relation_id": relation_id,
            "label": labels.get(relation_id, ""),
            "predicted": predicted,
            "gold": gold,
            "true_positive": counts["tp"],
            "false_positive": counts["fp"],
            "false_negative": counts["fn"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # Aggregate cumulative layer evaluation using micro counts at each layer.
    layer_buckets: dict[int, dict[str, Any]] = {}
    for result in completed:
        for row in result.get("cumulative_evaluation", []):
            index = int(row.get("layer_index", -1))
            bucket = layer_buckets.setdefault(index, {
                "layer_index": index,
                "layer_name": row.get("layer_name"),
                "predicted": 0,
                "gold": 0,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
            })
            for key in ["predicted", "gold", "true_positive", "false_positive", "false_negative"]:
                bucket[key] += int(row.get(key, 0))
    cumulative: list[dict[str, Any]] = []
    for index in sorted(layer_buckets):
        row = layer_buckets[index]
        p = row["true_positive"] / row["predicted"] if row["predicted"] else 0.0
        r = row["true_positive"] / row["gold"] if row["gold"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        cumulative.append({**row, "precision": p, "recall": r, "f1": f1})

    total_pipeline_seconds = sum(float(row.get("pipeline_seconds") or 0.0) for row in completed)
    summary = {
        "batch_version": BATCH_VERSION,
        "documents_requested": len(results),
        "documents_completed_or_resumed": len(completed),
        "documents_failed": len(failed),
        "micro_relation": micro_rel,
        "macro_relation": macro_rel,
        "micro_entity_inventory": micro_ent,
        "macro_entity_inventory": macro_ent,
        "micro_relation_endpoint_inventory": micro_end,
        "total_document_pipeline_seconds": total_pipeline_seconds,
        "mean_document_pipeline_seconds": total_pipeline_seconds / len(completed) if completed else 0.0,
        "median_document_pipeline_seconds": statistics.median(
            [float(row.get("pipeline_seconds") or 0.0) for row in completed]
        ) if completed else 0.0,
        "first_failure_counts": failure_counts,
        "failed_documents": failed,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_json(analysis_root / "batch_summary.json", summary)
    write_json(analysis_root / "document_results_compact.json", per_document)
    write_json(analysis_root / "per_relation_metrics.json", per_relation)
    write_json(analysis_root / "cumulative_layer_micro_evaluation.json", cumulative)
    _write_csv(analysis_root / "per_document_metrics.csv", per_document)
    _write_csv(analysis_root / "per_relation_metrics.csv", per_relation)
    _write_csv(analysis_root / "cumulative_layer_micro_evaluation.csv", cumulative)
    with (analysis_root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    with (analysis_root / "failed_documents.jsonl").open("w", encoding="utf-8") as handle:
        for row in failed:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    return {
        "summary": summary,
        "per_document": per_document,
        "per_relation": per_relation,
        "cumulative": cumulative,
        "predictions": prediction_rows,
        "failed": failed,
    }


def run_batch(
    *,
    dataset_jsonl: str | Path,
    task_guidance_path: str | Path,
    batch_root: str | Path,
    run_all_documents: bool,
    smoke_document_limit: int,
    type_filter: str | None,
    start_index: int,
    config: BatchRunConfig,
    api_key: str,
) -> dict[str, Any]:
    batch_root = Path(batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    documents = prepare_documents(
        dataset_jsonl=dataset_jsonl,
        task_guidance_path=task_guidance_path,
        batch_root=batch_root,
        run_all_documents=run_all_documents,
        smoke_document_limit=smoke_document_limit,
        type_filter=type_filter,
        start_index=start_index,
    )
    invocation = {
        "batch_version": BATCH_VERSION,
        "run_all_documents": run_all_documents,
        "smoke_document_limit": smoke_document_limit,
        "type_filter": type_filter,
        "start_index": start_index,
        "selected_documents": len(documents),
        "config": {**asdict(config), "api_key": "NOT_STORED"},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(batch_root / "batch_invocation.json", invocation)
    results = run_documents_parallel(
        documents=documents,
        config=config,
        api_key=api_key,
        batch_root=batch_root,
    )
    aggregate = aggregate_batch_results(
        results=results,
        batch_root=batch_root,
        relation_catalog_path=config.relation_catalog_path,
    )
    invocation["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    invocation["documents_completed_or_resumed"] = aggregate["summary"]["documents_completed_or_resumed"]
    invocation["documents_failed"] = aggregate["summary"]["documents_failed"]
    write_json(batch_root / "batch_invocation.json", invocation)
    return {
        "documents": [asdict(item) for item in documents],
        "results": results,
        **aggregate,
    }

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import statistics
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import docred_native_batch_v6_1 as base


ORCHESTRATOR_VERSION = "v6.2-dev-filtered-bounded-streaming"
EXACT_REQUIRED_TYPE = "dev"

BatchRunConfig = base.BatchRunConfig
PreparedDocument = base.PreparedDocument
read_json = base.read_json
write_json = base.write_json
append_jsonl = base.append_jsonl
iter_jsonl = base.iter_jsonl
sha256_bytes = base.sha256_bytes
sha256_file = base.sha256_file
document_id = base.document_id
safe_slug = base.safe_slug
strip_gold = base.strip_gold


def _json_default(value: Any) -> Any:
    return base._json_default(value)


def is_exact_type(record: dict[str, Any], required_type: str = EXACT_REQUIRED_TYPE) -> bool:
    """Match only the literal JSON key ``type``; never fall back to ``split``."""
    return str(record.get("type") or "").strip().lower() == str(required_type).strip().lower()


def count_exact_type_records(
    dataset_jsonl: str | Path,
    *,
    required_type: str = EXACT_REQUIRED_TYPE,
    first_n_ids: int = 5,
) -> dict[str, Any]:
    """Count matching records with a single streaming pass and O(1) document memory."""
    total = 0
    matching = 0
    first_ids: list[str] = []
    for source_index, record in enumerate(iter_jsonl(dataset_jsonl)):
        total += 1
        if not is_exact_type(record, required_type):
            continue
        matching += 1
        if len(first_ids) < max(0, int(first_n_ids)):
            first_ids.append(document_id(record, source_index))
    return {
        "total_records": total,
        "matching_records": matching,
        "required_type": required_type,
        "first_matching_ids": first_ids,
    }


def iter_prepared_documents_streaming(
    *,
    dataset_jsonl: str | Path,
    task_guidance_path: str | Path,
    batch_root: str | Path,
    run_all_documents: bool,
    smoke_document_limit: int = 5,
    required_type: str = EXACT_REQUIRED_TYPE,
    start_index: int = 0,
) -> Iterator[PreparedDocument]:
    """Yield one prepared matching document at a time.

    The caller controls how far this generator is advanced. The bounded launcher
    advances it only when a document-worker slot is available, so the parent
    process never constructs a list of the full dev split.
    """
    dataset_jsonl = Path(dataset_jsonl).resolve()
    task_guidance_path = Path(task_guidance_path).resolve()
    batch_root = Path(batch_root).resolve()
    selected_root = batch_root / "selected_documents"
    selected_root.mkdir(parents=True, exist_ok=True)
    task_guidance = read_json(task_guidance_path)

    selection_jsonl = batch_root / "selection_documents.jsonl"
    selection_jsonl.parent.mkdir(parents=True, exist_ok=True)
    selection_jsonl.write_text("", encoding="utf-8")

    selected_count = 0
    matching_index = 0
    for source_index, record in enumerate(iter_jsonl(dataset_jsonl)):
        if not is_exact_type(record, required_type):
            continue
        if matching_index < int(start_index):
            matching_index += 1
            continue
        if not run_all_documents and selected_count >= int(smoke_document_limit):
            break

        doc_id = document_id(record, source_index)
        slug = safe_slug(doc_id)
        selection_index = selected_count
        doc_root = selected_root / f"{selection_index:06d}_{slug}"
        doc_root.mkdir(parents=True, exist_ok=True)

        # Gold is stripped before the native pipeline input is serialized.
        input_record = strip_gold(record, task_guidance)
        input_line = json.dumps(input_record, ensure_ascii=False, separators=(",", ":"))
        gold_line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        input_path = doc_root / "input.jsonl"
        gold_path = doc_root / "gold.jsonl"
        input_path.write_text(input_line + "\n", encoding="utf-8")
        gold_path.write_text(gold_line + "\n", encoding="utf-8")

        prepared = PreparedDocument(
            selection_index=selection_index,
            source_index=source_index,
            document_id=doc_id,
            title=str(record.get("title")) if record.get("title") is not None else None,
            slug=slug,
            input_jsonl=str(input_path),
            gold_jsonl=str(gold_path),
            run_dir=str(batch_root / "document_runs" / f"{selection_index:06d}_{slug}"),
            input_sha256=sha256_bytes((input_line + "\n").encode("utf-8")),
            gold_sha256=sha256_bytes((gold_line + "\n").encode("utf-8")),
        )
        append_jsonl(selection_jsonl, asdict(prepared))
        yield prepared

        selected_count += 1
        matching_index += 1


def _load_parent_resumable_result(
    job: dict[str, Any],
    config: BatchRunConfig,
) -> dict[str, Any] | None:
    if not config.resume_completed:
        return None
    run_dir = Path(job["run_dir"]).resolve()
    result_path = run_dir / "document_result.json"
    if not result_path.is_file():
        return None
    try:
        previous = read_json(result_path)
        fingerprint = base._scientific_fingerprint(job, config)
    except Exception:
        return None
    if previous.get("status") != "completed":
        return None
    if previous.get("scientific_fingerprint") != fingerprint:
        return None
    resumed = dict(previous)
    resumed["status"] = "skipped_completed"
    resumed["resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return resumed


def _persist_parent_failure(job: dict[str, Any], config: BatchRunConfig, exc: BaseException) -> dict[str, Any]:
    run_dir = Path(job["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    failure = {
        "status": "failed_parent_process",
        "batch_version": base.BATCH_VERSION,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "scientific_fingerprint": base._scientific_fingerprint(job, config),
        "selection_index": job["selection_index"],
        "source_index": job["source_index"],
        "document_id": job["document_id"],
        "title": job.get("title"),
        "slug": job["slug"],
        "run_dir": str(run_dir),
        "transient": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir / "document_failure.json", failure)
    return failure


def _event_from_result(
    result: dict[str, Any],
    *,
    completed: int,
    total: int,
    started: float,
) -> dict[str, Any]:
    return {
        "completed": completed,
        "total": total,
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


def _print_progress(event: dict[str, Any], *, progress_every: int) -> None:
    completed = int(event["completed"])
    total = int(event["total"])
    status = str(event.get("status") or "")
    should_print = (
        completed == 1
        or completed == total
        or completed % max(1, int(progress_every)) == 0
        or status.startswith("failed")
    )
    if not should_print:
        return
    pipeline = event.get("pipeline_seconds")
    pipeline_text = f"{pipeline}s" if pipeline is not None else "n/a"
    suffix = " | reused saved pipeline" if event.get("pipeline_reused") else ""
    if status.startswith("failed"):
        suffix += (
            f" | {event.get('error_type')}: {event.get('error')}"
            f" | transient={event.get('transient')}"
        )
    print(
        f"[{completed}/{total}] {status}: {event.get('document_id')} "
        f"| pipeline={pipeline_text}{suffix}"
    )


def run_documents_bounded_streaming(
    *,
    documents: Iterator[PreparedDocument],
    expected_total: int,
    config: BatchRunConfig,
    api_key: str,
    batch_root: str | Path,
) -> dict[str, Any]:
    """Run with at most ``document_workers`` submitted jobs in parent memory."""
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    if not 1 <= int(config.document_workers) <= 5:
        raise ValueError("document_workers must be between 1 and 5")
    if int(config.layer_workers) < 1:
        raise ValueError("layer_workers must be positive")

    batch_root = Path(batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    events_path = batch_root / "batch_events.jsonl"
    events_path.write_text("", encoding="utf-8")

    started = time.time()
    prepared = 0
    completed = 0
    completed_ok = 0
    failed = 0
    resumed = 0
    exhausted = False
    config_dict = asdict(config)
    pending: dict[Any, dict[str, Any]] = {}

    def record_result(result: dict[str, Any]) -> None:
        nonlocal completed, completed_ok, failed, resumed
        completed += 1
        status = str(result.get("status") or "")
        if status in {"completed", "skipped_completed"}:
            completed_ok += 1
        else:
            failed += 1
        if status == "skipped_completed" or result.get("pipeline_reused"):
            resumed += 1
        event = _event_from_result(
            result,
            completed=completed,
            total=expected_total,
            started=started,
        )
        append_jsonl(events_path, event)
        _print_progress(event, progress_every=config.progress_every)

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=config.document_workers, mp_context=context) as pool:
        while not exhausted or pending:
            # Advance the JSONL generator only while a worker slot is available.
            while not exhausted and len(pending) < int(config.document_workers):
                try:
                    prepared_document = next(documents)
                except StopIteration:
                    exhausted = True
                    break
                prepared += 1
                job = asdict(prepared_document)

                parent_resume = _load_parent_resumable_result(job, config)
                if parent_resume is not None:
                    record_result(parent_resume)
                    continue

                future = pool.submit(base._worker_run_document, job, config_dict, api_key)
                pending[future] = job

            if not pending:
                continue

            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    result = future.result()
                except BaseException as exc:
                    result = _persist_parent_failure(job, config, exc)
                record_result(result)

    return {
        "expected_total": expected_total,
        "documents_prepared": prepared,
        "documents_finished": completed,
        "documents_completed_or_resumed": completed_ok,
        "documents_failed": failed,
        "documents_resumed": resumed,
        "max_pending_documents": int(config.document_workers),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def _metric_accumulator() -> dict[str, int]:
    return {"predicted": 0, "gold": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}


def _add_metrics(acc: dict[str, int], row: dict[str, Any]) -> None:
    for key in acc:
        acc[key] += int(row.get(key, 0) or 0)


def _finish_metrics(acc: dict[str, int]) -> dict[str, Any]:
    predicted = acc["predicted"]
    gold = acc["gold"]
    tp = acc["true_positive"]
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**acc, "precision": precision, "recall": recall, "f1": f1}


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
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


def aggregate_batch_results_streaming(
    *,
    batch_root: str | Path,
    relation_catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate per-document files without loading document results into a list."""
    batch_root = Path(batch_root).resolve()
    selection_path = batch_root / "selection_documents.jsonl"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing selection stream: {selection_path}")

    analysis_root = batch_root / "aggregate_analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    per_document_csv = analysis_root / "per_document_metrics.csv"
    per_document_jsonl = analysis_root / "document_results_compact.jsonl"
    predictions_jsonl = analysis_root / "predictions.jsonl"
    failures_jsonl = analysis_root / "failed_documents.jsonl"

    per_document_fields = [
        "selection_index", "source_index", "document_id", "title", "status",
        "pipeline_seconds", "wall_seconds", "relation_predicted", "relation_gold",
        "relation_tp", "relation_fp", "relation_fn", "relation_precision",
        "relation_recall", "relation_f1", "entity_precision", "entity_recall",
        "entity_f1", "endpoint_precision", "endpoint_recall", "endpoint_f1", "run_dir",
    ]

    relation_acc = _metric_accumulator()
    entity_acc = _metric_accumulator()
    endpoint_acc = _metric_accumulator()
    macro_relation_sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    macro_entity_sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    relation_counts: dict[str, dict[str, int]] = {}
    failure_counts: dict[str, int] = {}
    layer_buckets: dict[int, dict[str, Any]] = {}
    pipeline_seconds_values: list[float] = []  # numeric only; no document payloads
    failed_preview: list[dict[str, Any]] = []

    requested = 0
    completed_count = 0
    failed_count = 0

    with (
        per_document_csv.open("w", encoding="utf-8", newline="") as csv_handle,
        per_document_jsonl.open("w", encoding="utf-8") as compact_handle,
        predictions_jsonl.open("w", encoding="utf-8") as predictions_handle,
        failures_jsonl.open("w", encoding="utf-8") as failures_handle,
    ):
        csv_writer = csv.DictWriter(csv_handle, fieldnames=per_document_fields, extrasaction="ignore")
        csv_writer.writeheader()

        for job in iter_jsonl(selection_path):
            requested += 1
            run_dir = Path(job["run_dir"]).resolve()
            result_path = run_dir / "document_result.json"
            failure_path = run_dir / "document_failure.json"

            result: dict[str, Any] | None = None
            if result_path.is_file():
                candidate = read_json(result_path)
                if candidate.get("status") == "completed":
                    result = candidate

            if result is None:
                failed_count += 1
                if failure_path.is_file():
                    failure = read_json(failure_path)
                else:
                    failure = {
                        "status": "missing_result",
                        "selection_index": job.get("selection_index"),
                        "source_index": job.get("source_index"),
                        "document_id": job.get("document_id"),
                        "title": job.get("title"),
                        "run_dir": str(run_dir),
                        "error_type": "MissingResult",
                        "error": "No completed document_result.json or document_failure.json was found.",
                    }
                failures_handle.write(json.dumps(failure, ensure_ascii=False, default=_json_default) + "\n")
                if len(failed_preview) < 50:
                    failed_preview.append(failure)
                continue

            completed_count += 1
            rel = result["relation_metrics"]
            ent = result["entity_metrics"]["entity_inventory"]
            endpoint = result["entity_metrics"]["relation_endpoint_inventory"]
            _add_metrics(relation_acc, rel)
            _add_metrics(entity_acc, ent)
            _add_metrics(endpoint_acc, endpoint)
            for key in macro_relation_sums:
                macro_relation_sums[key] += float(rel.get(key, 0.0) or 0.0)
                macro_entity_sums[key] += float(ent.get(key, 0.0) or 0.0)

            pipeline_seconds = float(result.get("pipeline_seconds") or 0.0)
            pipeline_seconds_values.append(pipeline_seconds)
            row = {
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
            }
            csv_writer.writerow(row)
            compact_handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

            for prediction in result.get("predictions", []):
                predictions_handle.write(json.dumps({
                    "document_id": result.get("document_id"),
                    "title": result.get("title"),
                    **prediction,
                }, ensure_ascii=False, default=_json_default) + "\n")

            for bucket in ("tp", "fp", "fn"):
                for triple in rel.get(bucket, []):
                    if len(triple) != 3:
                        continue
                    relation_id = str(triple[1])
                    counts = relation_counts.setdefault(relation_id, {"tp": 0, "fp": 0, "fn": 0})
                    counts[bucket] += 1

            for reason, count in (result.get("failure_counts") or {}).items():
                failure_counts[str(reason)] = failure_counts.get(str(reason), 0) + int(count)

            for layer_row in result.get("cumulative_evaluation", []):
                index = int(layer_row.get("layer_index", -1))
                bucket = layer_buckets.setdefault(index, {
                    "layer_index": index,
                    "layer_name": layer_row.get("layer_name"),
                    "predicted": 0,
                    "gold": 0,
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                })
                for key in ["predicted", "gold", "true_positive", "false_positive", "false_negative"]:
                    bucket[key] += int(layer_row.get(key, 0) or 0)

    labels: dict[str, str] = {}
    if relation_catalog_path and Path(relation_catalog_path).is_file():
        catalog = read_json(relation_catalog_path)
        for relation in catalog.get("properties", []):
            relation_id = str(relation.get("property_id") or relation.get("id") or "")
            if relation_id:
                labels[relation_id] = str(relation.get("label") or "")

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

    cumulative: list[dict[str, Any]] = []
    for index in sorted(layer_buckets):
        row = layer_buckets[index]
        precision = row["true_positive"] / row["predicted"] if row["predicted"] else 0.0
        recall = row["true_positive"] / row["gold"] if row["gold"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        cumulative.append({**row, "precision": precision, "recall": recall, "f1": f1})

    micro_relation = _finish_metrics(relation_acc)
    micro_entity = _finish_metrics(entity_acc)
    micro_endpoint = _finish_metrics(endpoint_acc)
    macro_relation = {
        key: macro_relation_sums[key] / completed_count if completed_count else 0.0
        for key in macro_relation_sums
    }
    macro_entity = {
        key: macro_entity_sums[key] / completed_count if completed_count else 0.0
        for key in macro_entity_sums
    }
    total_pipeline_seconds = sum(pipeline_seconds_values)

    summary = {
        "batch_version": base.BATCH_VERSION,
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "required_type_key": "type",
        "required_type_value": EXACT_REQUIRED_TYPE,
        "documents_requested": requested,
        "documents_completed_or_resumed": completed_count,
        "documents_failed": failed_count,
        "micro_relation": micro_relation,
        "macro_relation": macro_relation,
        "micro_entity_inventory": micro_entity,
        "macro_entity_inventory": macro_entity,
        "micro_relation_endpoint_inventory": micro_endpoint,
        "total_document_pipeline_seconds": total_pipeline_seconds,
        "mean_document_pipeline_seconds": total_pipeline_seconds / completed_count if completed_count else 0.0,
        "median_document_pipeline_seconds": statistics.median(pipeline_seconds_values) if pipeline_seconds_values else 0.0,
        "first_failure_counts": failure_counts,
        "per_document_metrics_csv": str(per_document_csv),
        "failed_documents_jsonl": str(failures_jsonl),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_json(analysis_root / "batch_summary.json", summary)
    write_json(analysis_root / "per_relation_metrics.json", per_relation)
    write_json(analysis_root / "cumulative_layer_micro_evaluation.json", cumulative)
    _write_csv_rows(analysis_root / "per_relation_metrics.csv", per_relation)
    _write_csv_rows(analysis_root / "cumulative_layer_micro_evaluation.csv", cumulative)

    return {
        "summary": summary,
        "per_relation": per_relation,
        "cumulative": cumulative,
        "failed_preview": failed_preview,
        "paths": {
            "per_document_csv": str(per_document_csv),
            "per_document_jsonl": str(per_document_jsonl),
            "predictions_jsonl": str(predictions_jsonl),
            "failed_documents_jsonl": str(failures_jsonl),
            "batch_summary": str(analysis_root / "batch_summary.json"),
        },
    }


def run_batch_streaming(
    *,
    dataset_jsonl: str | Path,
    task_guidance_path: str | Path,
    batch_root: str | Path,
    run_all_documents: bool,
    smoke_document_limit: int,
    start_index: int,
    config: BatchRunConfig,
    api_key: str,
    matching_record_count: int | None = None,
    required_type: str = EXACT_REQUIRED_TYPE,
) -> dict[str, Any]:
    batch_root = Path(batch_root).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)

    if matching_record_count is None:
        matching_record_count = int(count_exact_type_records(
            dataset_jsonl,
            required_type=required_type,
            first_n_ids=0,
        )["matching_records"])
    available = max(0, int(matching_record_count) - int(start_index))
    selected_count = available if run_all_documents else min(int(smoke_document_limit), available)
    if selected_count <= 0:
        raise ValueError(
            f"No records with exact key/value type={required_type!r} remain after start_index={start_index}."
        )

    selection_manifest = {
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "scientific_batch_version": base.BATCH_VERSION,
        "dataset_jsonl": str(Path(dataset_jsonl).resolve()),
        "dataset_sha256": sha256_file(dataset_jsonl),
        "task_guidance_path": str(Path(task_guidance_path).resolve()),
        "task_guidance_sha256": sha256_file(task_guidance_path),
        "filter": {"key": "type", "value": required_type, "fallback_to_split": False},
        "run_all_documents": run_all_documents,
        "smoke_document_limit": smoke_document_limit,
        "start_index": start_index,
        "matching_record_count": matching_record_count,
        "selected_count": selected_count,
        "selection_documents_jsonl": str(batch_root / "selection_documents.jsonl"),
        "memory_policy": {
            "full_document_list_materialized": False,
            "maximum_pending_document_jobs": int(config.document_workers),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(batch_root / "selection_manifest.json", selection_manifest)

    invocation = {
        **selection_manifest,
        "config": {**asdict(config), "api_key": "NOT_STORED"},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(batch_root / "batch_invocation.json", invocation)

    documents = iter_prepared_documents_streaming(
        dataset_jsonl=dataset_jsonl,
        task_guidance_path=task_guidance_path,
        batch_root=batch_root,
        run_all_documents=run_all_documents,
        smoke_document_limit=smoke_document_limit,
        required_type=required_type,
        start_index=start_index,
    )
    scheduler = run_documents_bounded_streaming(
        documents=documents,
        expected_total=selected_count,
        config=config,
        api_key=api_key,
        batch_root=batch_root,
    )
    aggregate = aggregate_batch_results_streaming(
        batch_root=batch_root,
        relation_catalog_path=config.relation_catalog_path,
    )

    invocation["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    invocation["scheduler"] = scheduler
    invocation["documents_completed_or_resumed"] = aggregate["summary"]["documents_completed_or_resumed"]
    invocation["documents_failed"] = aggregate["summary"]["documents_failed"]
    write_json(batch_root / "batch_invocation.json", invocation)
    return {"scheduler": scheduler, **aggregate}

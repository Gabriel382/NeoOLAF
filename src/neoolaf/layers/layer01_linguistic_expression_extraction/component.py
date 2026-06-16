from __future__ import annotations

# Standard library imports
import json
import re
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Tuple

from neoolaf.core.base_layer import BaseLayer
from neoolaf.config.prompt_loader import load_prompt_template, render_prompt_template
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.grounding.rag.base import RAGBackend, RAGRequest
from neoolaf.profiles.identity_policy import apply_record_identity_policy
from neoolaf.schemas.structured_output import (
    StructuredOutputConfig,
    build_litellm_response_format,
    validate_with_pydantic,
)
from neoolaf.domain.linguistic_expression import LinguisticExpression, Evidence
from neoolaf.layers.layer01_linguistic_expression_extraction.prompt import (
    build_system_prompt,
    build_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend
# Third-party imports
from tqdm.auto import tqdm

class LinguisticExpressionExtractionLayer(BaseLayer):
    """
    Layer 01: linguistic expression extraction

    Responsibilities:
    - inspect chunks
    - call the LLM
    - extract semantically relevant linguistic expressions
    - locate their positions in the chunk/document
    - deduplicate results
    """

    name = "layer01_linguistic_expression_extraction"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        max_chunks: int | None = None,
        temperature: float = 0.0,
        save_intermediate: bool = True,
        verbose: bool = False,
        rag_backend: RAGBackend | None = None,
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
        rag_enabled: bool | None = None,
        rag_top_k: int = 0,
        rag_max_chars: int = 0,
        failed_chunks_file: str | None = None,
    ) -> None:
        """
        Initialize Layer 1.

        Args:
            ollama_backend:
                LLM backend used for chat calls.
            max_chunks:
                Optional chunk limit for quick testing.
            temperature:
                Generation temperature.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Wheter to show logs or not.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.max_chunks = max_chunks
        self.temperature = temperature
        self.rag_backend = rag_backend
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 0.0)
        self.rag_enabled = rag_enabled
        self.rag_top_k = max(0, int(rag_top_k or 0))
        self.rag_max_chars = max(0, int(rag_max_chars or 0))
        self.failed_chunks_file = failed_chunks_file
        self._failure_lock = threading.Lock()

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run expression extraction over document chunks.
        """
        strategy = (state.profile_config or {}).get("layers", {}).get(
            self.name, {}
        ).get("strategy", "generic")
        if strategy == "alarm_table_record":
            return self._run_alarm_table_record_extraction(state)

        expressions: List[LinguisticExpression] = []
        chunks = state.document.chunks

        # Optional chunk limit for faster debugging
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]

        expr_counter = 0

        chunk_iterator = chunks
        if self.verbose:
            chunk_iterator = tqdm(chunks, desc="Layer 1 - chunks", leave=False)

        for chunk in chunk_iterator:
            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(chunk, state.user_guidance, state.seed_ontology)},
            ]

            raw_response = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )

            # For ablation/debugging, never let one malformed or empty LLM
            # response kill the whole layer. Save the raw response and the
            # parsing error, then continue with the next chunk.
            parsed = self._safe_extract_json(
                raw_response=raw_response,
                state=state,
                chunk_id=chunk.chunk_id,
                messages=messages,
            )
            if parsed is None or not isinstance(parsed, dict):
                continue

            items = parsed.get("expressions", [])

            for item in items:
                text = item["text"].strip()
                label = item["label"].strip()
                justification = item["justification"].strip()

                # Try to locate the extracted expression in the chunk
                match_span = self._find_expression_span(
                    expression_text=text,
                    chunk_text=chunk.text,
                )

                if match_span is not None:
                    chunk_start_char, chunk_end_char = match_span
                    doc_start_char = chunk.start_char + chunk_start_char
                    doc_end_char = chunk.start_char + chunk_end_char
                    snippet = self._build_snippet(chunk.text, chunk_start_char, chunk_end_char)
                else:
                    # Fallback when the extracted string cannot be found exactly
                    chunk_start_char = -1
                    chunk_end_char = -1
                    doc_start_char = -1
                    doc_end_char = -1
                    snippet = chunk.text[:300]

                expressions.append(
                    LinguisticExpression(
                        expr_id=f"expr_{expr_counter:05d}",
                        text=text,
                        label=label,
                        justification=justification,
                        evidence=[
                            Evidence(
                                chunk_id=chunk.chunk_id,
                                chunk_start_char=chunk_start_char,
                                chunk_end_char=chunk_end_char,
                                doc_start_char=doc_start_char,
                                doc_end_char=doc_end_char,
                                snippet=snippet,
                            )
                        ],
                    )
                )
                expr_counter += 1

        # Deduplicate by normalized text + label
        dedup = {}
        for expr in expressions:
            key = (expr.text.lower(), expr.label.lower())
            if key not in dedup:
                dedup[key] = expr

        state.linguistic_expressions = list(dedup.values())
        state.log(
            f"[layer01_linguistic_expression_extraction] extracted "
            f"{len(state.linguistic_expressions)} unique expressions"
        )
        return state


    def _run_alarm_table_record_extraction(self, state: PipelineState) -> PipelineState:
        """Extract structured alarm records from table chunks.

        This is profile-specific behavior used by ``xquality_machine32``.  It is
        intentionally selected by the profile, not hard-coded as the default
        Layer 1 behavior.
        """
        prompt_path = (state.profile_config or {}).get("prompts", {}).get(self.name)
        template = load_prompt_template(
            prompt_path or "xquality_machine32/layer01_alarm_table_record_extraction.md",
            fallback=""
        )
        if not template:
            raise FileNotFoundError(
                "Missing alarm-table prompt template for Layer 1. "
                "Expected prompts/xquality_machine32/layer01_alarm_table_record_extraction.md"
            )

        # Avoid mixing stale JSON errors from older runs with the current run.
        # This makes the ablation artifacts much easier to inspect.
        self._clear_layer_runtime_artifacts(state)

        chunks = state.document.chunks
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]

        records: list[dict[str, Any]] = []
        expressions: list[LinguisticExpression] = []
        expr_counter = 0

        selected_chunks = []
        for chunk in chunks:
            # Skip non-table chunks when the profile selected table-aware extraction.
            if getattr(chunk, "metadata", {}).get("chunk_type") not in {"table", None}:
                continue
            selected_chunks.append(chunk)

        failed_chunk_ids = self._load_failed_chunk_filter()
        if failed_chunk_ids is not None:
            before = len(selected_chunks)
            selected_chunks = [chunk for chunk in selected_chunks if chunk.chunk_id in failed_chunk_ids]
            state.log({
                "layer": self.name,
                "status": "filtered_to_failed_chunks",
                "failed_chunks_file": self.failed_chunks_file,
                "requested_failed_chunks": sorted(failed_chunk_ids),
                "selected_before_filter": before,
                "selected_after_filter": len(selected_chunks),
            })

        # Layer 1 units are independent, so they can be processed in parallel.
        # Keep max_concurrency=1 for deterministic/debug runs.
        if self.max_concurrency > 1 and len(selected_chunks) > 1:
            if self.verbose:
                print(
                    f"[NeoOLAF][{self.name}] Processing {len(selected_chunks)} units "
                    f"with max_concurrency={self.max_concurrency}"
                )
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                future_to_index = {
                    executor.submit(self._process_alarm_table_chunk, chunk, state, template): index
                    for index, chunk in enumerate(selected_chunks)
                }
                iterator = as_completed(future_to_index)
                if self.verbose:
                    iterator = tqdm(iterator, total=len(future_to_index), desc="Layer 1 - alarm tables", leave=False)
                results: list[tuple[int, Any, list[dict[str, Any]]]] = []
                for future in iterator:
                    index = future_to_index[future]
                    chunk = selected_chunks[index]
                    try:
                        results.append((index, chunk, future.result()))
                    except Exception as exc:
                        error = f"Unhandled parallel worker error: {type(exc).__name__}: {exc}"
                        self._save_failed_response(
                            state=state,
                            chunk_id=chunk.chunk_id,
                            raw_response="",
                            messages=[],
                            error=error,
                        )
                        self._record_failed_chunk(
                            state=state,
                            chunk=chunk,
                            attempts=1 + self.retry_failed_calls,
                            error=error,
                            raw_response="",
                            messages=[],
                        )
                        state.log({
                            "layer": self.name,
                            "status": "worker_failed",
                            "chunk_id": chunk.chunk_id,
                            "error": str(exc),
                        })
                results.sort(key=lambda item: item[0])
                for _, chunk, chunk_records in results:
                    for record in chunk_records:
                        records.append(record)
                        for text, label in self._record_expression_items(record):
                            if not text:
                                continue
                            expr = self._make_linguistic_expression(
                                expr_id=f"expr_{expr_counter:05d}",
                                text=text,
                                label=label,
                                chunk=chunk,
                            )
                            expressions.append(expr)
                            expr_counter += 1
        else:
            chunk_iterator = selected_chunks
            if self.verbose:
                chunk_iterator = tqdm(selected_chunks, desc="Layer 1 - alarm tables", leave=False)
            for chunk in chunk_iterator:
                chunk_records = self._process_alarm_table_chunk(chunk, state, template)
                for record in chunk_records:
                    records.append(record)
                    for text, label in self._record_expression_items(record):
                        if not text:
                            continue
                        expr = self._make_linguistic_expression(
                            expr_id=f"expr_{expr_counter:05d}",
                            text=text,
                            label=label,
                            chunk=chunk,
                        )
                        expressions.append(expr)
                        expr_counter += 1

        # Deduplicate expressions but keep all alarm records.
        dedup: dict[tuple[str, str], LinguisticExpression] = {}
        for expr in expressions:
            key = (expr.text.lower(), expr.label.lower())
            dedup.setdefault(key, expr)

        state.document.alarm_records = records
        state.linguistic_expressions = list(dedup.values())
        state.log(
            f"[{self.name}] extracted {len(records)} alarm records and "
            f"{len(state.linguistic_expressions)} unique expressions"
        )
        return state


    def _process_alarm_table_chunk(
        self,
        chunk,
        state: PipelineState,
        template: str,
    ) -> list[dict[str, Any]]:
        """Process one table/record unit and return post-processed records.

        This method is intentionally side-effect-light so it can be called from
        a ThreadPoolExecutor. It retries the whole unit on transport/API errors,
        empty responses, JSON parsing failures, or schema problems. Only after
        all attempts fail is the chunk written to ``failed_chunks.json``.
        """
        total_attempts = 1 + self.retry_failed_calls
        last_error = ""
        last_messages: list[dict[str, str]] = []
        last_raw_response = ""

        for attempt in range(total_attempts):
            try:
                compact_unit = self._build_prompt_unit(chunk, state)
                rag_context = self._retrieve_optional_rag_context(compact_unit, chunk, state)
                if rag_context:
                    compact_unit["optional_rag_guidance"] = rag_context

                language_cfg = (state.profile_config or {}).get("language", {})
                output_language = str(language_cfg.get("target") or "en")
                translate_inside_layer = bool(language_cfg.get("llm_translate_inside_extraction", True))
                translation_instruction = (
                    f"Translate extracted source-language content to concise {output_language}."
                    if translate_inside_layer
                    else "Do not translate extracted content. Keep node labels in the source language."
                )

                user_prompt = render_prompt_template(
                    template,
                    table_unit_json=json.dumps(compact_unit, ensure_ascii=False, indent=2),
                    output_language=output_language,
                    translation_instruction=translation_instruction,
                    few_shot_examples=self._few_shot_examples_text(state),
                    # Backward-compatible variables for older prompt templates.
                    chunk_metadata=json.dumps(self._compact_metadata_for_prompt(chunk), ensure_ascii=False, indent=2),
                    chunk_text=chunk.text,
                )
                messages = [
                    {"role": "system", "content": "You are a strict JSON extractor for industrial alarm tables."},
                    {"role": "user", "content": user_prompt},
                ]
                last_messages = messages

                max_output_tokens = self._layer_max_output_tokens(state)
                structured_cfg = StructuredOutputConfig.from_profile(state.profile_config, self.name)
                response_format = build_litellm_response_format(structured_cfg)
                raw_response = self._chat_with_optional_max_tokens(
                    model=state.llm_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_output_tokens,
                    response_format=response_format,
                    fallback_to_json_parse=structured_cfg.fallback_to_json_parse,
                )
                last_raw_response = raw_response or ""

                parsed = self._safe_extract_json(
                    raw_response=raw_response,
                    state=state,
                    chunk_id=chunk.chunk_id,
                    messages=messages,
                )
                if parsed is None:
                    last_error = "JSON parse failed or model returned an empty/non-JSON response."
                    raise ValueError(last_error)

                chunk_records = self._extract_alarm_records_from_parsed(parsed)
                if not chunk_records:
                    last_error = "Parsed JSON did not contain an alarm_record object or list of records."
                    self._save_failed_response(
                        state=state,
                        chunk_id=chunk.chunk_id,
                        raw_response=raw_response or "",
                        messages=messages,
                        error=last_error,
                    )
                    raise ValueError(last_error)

                validated_records = []
                for record in chunk_records:
                    validated_records.append(
                        self._validate_alarm_record_with_optional_schema(
                            record=record,
                            state=state,
                            chunk_id=chunk.chunk_id,
                            messages=messages,
                            raw_response=raw_response or "",
                            structured_cfg=structured_cfg,
                        )
                    )

                return [
                    self._postprocess_alarm_record(record, chunk, state, compact_unit)
                    for record in validated_records
                ]

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                state.log({
                    "layer": self.name,
                    "status": "unit_attempt_failed",
                    "chunk_id": getattr(chunk, "chunk_id", None),
                    "attempt": attempt + 1,
                    "max_attempts": total_attempts,
                    "error": last_error,
                })
                self._save_failed_response(
                    state=state,
                    chunk_id=f"{chunk.chunk_id}.attempt_{attempt + 1}",
                    raw_response=last_raw_response,
                    messages=last_messages,
                    error=last_error,
                )
                if attempt < total_attempts - 1 and self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)

        self._record_failed_chunk(
            state=state,
            chunk=chunk,
            attempts=total_attempts,
            error=last_error or "Unknown Layer 1 unit failure.",
            raw_response=last_raw_response,
            messages=last_messages,
        )
        state.log({
            "layer": self.name,
            "status": "unit_failed_after_retries",
            "chunk_id": chunk.chunk_id,
            "attempts": total_attempts,
            "error": last_error,
        })
        return []

    def _few_shot_examples_text(self, state: PipelineState) -> str:
        """Return optional tiny few-shot examples from the active profile.

        Few-shot guidance can improve the alarm/message identity decision, but
        it also increases prompt size.  Keeping this profile-controlled makes it
        easy to disable for cheaper runs or other datasets.
        """
        layer_cfg = (state.profile_config or {}).get("layers", {}).get(self.name, {}) or {}
        prompt_cfg = layer_cfg.get("prompt_options", {}) or {}
        fewshot_cfg = prompt_cfg.get("few_shot_examples", {}) or {}
        if not bool(fewshot_cfg.get("enabled", False)):
            return ""

        mode = str(fewshot_cfg.get("mode", "tiny")).lower()
        if mode == "off":
            return ""

        examples = fewshot_cfg.get("examples")
        if isinstance(examples, list) and examples:
            lines = ["Tiny examples:"]
            for example in examples:
                if isinstance(example, str):
                    lines.append(f"- {example.strip()}")
                elif isinstance(example, dict):
                    source = str(example.get("source") or example.get("input") or "").strip()
                    target = str(example.get("target") or example.get("output") or "").strip()
                    if source and target:
                        lines.append(f"- {source} -> {target}")
            return "\n".join(lines).strip() + "\n" if len(lines) > 1 else ""

        # Default ultra-short examples for profiles that ask for tiny few-shot
        # guidance but do not define custom examples.
        return (
            "Tiny examples:\n"
            "- Alarm: header `Alarme n°: 1001`, label `URGENCE ACTIVE` "
            "-> record_type=alarm, record_id=1001, alarm_no=1001, message_no=null.\n"
            "- Message: header `message n°: 2060`, body mentions `voir alarme n° 1083` "
            "-> record_type=message, record_id=2060, message_no=2060, alarm_no=null; 1083 is only a reference.\n"
        )

    def _failed_chunks_manifest_path(self, state: PipelineState) -> Path | None:
        """Return the failed-chunk manifest path for the current run."""
        if state.artifact_dir is None:
            return None
        return Path(state.artifact_dir) / self.name / "failed_chunks.json"

    def _record_failed_chunk(
        self,
        *,
        state: PipelineState,
        chunk,
        attempts: int,
        error: str,
        raw_response: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Append a final failed unit to failed_chunks.json and detail files.

        This manifest allows a later run to process only failed chunks through
        ``--layer01-failed-chunks-file`` instead of rerunning the entire layer.
        """
        manifest_path = self._failed_chunks_manifest_path(state)
        if manifest_path is None:
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        details_dir = manifest_path.parent / "failed_chunk_details"
        details_dir.mkdir(parents=True, exist_ok=True)

        metadata = getattr(chunk, "metadata", {}) or {}
        entry = {
            "chunk_id": getattr(chunk, "chunk_id", None),
            "page": metadata.get("page"),
            "title": metadata.get("title"),
            "subsection_key": metadata.get("subsection_key"),
            "attempts": attempts,
            "error": error,
            "text_preview": str(getattr(chunk, "text", "") or "")[:500],
        }

        safe_chunk_id = self._safe_artifact_name(str(entry.get("chunk_id") or "unknown"))
        (details_dir / f"{safe_chunk_id}.error.txt").write_text(error or "", encoding="utf-8")
        (details_dir / f"{safe_chunk_id}.raw_response.txt").write_text(raw_response or "", encoding="utf-8")
        (details_dir / f"{safe_chunk_id}.messages.json").write_text(
            json.dumps(messages or [], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self._failure_lock:
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    manifest = {"failed_chunks": []}
            else:
                manifest = {"failed_chunks": []}

            failed_chunks = manifest.setdefault("failed_chunks", [])
            # Replace an existing entry for the same chunk id instead of
            # duplicating it when retries/parallel workers are involved.
            failed_chunks = [item for item in failed_chunks if item.get("chunk_id") != entry.get("chunk_id")]
            failed_chunks.append(entry)
            manifest["failed_chunks"] = sorted(failed_chunks, key=lambda item: str(item.get("chunk_id") or ""))
            manifest["count"] = len(manifest["failed_chunks"])
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_failed_chunk_filter(self) -> set[str] | None:
        """Load a chunk-id filter for rerunning only failed Layer 1 units."""
        if not self.failed_chunks_file:
            return None
        path = Path(self.failed_chunks_file)
        if not path.exists():
            raise FileNotFoundError(f"Layer 1 failed chunks file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            chunk_ids = [str(item) for item in data]
        else:
            chunk_ids = []
            for item in data.get("failed_chunks", []):
                if isinstance(item, str):
                    chunk_ids.append(item)
                elif isinstance(item, dict) and item.get("chunk_id"):
                    chunk_ids.append(str(item["chunk_id"]))
        return set(chunk_ids)

    def _safe_artifact_name(self, value: str) -> str:
        """Return a filesystem-safe artifact stem."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"

    def _validate_alarm_record_with_optional_schema(
        self,
        *,
        record: dict[str, Any],
        state: PipelineState,
        chunk_id: str,
        messages: list[dict[str, str]],
        raw_response: str,
        structured_cfg: StructuredOutputConfig,
    ) -> dict[str, Any]:
        """Validate one Layer 1 record with optional Pydantic schema.

        Validation is profile-controlled.  In non-strict mode, validation
        failures are logged and the original record continues through the
        deterministic profile post-processors.
        """
        if not structured_cfg.enabled:
            return record

        wrapped = {"alarm_record": record}
        try:
            validated = validate_with_pydantic(wrapped, structured_cfg)
            if isinstance(validated, dict) and isinstance(validated.get("alarm_record"), dict):
                return validated["alarm_record"]
            return record
        except Exception as exc:
            error = f"Pydantic structured-output validation failed: {exc}"
            self._save_failed_response(
                state=state,
                chunk_id=f"{chunk_id}.schema_validation",
                raw_response=raw_response,
                messages=messages,
                error=error,
            )
            state.log({
                "layer": self.name,
                "status": "schema_validation_failed",
                "chunk_id": chunk_id,
                "schema_name": structured_cfg.schema_name,
                "strict_validation": structured_cfg.strict_validation,
                "error": str(exc),
            })
            if structured_cfg.strict_validation:
                raise
            return record


    def _make_linguistic_expression(
        self,
        *,
        expr_id: str,
        text: str,
        label: str,
        chunk,
    ) -> LinguisticExpression:
        """Build a LinguisticExpression from one record item."""
        match_span = self._find_expression_span(text, chunk.text)
        if match_span is not None:
            chunk_start_char, chunk_end_char = match_span
            doc_start_char = chunk.start_char + chunk_start_char
            doc_end_char = chunk.start_char + chunk_end_char
            snippet = self._build_snippet(chunk.text, chunk_start_char, chunk_end_char)
        else:
            chunk_start_char = chunk_end_char = doc_start_char = doc_end_char = -1
            snippet = chunk.text[:300]

        return LinguisticExpression(
            expr_id=expr_id,
            text=text,
            label=label,
            justification="Extracted from structured alarm/message table record.",
            evidence=[
                Evidence(
                    chunk_id=chunk.chunk_id,
                    chunk_start_char=chunk_start_char,
                    chunk_end_char=chunk_end_char,
                    doc_start_char=doc_start_char,
                    doc_end_char=doc_end_char,
                    snippet=snippet,
                )
            ],
        )

    def _retrieve_optional_rag_context(
        self,
        compact_unit: dict[str, Any],
        chunk,
        state: PipelineState,
    ) -> str:
        """Return a tiny optional RAG hint for Layer 1, or an empty string.

        Layer 1 should remain source-driven.  RAG is therefore disabled by
        default and, when enabled, capped by both top_k and max_chars.  This
        makes it a small guidance channel rather than a large context injection.
        """
        layer_cfg = (state.profile_config or {}).get("layers", {}).get(self.name, {}) or {}
        rag_cfg = (state.profile_config or {}).get("rag", {}) or {}
        enabled = self.rag_enabled
        if enabled is None:
            enabled = bool(layer_cfg.get("rag_enabled", False) or rag_cfg.get("enabled_for_layer01", False))
        if not enabled or self.rag_backend is None:
            return ""

        top_k = self.rag_top_k or int(layer_cfg.get("rag_top_k") or rag_cfg.get("top_k_layer01") or 0)
        max_chars = self.rag_max_chars or int(layer_cfg.get("rag_max_chars") or rag_cfg.get("max_chars_layer01") or 0)
        if top_k <= 0 or max_chars <= 0:
            return ""

        snippets = self._build_lightweight_layer01_rag_snippets(state)
        request = RAGRequest(
            query="Guidance for extracting one structured record from an industrial table.",
            layer_name=self.name,
            document_id=getattr(state.document, "doc_id", None),
            allowed_spaces=(rag_cfg.get("allowed_spaces", {}) or {}).get(self.name, []),
            top_k=top_k,
            metadata={
                "chunk_id": getattr(chunk, "chunk_id", None),
                "record_id_hint": compact_unit.get("record_id_hint"),
                "record_type_hint": compact_unit.get("record_type_hint"),
                "lightweight_profile_context": snippets,
            },
        )
        try:
            result = self.rag_backend.retrieve(request)
        except Exception as exc:
            state.log({
                "layer": self.name,
                "status": "rag_failed",
                "chunk_id": getattr(chunk, "chunk_id", None),
                "error": str(exc),
            })
            return ""

        context = (result.context or "").strip()
        if not context:
            return ""
        return context[:max_chars]

    def _build_lightweight_layer01_rag_snippets(self, state: PipelineState) -> list[str]:
        """Build tiny profile-derived snippets for the lightweight RAG stub.

        Layer 1 already receives field aliases through ``profile_guidance``.
        Repeating those aliases through RAG increases tokens and can make the
        prompt look more important than the source table.  The lightweight RAG
        context is therefore intentionally limited to short behavioral guidance.
        """
        return [
            (
                "Identifier rule: record_id_hint and record_type_hint are the source of truth. "
                "Mentions such as 'voir alarme n° 1083' inside a cause/intervention cell are cross-references, "
                "not the current record identifier."
            ),
            (
                "Reference rule: page/input/diagram mentions and cross-references to other alarms/messages "
                "should be concise reference_items, not duplicated as actions."
            ),
        ]

    def _layer_max_output_tokens(self, state: PipelineState) -> int | None:
        """Return per-layer output-token budget from the active profile."""
        layer_cfg = (state.profile_config or {}).get("layers", {}).get(self.name, {}) or {}
        value = layer_cfg.get("max_output_tokens")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _chat_with_optional_max_tokens(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None = None,
        fallback_to_json_parse: bool = True,
    ) -> str:
        """Call the backend with max_tokens when the backend supports it.

        OpenAI/LiteLLM-compatible backends accept ``max_tokens``.  Some old
        local backends do not, so we fall back cleanly to the legacy call.
        """
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
            kwargs["fallback_to_json_parse"] = fallback_to_json_parse
        try:
            return self.ollama_backend.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
        except TypeError:
            # Legacy backends may not accept max_tokens/response_format.  Retry
            # with the old minimal signature.
            return self.ollama_backend.chat(
                model=model,
                messages=messages,
                temperature=temperature,
            )

    def _clear_layer_runtime_artifacts(self, state: PipelineState) -> None:
        """Remove stale Layer 1 runtime artifacts before a new run.

        This clears old JSON/API errors and the current run's failed chunk
        manifest. Without this cleanup, stale failures from older experiments
        are easily mistaken for current failures.
        """
        if state.artifact_dir is None:
            return
        layer_dir = Path(state.artifact_dir) / self.name
        errors_dir = layer_dir / "json_errors"
        if errors_dir.exists():
            shutil.rmtree(errors_dir)
        errors_dir.mkdir(parents=True, exist_ok=True)

        failed_details_dir = layer_dir / "failed_chunk_details"
        if failed_details_dir.exists():
            shutil.rmtree(failed_details_dir)
        failed_details_dir.mkdir(parents=True, exist_ok=True)

        failed_manifest = layer_dir / "failed_chunks.json"
        if failed_manifest.exists():
            failed_manifest.unlink()

    # Backward-compatible alias used by older tests or notebooks.
    def _clear_json_errors(self, state: PipelineState) -> None:
        self._clear_layer_runtime_artifacts(state)

    def _build_prompt_unit(self, chunk, state: PipelineState) -> dict[str, Any]:
        """Build the compact structured unit sent to the LLM.

        The model should see table structure, but not pay for duplicated raw
        HTML, full rows, and fallback text unless strictly needed.  This method
        therefore sends the smallest useful representation:
        - unit identity and provenance,
        - field/value rows when available,
        - optional minimal profile guidance,
        - a deterministic alarm-number hint when recoverable.

        Raw HTML and full table rows stay in the saved state/debug artifacts.
        """
        metadata = getattr(chunk, "metadata", {}) or {}
        profile = state.profile_config or {}
        table_cfg = profile.get("table_extraction", {}) or {}
        compact_table = metadata.get("compact_table") or {}

        field_value_rows = compact_table.get("field_value_rows") or []
        identifier_hint = self._recover_record_identifier_from_chunk(chunk)

        guidance = {
            "profile_name": state.profile_name,
            "unit_type": metadata.get("chunk_type"),
            "table_layout_hint": table_cfg.get("layout_hint", "unknown"),
            # Keep aliases because they are profile-level, not core-level, but
            # avoid sending unrelated ontology/KG content here.
            "field_aliases": table_cfg.get("field_aliases", {}),
            "output_language": profile.get("language", {}).get("target", "en"),
            "llm_translate_inside_extraction": profile.get("language", {}).get("llm_translate_inside_extraction", True),
        }

        current_header_text = self._current_record_header_text(chunk, field_value_rows)
        unit: dict[str, Any] = {
            "unit_id": chunk.chunk_id,
            "unit_type": metadata.get("chunk_type"),
            "page": metadata.get("page"),
            "title": metadata.get("title"),
            "section_key": metadata.get("section_key"),
            "subsection_key": metadata.get("subsection_key"),
            "current_header_text": current_header_text,
            "record_identity_source": {
                "current_header_text": current_header_text,
                "record_type_hint": identifier_hint.get("record_type"),
                "record_id_hint": identifier_hint.get("record_id"),
                "alarm_no_hint": identifier_hint.get("alarm_no"),
                "message_no_hint": identifier_hint.get("message_no"),
            },
            "record_type_hint": identifier_hint.get("record_type"),
            "record_id_hint": identifier_hint.get("record_id"),
            "alarm_no_hint": identifier_hint.get("alarm_no"),
            "message_no_hint": identifier_hint.get("message_no"),
            "table_shape": compact_table.get("shape"),
            "field_value_rows": field_value_rows,
            "profile_guidance": guidance,
        }

        # Only keep a compact fallback when the structural extraction is weak.
        # In normal Machine32 table chunks, field_value_rows is enough and saves
        # a lot of tokens.
        if not field_value_rows:
            unit["fallback_text"] = str(getattr(chunk, "text", "") or "")[:1200]

        return unit

    def _current_record_header_text(self, chunk, field_value_rows: list[dict[str, Any]]) -> str | None:
        """Return a short header identifying the current structural record.

        This is not XQuality-specific.  It simply prefers the first row/header
        and falls back to text previews.  The profile identity policy interprets
        the header patterns.
        """
        metadata = getattr(chunk, "metadata", {}) or {}
        compact_table = metadata.get("compact_table") or {}
        rows = compact_table.get("rows") or []
        if rows:
            first_row = rows[0]
            texts = []
            for cell in first_row:
                if isinstance(cell, dict) and cell.get("text"):
                    texts.append(str(cell.get("text")))
            if texts:
                # Add the recovered number if the table extraction stored it in
                # preview text but not in the first compact row.
                preview = str(metadata.get("html_text") or metadata.get("html_text_preview") or getattr(chunk, "text", "") or "")
                first_line = preview.splitlines()[0] if preview else ""
                number = re.search(r"\d{2,8}", first_line or preview[:80])
                suffix = f" {number.group(0)}" if number and not re.search(r"\d", " ".join(texts)) else ""
                return " ".join(texts).strip() + suffix

        preview = str(metadata.get("html_text") or metadata.get("html_text_preview") or getattr(chunk, "text", "") or "")
        return preview[:120].strip() or None


    def _compact_metadata_for_prompt(self, chunk) -> dict[str, Any]:
        """Return prompt-safe metadata without verbose raw HTML duplication."""
        metadata = getattr(chunk, "metadata", {}) or {}
        return {
            "chunk_type": metadata.get("chunk_type"),
            "page": metadata.get("page"),
            "title": metadata.get("title"),
            "section_key": metadata.get("section_key"),
            "subsection_key": metadata.get("subsection_key"),
            "table_index": metadata.get("table_index"),
            "compact_table": metadata.get("compact_table"),
        }

    def _compact_metadata_for_state(self, chunk) -> dict[str, Any]:
        """Keep provenance in state while avoiding repeated huge HTML in records."""
        metadata = getattr(chunk, "metadata", {}) or {}
        return {
            "chunk_type": metadata.get("chunk_type"),
            "page": metadata.get("page"),
            "title": metadata.get("title"),
            "document_key": metadata.get("document_key"),
            "section_key": metadata.get("section_key"),
            "subsection_key": metadata.get("subsection_key"),
            "table_index": metadata.get("table_index"),
            "compact_table": metadata.get("compact_table"),
            "html_text_preview": str(metadata.get("html_text") or "")[:1000],
        }

    def _extract_alarm_records_from_parsed(self, parsed: Any) -> list[dict[str, Any]]:
        """Normalize model JSON into a list of alarm-record dictionaries.

        Some models return {"alarm_record": {...}}, others return {...}, and
        others return [{"alarm_record": {...}}].  During ablation we accept all
        three forms so one valid response shape does not become a false failure.
        """
        records: list[dict[str, Any]] = []

        if isinstance(parsed, list):
            for item in parsed:
                records.extend(self._extract_alarm_records_from_parsed(item))
            return records

        if not isinstance(parsed, dict):
            return records

        if isinstance(parsed.get("alarm_record"), dict):
            records.append(parsed["alarm_record"])
            return records

        if isinstance(parsed.get("alarm_records"), list):
            for item in parsed["alarm_records"]:
                records.extend(self._extract_alarm_records_from_parsed(item))
            return records

        # Fallback: treat a dict that already looks like a record as the record.
        if any(key in parsed for key in ("alarm_no", "alarm_label_en", "alarm_label_fr", "cause_items")):
            records.append(parsed)

        return records

    def _postprocess_alarm_record(
        self,
        record: dict[str, Any],
        chunk,
        state: PipelineState,
        compact_unit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply deterministic, profile-aware cleanup after LLM extraction.

        The LLM is useful for semantic splitting and translation, but some
        values should not depend on the model.  In particular, ``alarm_no`` is
        an evaluation anchor and is usually recoverable from the chunk text or
        table provenance.  Responsible actors are also often compact lists like
        ``opérateur/chargé de l’entretien`` and can be split deterministically.
        """
        metadata = getattr(chunk, "metadata", {}) or {}
        profile = state.profile_config or {}

        # 1) Deterministic record identifier recovery.  Do not rely on the LLM
        # for identifiers used as evaluation anchors.
        record = self._recover_record_identifiers(record, chunk)

        # 1b) Profile-driven identity policy. This is the generic hook that lets
        # each dataset decide whether unit hints override the LLM identity.
        # It prevents cross-references inside body text from replacing the
        # current record id, without hard-coding XQuality-specific logic here.
        record = apply_record_identity_policy(
            record=record,
            unit=compact_unit or self._build_prompt_unit(chunk, state),
            profile_config=state.profile_config,
        )

        # 2) Stable provenance fields.
        record.setdefault("chunk_id", chunk.chunk_id)
        record.setdefault("page", metadata.get("page"))
        record.setdefault("source_metadata", self._compact_metadata_for_state(chunk))

        # 3) Ensure item-list fields are lists of dicts.
        for field_name in [
            "cause_items",
            "effect_items",
            "intervention_items",
            "responsible_items",
            "reference_items",
        ]:
            record[field_name] = self._normalize_item_list(record.get(field_name), field_name)

        # 4) Profile-aware responsible splitting/canonicalization.
        record["responsible_items"] = self._split_and_canonicalize_responsibles(
            record.get("responsible_items") or [], profile
        )

        return record

    def _recover_alarm_no(self, chunk) -> str | None:
        """Backward-compatible helper returning only alarm_no."""
        return self._recover_record_identifier_from_chunk(chunk).get("alarm_no")

    def _recover_record_identifiers(self, record: dict[str, Any], chunk) -> dict[str, Any]:
        """Recover record_id, record_type, alarm_no and message_no.

        The chunk-level hints are the trusted source for the *current* record.
        The LLM may see cross-references such as ``voir alarme n° 1083`` inside
        a cause/intervention cell and accidentally replace the current message
        id with that referenced alarm id.  This method prevents that: when a
        chunk hint exists, it overrides any conflicting model identifier.
        """
        hints = self._recover_record_identifier_from_chunk(chunk)
        hint_type = hints.get("record_type")
        hint_record_id = hints.get("record_id")
        hint_alarm_no = hints.get("alarm_no")
        hint_message_no = hints.get("message_no")

        # Highest-priority path: the preprocessor/chunk metadata identified the
        # current record.  Trust it over the LLM output.
        if hint_record_id:
            if hint_type == "message":
                record["record_type"] = "message"
                record["record_id"] = hint_record_id
                record["message_no"] = hint_message_no or hint_record_id
                record["alarm_no"] = None
                return record
            if hint_type == "alarm":
                record["record_type"] = "alarm"
                record["record_id"] = hint_record_id
                record["alarm_no"] = hint_alarm_no or hint_record_id
                record["message_no"] = None
                return record

            # Generic future-proof fallback when a profile has an identifier but
            # not a known alarm/message type.
            record["record_type"] = record.get("record_type") or hint_type or "unknown"
            record["record_id"] = hint_record_id
            record["alarm_no"] = hint_alarm_no
            record["message_no"] = hint_message_no
            return record

        # No reliable chunk hint. Fall back to the model output, while keeping
        # the generic fields consistent.
        record_type = record.get("record_type") or "unknown"
        alarm_no = record.get("alarm_no")
        message_no = record.get("message_no")
        record_id = record.get("record_id") or alarm_no or message_no

        if record_type in (None, "", "unknown"):
            if message_no:
                record_type = "message"
            elif alarm_no:
                record_type = "alarm"
            else:
                record_type = "unknown"

        if record_type == "message":
            record["record_type"] = "message"
            record["record_id"] = record_id or message_no
            record["message_no"] = message_no or record_id
            record["alarm_no"] = None
        elif record_type == "alarm":
            record["record_type"] = "alarm"
            record["record_id"] = record_id or alarm_no
            record["alarm_no"] = alarm_no or record_id
            record["message_no"] = None
        else:
            record["record_type"] = "unknown"
            record["record_id"] = record_id
            record["alarm_no"] = alarm_no
            record["message_no"] = message_no
        return record

    def _recover_record_identifier_from_chunk(self, chunk) -> dict[str, str | None]:
        """Recover record identity from structured rows, metadata or text.

        Returns a dict with: record_type, record_id, alarm_no, message_no.
        Works for documents with either alarm-like or message-like records.
        """
        metadata = getattr(chunk, "metadata", {}) or {}
        compact_table = metadata.get("compact_table") or {}

        def result(record_type: str | None, number: str | None) -> dict[str, str | None]:
            if not number:
                return {
                    "record_type": record_type or None,
                    "record_id": None,
                    "alarm_no": None,
                    "message_no": None,
                }
            if record_type == "message":
                return {
                    "record_type": "message",
                    "record_id": number,
                    "alarm_no": None,
                    "message_no": number,
                }
            return {
                "record_type": "alarm" if record_type == "alarm" else record_type,
                "record_id": number,
                "alarm_no": number if record_type == "alarm" else None,
                "message_no": number if record_type == "message" else None,
            }

        # Direct metadata, if a future chunker provides it.
        for key in ("record_id", "alarm_no", "alarm_number", "alarm_id", "message_no", "message_number"):
            value = metadata.get(key) or compact_table.get(key)
            if value is not None and str(value).strip():
                m = re.search(r"\d+", str(value))
                if m:
                    if "message" in key:
                        return result("message", m.group(0))
                    if "alarm" in key:
                        return result("alarm", m.group(0))
                    return result(str(metadata.get("record_type") or "unknown"), m.group(0))

        # Current header and source preview have priority over body rows.
        # Body rows may contain cross-references such as "voir alarme n° 1083";
        # these must not become the current record identity when the table starts
        # with "message n°: 2060".
        current_header = self._current_record_header_text(chunk, compact_table.get("field_value_rows") or [])
        priority_texts = [
            str(current_header or ""),
            str(metadata.get("title") or ""),
            str(metadata.get("html_text") or "")[:300],
            str(getattr(chunk, "text", "") or "")[:300],
        ]
        priority_combined = "\n".join(t for t in priority_texts if t)
        found = self._extract_record_identifier_from_text(priority_combined)
        if found.get("record_id"):
            return found

        # Fallback text and previews. This fixes rows where the compact header
        # was kept but the id value was lost.
        texts = [
            str(getattr(chunk, "text", "") or ""),
            str(metadata.get("html_text") or ""),
            str(metadata.get("html_text_preview") or ""),
            str(metadata.get("title") or ""),
            str(metadata.get("subsection_key") or ""),
            str(getattr(chunk, "chunk_id", "") or ""),
        ]
        combined = "\n".join(t for t in texts if t)
        found = self._extract_record_identifier_from_text(combined)
        if found.get("record_id"):
            return found

        # Last resort: scan field/value rows. This is deliberately after the
        # current header/source preview to avoid cross-reference overrides.
        for row in compact_table.get("field_value_rows") or []:
            field = str(row.get("field") or "")
            values = row.get("values") or []
            joined = " ".join(str(v) for v in values if v is not None)
            haystack = f"{field} {joined}"
            found = self._extract_record_identifier_from_text(haystack)
            if found.get("record_id"):
                return found

        return result(None, None)

    def _extract_record_identifier_from_text(self, text: str) -> dict[str, str | None]:
        """Extract alarm/message identifiers from text."""
        if not text:
            return {"record_type": None, "record_id": None, "alarm_no": None, "message_no": None}

        message_patterns = [
            r"(?i)\bmessage\s*n\s*[°o.]?\s*[:#-]?\s*(\d{3,6})\b",
            r"(?i)\bmessage\s*(?:no\.?|number|#)?\s*[:#-]?\s*(\d{3,6})\b",
        ]
        alarm_patterns = [
            r"(?i)\balarme\s*n\s*[°o.]?\s*[:#-]?\s*(\d{3,6})\b",
            r"(?i)\balarm\s*(?:no\.?|number|#)?\s*[:#-]?\s*(\d{3,6})\b",
            r"(?i)\bn[°o.]\s*alarme\s*[:#-]?\s*(\d{3,6})\b",
        ]
        for pattern in message_patterns:
            m = re.search(pattern, text)
            if m:
                number = m.group(1)
                return {"record_type": "message", "record_id": number, "alarm_no": None, "message_no": number}
        for pattern in alarm_patterns:
            m = re.search(pattern, text)
            if m:
                number = m.group(1)
                return {"record_type": "alarm", "record_id": number, "alarm_no": number, "message_no": None}
        return {"record_type": None, "record_id": None, "alarm_no": None, "message_no": None}

    def _normalize_item_list(self, value: Any, field_name: str) -> list[dict[str, Any]]:
        """Normalize item-list fields to a list of dictionaries."""
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if not isinstance(value, list):
            return [{"text_en": str(value), "text_fr": None, "evidence_field": field_name}]

        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(item)
            elif item is not None:
                normalized.append({"text_en": str(item), "text_fr": None, "evidence_field": field_name})
        return normalized

    def _split_and_canonicalize_responsibles(
        self,
        responsible_items: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Split compact responsible values and apply profile mappings.

        This remains profile-driven.  The core only knows how to split list-like
        values; the actual canonical labels live in the document profile.
        """
        mappings = (
            profile.get("canonical_value_mappings", {})
            .get("responsible", {})
        )
        # Also accept the shorter name for future profiles.
        mappings.update(profile.get("responsible_mappings", {}) or {})

        output: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for item in responsible_items:
            text_fr = str(item.get("text_fr") or "").strip()
            text_en = str(item.get("text_en") or "").strip()
            raw = text_fr or text_en
            if not raw:
                continue

            # Split common compact-list separators without splitting technical
            # identifiers.  Responsible cells are usually short actor lists.
            parts = [p.strip(" .;,") for p in re.split(r"\s*/\s*|\s*;\s*|\s+et\s+|\s+and\s+", raw) if p.strip(" .;, ")]
            if not parts:
                parts = [raw]

            for part in parts:
                canonical_en = self._canonicalize_value(part, mappings) or part
                # If the model already gave English and no mapping exists, keep it.
                if part == raw and text_en and not self._canonicalize_value(part, mappings):
                    canonical_en = text_en
                key = (canonical_en.lower(), part.lower())
                if key in seen:
                    continue
                seen.add(key)
                output.append({
                    "text_en": canonical_en,
                    "text_fr": part if text_fr else None,
                    "evidence_field": item.get("evidence_field") or "responsible",
                })

        return output

    def _canonicalize_value(self, value: str, mappings: dict[str, str]) -> str | None:
        """Return a profile canonical value if one matches normalized text."""
        if not value:
            return None
        norm = self._norm_key(value)
        for source, target in mappings.items():
            if self._norm_key(source) == norm:
                return target
        return None

    def _norm_key(self, value: str) -> str:
        """Small normalization helper for profile mappings."""
        value = value.lower().strip()
        value = value.replace("’", "'")
        value = re.sub(r"\s+", " ", value)
        value = value.strip(" .:;,")
        return value

    def _record_expression_items(self, record: dict[str, Any]) -> list[tuple[str, str]]:
        """Return (text, label) pairs from one structured alarm record."""
        items: list[tuple[str, str]] = []
        alarm_label = str(record.get("alarm_label_en") or record.get("alarm_label_fr") or "").strip()
        if alarm_label:
            items.append((alarm_label, str(record.get("record_type") or "record")))
        for field_name, label in [
            ("cause_items", "cause"),
            ("effect_items", "effect"),
            ("intervention_items", "intervention"),
            ("responsible_items", "responsible"),
            ("reference_items", "reference"),
        ]:
            for item in record.get(field_name, []) or []:
                if isinstance(item, dict):
                    text = str(item.get("text_en") or item.get("text_fr") or "").strip()
                else:
                    text = str(item).strip()
                if text:
                    items.append((text, label))
        return items


    def _safe_extract_json(
        self,
        *,
        raw_response: str | None,
        state: PipelineState,
        chunk_id: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        """Parse a layer-1 LLM response without breaking the whole run.

        Empty responses are common when a provider silently fails or when the
        model does not respect JSON output constraints. During ablation, it is
        more useful to save the failed response and continue than to lose the
        whole run.
        """
        try:
            parsed = self.ollama_backend.extract_json(raw_response or "")
            if isinstance(parsed, list):
                # Valid for profile-specific extractors where a model may return
                # a list of records instead of one object. Generic expression
                # extraction will simply see no "expressions" key and skip it.
                return parsed
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected a JSON object or list, got {type(parsed).__name__}.")
            if "expressions" not in parsed:
                parsed["expressions"] = []
            return parsed
        except Exception as exc:
            self._save_failed_response(
                state=state,
                chunk_id=chunk_id,
                raw_response=raw_response or "",
                messages=messages,
                error=str(exc),
            )
            state.log(
                {
                    "layer": self.name,
                    "status": "json_parse_failed",
                    "chunk_id": chunk_id,
                    "error": str(exc),
                    "raw_response_chars": len(raw_response or ""),
                }
            )
            if self.verbose:
                print(
                    f"[NeoOLAF][{self.name}] JSON parse failed for chunk {chunk_id}; "
                    "saved raw response and continued."
                )
            return None

    def _save_failed_response(
        self,
        *,
        state: PipelineState,
        chunk_id: str,
        raw_response: str,
        messages: list[dict[str, str]],
        error: str,
    ) -> None:
        """Save malformed LLM outputs for inspection."""
        if state.artifact_dir is None:
            return

        errors_dir = Path(state.artifact_dir) / self.name / "json_errors"
        errors_dir.mkdir(parents=True, exist_ok=True)
        safe_chunk_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(chunk_id))
        base = errors_dir / f"{safe_chunk_id}"

        (base.with_suffix(".raw_response.txt")).write_text(raw_response, encoding="utf-8")
        (base.with_suffix(".error.txt")).write_text(error, encoding="utf-8")
        (base.with_suffix(".messages.json")).write_text(
            json.dumps(messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _find_expression_span(
        self,
        expression_text: str,
        chunk_text: str,
    ) -> Optional[Tuple[int, int]]:
        """
        Try to locate an extracted expression inside the chunk text.

        Strategy:
        1. exact case-insensitive match
        2. whitespace-normalized match
        3. regex-escaped case-insensitive search

        Args:
            expression_text:
                Expression returned by the LLM.
            chunk_text:
                Original chunk text.

        Returns:
            (start, end) span in chunk coordinates if found, else None.
        """
        if not expression_text:
            return None

        # Exact case-insensitive search
        lower_chunk = chunk_text.lower()
        lower_expr = expression_text.lower()

        idx = lower_chunk.find(lower_expr)
        if idx != -1:
            return idx, idx + len(expression_text)

        # Regex-safe search
        pattern = re.escape(expression_text.strip())
        match = re.search(pattern, chunk_text, flags=re.IGNORECASE)
        if match:
            return match.start(), match.end()

        # Whitespace-normalized fallback
        normalized_expr = re.sub(r"\s+", " ", expression_text.strip().lower())
        normalized_chunk = re.sub(r"\s+", " ", chunk_text.lower())

        idx_norm = normalized_chunk.find(normalized_expr)
        if idx_norm != -1:
            # This fallback gives only approximate recovery in normalized text space.
            # Because mapping back exactly is non-trivial, we return None here rather
            # than a potentially wrong span.
            return None

        return None

    def _build_snippet(
        self,
        chunk_text: str,
        start: int,
        end: int,
        window: int = 80,
    ) -> str:
        """
        Build a local snippet around a matched expression.

        Args:
            chunk_text:
                Full chunk text.
            start:
                Start character in chunk coordinates.
            end:
                End character in chunk coordinates.
            window:
                Number of context characters kept around the expression.

        Returns:
            Local textual snippet.
        """
        left = max(0, start - window)
        right = min(len(chunk_text), end + window)
        return chunk_text[left:right].strip()

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize the extracted expressions with positional evidence.
        """
        return {
            "layer": self.name,
            "profile_name": state.profile_name,
            "num_alarm_records": len(getattr(state.document, "alarm_records", []) or []),
            "alarm_records": getattr(state.document, "alarm_records", []) or [],
            "num_expressions": len(state.linguistic_expressions),
            "expressions": [
                {
                    "expr_id": expr.expr_id,
                    "text": expr.text,
                    "label": expr.label,
                    "justification": expr.justification,
                    "evidence": [
                        {
                            "chunk_id": ev.chunk_id,
                            "chunk_start_char": ev.chunk_start_char,
                            "chunk_end_char": ev.chunk_end_char,
                            "doc_start_char": ev.doc_start_char,
                            "doc_end_char": ev.doc_end_char,
                            "snippet": ev.snippet,
                        }
                        for ev in expr.evidence
                    ],
                }
                for expr in state.linguistic_expressions
            ],
        }
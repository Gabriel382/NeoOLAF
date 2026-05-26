from __future__ import annotations

# Standard library imports
import json
import re
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

from neoolaf.core.base_layer import BaseLayer
from neoolaf.config.prompt_loader import load_prompt_template, render_prompt_template
from neoolaf.core.pipeline_state import PipelineState
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
        self._clear_json_errors(state)

        chunks = state.document.chunks
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]

        records: list[dict[str, Any]] = []
        expressions: list[LinguisticExpression] = []
        expr_counter = 0

        chunk_iterator = chunks
        if self.verbose:
            chunk_iterator = tqdm(chunks, desc="Layer 1 - alarm tables", leave=False)

        for chunk in chunk_iterator:
            # Skip non-table chunks when the profile selected table-aware extraction.
            if getattr(chunk, "metadata", {}).get("chunk_type") not in {"table", None}:
                continue

            compact_unit = self._build_prompt_unit(chunk, state)
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
                # Backward-compatible variables for older prompt templates.
                chunk_metadata=json.dumps(self._compact_metadata_for_prompt(chunk), ensure_ascii=False, indent=2),
                chunk_text=chunk.text,
            )
            messages = [
                {"role": "system", "content": "You are a strict JSON extractor for industrial alarm tables."},
                {"role": "user", "content": user_prompt},
            ]

            max_output_tokens = self._layer_max_output_tokens(state)
            raw_response = self._chat_with_optional_max_tokens(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_output_tokens,
            )
            parsed = self._safe_extract_json(
                raw_response=raw_response,
                state=state,
                chunk_id=chunk.chunk_id,
                messages=messages,
            )
            if parsed is None:
                continue

            chunk_records = self._extract_alarm_records_from_parsed(parsed)
            if not chunk_records:
                self._save_failed_response(
                    state=state,
                    chunk_id=chunk.chunk_id,
                    raw_response=raw_response or "",
                    messages=messages,
                    error="Parsed JSON did not contain an alarm_record object or list of records.",
                )
                continue

            for record in chunk_records:
                record = self._postprocess_alarm_record(record, chunk, state)
                records.append(record)

                # Also expose record items as Layer-1 linguistic expressions so the
                # generic downstream/evaluation machinery still sees Layer-1 labels.
                for text, label in self._record_expression_items(record):
                    if not text:
                        continue
                    match_span = self._find_expression_span(text, chunk.text)
                    if match_span is not None:
                        chunk_start_char, chunk_end_char = match_span
                        doc_start_char = chunk.start_char + chunk_start_char
                        doc_end_char = chunk.start_char + chunk_end_char
                        snippet = self._build_snippet(chunk.text, chunk_start_char, chunk_end_char)
                    else:
                        chunk_start_char = chunk_end_char = doc_start_char = doc_end_char = -1
                        snippet = chunk.text[:300]

                    expressions.append(
                        LinguisticExpression(
                            expr_id=f"expr_{expr_counter:05d}",
                            text=text,
                            label=label,
                            justification="Extracted from structured alarm table record.",
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
    ) -> str:
        """Call the backend with max_tokens when the backend supports it.

        OpenAI/LiteLLM-compatible backends accept ``max_tokens``.  Some old
        local backends do not, so we fall back cleanly to the legacy call.
        """
        if max_tokens is None:
            return self.ollama_backend.chat(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        try:
            return self.ollama_backend.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except TypeError:
            return self.ollama_backend.chat(
                model=model,
                messages=messages,
                temperature=temperature,
            )

    def _clear_json_errors(self, state: PipelineState) -> None:
        """Remove stale JSON error files for this layer before a new run."""
        if state.artifact_dir is None:
            return
        errors_dir = Path(state.artifact_dir) / self.name / "json_errors"
        if errors_dir.exists():
            shutil.rmtree(errors_dir)
        errors_dir.mkdir(parents=True, exist_ok=True)

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

        unit: dict[str, Any] = {
            "unit_id": chunk.chunk_id,
            "unit_type": metadata.get("chunk_type"),
            "page": metadata.get("page"),
            "title": metadata.get("title"),
            "section_key": metadata.get("section_key"),
            "subsection_key": metadata.get("subsection_key"),
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

        The LLM can translate and segment text, but identifiers must be
        deterministic because they anchor the evaluation.  This method supports
        both alarm and message tables without making Layer 1 Machine32-only.
        """
        hints = self._recover_record_identifier_from_chunk(chunk)

        record_type = record.get("record_type") or hints.get("record_type") or "unknown"
        record_id = record.get("record_id") or hints.get("record_id")
        alarm_no = record.get("alarm_no") or hints.get("alarm_no")
        message_no = record.get("message_no") or hints.get("message_no")

        # If the model only filled alarm_no/message_no, infer the generic fields.
        if not record_id:
            record_id = alarm_no or message_no
        if record_type in (None, "", "unknown"):
            if alarm_no:
                record_type = "alarm"
            elif message_no:
                record_type = "message"
            else:
                record_type = "unknown"

        record["record_type"] = record_type
        record["record_id"] = record_id
        record["alarm_no"] = alarm_no if record_type == "alarm" or alarm_no else record.get("alarm_no")
        record["message_no"] = message_no if record_type == "message" or message_no else record.get("message_no")
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

        # Field/value rows: robust to table schemas that explicitly keep the id.
        for row in compact_table.get("field_value_rows") or []:
            field = str(row.get("field") or "")
            values = row.get("values") or []
            joined = " ".join(str(v) for v in values if v is not None)
            haystack = f"{field} {joined}"
            found = self._extract_record_identifier_from_text(haystack)
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
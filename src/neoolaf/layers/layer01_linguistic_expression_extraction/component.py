from __future__ import annotations

# Standard library imports
import json
import re
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
            if parsed is None:
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

            user_prompt = render_prompt_template(
                template,
                chunk_metadata=json.dumps(getattr(chunk, "metadata", {}), ensure_ascii=False, indent=2),
                chunk_text=chunk.text,
            )
            messages = [
                {"role": "system", "content": "You are a strict JSON extractor for industrial alarm tables."},
                {"role": "user", "content": user_prompt},
            ]

            raw_response = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self._safe_extract_json(
                raw_response=raw_response,
                state=state,
                chunk_id=chunk.chunk_id,
                messages=messages,
            )
            if parsed is None:
                continue

            record = parsed.get("alarm_record", parsed)
            if not isinstance(record, dict):
                continue
            record.setdefault("chunk_id", chunk.chunk_id)
            record.setdefault("page", getattr(chunk, "metadata", {}).get("page"))
            record.setdefault("source_metadata", getattr(chunk, "metadata", {}))
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

    def _record_expression_items(self, record: dict[str, Any]) -> list[tuple[str, str]]:
        """Return (text, label) pairs from one structured alarm record."""
        items: list[tuple[str, str]] = []
        alarm_label = str(record.get("alarm_label_en") or record.get("alarm_label_fr") or "").strip()
        if alarm_label:
            items.append((alarm_label, "alarm"))
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
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
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
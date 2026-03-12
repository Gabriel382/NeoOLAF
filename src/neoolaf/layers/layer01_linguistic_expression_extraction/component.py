from __future__ import annotations

# Standard library imports
import re
from typing import List, Optional, Tuple

from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.linguistic_expression import LinguisticExpression, Evidence
from neoolaf.layers.layer01_linguistic_expression_extraction.prompt import (
    build_system_prompt,
    build_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend


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
        """
        super().__init__(save_intermediate=save_intermediate)
        self.ollama_backend = ollama_backend
        self.max_chunks = max_chunks
        self.temperature = temperature

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Run expression extraction over document chunks.
        """
        expressions: List[LinguisticExpression] = []
        chunks = state.document.chunks

        # Optional chunk limit for faster debugging
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]

        expr_counter = 0

        for chunk in chunks:
            messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(chunk, state.user_guidance)},
            ]

            raw_response = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )

            parsed = self.ollama_backend.extract_json(raw_response)
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
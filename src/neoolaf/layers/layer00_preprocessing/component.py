from __future__ import annotations

from typing import Optional

from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.preprocessing.chunking import chunk_text
from neoolaf.preprocessing.cleaners import clean_plain_text
from neoolaf.preprocessing.pdf_parsing import (
    extract_content_blocks,
    extract_pdf,
    extract_tables_for_export,
    flatten_content_blocks,
)
from neoolaf.preprocessing.structured_chunking import build_structured_chunks
from neoolaf.resources.ocr.base_engine import BaseOCREngine
from neoolaf.resources.translation.base_backend import BaseTranslationBackend


class PreprocessingLayer(BaseLayer):
    """
    Layer 00: preprocessing.

    Responsibilities:
    - extract raw content from PDF sources (text-based or OCR-based)
    - clean and normalize extracted plain text
    - optionally translate content blocks using a pluggable backend
    - create fixed, page-level, subsection-level, and table-level chunks
    - select the active downstream chunk view from the document profile
    """

    name = "layer00_preprocessing"

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 200,
        enable_chunking: bool = True,
        translate: bool = False,
        translator: Optional[BaseTranslationBackend] = None,
        source_language: str | None = None,
        target_language: str = "en",
        ocr_engine: Optional[BaseOCREngine] = None,
        ocr_dpi: int = 300,
        save_intermediate: bool = True,
        verbose: bool = False,
        profile_config: dict | None = None,
    ) -> None:
        """Initialize Layer 00."""
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_chunking = enable_chunking
        self.translate = translate
        self.source_language = source_language
        self.target_language = target_language
        self.ocr_engine = ocr_engine
        self.ocr_dpi = ocr_dpi
        self._translator = translator
        self.profile_config = profile_config or {}

        if self.translate and self._translator is None:
            raise ValueError(
                "translate=True requires a translator backend. "
                "Pass translator=DeepTranslatorBackend() or "
                "translator=NLLB200TranslatorBackend()."
            )

    def _get_translator(self) -> BaseTranslationBackend:
        """Return the translation backend provided at init."""
        return self._translator

    def _active_profile(self, state: PipelineState) -> dict:
        return state.profile_config or self.profile_config or {}

    def _run(self, state: PipelineState) -> PipelineState:
        """Extract, clean, translate, and chunk document content."""
        source = state.document.source_path
        profile = self._active_profile(state)

        # ---------------------------------------------------------
        # 1. PDF extraction
        # ---------------------------------------------------------
        if source and source.lower().endswith(".pdf"):
            result = extract_pdf(source, ocr_engine=self.ocr_engine, dpi=self.ocr_dpi)
            state.document.pdf_type = result["pdf_type"]
            state.document.extraction_result = result["content"]
            state.log(f"[{self.name}] PDF detected as {result['pdf_type']}")

            ordered_blocks = extract_content_blocks(result["pdf_type"], result["content"])
            state.document.content_blocks = ordered_blocks
            state.document.raw_text = flatten_content_blocks(ordered_blocks)

        # ---------------------------------------------------------
        # 2. Cleaning
        # ---------------------------------------------------------
        cleaned = clean_plain_text(state.document.raw_text)
        state.document.cleaned_text = cleaned
        text_for_chunking = cleaned

        # ---------------------------------------------------------
        # 3. Translation
        # ---------------------------------------------------------
        if self.translate:
            translator = self._get_translator()
            source_lang = self.source_language
            if source_lang is None:
                source_lang = self._detect_language(cleaned)
            state.document.content_blocks = self._translate_content_blocks(
                state.document.content_blocks, translator, source_lang
            )
            translated = flatten_content_blocks(
                state.document.content_blocks, use_translated_text=True
            )
            state.document.translated_text = translated
            text_for_chunking = translated
            state.log(
                f"[{self.name}] translation applied "
                f"({self.source_language or 'auto'} -> {self.target_language}) "
                f"using {translator.name}"
            )

        # ---------------------------------------------------------
        # 4. Fixed + structured chunking
        # ---------------------------------------------------------
        fixed_chunks = []
        if self.enable_chunking:
            fixed_chunks = chunk_text(text_for_chunking, chunk_size=self.chunk_size, overlap=self.overlap)

        prefer_translated = bool(self.translate and getattr(state.document, "translated_text", None))
        structured = build_structured_chunks(
            state.document.content_blocks or [],
            profile_config=profile,
            prefer_translated=prefer_translated,
        )
        state.document.page_chunks = structured.get("page", [])
        state.document.subsection_chunks = structured.get("subsection", [])
        state.document.table_chunks = structured.get("table", [])
        state.document.structured_units = [
            {
                "unit_type": key,
                "num_chunks": len(value),
                "chunk_ids": [chunk.chunk_id for chunk in value],
            }
            for key, value in structured.items()
        ]

        preferred_unit = profile.get("chunking", {}).get("preferred_unit_for_extraction", "chunk")
        if preferred_unit == "page" and state.document.page_chunks:
            state.document.chunks = state.document.page_chunks
        elif preferred_unit == "subsection" and state.document.subsection_chunks:
            state.document.chunks = state.document.subsection_chunks
        elif preferred_unit == "table" and state.document.table_chunks:
            state.document.chunks = state.document.table_chunks
        elif self.enable_chunking:
            state.document.chunks = fixed_chunks
        else:
            state.document.chunks = []

        state.log(
            f"[{self.name}] produced active={len(state.document.chunks)} chunks "
            f"(preferred_unit={preferred_unit}; fixed={len(fixed_chunks)}, "
            f"page={len(state.document.page_chunks)}, "
            f"subsection={len(state.document.subsection_chunks)}, "
            f"table={len(state.document.table_chunks)})"
        )

        return state

    @staticmethod
    def _detect_language(text: str) -> str | None:
        """Auto-detect source language using langdetect."""
        try:
            from langdetect import detect
            return detect(text[:2000])
        except Exception:
            return None

    def _translate_content_blocks(self, content_blocks, translator, source_language):
        """Translate each content block using the provided translator backend."""
        translated_blocks = []
        for block in content_blocks:
            updated = dict(block)
            source_text = (
                block.get("text", "") if block.get("type") == "text"
                else block.get("html_text", "")
            )
            if source_text:
                updated["translated_text"] = translator.translate(
                    source_text,
                    source_language=source_language,
                    target_language=self.target_language,
                )
            else:
                updated["translated_text"] = ""
            if block.get("type") == "table":
                updated["translated_html"] = None
            translated_blocks.append(updated)
        return translated_blocks

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """Serialize Layer 00 outputs for debugging and reproducibility."""
        payload = {
            "layer": self.name,
            "doc_id": state.document.doc_id,
            "source_path": state.document.source_path,
            "profile_name": state.profile_name,
            "pdf_type": state.document.pdf_type,
            "content_blocks": state.document.content_blocks,
            "tables": extract_tables_for_export(state.document.extraction_result),
        }
        if self.enable_chunking:
            payload["num_chunks"] = len(state.document.chunks)
            payload["chunks"] = [
                {
                    "chunk_id": c.chunk_id,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "metadata": getattr(c, "metadata", {}),
                    "text_preview": c.text[:300],
                }
                for c in state.document.chunks
            ]
        payload["structured_chunk_counts"] = {
            "page": len(getattr(state.document, "page_chunks", []) or []),
            "subsection": len(getattr(state.document, "subsection_chunks", []) or []),
            "table": len(getattr(state.document, "table_chunks", []) or []),
        }
        payload["page_chunks"] = [
            {"chunk_id": c.chunk_id, "metadata": c.metadata, "text_preview": c.text[:300]}
            for c in getattr(state.document, "page_chunks", []) or []
        ]
        payload["subsection_chunks"] = [
            {"chunk_id": c.chunk_id, "metadata": c.metadata, "text_preview": c.text[:300]}
            for c in getattr(state.document, "subsection_chunks", []) or []
        ]
        payload["table_chunks"] = [
            {"chunk_id": c.chunk_id, "metadata": c.metadata, "text_preview": c.text[:300]}
            for c in getattr(state.document, "table_chunks", []) or []
        ]
        return payload

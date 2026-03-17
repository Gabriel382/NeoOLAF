from __future__ import annotations

from typing import Optional

from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.preprocessing.normalization import normalize_text
from neoolaf.preprocessing.chunking import chunk_text
from neoolaf.preprocessing.pdf_parsing import extract_pdf
from neoolaf.resources.ocr.base_engine import BaseOCREngine


class PreprocessingLayer(BaseLayer):
    """
    Layer 00: preprocessing

    Responsibilities:
    - detect PDF type (scanned vs textual) and run the appropriate extraction
    - normalize raw text
    - optionally translate text
    - create chunks
    """

    name = "layer00_preprocessing"

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 200,
        translate: bool = False,
        translator=None,
        source_language: str | None = None,
        target_language: str = "en",
        ocr_engine: Optional[BaseOCREngine] = None,
        ocr_dpi: int = 300,
        save_intermediate: bool = True,
    ) -> None:
        """
        Args:
            chunk_size:
                Maximum character length of each chunk.
            overlap:
                Overlap between consecutive chunks.
            translate:
                Whether to apply translation after cleaning.
            translator:
                Translator backend with a `.translate(...)` method.
            source_language:
                Optional source language hint.
            target_language:
                Target language used if translation is enabled.
            ocr_engine:
                OCR engine for scanned PDFs. If None, scanned PDFs will raise.
            ocr_dpi:
                Rendering resolution for scanned page images.
            save_intermediate:
                Whether to save intermediate artifacts.
        """
        super().__init__(save_intermediate=save_intermediate)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.translate = translate
        self.translator = translator
        self.source_language = source_language
        self.target_language = target_language
        self.ocr_engine = ocr_engine
        self.ocr_dpi = ocr_dpi

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Execute preprocessing.

        If the document source_path points to a PDF, runs the unified
        extraction pipeline (scanned detection + appropriate extraction).
        Otherwise falls back to normalizing the provided raw_text.
        """
        source = state.document.source_path

        # Run unified PDF extraction if source is a PDF
        if source and source.lower().endswith(".pdf"):
            result = extract_pdf(
                source,
                ocr_engine=self.ocr_engine,
                dpi=self.ocr_dpi,
            )
            state.document.pdf_type = result["pdf_type"]
            state.document.extraction_result = result["content"]
            state.log(
                f"[layer00_preprocessing] PDF detected as {result['pdf_type']}"
            )

            # Build raw_text from extraction result for downstream layers
            if result["pdf_type"] == "textual":
                state.document.raw_text = self._flatten_textual_result(
                    result["content"]
                )
            else:
                state.document.raw_text = self._flatten_scanned_result(
                    result["content"]
                )

        # Normalize the raw text
        cleaned = normalize_text(state.document.raw_text)
        state.document.cleaned_text = cleaned

        text_for_chunking = cleaned

        # Optional translation step
        if self.translate:
            if self.translator is None:
                raise ValueError(
                    "Translation was requested but no translator backend was provided."
                )

            translated = self.translator.translate(
                cleaned,
                source_language=self.source_language,
                target_language=self.target_language,
            )

            state.document.translated_text = translated
            text_for_chunking = translated
            state.log("[layer00_preprocessing] translation applied")

        # Chunk the final text version used downstream
        chunks = chunk_text(
            text_for_chunking,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        state.document.chunks = chunks
        state.log(f"[layer00_preprocessing] produced {len(chunks)} chunks")

        return state

    @staticmethod
    def _flatten_textual_result(content: dict) -> str:
        """
        Flatten a textual extraction result dict into plain text
        for downstream normalization and chunking.
        """
        parts = []
        for doc_key, chapter in content.items():
            for sec_key, section in chapter.get("sections", {}).items():
                contenu = section.get("contenu", "")
                if contenu:
                    parts.append(contenu)
        return "\n\n".join(parts)

    @staticmethod
    def _flatten_scanned_result(content: list) -> str:
        """
        Flatten a scanned extraction result (list of page dicts)
        into plain text for downstream normalization and chunking.
        """
        parts = []
        for page in content:
            text = page.get("content", {}).get("text", "")
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize relevant preprocessing outputs.
        """
        return {
            "layer": self.name,
            "doc_id": state.document.doc_id,
            "pdf_type": state.document.pdf_type,
            "cleaned_text_preview": (state.document.cleaned_text or "")[:1000],
            "translated_text_preview": (state.document.translated_text or "")[:1000],
            "num_chunks": len(state.document.chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "text_preview": c.text[:300],
                }
                for c in state.document.chunks
            ],
        }

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
from neoolaf.resources.ocr.base_engine import BaseOCREngine


class PreprocessingLayer(BaseLayer):
    """
    Layer 00: preprocessing.

    Responsibilities:
    - extract raw content from PDF sources (text-based or OCR-based)
    - clean and normalize extracted plain text
    - translate content blocks using NLLB-200
    - chunk the final text for downstream consumption
    """

    name = "layer00_preprocessing"

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 200,
        enable_chunking: bool = True,
        translate: bool = True,
        source_language: str | None = None,
        target_language: str = "en",
        nllb_model: str = "facebook/nllb-200-distilled-600M",
        nllb_device: str | None = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        ocr_dpi: int = 300,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize Layer 00.

        Args:
            chunk_size:
                Maximum character size of each text chunk.
            overlap:
                Number of overlapping characters between consecutive chunks.
            enable_chunking:
                Whether to split the final text into chunks.
            translate:
                Whether to apply NLLB-200 translation to content blocks.
            source_language:
                ISO language code of the source document.
                If None, language is detected automatically.
            target_language:
                ISO language code of the translation target (default: English).
            nllb_model:
                HuggingFace model identifier for the NLLB-200 translation backend.
            nllb_device:
                Device string for running the translation model (e.g. "cuda", "cpu").
                If None, the backend selects automatically.
            ocr_engine:
                Optional OCR engine instance for image-based PDFs.
                If None, OCR is skipped.
            ocr_dpi:
                DPI resolution used when rasterizing PDF pages for OCR.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_chunking = enable_chunking
        self.translate = translate
        self.source_language = source_language
        self.target_language = target_language
        self.nllb_model = nllb_model
        self.nllb_device = nllb_device
        self.ocr_engine = ocr_engine
        self.ocr_dpi = ocr_dpi
        self._translator = None

    def _get_translator(self):
        if self._translator is None:
            from neoolaf.resources.translation.nllb_backend import NLLB200TranslatorBackend
            self._translator = NLLB200TranslatorBackend(
                model_name=self.nllb_model,
                device=self.nllb_device,
            )
        return self._translator

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Extract, clean, translate, and chunk document content.
        """
        source = state.document.source_path

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
            # Detect language once on full text, then reuse for all blocks
            source_lang = self.source_language
            if source_lang is None:
                from neoolaf.resources.translation.nllb_backend import detect_language
                source_lang = detect_language(cleaned)
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
                f"using {self.nllb_model}"
            )

        # ---------------------------------------------------------
        # 4. Chunking
        # ---------------------------------------------------------
        if self.enable_chunking:
            chunks = chunk_text(text_for_chunking, chunk_size=self.chunk_size, overlap=self.overlap)
            state.document.chunks = chunks
            state.log(f"[{self.name}] produced {len(chunks)} chunks")
        else:
            state.document.chunks = []
            state.log(f"[{self.name}] chunking disabled")

        return state

    def _translate_content_blocks(self, content_blocks, translator, source_language):
        """
        Translate each content block using NLLB-200.

        Text blocks are translated from their ``text`` field; table blocks
        use the ``html_text`` field as the source. The translated output is
        stored in ``translated_text`` for all block types.

        Args:
            content_blocks:
                List of content block dicts as produced by the PDF extraction step.
            translator:
                Instantiated NLLB-200 translator backend.
            source_language:
                ISO language code of the source text (already detected or provided).

        Returns:
            List of updated content block dicts with a ``translated_text`` key.
        """
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
        """
        Serialize Layer 00 outputs for debugging and reproducibility.
        """
        payload = {
            "layer": self.name,
            "doc_id": state.document.doc_id,
            "source_path": state.document.source_path,
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
                    "text_preview": c.text[:300],
                }
                for c in state.document.chunks
            ]
        return payload
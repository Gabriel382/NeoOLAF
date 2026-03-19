from __future__ import annotations

from typing import Optional

from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.preprocessing.normalization import normalize_text
from neoolaf.preprocessing.chunking import chunk_text
from neoolaf.preprocessing.cleaners import table_html_to_text
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
        enable_chunking: bool = True,
        translate: bool = False,
        translator=None,
        source_language: str | None = None,
        target_language: str = "en",
        ocr_engine: Optional[BaseOCREngine] = None,
        ocr_dpi: int = 300,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            chunk_size:
                Maximum character length of each chunk.
            overlap:
                Overlap between consecutive chunks.
            enable_chunking:
                Whether to split the document into chunks.
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
            verbose:
                Wheter to show logs or not.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_chunking = enable_chunking
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
                ordered_blocks = self._extract_textual_blocks(result["content"])
            else:
                ordered_blocks = self._extract_scanned_blocks(result["content"])

            state.document.content_blocks = ordered_blocks
            state.document.raw_text = self._flatten_content_blocks(ordered_blocks)

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

            state.document.content_blocks = self._translate_content_blocks(
                state.document.content_blocks
            )
            translated = self._flatten_content_blocks(
                state.document.content_blocks,
                use_translated_text=True,
            )
            state.document.translated_text = translated
            text_for_chunking = translated
            state.log("[layer00_preprocessing] translation applied")

        if self.enable_chunking:
            chunks = chunk_text(
                text_for_chunking,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
            state.document.chunks = chunks
            state.log(f"[layer00_preprocessing] produced {len(chunks)} chunks")
        else:
            state.document.chunks = []
            state.log("[layer00_preprocessing] chunking disabled")

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
    def _extract_textual_blocks(content: dict) -> list[dict]:
        """
        Extract ordered content blocks from textual PDF extraction output.
        """
        blocks: list[dict] = []
        order = 1

        for doc_key, chapter in content.items():
            sections = chapter.get("sections", {})
            chapter_blocks = chapter.get("content_blocks", [])
            if chapter_blocks:
                for item in chapter_blocks:
                    block = dict(item)
                    block.setdefault("document_key", doc_key)
                    block.setdefault("order", order)
                    blocks.append(block)
                    order = max(order, int(block["order"]) + 1)
                continue

            for section_key, section in sections.items():
                text = section.get("contenu", "")
                if text:
                    blocks.append(
                        {
                            "block_id": f"block_{order:05d}",
                            "type": "text",
                            "order": order,
                            "page": section.get("page"),
                            "document_key": doc_key,
                            "section_key": section_key,
                            "html": None,
                            "text": text,
                        }
                    )
                    order += 1

                for subsection_key, subsection in section.get("sous_sections", {}).items():
                    html = subsection.get("table_html", "")
                    blocks.append(
                        {
                            "block_id": f"block_{order:05d}",
                            "type": "table",
                            "order": order,
                            "page": subsection.get("page"),
                            "document_key": doc_key,
                            "section_key": section_key,
                            "subsection_key": subsection_key,
                            "text": None,
                            "title": subsection.get("titre", ""),
                            "html": html,
                            "html_text": table_html_to_text(html),
                        }
                    )
                    order += 1

        return sorted(
            blocks,
            key=lambda item: (item.get("page") or 0, item.get("order") or 0),
        )

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

    @staticmethod
    def _extract_scanned_blocks(content: list) -> list[dict]:
        """
        Extract ordered content blocks from scanned PDF extraction output.
        """
        blocks: list[dict] = []

        for page in content:
            page_number = page.get("page")
            page_blocks = page.get("content", {}).get("content_blocks", [])
            if page_blocks:
                blocks.extend(page_blocks)
                continue

            order = 1
            text = page.get("content", {}).get("text", "")
            if text:
                blocks.append(
                    {
                        "block_id": f"block_{page_number:05d}_{order:03d}",
                        "type": "text",
                        "page": page_number,
                        "order": order,
                        "html": None,
                        "text": text,
                    }
                )
                order += 1

            for idx, table in enumerate(page.get("content", {}).get("tables", [])):
                html = table.get("html", "")
                blocks.append(
                    {
                        "block_id": f"block_{page_number:05d}_{order:03d}",
                        "type": "table",
                        "page": page_number,
                        "order": order,
                        "table_index": idx,
                        "text": None,
                        "html": html,
                        "html_text": table_html_to_text(html),
                        "bbox": table.get("bbox"),
                    }
                )
                order += 1

        return sorted(
            blocks,
            key=lambda item: (item.get("page") or 0, item.get("order") or 0),
        )

    @staticmethod
    def _flatten_content_blocks(
        content_blocks: list[dict],
        use_translated_text: bool = False,
    ) -> str:
        """
        Flatten ordered content blocks into one text stream.
        """
        parts = []
        for block in content_blocks:
            if use_translated_text:
                text = block.get("translated_text", "")
            elif block.get("type") == "text":
                text = block.get("text", "")
            else:
                text = block.get("html_text", "")
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _translate_content_blocks(self, content_blocks: list[dict]) -> list[dict]:
        """
        Translate each content block independently while preserving PDF order.
        """
        translated_blocks: list[dict] = []
        for block in content_blocks:
            updated_block = dict(block)
            source_text = (
                block.get("text", "")
                if block.get("type") == "text"
                else block.get("html_text", "")
            )
            translated_text = source_text
            if source_text:
                translated_text = self.translator.translate(
                    source_text,
                    source_language=self.source_language,
                    target_language=self.target_language,
                )
            updated_block["translated_text"] = translated_text
            translated_blocks.append(updated_block)
        return translated_blocks

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize relevant preprocessing outputs.
        """
        extraction_tables = self._extract_tables_for_export(state.document.extraction_result)

        payload = {
            "layer": self.name,
            "doc_id": state.document.doc_id,
            "source_path": state.document.source_path,
            "pdf_type": state.document.pdf_type,
            "content_blocks": state.document.content_blocks,
            "tables": extraction_tables,
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

    @staticmethod
    def _extract_tables_for_export(extraction_result) -> list[dict]:
        """
        Extract table HTML snippets from the raw preprocessing result.

        For textual PDFs, tables are nested under chapter sections.
        For scanned PDFs, tables are stored page by page.
        """
        tables: list[dict] = []

        if isinstance(extraction_result, dict):
            for doc_key, chapter in extraction_result.items():
                sections = chapter.get("sections", {})
                for section_key, section in sections.items():
                    for subsection_key, subsection in section.get("sous_sections", {}).items():
                        html = subsection.get("table_html", "")
                        tables.append(
                            {
                                "document_key": doc_key,
                                "section_key": section_key,
                                "subsection_key": subsection_key,
                                "title": subsection.get("titre", ""),
                                "page": subsection.get("page"),
                                "html": html,
                                "html_text": table_html_to_text(html),
                            }
                        )

        elif isinstance(extraction_result, list):
            for page in extraction_result:
                page_number = page.get("page")
                page_tables = page.get("content", {}).get("tables", [])
                for idx, table in enumerate(page_tables):
                    html = table.get("html", "")
                    tables.append(
                        {
                            "page": page_number,
                            "table_index": idx,
                            "html": html,
                            "html_text": table_html_to_text(html),
                            "bbox": table.get("bbox"),
                        }
                    )

        return tables

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Optional

import pdfplumber

from neoolaf.preprocessing.cleaners import (
    clean_ocr_table_output,
    clean_ocr_text_output,
    table_html_to_text,
)
from neoolaf.preprocessing.image_conversion import pdf_to_images
from neoolaf.preprocessing.image_preprocessing import preprocess_page
from neoolaf.preprocessing.textual_extraction import extract_textual_pdf
from neoolaf.resources.ocr.base_engine import BaseOCREngine


# ── PDF type detection ───────────────────────────────────────────────────────


def is_scanned(
    pdf_path: str,
    sample_pages: int = 5,
    min_chars_per_page: int = 30,
    scanned_ratio: float = 0.8,
) -> bool:
    """
    Detect whether a PDF is scanned (image-based) or textual.

    Samples up to ``sample_pages`` pages and checks whether each yields
    fewer than ``min_chars_per_page`` characters of extractable text.
    If the proportion of low-text pages meets or exceeds ``scanned_ratio``,
    the PDF is classified as scanned.

    Args:
        pdf_path:
            Path to the PDF file.
        sample_pages:
            Maximum number of pages to sample for detection.
        min_chars_per_page:
            Character threshold below which a page is considered image-only.
        scanned_ratio:
            Proportion of low-text pages required to classify as scanned.

    Returns:
        True if the PDF is scanned, False if it contains extractable text.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        pages_to_check = min(sample_pages, total_pages)

        low_text_count = 0
        for page in pdf.pages[:pages_to_check]:
            text = (page.extract_text() or "").strip()
            if len(text) < min_chars_per_page:
                low_text_count += 1

    return low_text_count / pages_to_check >= scanned_ratio


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a textual PDF and concatenate the pages.

    Opens the PDF with pdfplumber, extracts text from every page, and
    joins the results with newlines. Pages that yield no text contribute
    an empty string to the join.

    Args:
        pdf_path:
            Path to the PDF file.

    Returns:
        Full document text as a single newline-separated string.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(path) as pdf:
        pages = []
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    return "\n".join(pages)


# ── Scanned PDF extraction ───────────────────────────────────────────────────


def _extract_scanned_pdf(
    pdf_path: str,
    engine: BaseOCREngine,
    dpi: int = 300,
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Extract content from a scanned PDF using OCR.

    Converts each page to a rasterized image at the requested DPI,
    preprocesses it for cleaner OCR input, runs the OCR engine, and
    applies postprocessing to both text and table outputs. Page results
    are assembled into structured content-block dicts. If a page fails,
    an error entry is appended and processing continues.

    Args:
        pdf_path:
            Path to the scanned PDF file.
        engine:
            Instantiated OCR engine (e.g. PaddleOCR or LightOnOCR).
        dpi:
            Rendering resolution used when rasterizing page images.
        output_path:
            Optional file path; if provided, the full results list is
            serialized to JSON at this location.

    Returns:
        List of per-page result dicts, each containing ``page``,
        ``page_size``, and ``content`` keys. Failed pages contain
        ``page`` and ``error`` keys instead.
    """
    pages = pdf_to_images(pdf_path, dpi=dpi)
    results = []

    for i, raw_page in enumerate(pages):
        page_number = i + 1
        try:
            proc_page = preprocess_page(raw_page)
            page_result = engine.ocr_page(proc_page)

            w, h = raw_page.size
            cleaned_text = clean_ocr_text_output(page_result["text"])
            cleaned_tables = clean_ocr_table_output(page_result.get("tables", []))
            content_blocks = []
            block_order = 1

            if cleaned_text:
                content_blocks.append(
                    {
                        "block_id": f"block_{page_number:05d}_{block_order:03d}",
                        "type": "text",
                        "page": page_number,
                        "order": block_order,
                        "text": cleaned_text,
                    }
                )
                block_order += 1

            for table_index, table in enumerate(cleaned_tables):
                html = table.get("html", "")
                content_blocks.append(
                    {
                        "block_id": f"block_{page_number:05d}_{block_order:03d}",
                        "type": "table",
                        "page": page_number,
                        "order": block_order,
                        "table_index": table_index,
                        "html": html,
                        "html_text": table_html_to_text(html),
                        "bbox": table.get("bbox"),
                    }
                )
                block_order += 1

            results.append({
                "page": page_number,
                "page_size": {"width": w, "height": h, "dpi": dpi},
                "content": {
                    "text": cleaned_text,
                    "tables": cleaned_tables,
                    "content_blocks": content_blocks,
                },
            })

        except Exception as e:
            results.append({"page": page_number, "error": str(e)})

        finally:
            del raw_page
            gc.collect()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ── Block extraction utilities ───────────────────────────────────────────────


def extract_content_blocks(pdf_type: str, content) -> list[dict]:
    """
    Convert raw extraction output into ordered content blocks.

    Routes to ``_blocks_from_textual`` or ``_blocks_from_scanned``
    depending on ``pdf_type``.

    Args:
        pdf_type:
            ``"textual"`` or ``"scanned"``, as returned by ``extract_pdf``.
        content:
            Raw extraction result — a dict for textual PDFs, a list for
            scanned PDFs.

    Returns:
        Sorted list of content block dicts ordered by page then insertion
        order.
    """
    if pdf_type == "textual":
        return _blocks_from_textual(content)
    return _blocks_from_scanned(content)


def _blocks_from_textual(content: dict) -> list[dict]:
    """
    Convert a textual PDF extraction dict into ordered content blocks.

    Iterates over chapters and their sections. If a chapter already carries
    pre-built ``content_blocks``, those are used directly; otherwise, section
    text and subsection table HTML are each converted into individual block
    dicts. All blocks are sorted by page then order before being returned.

    Args:
        content:
            Nested extraction dict as produced by ``extract_textual_pdf``.

    Returns:
        Sorted list of content block dicts.
    """
    blocks: list[dict] = []
    order = 1

    for doc_key, chapter in content.items():
        chapter_blocks = chapter.get("content_blocks", [])
        if chapter_blocks:
            for item in chapter_blocks:
                block = dict(item)
                block.setdefault("document_key", doc_key)
                block.setdefault("order", order)
                blocks.append(block)
                order = max(order, int(block["order"]) + 1)
            continue

        for section_key, section in chapter.get("sections", {}).items():
            text = section.get("contenu", "")
            if text:
                blocks.append({
                    "block_id": f"block_{order:05d}",
                    "type": "text",
                    "order": order,
                    "page": section.get("page"),
                    "document_key": doc_key,
                    "section_key": section_key,
                    "text": text,
                })
                order += 1

            for sub_key, sub in section.get("sous_sections", {}).items():
                html = sub.get("table_html", "")
                blocks.append({
                    "block_id": f"block_{order:05d}",
                    "type": "table",
                    "order": order,
                    "page": sub.get("page"),
                    "document_key": doc_key,
                    "section_key": section_key,
                    "subsection_key": sub_key,
                    "title": sub.get("titre", ""),
                    "html": html,
                    "html_text": table_html_to_text(html),
                })
                order += 1

    return sorted(blocks, key=lambda b: (b.get("page") or 0, b.get("order") or 0))


def _blocks_from_scanned(content: list) -> list[dict]:
    """
    Convert a scanned PDF extraction list into ordered content blocks.

    Iterates over per-page result dicts. If a page already carries
    pre-built ``content_blocks``, those are used directly; otherwise,
    page-level text and table entries are each converted into individual
    block dicts. All blocks are sorted by page then order before being
    returned.

    Args:
        content:
            List of per-page result dicts as produced by
            ``_extract_scanned_pdf``.

    Returns:
        Sorted list of content block dicts.
    """
    blocks: list[dict] = []

    for page in content:
        page_blocks = page.get("content", {}).get("content_blocks", [])
        if page_blocks:
            blocks.extend(page_blocks)
            continue

        page_number = page.get("page")
        order = 1
        text = page.get("content", {}).get("text", "")
        if text:
            blocks.append({
                "block_id": f"block_{page_number:05d}_{order:03d}",
                "type": "text",
                "page": page_number,
                "order": order,
                "text": text,
            })
            order += 1

        for idx, table in enumerate(page.get("content", {}).get("tables", [])):
            html = table.get("html", "")
            blocks.append({
                "block_id": f"block_{page_number:05d}_{order:03d}",
                "type": "table",
                "page": page_number,
                "order": order,
                "table_index": idx,
                "html": html,
                "html_text": table_html_to_text(html),
                "bbox": table.get("bbox"),
            })
            order += 1

    return sorted(blocks, key=lambda b: (b.get("page") or 0, b.get("order") or 0))


def flatten_content_blocks(
    content_blocks: list[dict],
    use_translated_text: bool = False,
) -> str:
    """
    Flatten ordered content blocks into one plain-text stream.

    When ``use_translated_text`` is True, the ``translated_text`` field
    is used for every block type. Otherwise, ``text`` is used for text
    blocks and ``html_text`` for table blocks. Non-empty parts are joined
    with a double newline.

    Args:
        content_blocks:
            Ordered list of content block dicts.
        use_translated_text:
            If True, read the ``translated_text`` field instead of the
            type-specific default field.

    Returns:
        Single string with block texts joined by ``"\\n\\n"``.
    """
    parts = []
    for block in content_blocks:
        if use_translated_text:
            text = block.get("translated_text") or ""
        elif block.get("type") == "text":
            text = block.get("text") or ""
        else:
            text = block.get("html_text") or ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def extract_tables_for_export(extraction_result) -> list[dict]:
    """
    Extract table HTML snippets from a raw extraction result.

    Handles both textual (dict) and scanned (list) extraction results.
    For textual results, tables are drawn from subsection ``table_html``
    fields. For scanned results, tables are drawn from per-page
    ``content.tables`` entries.

    Args:
        extraction_result:
            Raw extraction result as returned by ``extract_pdf`` — either
            a nested dict (textual) or a list of page dicts (scanned).

    Returns:
        List of table dicts, each containing ``html``, ``html_text``,
        and provenance keys (``page``, ``document_key``, etc.).
    """
    tables: list[dict] = []

    if isinstance(extraction_result, dict):
        for doc_key, chapter in extraction_result.items():
            for sec_key, section in chapter.get("sections", {}).items():
                for sub_key, sub in section.get("sous_sections", {}).items():
                    html = sub.get("table_html", "")
                    tables.append({
                        "document_key": doc_key,
                        "section_key": sec_key,
                        "subsection_key": sub_key,
                        "title": sub.get("titre", ""),
                        "page": sub.get("page"),
                        "html": html,
                        "html_text": table_html_to_text(html),
                    })

    elif isinstance(extraction_result, list):
        for page in extraction_result:
            page_number = page.get("page")
            for idx, table in enumerate(page.get("content", {}).get("tables", [])):
                html = table.get("html", "")
                tables.append({
                    "page": page_number,
                    "table_index": idx,
                    "html": html,
                    "html_text": table_html_to_text(html),
                    "bbox": table.get("bbox"),
                })

    return tables


# ── Unified extraction ───────────────────────────────────────────────────────


def extract_pdf(
    pdf_path: str,
    ocr_engine: Optional[BaseOCREngine] = None,
    dpi: int = 300,
) -> dict:
    """
    Unified PDF extraction: detect type and route to the appropriate pipeline.

    Textual PDFs are handled by the direct text-layer extraction path
    (pdfplumber). Scanned PDFs are rasterized and processed through the
    provided OCR engine. Raises ``ValueError`` if a scanned PDF is
    encountered without an engine.

    Args:
        pdf_path:
            Path to the PDF file.
        ocr_engine:
            Instantiated OCR engine, required when the PDF is scanned.
            If None and the PDF is scanned, a ``ValueError`` is raised.
        dpi:
            Rendering resolution used when rasterizing scanned page images.

    Returns:
        Dict with keys:

        - ``pdf_type`` (str): ``"textual"`` or ``"scanned"``.
        - ``content`` (dict | list): extraction result in the format
          produced by the chosen pipeline.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    scanned = is_scanned(pdf_path)

    if scanned:
        if ocr_engine is None:
            raise ValueError(
                f"PDF '{path.name}' is scanned but no OCR engine was provided."
            )
        content = _extract_scanned_pdf(pdf_path, ocr_engine, dpi=dpi)
        return {"pdf_type": "scanned", "content": content}
    else:
        content = extract_textual_pdf(pdf_path)
        return {"pdf_type": "textual", "content": content}
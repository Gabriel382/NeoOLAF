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


def is_scanned(
    pdf_path: str,
    sample_pages: int = 5,
    min_chars_per_page: int = 30,
    scanned_ratio: float = 0.8,
) -> bool:
    """
    Detect whether a PDF is scanned (image-based) or textual.

    Samples up to `sample_pages` pages and checks if each page yields
    fewer than `min_chars_per_page` characters of extractable text.
    If the proportion of low-text pages exceeds `scanned_ratio`, the
    PDF is classified as scanned.

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

    Converts pages to images, preprocesses them, runs OCR,
    and applies postprocessing to the results.

    Args:
        pdf_path:   Path to the scanned PDF.
        engine:     OCR engine instance (PaddleOCR or LightOnOCR).
        dpi:        Rendering resolution for page images.
        output_path: Optional path to save results as JSON.

    Returns:
        List of page result dicts with text, tables, and metadata.
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
                        "html": None,
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
                        "text": None,
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


# ── Unified extraction ───────────────────────────────────────────────────────


def extract_pdf(
    pdf_path: str,
    ocr_engine: Optional[BaseOCREngine] = None,
    dpi: int = 300,
) -> dict:
    """
    Unified PDF extraction: detects type and routes to the appropriate pipeline.

    - Textual PDFs → direct text-layer extraction (pdfplumber)
    - Scanned PDFs → image conversion + OCR engine

    Args:
        pdf_path:    Path to the PDF file.
        ocr_engine:  OCR engine for scanned PDFs. Required if PDF is scanned.
        dpi:         Rendering resolution for scanned page images.

    Returns:
        dict with keys:
            - pdf_type  (str)  : "textual" or "scanned"
            - content   (dict | list) : extraction result
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

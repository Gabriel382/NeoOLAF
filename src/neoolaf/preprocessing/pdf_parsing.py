"""
Utilities for extracting text from textual PDFs.
This version assumes the PDF already contains machine-readable text.
"""
from __future__ import annotations

from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF and concatenate the pages.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages)
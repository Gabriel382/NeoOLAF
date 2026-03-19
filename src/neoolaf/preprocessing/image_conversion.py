from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image
from pdf2image import convert_from_path


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert each page of a PDF into a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Rendering resolution

    Returns:
        List of PIL Image objects, one per page.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    return convert_from_path(str(path), dpi=dpi)

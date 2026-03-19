from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocumentChunk:
    """
    A chunk extracted from a document after preprocessing.
    """
    chunk_id: str
    text: str
    start_char: int
    end_char: int


@dataclass
class Document:
    """
    Main document container.
    Stores the raw text, cleaned text, optional translated text,
    and the chunks generated during preprocessing.
    """
    doc_id: str
    source_path: str
    raw_text: str

    # PDF type detected during preprocessing ("textual" or "scanned")
    pdf_type: Optional[str] = None

    # Structured extraction result from the PDF pipeline
    extraction_result: Optional[dict] = None

    # Ordered document blocks following PDF reading order
    content_blocks: List[dict] = field(default_factory=list)

    # Cleaned text after normalization
    cleaned_text: Optional[str] = None

    # Optional translated version used for downstream processing
    translated_text: Optional[str] = None

    # Chunks created after preprocessing
    chunks: List[DocumentChunk] = field(default_factory=list)

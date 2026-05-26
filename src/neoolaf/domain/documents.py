from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class DocumentChunk:
    """
    A chunk extracted from a document after preprocessing.
    """
    chunk_id: str
    text: str
    start_char: int
    end_char: int

    # Optional structured metadata, e.g. page/subsection/table provenance.
    metadata: dict[str, Any] = field(default_factory=dict)


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

    # Chunks created after preprocessing.  ``chunks`` is the active view used by
    # downstream layers.  The specialized views below are kept so a run can switch
    # from page-level to subsection/table-level extraction without re-parsing the PDF.
    chunks: List[DocumentChunk] = field(default_factory=list)
    page_chunks: List[DocumentChunk] = field(default_factory=list)
    subsection_chunks: List[DocumentChunk] = field(default_factory=list)
    table_chunks: List[DocumentChunk] = field(default_factory=list)

    # Structured units produced by profile-specific preprocessing.
    structured_units: List[dict[str, Any]] = field(default_factory=list)

    # Optional profile-specific records, e.g. XQuality alarm records extracted
    # from alarm tables before triple generation.
    alarm_records: List[dict[str, Any]] = field(default_factory=list)

from __future__ import annotations

"""Profile-aware structured chunk creation.

The generic fixed-size chunks remain available, but profiles can also request
page-, subsection-, and table-level chunks.  This is especially useful for
technical manuals whose tables map almost directly to KG triples.
"""

import re
from collections import defaultdict
from typing import Any

from neoolaf.domain.documents import DocumentChunk


def _block_text(block: dict[str, Any], *, prefer_translated: bool = False) -> str:
    if prefer_translated and block.get("translated_text"):
        return str(block.get("translated_text") or "")
    if block.get("type") == "text":
        return str(block.get("text") or "")
    return str(block.get("html_text") or block.get("text") or "")


def _clean_id(value: Any) -> str:
    raw = str(value or "unknown").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return raw or "unknown"


def _make_chunk(chunk_id: str, text: str, offset: int, metadata: dict[str, Any]) -> tuple[DocumentChunk, int]:
    text = text or ""
    start = offset
    end = offset + len(text)
    return DocumentChunk(chunk_id=chunk_id, text=text, start_char=start, end_char=end, metadata=metadata), end + 2


def build_page_chunks(content_blocks: list[dict[str, Any]], *, prefer_translated: bool = False) -> list[DocumentChunk]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for block in content_blocks:
        grouped[block.get("page") or "unknown"].append(block)

    chunks: list[DocumentChunk] = []
    offset = 0
    for page in sorted(grouped, key=lambda p: (999999 if p == "unknown" else int(p))):
        blocks = sorted(grouped[page], key=lambda b: b.get("order") or 0)
        text = "\n\n".join(part for part in (_block_text(b, prefer_translated=prefer_translated) for b in blocks) if part)
        chunk, offset = _make_chunk(
            f"page_{int(page):04d}" if str(page).isdigit() else f"page_{_clean_id(page)}",
            text,
            offset,
            {"chunk_type": "page", "page": page, "num_blocks": len(blocks)},
        )
        chunks.append(chunk)
    return chunks


def build_table_chunks(content_blocks: list[dict[str, Any]], *, prefer_translated: bool = False) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    offset = 0
    for idx, block in enumerate(content_blocks):
        if block.get("type") != "table":
            continue
        page = block.get("page")
        subsection = block.get("subsection_key") or block.get("title") or block.get("table_index") or idx
        title = block.get("title") or ""
        text = _block_text(block, prefer_translated=prefer_translated)
        if not text:
            continue
        chunk_id = f"table_p{str(page or 'unknown').zfill(4)}_{_clean_id(subsection)}"
        metadata = {
            "chunk_type": "table",
            "page": page,
            "title": title,
            "document_key": block.get("document_key"),
            "section_key": block.get("section_key"),
            "subsection_key": block.get("subsection_key"),
            "table_index": block.get("table_index"),
            "html": block.get("html"),
            "html_text": block.get("html_text"),
            "translated_text": block.get("translated_text"),
        }
        chunk, offset = _make_chunk(chunk_id, text, offset, metadata)
        chunks.append(chunk)
    return chunks


def build_subsection_chunks(
    content_blocks: list[dict[str, Any]],
    *,
    prefer_translated: bool = False,
    subsection_patterns: list[str] | None = None,
) -> list[DocumentChunk]:
    """Build subsection chunks from explicit PDF structure or regex fallbacks."""
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for block in content_blocks:
        key = (
            block.get("document_key"),
            block.get("section_key"),
            block.get("subsection_key") or block.get("title") or block.get("page"),
        )
        grouped[key].append(block)

    chunks: list[DocumentChunk] = []
    offset = 0
    for idx, (key, blocks) in enumerate(grouped.items()):
        blocks = sorted(blocks, key=lambda b: (b.get("page") or 0, b.get("order") or 0))
        parts = [_block_text(b, prefer_translated=prefer_translated) for b in blocks]
        text = "\n\n".join(part for part in parts if part)
        if not text:
            continue
        pages = sorted({b.get("page") for b in blocks if b.get("page") is not None})
        alarm_no = _extract_alarm_no(text, subsection_patterns or [])
        chunk_id = f"subsection_{alarm_no}" if alarm_no else f"subsection_{idx:04d}_{_clean_id(key[-1])}"
        metadata = {
            "chunk_type": "subsection",
            "document_key": key[0],
            "section_key": key[1],
            "subsection_key": key[2],
            "page_start": pages[0] if pages else None,
            "page_end": pages[-1] if pages else None,
            "alarm_no": alarm_no,
            "num_blocks": len(blocks),
        }
        chunk, offset = _make_chunk(chunk_id, text, offset, metadata)
        chunks.append(chunk)
    return chunks


def _extract_alarm_no(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        try:
            match = re.search(pattern, text, flags=re.IGNORECASE)
        except re.error:
            continue
        if match:
            if "alarm_no" in match.groupdict():
                return match.group("alarm_no")
            return match.group(0)
    # Fallback for common French/English forms.
    match = re.search(r"(?:alarme|alarm)\s*n?[°o]?\s*(\d{3,5})", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def build_structured_chunks(
    content_blocks: list[dict[str, Any]],
    *,
    profile_config: dict[str, Any] | None = None,
    prefer_translated: bool = False,
) -> dict[str, list[DocumentChunk]]:
    profile_config = profile_config or {}
    subsection_patterns = (
        profile_config.get("chunking", {}).get("subsection_patterns", [])
        if isinstance(profile_config.get("chunking"), dict)
        else []
    )
    return {
        "page": build_page_chunks(content_blocks, prefer_translated=prefer_translated),
        "subsection": build_subsection_chunks(
            content_blocks,
            prefer_translated=prefer_translated,
            subsection_patterns=list(subsection_patterns or []),
        ),
        "table": build_table_chunks(content_blocks, prefer_translated=prefer_translated),
    }

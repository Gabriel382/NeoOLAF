from __future__ import annotations

"""Profile-aware structured chunk creation.

The generic fixed-size chunks remain available, but profiles can also request
page-, subsection-, and table-level chunks.  Table chunks keep the original raw
HTML for traceability, while downstream prompts receive a compact, generic table
representation (rows/cells) so prompts are not tied to one document schema and do
not waste tokens on duplicated HTML/text.
"""

import html
import re
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any

from neoolaf.domain.documents import DocumentChunk


class _SimpleTableHTMLParser(HTMLParser):
    """Small HTML table parser using only the Python standard library.

    It extracts row/cell structure from simple HTML tables produced by the PDF
    parser.  We intentionally keep this generic: it does not know about Machine
    32, alarms, causes, effects, or any domain-specific field name.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict[str, Any]]] = []
        self._current_row: list[dict[str, Any]] | None = None
        self._current_cell: dict[str, Any] | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {k.lower(): v for k, v in attrs}
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_text = []
            self._current_cell = {
                "role": "header" if tag == "th" else "value",
                "rowspan": _safe_int(attrs_dict.get("rowspan"), default=1),
                "colspan": _safe_int(attrs_dict.get("colspan"), default=1),
            }

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            text = _normalize_ws(" ".join(self._current_text))
            self._current_cell["text"] = html.unescape(text)
            self._current_row.append(self._current_cell)
            self._current_cell = None
            self._current_text = []
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None


def _safe_int(value: str | None, *, default: int = 1) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


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


def _parse_html_table(html_text: str | None) -> list[list[dict[str, Any]]]:
    if not html_text:
        return []
    parser = _SimpleTableHTMLParser()
    try:
        parser.feed(str(html_text))
        parser.close()
    except Exception:
        return []
    return parser.rows


def _compact_table_from_block(block: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, generic table representation for prompting.

    The result preserves row/column structure and cell roles, but it avoids
    embedding full raw HTML.  Raw HTML remains separately available in metadata
    for debugging/provenance.
    """
    rows = _parse_html_table(block.get("html"))
    compact_rows: list[list[dict[str, Any]]] = []
    max_cols = 0
    for row_idx, row in enumerate(rows):
        compact_row: list[dict[str, Any]] = []
        logical_col = 0
        for cell in row:
            text = _normalize_ws(cell.get("text", ""))
            if not text:
                logical_col += int(cell.get("colspan") or 1)
                continue
            compact_row.append(
                {
                    "row": row_idx,
                    "col": logical_col,
                    "role": cell.get("role") or "value",
                    "text": text,
                    "rowspan": int(cell.get("rowspan") or 1),
                    "colspan": int(cell.get("colspan") or 1),
                }
            )
            logical_col += int(cell.get("colspan") or 1)
        max_cols = max(max_cols, logical_col)
        if compact_row:
            compact_rows.append(compact_row)

    # A very common technical-document layout is a field/value table.  This is a
    # generic structural hint only: it does not require any predefined field name.
    # It helps the LLM see that the first cell is often a row header and the rest
    # are values, while still allowing other table layouts.
    field_value_rows: list[dict[str, Any]] = []
    for row in compact_rows:
        if not row:
            continue
        header = row[0]
        values = row[1:] if len(row) > 1 else []
        if header.get("role") == "header" and values:
            field_value_rows.append(
                {
                    "field": header.get("text", ""),
                    "values": [v.get("text", "") for v in values if v.get("text")],
                    "row": header.get("row"),
                }
            )

    return {
        "unit_type": "table",
        "page": block.get("page"),
        "title": block.get("title") or "",
        "section_key": block.get("section_key") or "",
        "subsection_key": block.get("subsection_key") or "",
        "table_index": block.get("table_index"),
        "shape": {"rows": len(compact_rows), "cols": max_cols},
        "rows": compact_rows,
        "field_value_rows": field_value_rows,
    }


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
        compact_table = _compact_table_from_block(block)
        metadata = {
            "chunk_type": "table",
            "page": page,
            "title": title,
            "document_key": block.get("document_key"),
            "section_key": block.get("section_key"),
            "subsection_key": block.get("subsection_key"),
            "table_index": block.get("table_index"),
            # Compact structure for prompts.
            "compact_table": compact_table,
            # Raw/provenance/debug fields. Prompts should avoid these by default.
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

from __future__ import annotations

import re
from dataclasses import dataclass

from neoolaf.preprocessing.cleaners import normalize_line
from neoolaf.preprocessing.structural_detection import (
    GENERIC_FOOTER_PATTERNS,
    PAGE_REF_RE,
    DocumentStructure,
    _normalize_for_comparison,
    detect_toc_by_structure,
)


@dataclass(frozen=True)
class Word:
    """One extracted word with its page coordinates."""
    top: float
    bottom: float
    x0: float
    x1: float
    text: str


def remove_footers(
    lines: list[str],
    structure: DocumentStructure | None = None,
) -> list[str]:
    """Remove lines that look like page footers."""
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if any(pattern.match(stripped) for pattern in GENERIC_FOOTER_PATTERNS):
            continue
        if structure and structure.repeated_footer_lines:
            if _normalize_for_comparison(stripped) in structure.repeated_footer_lines:
                continue
        kept.append(stripped)
    return kept


def remove_headers(
    lines: list[str],
    structure: DocumentStructure | None = None,
) -> list[str]:
    """Remove lines that look like page headers (detected by repetition)."""
    if not structure or not structure.repeated_header_lines:
        return lines
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _normalize_for_comparison(stripped) in structure.repeated_header_lines:
            continue
        kept.append(stripped)
    return kept


def split_lines(
    text: str,
    structure: DocumentStructure | None = None,
) -> list[str]:
    """Split extracted page text into cleaned lines."""
    lines = remove_footers(text.splitlines(), structure)
    return remove_headers(lines, structure)


def join_lines(lines: list[str]) -> str:
    """Join non-empty lines into one text block."""
    return "\n".join(line for line in lines if line).strip()


def keep_word(
    word,
    page_height: float,
    structure: DocumentStructure | None = None,
) -> bool:
    """Keep only body words and ignore header/footer words."""
    top = float(word.get("top", 0.0))
    bottom = float(word.get("bottom", top))
    text = str(word.get("text", "")).strip()
    if not text:
        return False
    if structure:
        top_margin = structure.margins.top
        bottom_margin = structure.margins.bottom_offset
    else:
        top_margin = 50.0
        bottom_margin = 40.0
    return top >= top_margin and bottom <= page_height - bottom_margin


def body_lines(
    page,
    structure: DocumentStructure | None = None,
) -> list[str]:
    """Rebuild cleaner body lines from word coordinates."""
    words = page.extract_words(
        x_tolerance=1, y_tolerance=3, use_text_flow=False, keep_blank_chars=False
    )
    items = [
        Word(
            top=round(float(word.get("top", 0.0)), 1),
            bottom=float(word.get("bottom", 0.0)),
            x0=float(word.get("x0", 0.0)),
            x1=float(word.get("x1", 0.0)),
            text=str(word.get("text", "")).strip(),
        )
        for word in words
        if keep_word(word, page.height, structure)
    ]
    if not items:
        return []

    rows = []
    current = [items[0]]
    current_top = items[0].top
    for item in items[1:]:
        if abs(item.top - current_top) > 4:
            rows.append(current)
            current = [item]
            current_top = item.top
        else:
            current.append(item)
    rows.append(current)

    lines = []
    previous = None
    for row in rows:
        row.sort(key=lambda item: item.x0)
        line = normalize_line(" ".join(item.text for item in row))
        if not line or line == previous:
            continue
        lines.append(line)
        previous = line
    return remove_footers(lines, structure)


def extract_page_lines(
    page,
    body_only: bool = False,
    structure: DocumentStructure | None = None,
) -> list[str]:
    """Extract cleaned lines from one page."""
    if body_only:
        return body_lines(page, structure)
    return split_lines(page.extract_text() or "", structure)


def extract_page_text(
    page,
    body_only: bool = False,
    structure: DocumentStructure | None = None,
) -> str:
    """Extract cleaned text from one page."""
    return join_lines(extract_page_lines(page, body_only=body_only, structure=structure))


# ── Page classification helpers ──────────────────────────────────────────────


def local_index(lines: list[str]) -> bool:
    """Detect local index pages inside a manual."""
    if not lines:
        return False
    if lines[0].strip().upper() == "INDEX":
        return True
    page_refs = sum(len(PAGE_REF_RE.findall(line)) for line in lines[:25])
    toc_like = sum(
        1
        for line in lines[:25]
        if re.match(r"^\s*\d+(?:\.\d+)+(?:\.?[^\n]*)?\s+\d+(?:-\d+)?\s*$", line)
    )
    return page_refs >= 6 or toc_like >= 4


def toc_page(lines: list[str]) -> bool:
    """Detect whether a page is a table-of-contents page."""
    return detect_toc_by_structure(lines)

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
    """
    One extracted word with its page coordinates.

    Attributes:
        top:
            Y-coordinate of the word's top edge in PDF points.
        bottom:
            Y-coordinate of the word's bottom edge in PDF points.
        x0:
            X-coordinate of the word's left edge in PDF points.
        x1:
            X-coordinate of the word's right edge in PDF points.
        text:
            Stripped string content of the word.
    """

    top: float
    bottom: float
    x0: float
    x1: float
    text: str


# ── Header and footer removal ────────────────────────────────────────────────


def remove_footers(
    lines: list[str],
    structure: DocumentStructure | None = None,
) -> list[str]:
    """
    Remove lines that look like page footers.

    Drops any line matching a pattern in ``GENERIC_FOOTER_PATTERNS``.
    If a ``DocumentStructure`` is provided, also drops lines whose
    normalized form appears in ``structure.repeated_footer_lines``.

    Args:
        lines:
            Raw lines extracted from a single page.
        structure:
            Optional document-level structure carrying detected repeated
            footer signatures. If None, only generic pattern matching is used.

    Returns:
        Filtered list of lines with footer lines removed.
    """
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
    """
    Remove lines that look like page headers (detected by repetition).

    Only acts when ``structure`` provides a non-empty
    ``repeated_header_lines`` set; returns the input unchanged otherwise.

    Args:
        lines:
            Raw lines extracted from a single page.
        structure:
            Optional document-level structure carrying detected repeated
            header signatures. If None, the lines are returned as-is.

    Returns:
        Filtered list of lines with repeated header lines removed.
    """
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


# ── Line splitting and joining ───────────────────────────────────────────────


def split_lines(
    text: str,
    structure: DocumentStructure | None = None,
) -> list[str]:
    """
    Split extracted page text into cleaned lines.

    Splits on newlines then passes the result through ``remove_footers``
    and ``remove_headers`` in sequence.

    Args:
        text:
            Raw text string extracted from a single PDF page.
        structure:
            Optional document-level structure used for header/footer removal.

    Returns:
        Cleaned list of lines with headers and footers stripped.
    """
    lines = remove_footers(text.splitlines(), structure)
    return remove_headers(lines, structure)


def join_lines(lines: list[str]) -> str:
    """
    Join non-empty lines into one text block.

    Filters out empty strings and joins the remaining lines with newlines,
    stripping leading and trailing whitespace from the result.
    """
    return "\n".join(line for line in lines if line).strip()


# ── Word-level body extraction ───────────────────────────────────────────────


def keep_word(
    word,
    page_height: float,
    structure: DocumentStructure | None = None,
) -> bool:
    """
    Keep only body words and ignore header and footer words.

    Accepts a word if its vertical span falls within the body region
    defined by the top and bottom margins. Falls back to default margins
    of 50 pt (top) and 40 pt (bottom) when no structure is provided.

    Args:
        word:
            Word dict as returned by pdfplumber's ``extract_words``.
        page_height:
            Total height of the page in PDF points.
        structure:
            Optional document-level structure carrying detected margin
            offsets. If None, default margin values are used.

    Returns:
        True if the word falls inside the body region, False otherwise.
    """
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
    """
    Rebuild cleaner body lines from word coordinates.

    Extracts words via pdfplumber, filters them through ``keep_word``,
    groups them into rows by proximity of their top Y-coordinate (tolerance
    of 4 pt), sorts each row left-to-right, normalizes spacing, deduplicates
    adjacent identical lines, and finally applies ``remove_footers``.

    Args:
        page:
            A pdfplumber page object.
        structure:
            Optional document-level structure used for margin detection
            and footer removal.

    Returns:
        Ordered list of cleaned body text lines for the page.
    """
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


# ── Page-level extraction entrypoints ────────────────────────────────────────


def extract_page_lines(
    page,
    body_only: bool = False,
    structure: DocumentStructure | None = None,
) -> list[str]:
    """
    Extract cleaned lines from one page.

    Delegates to ``body_lines`` when ``body_only`` is True (word-coordinate
    path), or to ``split_lines`` over the raw page text otherwise.

    Args:
        page:
            A pdfplumber page object.
        body_only:
            If True, use the word-coordinate extraction path which respects
            margin boundaries. If False, use the raw text extraction path.
        structure:
            Optional document-level structure passed through to the chosen
            extraction path.

    Returns:
        Ordered list of cleaned text lines for the page.
    """
    if body_only:
        return body_lines(page, structure)
    return split_lines(page.extract_text() or "", structure)


def extract_page_text(
    page,
    body_only: bool = False,
    structure: DocumentStructure | None = None,
) -> str:
    """
    Extract cleaned text from one page.

    Calls ``extract_page_lines`` and joins the result into a single
    string via ``join_lines``.

    Args:
        page:
            A pdfplumber page object.
        body_only:
            Passed through to ``extract_page_lines``; see its docstring
            for details.
        structure:
            Optional document-level structure passed through to the
            extraction path.

    Returns:
        Cleaned page text as a single newline-separated string.
    """
    return join_lines(extract_page_lines(page, body_only=body_only, structure=structure))


# ── Page classification helpers ──────────────────────────────────────────────


def local_index(lines: list[str]) -> bool:
    """
    Detect local index pages inside a manual.

    Returns True if the first line is ``"INDEX"`` (case-insensitive), or
    if the first 25 lines contain at least 6 page references or at least
    4 numbered-section TOC-style entries.

    Args:
        lines:
            Lines extracted from a candidate page.

    Returns:
        True if the page is identified as a local index page.
    """
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
    """
    Detect whether a page is a table-of-contents page.

    Delegates to ``detect_toc_by_structure`` for the full structural
    heuristic defined in the structural detection module.

    Args:
        lines:
            Lines extracted from a candidate page.

    Returns:
        True if the page is identified as a table-of-contents page.
    """
    return detect_toc_by_structure(lines)
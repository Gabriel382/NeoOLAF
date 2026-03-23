from __future__ import annotations

import re
from html import escape

from neoolaf.preprocessing.structural_detection import (
    HEADING_RE,
    REVISION_RE,
    clean_heading_title,
)


# ── Cell and row helpers ─────────────────────────────────────────────────────


def clean_cell(value) -> str:
    """
    Normalize whitespace inside one table cell.

    Converts the value to a string, replaces carriage returns with
    newlines, collapses runs of spaces and tabs to a single space,
    and collapses runs of two or more newlines to one before stripping.
    Returns an empty string for None inputs.
    """
    if value is None:
        return ""
    text = str(value).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def row_text(row: list[str]) -> str:
    """
    Join a row into one comparable string.

    Concatenates non-empty cells with a single space and strips the
    result, producing a canonical representation suitable for noise
    detection and deduplication.
    """
    return " ".join(cell for cell in row if cell).strip()


def noise_row(row: list[str]) -> bool:
    """
    Detect rows that are only revision or layout noise.

    Returns True if the joined row text is non-empty and matches
    ``REVISION_RE`` (e.g. revision markers, sheet numbers, or ``rev.``
    annotations).

    Args:
        row:
            List of cleaned cell strings for one table row.

    Returns:
        True if the row is identified as noise and should be discarded.
    """
    text = row_text(row)
    return bool(text and REVISION_RE.search(text))


# ── Table quality filter ─────────────────────────────────────────────────────


def good_table(rows: list[list[str]]) -> bool:
    """
    Filter out poor-quality tables.

    A table is accepted only if, after removing empty and noise rows,
    it has at least two rows, at least four non-empty cells in total,
    no more than two cells exceeding 800 characters, and no revision
    noise in the first six rows.

    Args:
        rows:
            List of cleaned rows, where each row is a list of cell strings.

    Returns:
        True if the table meets the minimum quality thresholds.
    """
    rows = [row for row in rows if any(row) and not noise_row(row)]
    if len(rows) < 2:
        return False
    non_empty = sum(len([cell for cell in row if cell]) for row in rows)
    giant_cells = sum(
        1 for row in rows for cell in row if cell and len(cell) > 800
    )
    joined = " ".join(row_text(row) for row in rows[:6])
    if giant_cells > 2 or REVISION_RE.search(joined):
        return False
    return non_empty >= 4


# ── Page table extraction ────────────────────────────────────────────────────


def tables(page) -> list[list[list[str]]]:
    """
    Extract valid tables from one page.

    Iterates over all tables found by pdfplumber, cleans each cell via
    ``clean_cell``, drops empty and noise rows, and retains only tables
    that pass ``good_table``.

    Args:
        page:
            A pdfplumber page object.

    Returns:
        List of accepted tables, where each table is a list of rows and
        each row is a list of cleaned cell strings.
    """
    result = []
    for found in page.find_tables():
        rows = []
        for row in found.extract():
            clean_row = [clean_cell(cell) for cell in row or []]
            if any(clean_row) and not noise_row(clean_row):
                rows.append(clean_row)
        if good_table(rows):
            result.append(rows)
    return result


def extract_page_tables(page) -> list[list[list[str]]]:
    """Public high-level entrypoint for extracting valid tables from one page."""
    return tables(page)


# ── HTML conversion ──────────────────────────────────────────────────────────


def html_table(rows: list[list[str]]) -> str:
    """
    Convert extracted rows into deterministic HTML.

    Empty rows are skipped. Single-cell rows are rendered as a plain
    ``<td>``; multi-cell rows use the first cell as a ``<th>`` header
    and the remaining cells as ``<td>`` data cells. Returns a minimal
    placeholder table if all rows are empty.

    Args:
        rows:
            List of rows, where each row is a list of cleaned cell strings.

    Returns:
        HTML string representing the table, always wrapped in a
        ``<table>`` element.
    """
    parts = []
    for row in rows:
        if not any(row):
            continue
        if len(row) == 1:
            parts.append(f"<tr><td>{escape(row[0])}</td></tr>")
            continue
        first = escape(row[0])
        rest = "".join(f"<td>{escape(cell)}</td>" for cell in row[1:])
        parts.append(f"<tr><th>{first}</th>{rest}</tr>")
    if not parts:
        return "<table><tr><td></td></tr></table>"
    return "<table>" + "".join(parts) + "</table>"


# ── Table title detection ────────────────────────────────────────────────────


def table_title(rows: list[list[str]], page_lines: list[str], fallback: str) -> str:
    """
    Pick the best title for one extracted table.

    Applies three candidate strategies in order of preference:

    1. A page line starting with ``"table "`` that passes ``valid_title``.
    2. The first non-empty cell of the first row, cleaned via
       ``clean_heading_title``, if it passes ``valid_title``.
    3. Any of the first eight page lines that passes ``valid_title`` and
       does not match ``HEADING_RE`` (to avoid chapter headings).

    Falls back to ``fallback`` if no candidate is found. Titles are
    rejected if they are shorter than 8 or longer than 80 characters,
    contain contact-info markers, or exceed 14 words.

    Args:
        rows:
            Cleaned rows of the table, used to inspect the first cell.
        page_lines:
            All lines extracted from the page, searched for a title
            appearing above the table.
        fallback:
            String returned when no suitable title candidate is found.

    Returns:
        Best available title string, or ``fallback`` if none qualifies.
    """
    def valid_title(title: str) -> bool:
        if not title:
            return False
        if len(title) < 8 or len(title) > 80:
            return False
        lowered = title.lower()
        if any(
            marker in lowered
            for marker in ("tel", "fax", "e-mail", "email", "www.", "@", "http")
        ):
            return False
        if len(title.split()) > 14:
            return False
        return True

    for line in page_lines:
        title = clean_heading_title(line)
        if title.lower().startswith("table ") and valid_title(title):
            return title
    if rows:
        first_row = [cell for cell in rows[0] if cell]
        if first_row:
            title = clean_heading_title(first_row[0])
            if valid_title(title):
                return title
    for line in page_lines[:8]:
        title = clean_heading_title(line)
        if valid_title(title) and not HEADING_RE.match(title):
            return title
    return fallback
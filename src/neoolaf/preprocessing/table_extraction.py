from __future__ import annotations

import re
from html import escape

from neoolaf.preprocessing.structural_detection import (
    HEADING_RE,
    REVISION_RE,
    clean_heading_title,
)


# ── Cell / row helpers ──────────────────────────────────────────────────────


def clean_cell(value) -> str:
    """Normalize whitespace inside one table cell."""
    if value is None:
        return ""
    text = str(value).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def row_text(row: list[str]) -> str:
    """Join a row into one comparable string."""
    return " ".join(cell for cell in row if cell).strip()


def noise_row(row: list[str]) -> bool:
    """Detect rows that are only revision or layout noise."""
    text = row_text(row)
    return bool(text and REVISION_RE.search(text))


# ── Table quality filter ────────────────────────────────────────────────────


def good_table(rows: list[list[str]]) -> bool:
    """Filter out poor-quality tables."""
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


# ── Page table extraction ───────────────────────────────────────────────────


def tables(page) -> list[list[list[str]]]:
    """Extract valid tables from one page."""
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
    """High-level entrypoint for extracting valid tables from one page."""
    return tables(page)


# ── HTML conversion ─────────────────────────────────────────────────────────


def html_table(rows: list[list[str]]) -> str:
    """Convert extracted rows into deterministic HTML."""
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


# ── Table title detection ───────────────────────────────────────────────────


def table_title(rows: list[list[str]], page_lines: list[str], fallback: str) -> str:
    """Pick the best title for one extracted table."""
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

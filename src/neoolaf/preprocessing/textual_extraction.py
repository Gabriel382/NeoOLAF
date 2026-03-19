from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from html import escape
from pathlib import Path

import pdfplumber

from neoolaf.preprocessing.cleaners import (
    finalize_extracted_document,
    normalize_compare,
    normalize_line,
    strip_chapter_lines,
    strip_repeated_title,
)


FOOTER_PATTERNS = [
    re.compile(r"^\s*Page\s+\d+(?:[-/]\d+)?\s+SPRINT.*$", re.IGNORECASE),
    re.compile(r"^\s*SPRINT.*\s+Page\s+\d+(?:[-/]\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
]
CHAPTER_RE = re.compile(
    r"^(?:chapitre|chapter|capitolo)\s+(?P<number>\d+)\s*$", re.IGNORECASE
)
CAP_RE = re.compile(r"^(?:cap\.?|chapitre)\s*\.?\s*(?P<number>\d+)\b", re.IGNORECASE)
TOP_NUMBER_RE = re.compile(r"^(?P<number>\d+)\s+(?P<title>.+?)\s*$")
NUMBERED_RE = re.compile(r"^(?P<number>\d+(?:\.\d+){1,3})\s+(?P<title>.+?)\s*$")
DOTS_RE = re.compile(r"\.{3,}")
END_PAGE_RE = re.compile(r"\s+\d+(?:-\d+)?\s*$")
PAGE_REF_RE = re.compile(r"\b\d+-\d+\b")
ADDRESS_RE = re.compile(
    r"(?:www\.|@|tel\.?|fax|via\s+|24030|brembate)", re.IGNORECASE
)
REVISION_RE = re.compile(
    r"revised for|sheet\s+\d+/\d+|admintool|a-\d+[a-z]?", re.IGNORECASE
)
SPRINT_RE = re.compile(r"SPRINT", re.IGNORECASE)


@dataclass(frozen=True)
class Word:
    """One extracted word with its page coordinates."""
    top: float
    bottom: float
    x0: float
    x1: float
    text: str


def slug(text: str) -> str:
    """Convert a file stem into a stable JSON key."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "document"


def score(first: str, second: str) -> float:
    """Return a fuzzy similarity score for two strings."""
    return SequenceMatcher(
        None, normalize_compare(first), normalize_compare(second)
    ).ratio()


# ── Page-level helpers ───────────────────────────────────────────────────────


def remove_footers(lines: list[str]) -> list[str]:
    """Remove lines that look like page footers."""
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if any(pattern.match(stripped) for pattern in FOOTER_PATTERNS):
            continue
        kept.append(stripped)
    return kept


def split_lines(text: str) -> list[str]:
    """Split extracted page text into cleaned lines."""
    return remove_footers(text.splitlines())


def join_lines(lines: list[str]) -> str:
    """Join non-empty lines into one text block."""
    return "\n".join(line for line in lines if line).strip()


def keep_word(word, page_height: float) -> bool:
    """Keep only body words and ignore header/footer words."""
    top = float(word.get("top", 0.0))
    bottom = float(word.get("bottom", top))
    text = str(word.get("text", "")).strip()
    return bool(text) and top >= 70 and bottom <= page_height - 45


def body_lines(page) -> list[str]:
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
        if keep_word(word, page.height)
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
    return remove_footers(lines)


def extract_page_lines(page, body_only: bool = False) -> list[str]:
    """High-level entrypoint for extracting cleaned lines from one page."""
    if body_only:
        return body_lines(page)
    return split_lines(page.extract_text() or "")


def extract_page_text(page, body_only: bool = False) -> str:
    """High-level entrypoint for extracting cleaned text from one page."""
    return join_lines(extract_page_lines(page, body_only=body_only))


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


def clean_title(text: str) -> str:
    """Remove page references and dot leaders from a title."""
    text = DOTS_RE.split(text)[0]
    text = END_PAGE_RE.sub("", text)
    return text.strip(" .:-|")


def toc_page(lines: list[str]) -> bool:
    """Detect whether a page is a table-of-contents page."""
    if not lines:
        return False
    head = " ".join(line.lower() for line in lines[:12] if line)
    dot_lines = sum(1 for line in lines if DOTS_RE.search(line))
    chapter_lines = sum(1 for line in lines if CHAPTER_RE.match(line.strip()))
    page_refs = sum(len(PAGE_REF_RE.findall(line)) for line in lines[:20])
    return (
        "sommaire" in head
        or "table des mat" in head
        or dot_lines >= 4
        or chapter_lines >= 3
        or page_refs >= 8
    )


def chapter_heading(lines: list[str]) -> dict | None:
    """Extract a chapter heading from the current page if present."""
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = CHAPTER_RE.match(stripped)
        if match:
            for next_index in range(index + 1, len(lines)):
                title = clean_title(lines[next_index].strip())
                if title and not CHAPTER_RE.match(title):
                    return {"number": match.group("number"), "title": title}
            return {
                "number": match.group("number"),
                "title": f"Chapitre {match.group('number')}",
            }

        short = CAP_RE.match(stripped)
        if short:
            if index + 1 < len(lines):
                title = clean_title(lines[index + 1].strip())
                if title and not NUMBERED_RE.match(title):
                    return {"number": short.group("number"), "title": title}
            return {
                "number": short.group("number"),
                "title": f"Chapitre {short.group('number')}",
            }
    return None


# ── TOC and chapter matching ─────────────────────────────────────────────────


def toc_chapters(pdf) -> OrderedDict:
    """Read chapter titles from early TOC pages."""
    chapters = OrderedDict()
    for page in pdf.pages[: min(len(pdf.pages), 12)]:
        lines = extract_page_lines(page)
        if not toc_page(lines):
            continue
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            match = CHAPTER_RE.match(line)
            if match:
                title = (
                    clean_title(lines[index + 1].strip())
                    if index + 1 < len(lines)
                    else ""
                )
                if title and match.group("number") not in chapters:
                    chapters[match.group("number")] = title
                index += 2
                continue
            index += 1
    return chapters


def match_chapter(
    lines: list[str], chapters: OrderedDict, current: str | None
) -> str | None:
    """Match the current page to the best chapter from the TOC."""
    if not chapters:
        return None
    top_lines = [clean_title(line) for line in lines[:8] if line.strip()]
    joined = " | ".join(top_lines)
    best_key = None
    best_score = 0.0

    for line in top_lines[:4]:
        match = TOP_NUMBER_RE.match(line)
        if not match:
            continue
        key = match.group("number")
        if key not in chapters or key == current:
            continue
        if score(match.group("title"), chapters[key]) >= 0.55:
            return key

    for key, title in chapters.items():
        for line in top_lines:
            value = score(line, title)
            if value > best_score:
                best_key = key
                best_score = value
        value = score(joined, f"{key} {title}")
        if value > best_score:
            best_key = key
            best_score = value

    if best_score >= 0.72 and best_key != current:
        return best_key
    return None


# ── Table extraction ─────────────────────────────────────────────────────────


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


# ── Document-level helpers ───────────────────────────────────────────────────


def doc_title(pdf, pdf_path: Path) -> str:
    """Pick a document title from the first pages."""
    sprint = []
    choices = []
    for page in pdf.pages[:3]:
        lines = split_lines(page.extract_text() or "")
        for line in lines:
            raw = line.strip()
            title = clean_title(raw)
            if not title or len(title) < 3:
                continue
            if SPRINT_RE.search(raw):
                sprint.append(title)
            if ADDRESS_RE.search(title):
                continue
            choices.append(title)
        if sprint:
            return sprint[0][:200]
        if choices:
            break
    return (choices[0] if choices else pdf_path.stem)[:200]


def family(pdf) -> str:
    """Classify a PDF as manual, table-heavy, or sparse."""
    chapter_pages = 0
    table_pages = 0
    table_count = 0
    text_size = 0
    toc_pages = 0
    for page in pdf.pages[: min(len(pdf.pages), 15)]:
        lines = extract_page_lines(page)
        text_size += len(join_lines(lines))
        if toc_page(lines):
            toc_pages += 1
        if chapter_heading(lines):
            chapter_pages += 1
        page_tables = extract_page_tables(page)
        if page_tables:
            table_pages += 1
            table_count += len(page_tables)
    if text_size < 120 and table_count == 0:
        return "sparse"
    if chapter_pages == 0 and table_pages >= 8 and table_count >= 12:
        return "table"
    if chapter_pages >= 1 or toc_pages >= 1:
        return "manual"
    if table_pages >= 2 or table_count >= 4:
        return "table"
    if text_size < 500:
        return "sparse"
    return "table"


def classify_textual_pdf(pdf) -> str:
    """High-level entrypoint for classifying a textual PDF family."""
    return family(pdf)


def table_title(rows: list[list[str]], page_lines: list[str], fallback: str) -> str:
    """Pick the best title for one extracted table."""
    def valid_title(title: str) -> bool:
        if not title:
            return False
        if len(title) < 8 or len(title) > 80:
            return False
        if title in {"VER/", "M"}:
            return False
        lowered = title.lower()
        if any(
            marker in lowered
            for marker in ("via ", "tel", "fax", "e-mail", "email", "www.", "24030", "brembate", "@", "cliente", "client")
        ):
            return False
        if title.count(",") >= 1 or title.count(":") >= 1:
            return False
        if len(title.split()) > 14:
            return False
        return True

    for line in page_lines:
        title = clean_title(line)
        if title.lower().startswith("table ") and valid_title(title):
            return title
    if rows:
        first_row = [cell for cell in rows[0] if cell]
        if first_row:
            title = clean_title(first_row[0])
            if valid_title(title):
                return title
    for line in page_lines[:8]:
        title = clean_title(line)
        if (
            valid_title(title)
            and not CHAPTER_RE.match(title)
            and not CAP_RE.match(title)
        ):
            return title
    return fallback


# ── Section assembly helpers ─────────────────────────────────────────────────


def append_text(section: dict, text: str) -> None:
    """Append text to a section while keeping order."""
    text = text.strip()
    if not text:
        return
    section["contenu"] = (
        f"{section['contenu']}\n{text}".strip() if section["contenu"] else text
    )


def ensure_section(chapter: dict, key: str, title: str, page: int) -> dict:
    """Create a section if it does not exist yet."""
    if key not in chapter["sections"]:
        chapter["sections"][key] = {
            "titre": title,
            "page": page,
            "contenu": "",
            "sous_sections": OrderedDict(),
        }
    return chapter["sections"][key]


def add_table(
    section: dict, key: str, title: str, page: int, rows: list[list[str]]
) -> None:
    """Attach one table subsection to a section."""
    section["sous_sections"][key] = {
        "titre": title,
        "page": page,
        "table_html": html_table(rows),
    }


def prune_sections(chapter: dict) -> dict:
    """Remove empty sections and keep stable ordering."""
    kept = []
    for key, section in chapter["sections"].items():
        if section["contenu"].strip() or section["sous_sections"]:
            kept.append((key, section))
    kept = kept or list(chapter["sections"].items())
    kept.sort(key=lambda item: int(item[0]) if str(item[0]).isdigit() else 9999)
    chapter["sections"] = OrderedDict(kept)
    return chapter


# ── Family-specific extraction strategies ────────────────────────────────────


def _extract_manual(pdf, pdf_path: Path) -> dict:
    """Extract a manual-style PDF into the target schema."""
    chapters = toc_chapters(pdf)
    title = doc_title(pdf, pdf_path)
    chapter = {"numero": "1", "titre": title, "sections": OrderedDict()}
    current = None
    table_counts: dict[str, int] = {}

    for page_number, page in enumerate(pdf.pages, start=1):
        lines = extract_page_lines(page, body_only=True)
        if not join_lines(lines):
            continue
        if toc_page(lines) or local_index(lines):
            continue

        explicit = chapter_heading(lines)
        matched = match_chapter(lines, chapters, current)
        next_key = None
        next_title = None

        if explicit and (not chapters or explicit["number"] in chapters):
            next_key = explicit["number"]
            next_title = chapters.get(next_key, explicit["title"])
        elif matched:
            next_key = matched
            next_title = chapters[matched]

        if next_key is not None:
            current = next_key
            ensure_section(chapter, current, next_title, page_number)

        if current is None:
            if chapters and page_number > 8:
                current = next(iter(chapters))
                ensure_section(chapter, current, chapters[current], page_number)
            else:
                current = "1"
                ensure_section(chapter, current, title, page_number)

        section = chapter["sections"][current]
        page_tables = extract_page_tables(page)

        text_lines = strip_chapter_lines(lines, CHAPTER_RE)
        text_lines = strip_repeated_title(text_lines, current, section["titre"], score)
        text = join_lines(text_lines)
        if text:
            append_text(section, text)

        if page_tables:
            count = table_counts.get(current, 0)
            for rows in page_tables:
                count += 1
                add_table(
                    section,
                    f"{current}.{count}",
                    table_title(rows, text_lines, f"Table {count}"),
                    page_number,
                    rows,
                )
            table_counts[current] = count

    if not chapter["sections"]:
        chapter["sections"]["1"] = {
            "titre": title,
            "page": 1,
            "contenu": title,
            "sous_sections": OrderedDict(),
        }
    return prune_sections(chapter)


def _extract_table_doc(pdf, pdf_path: Path) -> dict:
    """Extract a table-heavy PDF into the target schema."""
    title = doc_title(pdf, pdf_path)
    chapter = {
        "numero": "1",
        "titre": title,
        "sections": OrderedDict(
            {
                "1": {
                    "titre": title,
                    "page": 1,
                    "contenu": "",
                    "sous_sections": OrderedDict(),
                }
            }
        ),
    }
    section = chapter["sections"]["1"]
    count = 0
    for page_number, page in enumerate(pdf.pages, start=1):
        lines = extract_page_lines(page)
        text = join_lines(lines)
        page_tables = extract_page_tables(page)
        if text and (not page_tables or len(text) < 1200):
            append_text(section, text)
        for rows in page_tables:
            count += 1
            add_table(
                section,
                f"1.{count}",
                table_title(rows, lines, f"Table {count}"),
                page_number,
                rows,
            )
    if not section["contenu"]:
        section["contenu"] = title
    return chapter


def _extract_sparse(pdf, pdf_path: Path) -> dict:
    """Build a minimal structure for sparse PDFs."""
    title = doc_title(pdf, pdf_path)
    section = {
        "titre": title,
        "page": 1,
        "contenu": title,
        "sous_sections": OrderedDict(),
    }
    return {"numero": "1", "titre": title, "sections": OrderedDict({"1": section})}


def extract_textual_document_structure(pdf, pdf_path: Path) -> dict:
    """High-level entrypoint for extracting one textual PDF into the target schema."""
    kind = classify_textual_pdf(pdf)
    if kind == "manual":
        return _extract_manual(pdf, pdf_path)
    if kind == "table":
        return _extract_table_doc(pdf, pdf_path)
    return _extract_sparse(pdf, pdf_path)


# ── Public API ───────────────────────────────────────────────────────────────


def extract_textual_pdf(pdf_path: str) -> dict:
    """
    Extract a textual PDF into a structured, cleaned document dict.

    Classifies the PDF by family (manual / table-heavy / sparse) and
    applies the appropriate extraction strategy.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A JSON-serializable dict with the extracted document structure.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(path) as pdf:
        chapter = extract_textual_document_structure(pdf, path)
        return finalize_extracted_document(
            OrderedDict({f"document_{slug(path.stem)}": chapter})
        )

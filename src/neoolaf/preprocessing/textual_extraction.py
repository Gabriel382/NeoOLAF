from __future__ import annotations

import re
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path

import pdfplumber

from neoolaf.preprocessing.cleaners import (
    finalize_extracted_document,
    normalize_compare,
    strip_chapter_lines,
    strip_repeated_title,
    table_html_to_text,
)
from neoolaf.preprocessing.page_extraction import (
    extract_page_lines,
    join_lines,
    local_index,
    split_lines,
    toc_page,
)
from neoolaf.preprocessing.structural_detection import (
    CONTACT_RE,
    HEADING_RE,
    NUMBERED_SECTION_RE,
    DocumentStructure,
    analyze_document,
    clean_heading_title,
    detect_chapter_heading,
    detect_headings_by_font,
)
from neoolaf.preprocessing.table_extraction import (
    extract_page_tables,
    html_table,
    table_title,
)


# ── Generic patterns ─────────────────────────────────────────────────────────

TOP_NUMBER_RE = re.compile(r"^(?P<number>\d+)\s+(?P<title>.+?)\s*$")


def slug(text: str) -> str:
    """
    Convert a file stem into a stable JSON key.

    Lowercases the input, replaces all non-alphanumeric character runs
    with underscores, and strips leading and trailing underscores.
    Falls back to ``"document"`` if the result is empty.
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "document"


def score(first: str, second: str) -> float:
    """
    Return a fuzzy similarity score for two strings.

    Normalizes both inputs via ``normalize_compare`` before passing them
    to ``SequenceMatcher``, so the score is case- and punctuation-agnostic.

    Args:
        first:
            First string to compare.
        second:
            Second string to compare.

    Returns:
        Similarity ratio in [0.0, 1.0].
    """
    return SequenceMatcher(
        None, normalize_compare(first), normalize_compare(second)
    ).ratio()


def clean_title(text: str) -> str:
    """
    Remove page references and dot leaders from a title.

    Delegates to ``clean_heading_title`` from the structural detection
    module.
    """
    return clean_heading_title(text)


# ── TOC and chapter matching ─────────────────────────────────────────────────


def chapter_heading(
    lines: list[str],
    page_headings: list[dict] | None = None,
) -> dict | None:
    """
    Extract a chapter heading from the current page if present.

    Delegates to ``detect_chapter_heading`` from the structural detection
    module, passing through both text lines and optional font-based
    heading candidates.

    Args:
        lines:
            Lines extracted from the current page.
        page_headings:
            Optional list of font-based heading dicts as returned by
            ``detect_headings_by_font``.

    Returns:
        Dict with keys ``number`` and ``title`` if a heading is found,
        or None otherwise.
    """
    return detect_chapter_heading(lines, page_headings)


def toc_chapters(
    pdf,
    structure: DocumentStructure | None = None,
) -> OrderedDict:
    """
    Read chapter titles from early TOC pages.

    Scans the first 12 pages for TOC pages, then extracts chapter numbers
    and titles using ``HEADING_RE`` (multilingual keyword pattern) and
    ``NUMBERED_SECTION_RE`` (numeric section pattern). Only top-level
    sections (no dot in the number) are retained.

    Args:
        pdf:
            An open pdfplumber PDF object.
        structure:
            Optional document-level structure passed to ``extract_page_lines``
            for header/footer removal.

    Returns:
        ``OrderedDict`` mapping section number strings to title strings,
        in the order they appear in the TOC.
    """
    chapters = OrderedDict()
    for page in pdf.pages[: min(len(pdf.pages), 12)]:
        lines = extract_page_lines(page, structure=structure)
        if not toc_page(lines):
            continue
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            match = HEADING_RE.match(line)
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
            num_match = NUMBERED_SECTION_RE.match(line)
            if num_match:
                number = num_match.group("number")
                title = clean_title(num_match.group("title"))
                if title and number not in chapters and "." not in number:
                    chapters[number] = title
            index += 1
    return chapters


def match_chapter(
    lines: list[str], chapters: OrderedDict, current: str | None
) -> str | None:
    """
    Match the current page to the best chapter from the TOC.

    Applies two strategies in order:

    1. Exact number match — if one of the first four page lines contains a
       leading number that exists in ``chapters`` (excluding ``current``)
       and its title fuzzy-scores ≥ 0.55 against the known chapter title,
       that chapter key is returned immediately.
    2. Best fuzzy match — scores all chapter titles against all top lines
       and the joined line string; returns the best key if its score is
       ≥ 0.72 and the key differs from ``current``.

    Returns None when ``chapters`` is empty or no candidate meets the
    threshold.

    Args:
        lines:
            Lines extracted from the current page.
        chapters:
            TOC chapter map as returned by ``toc_chapters``.
        current:
            Chapter key currently active, excluded from matching to avoid
            false re-assignments.

    Returns:
        Matched chapter key string, or None if no match qualifies.
    """
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


# ── Document-level helpers ───────────────────────────────────────────────────


def doc_title(
    pdf,
    pdf_path: Path,
    structure: DocumentStructure | None = None,
) -> str:
    """
    Pick a document title from the first pages using font-based detection.

    When a ``DocumentStructure`` is provided, attempts font-based heading
    detection on the first three pages first. Falls back to scanning raw
    text lines from the first three pages for the first non-contact,
    non-empty line. Uses the file stem as a last resort.

    Args:
        pdf:
            An open pdfplumber PDF object.
        pdf_path:
            Path to the PDF file, used as a fallback title source.
        structure:
            Optional document-level structure carrying the median font size
            needed for font-based heading detection.

    Returns:
        Cleaned title string truncated to 200 characters.
    """
    if structure:
        for page in pdf.pages[:3]:
            headings = detect_headings_by_font(page, structure.median_font_size)
            if headings:
                for h in headings:
                    title = clean_title(h["text"])
                    if title and len(title) >= 3 and not CONTACT_RE.search(title):
                        return title[:200]

    choices = []
    for page in pdf.pages[:3]:
        lines = split_lines(page.extract_text() or "", structure)
        for line in lines:
            raw = line.strip()
            title = clean_title(raw)
            if not title or len(title) < 3:
                continue
            if CONTACT_RE.search(title):
                continue
            choices.append(title)
        if choices:
            break
    return (choices[0] if choices else pdf_path.stem)[:200]


def family(
    pdf,
    structure: DocumentStructure | None = None,
) -> str:
    """
    Classify a PDF as ``"manual"``, ``"table"``, or ``"sparse"``.

    Samples up to 15 pages and counts chapter headings, table pages,
    TOC pages, heading pages, and total extracted text size. Classification
    rules applied in order:

    - ``"sparse"`` — total text under 120 characters and no tables found.
    - ``"table"`` — no chapter or heading pages, but ≥ 8 table pages and
      ≥ 12 tables total.
    - ``"manual"`` — at least one chapter page, one TOC page, or two
      heading pages.
    - ``"table"`` — at least two table pages or four tables total.
    - ``"sparse"`` — total text under 500 characters.
    - ``"table"`` — default fallback.

    Args:
        pdf:
            An open pdfplumber PDF object.
        structure:
            Optional document-level structure used for heading detection
            and page line extraction.

    Returns:
        Classification string: ``"manual"``, ``"table"``, or ``"sparse"``.
    """
    chapter_pages = 0
    table_pages = 0
    table_count = 0
    text_size = 0
    toc_pages = 0
    heading_pages = 0

    sample_count = min(len(pdf.pages), 15)
    for page in pdf.pages[:sample_count]:
        lines = extract_page_lines(page, structure=structure)
        text_size += len(join_lines(lines))
        if toc_page(lines):
            toc_pages += 1

        page_headings = None
        if structure:
            page_headings = detect_headings_by_font(page, structure.median_font_size)
            if page_headings:
                heading_pages += 1
        if chapter_heading(lines, page_headings):
            chapter_pages += 1

        page_tables = extract_page_tables(page)
        if page_tables:
            table_pages += 1
            table_count += len(page_tables)

    if text_size < 120 and table_count == 0:
        return "sparse"
    if chapter_pages == 0 and heading_pages == 0 and table_pages >= 8 and table_count >= 12:
        return "table"
    if chapter_pages >= 1 or toc_pages >= 1 or heading_pages >= 2:
        return "manual"
    if table_pages >= 2 or table_count >= 4:
        return "table"
    if text_size < 500:
        return "sparse"
    return "table"


def classify_textual_pdf(
    pdf,
    structure: DocumentStructure | None = None,
) -> str:
    """Public high-level entrypoint for classifying a textual PDF family."""
    return family(pdf, structure)


# ── Section assembly helpers ─────────────────────────────────────────────────


def append_text(section: dict, text: str) -> None:
    """
    Append text to a section while keeping order.

    Strips the input and no-ops on empty strings. Concatenates with a
    newline separator if the section already has content.

    Args:
        section:
            Section dict carrying a ``contenu`` key.
        text:
            Text to append.
    """
    text = text.strip()
    if not text:
        return
    section["contenu"] = (
        f"{section['contenu']}\n{text}".strip() if section["contenu"] else text
    )


def ensure_section(chapter: dict, key: str, title: str, page: int) -> dict:
    """
    Create a section if it does not exist yet.

    Inserts a new blank section dict under ``key`` in
    ``chapter["sections"]`` only when the key is absent, leaving any
    existing section untouched.

    Args:
        chapter:
            Chapter dict carrying a ``sections`` OrderedDict.
        key:
            Section number string used as the dict key.
        title:
            Human-readable section title stored on creation.
        page:
            Page number where the section was first detected.

    Returns:
        The section dict for ``key`` (newly created or pre-existing).
    """
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
    """
    Attach one table subsection to a section.

    Converts ``rows`` to HTML via ``html_table`` and stores the result
    under ``section["sous_sections"][key]``.

    Args:
        section:
            Section dict carrying a ``sous_sections`` OrderedDict.
        key:
            Subsection key string (e.g. ``"3.2"``).
        title:
            Human-readable table title.
        page:
            Page number where the table was found.
        rows:
            Cleaned table rows as returned by ``extract_page_tables``.
    """
    section["sous_sections"][key] = {
        "titre": title,
        "page": page,
        "table_html": html_table(rows),
    }


def append_block(
    blocks: list[dict],
    block_type: str,
    page: int,
    order: int,
    *,
    text: str = "",
    title: str = "",
    html: str = "",
    html_text: str = "",
    section_key: str | None = None,
    subsection_key: str | None = None,
) -> int:
    """
    Append one ordered content block and return the next order value.

    Constructs a block dict with a zero-padded ``block_id``, sets
    type-specific fields (``text`` for text blocks; ``title``, ``html``,
    and ``html_text`` for table blocks), and appends it to ``blocks``.

    Args:
        blocks:
            Mutable list of content block dicts to append to.
        block_type:
            ``"text"`` or ``"table"``.
        page:
            Page number where the block originates.
        order:
            Current sequential order value for the block.
        text:
            Text content, used when ``block_type`` is ``"text"``.
        title:
            Table title, used when ``block_type`` is ``"table"``.
        html:
            Table HTML string, used when ``block_type`` is ``"table"``.
        html_text:
            Plain-text representation of the table, used when
            ``block_type`` is ``"table"``.
        section_key:
            Optional section number string to attach as provenance.
        subsection_key:
            Optional subsection key string to attach as provenance.

    Returns:
        ``order + 1``, ready for the next block.
    """
    payload = {
        "block_id": f"block_{order:05d}",
        "type": block_type,
        "page": page,
        "order": order,
    }
    if section_key is not None:
        payload["section_key"] = section_key
    if subsection_key is not None:
        payload["subsection_key"] = subsection_key
    if block_type == "text":
        payload["text"] = text
    elif block_type == "table":
        payload["title"] = title
        payload["html"] = html
        payload["html_text"] = html_text

    blocks.append(payload)
    return order + 1


def prune_sections(chapter: dict) -> dict:
    """
    Remove empty sections and keep stable ordering.

    Drops sections that have neither body text nor subsections. If all
    sections are empty, retains them all to avoid an empty chapter.
    Remaining sections are sorted numerically (non-numeric keys sort last
    at 9999).

    Args:
        chapter:
            Chapter dict carrying a ``sections`` OrderedDict.

    Returns:
        The same chapter dict with its ``sections`` replaced by a pruned,
        sorted ``OrderedDict``.
    """
    kept = []
    for key, section in chapter["sections"].items():
        if section["contenu"].strip() or section["sous_sections"]:
            kept.append((key, section))
    kept = kept or list(chapter["sections"].items())
    kept.sort(key=lambda item: int(item[0]) if str(item[0]).isdigit() else 9999)
    chapter["sections"] = OrderedDict(kept)
    return chapter


# ── Family-specific extraction strategies ────────────────────────────────────


def _extract_manual(
    pdf,
    pdf_path: Path,
    structure: DocumentStructure | None = None,
) -> dict:
    """
    Extract a manual-style PDF into the target schema.

    Reads the TOC for chapter titles, then iterates over all pages. Each
    page is matched to a chapter via ``chapter_heading`` and
    ``match_chapter``. Body text and tables are appended to the active
    section and recorded as ordered content blocks. TOC and local-index
    pages are skipped.

    Args:
        pdf:
            An open pdfplumber PDF object.
        pdf_path:
            Path to the PDF file, used for title detection.
        structure:
            Optional document-level structure used throughout extraction.

    Returns:
        Chapter dict conforming to the target schema, with pruned sections
        and an ordered ``content_blocks`` list.
    """
    chapters = toc_chapters(pdf, structure)
    title = doc_title(pdf, pdf_path, structure)
    chapter = {
        "numero": "1",
        "titre": title,
        "sections": OrderedDict(),
        "content_blocks": [],
    }
    current = None
    table_counts: dict[str, int] = {}
    block_order = 1

    for page_number, page in enumerate(pdf.pages, start=1):
        lines = extract_page_lines(page, body_only=True, structure=structure)
        if not join_lines(lines):
            continue
        if toc_page(lines) or local_index(lines):
            continue

        page_headings = None
        if structure:
            page_headings = detect_headings_by_font(page, structure.median_font_size)
        explicit = chapter_heading(lines, page_headings)
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

        text_lines = strip_chapter_lines(lines, HEADING_RE)
        text_lines = strip_repeated_title(text_lines, current, section["titre"], score)
        text = join_lines(text_lines)
        if text:
            append_text(section, text)
            block_order = append_block(
                chapter["content_blocks"],
                "text",
                page_number,
                block_order,
                text=text,
                section_key=current,
            )

        if page_tables:
            count = table_counts.get(current, 0)
            for rows in page_tables:
                count += 1
                table_key = f"{current}.{count}"
                table_name = table_title(rows, text_lines, f"Table {count}")
                html = html_table(rows)
                add_table(
                    section,
                    table_key,
                    table_name,
                    page_number,
                    rows,
                )
                block_order = append_block(
                    chapter["content_blocks"],
                    "table",
                    page_number,
                    block_order,
                    title=table_name,
                    html=html,
                    html_text=table_html_to_text(html),
                    section_key=current,
                    subsection_key=table_key,
                )
            table_counts[current] = count

    if not chapter["sections"]:
        chapter["sections"]["1"] = {
            "titre": title,
            "page": 1,
            "contenu": title,
            "sous_sections": OrderedDict(),
        }
        append_block(chapter["content_blocks"], "text", 1, block_order, text=title, section_key="1")
    return prune_sections(chapter)


def _extract_table_doc(
    pdf,
    pdf_path: Path,
    structure: DocumentStructure | None = None,
) -> dict:
    """
    Extract a table-heavy PDF into the target schema.

    Treats the entire document as a single section and iterates over all
    pages. Short text blocks (under 1 200 characters) are appended even
    when tables are present on the same page; longer text blocks on
    table-heavy pages are suppressed. All tables are recorded as ordered
    subsections and content blocks.

    Args:
        pdf:
            An open pdfplumber PDF object.
        pdf_path:
            Path to the PDF file, used for title detection.
        structure:
            Optional document-level structure used throughout extraction.

    Returns:
        Chapter dict conforming to the target schema with a single
        ``"1"`` section and an ordered ``content_blocks`` list.
    """
    title = doc_title(pdf, pdf_path, structure)
    chapter = {
        "numero": "1",
        "titre": title,
        "content_blocks": [],
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
    block_order = 1
    for page_number, page in enumerate(pdf.pages, start=1):
        lines = extract_page_lines(page, structure=structure)
        text = join_lines(lines)
        page_tables = extract_page_tables(page)
        if text and (not page_tables or len(text) < 1200):
            append_text(section, text)
            block_order = append_block(
                chapter["content_blocks"],
                "text",
                page_number,
                block_order,
                text=text,
                section_key="1",
            )
        for rows in page_tables:
            count += 1
            table_key = f"1.{count}"
            table_name = table_title(rows, lines, f"Table {count}")
            html = html_table(rows)
            add_table(
                section,
                table_key,
                table_name,
                page_number,
                rows,
            )
            block_order = append_block(
                chapter["content_blocks"],
                "table",
                page_number,
                block_order,
                title=table_name,
                html=html,
                html_text=table_html_to_text(html),
                section_key="1",
                subsection_key=table_key,
            )
    if not section["contenu"]:
        section["contenu"] = title
        append_block(chapter["content_blocks"], "text", 1, block_order, text=title, section_key="1")
    return chapter


def _extract_sparse(
    pdf,
    pdf_path: Path,
    structure: DocumentStructure | None = None,
) -> dict:
    """
    Build a minimal structure for sparse PDFs.

    Creates a single-section chapter whose body text and sole content
    block both contain only the document title. Used when the PDF yields
    insufficient text and no tables to warrant a richer extraction.

    Args:
        pdf:
            An open pdfplumber PDF object.
        pdf_path:
            Path to the PDF file, used for title detection.
        structure:
            Optional document-level structure passed to ``doc_title``.

    Returns:
        Minimal chapter dict conforming to the target schema.
    """
    title = doc_title(pdf, pdf_path, structure)
    section = {
        "titre": title,
        "page": 1,
        "contenu": title,
        "sous_sections": OrderedDict(),
    }
    return {
        "numero": "1",
        "titre": title,
        "content_blocks": [
            {
                "block_id": "block_00001",
                "type": "text",
                "page": 1,
                "order": 1,
                "section_key": "1",
                "text": title,
            }
        ],
        "sections": OrderedDict({"1": section}),
    }


def extract_textual_document_structure(
    pdf,
    pdf_path: Path,
    structure: DocumentStructure | None = None,
) -> dict:
    """
    High-level entrypoint for extracting one textual PDF into the target schema.

    Classifies the PDF via ``classify_textual_pdf`` and routes to
    ``_extract_manual``, ``_extract_table_doc``, or ``_extract_sparse``
    accordingly.

    Args:
        pdf:
            An open pdfplumber PDF object.
        pdf_path:
            Path to the PDF file.
        structure:
            Optional document-level structure passed to the chosen
            extraction strategy.

    Returns:
        Chapter dict conforming to the target schema.
    """
    kind = classify_textual_pdf(pdf, structure)
    if kind == "manual":
        return _extract_manual(pdf, pdf_path, structure)
    if kind == "table":
        return _extract_table_doc(pdf, pdf_path, structure)
    return _extract_sparse(pdf, pdf_path, structure)


# ── Public API ───────────────────────────────────────────────────────────────


def extract_textual_pdf(pdf_path: str) -> dict:
    """
    Extract a textual PDF into a structured, cleaned document dict.

    Analyzes document structure (margins, repeated headers/footers, font
    sizes) then classifies the PDF by family (manual / table-heavy /
    sparse) and applies the appropriate extraction strategy. The result
    is finalized via ``finalize_extracted_document`` before being returned.

    Args:
        pdf_path:
            Path to the PDF file.

    Returns:
        JSON-serializable dict with the extracted document structure,
        keyed by a slug derived from the file stem.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdfplumber.open(path) as pdf:
        structure = analyze_document(pdf)

        chapter = extract_textual_document_structure(pdf, path, structure)
        return finalize_extracted_document(
            OrderedDict({f"document_{slug(path.stem)}": chapter})
        )
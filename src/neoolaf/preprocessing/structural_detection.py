from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ── Multilingual TOC keywords ────────────────────────────────────────────────

TOC_KEYWORDS = [
    # French
    "sommaire",
    "table des matières",
    "table des matieres",
    # English
    "table of contents",
    "contents",
    # Spanish
    "tabla de contenidos",
    "índice",
    # German
    "inhaltsverzeichnis",
    # Italian
    "indice",
    "indice generale",
    # Generic
    "index",
]

# ── Generic page-number footer patterns ──────────────────────────────────────

GENERIC_FOOTER_PATTERNS = [
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),              # "3 / 10"
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),                # "- 5 -"
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),  # "Page 12"
    re.compile(r"^\s*\d+\s*$"),                         # bare page number
]

# ── Generic contact-info filter ──────────────────────────────────────────────

CONTACT_RE = re.compile(r"(?:www\.|@|tel\.?|fax|https?://)", re.IGNORECASE)

# ── Generic revision noise ───────────────────────────────────────────────────

REVISION_RE = re.compile(
    r"revised\s+for|sheet\s+\d+/\d+|rev\.?\s*\d+", re.IGNORECASE
)

# ── Multilingual chapter/section heading pattern ─────────────────────────────

HEADING_RE = re.compile(
    r"^(?:"
    r"chapitre|chapter|capitolo|kapitel|capítulo|capitulo|"
    r"partie|part|parte|teil|"
    r"section|sezione|sección|abschnitt|"
    r"cap\.?"
    r")\s*\.?\s*(?P<number>\d+)\s*$",
    re.IGNORECASE,
)

# ── Generic numbered section pattern ─────────────────────────────────────────

NUMBERED_SECTION_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+){0,3})\s+(?P<title>.+?)\s*$"
)

# ── Dot leader pattern (TOC indicator) ───────────────────────────────────────

DOTS_RE = re.compile(r"\.{3,}")

# ── Page reference pattern ───────────────────────────────────────────────────

PAGE_REF_RE = re.compile(r"\b\d+-\d+\b")


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class PageMargins:
    """
    Adaptive top and bottom margins computed from word positions.

    Attributes:
        top:
            Y-coordinate (in PDF points) where the body region begins.
        bottom_offset:
            Distance in PDF points from the page bottom where the body
            region ends.
    """

    top: float = 50.0
    bottom_offset: float = 40.0


@dataclass
class DocumentStructure:
    """
    Pre-computed structural information about a PDF.

    Produced once by ``analyze_document`` and consumed by extraction and
    cleaning helpers throughout the preprocessing pipeline.

    Attributes:
        margins:
            Adaptive page margins derived from word position distribution.
        repeated_header_lines:
            Normalized text of lines detected as repeating page headers.
        repeated_footer_lines:
            Normalized text of lines detected as repeating page footers.
        median_font_size:
            Most common font size across sampled pages, used as the
            baseline for heading detection.
    """

    margins: PageMargins = field(default_factory=PageMargins)
    repeated_header_lines: set[str] = field(default_factory=set)
    repeated_footer_lines: set[str] = field(default_factory=set)
    median_font_size: float = 10.0


# ── Margin detection ─────────────────────────────────────────────────────────


def detect_margins(pdf, sample_pages: int = 10) -> PageMargins:
    """
    Compute adaptive top and bottom margins from word position distribution.

    Samples up to ``sample_pages`` pages, collects the top and
    bottom Y-coordinates of every extracted word, and uses the 5th
    percentile of each distribution as the margin boundary. Results are
    clamped to [20, 120] pt (top) and [20, 100] pt (bottom).

    Args:
        pdf:
            An open pdfplumber PDF object.
        sample_pages:
            Maximum number of pages to sample for margin estimation.

    Returns:
        ``PageMargins`` instance with estimated top and bottom offsets.
        Falls back to default values if no words are found.
    """
    tops: list[float] = []
    bottoms: list[float] = []

    pages_to_check = min(sample_pages, len(pdf.pages))
    for page in pdf.pages[:pages_to_check]:
        words = page.extract_words()
        if not words:
            continue
        page_height = page.height
        for w in words:
            top = float(w.get("top", 0))
            bottom = float(w.get("bottom", top))
            tops.append(top)
            bottoms.append(page_height - bottom)

    if not tops:
        return PageMargins()

    tops.sort()
    bottoms.sort()

    # Use 5th percentile as the margin boundary
    top_margin = tops[max(0, len(tops) // 20)]
    bottom_margin = bottoms[max(0, len(bottoms) // 20)]

    # Clamp to reasonable ranges
    top_margin = max(20.0, min(top_margin, 120.0))
    bottom_margin = max(20.0, min(bottom_margin, 100.0))

    return PageMargins(top=top_margin, bottom_offset=bottom_margin)


# ── Repeated line detection (headers/footers) ────────────────────────────────


def _normalize_for_comparison(text: str) -> str:
    """
    Normalize text for repetition comparison.

    Lowercases the input, collapses whitespace, and replaces all digit
    sequences with ``#`` so that lines differing only in page numbers
    are treated as identical.
    """
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


def detect_repeated_lines(
    pdf,
    sample_pages: int = 15,
    min_ratio: float = 0.4,
) -> tuple[set[str], set[str]]:
    """
    Find header and footer text by detecting lines that repeat across pages.

    Examines the top three and bottom three lines of each sampled page.
    Lines whose normalized form appears on at least ``min_ratio`` of
    sampled pages are classified as repeated headers or footers
    respectively. Returns empty sets when fewer than three pages are
    available.

    Args:
        pdf:
            An open pdfplumber PDF object.
        sample_pages:
            Maximum number of pages to sample.
        min_ratio:
            Minimum fraction of sampled pages on which a line must appear
            to be considered a repeated header or footer.

    Returns:
        Tuple of ``(repeated_header_lines, repeated_footer_lines)`` — each
        a set of normalized text strings.
    """
    pages_to_check = min(sample_pages, len(pdf.pages))
    if pages_to_check < 3:
        return set(), set()

    top_lines: list[str] = []
    bottom_lines: list[str] = []

    for page in pdf.pages[:pages_to_check]:
        text = page.extract_text() or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            continue

        # Top 3 lines → potential headers
        for line in lines[:3]:
            top_lines.append(_normalize_for_comparison(line))

        # Bottom 3 lines → potential footers
        for line in lines[-3:]:
            bottom_lines.append(_normalize_for_comparison(line))

    threshold = int(pages_to_check * min_ratio)

    header_counts = Counter(top_lines)
    footer_counts = Counter(bottom_lines)

    headers = {text for text, count in header_counts.items() if count >= threshold and len(text) > 2}
    footers = {text for text, count in footer_counts.items() if count >= threshold and len(text) > 2}

    return headers, footers


# ── Font-based heading detection ─────────────────────────────────────────────


def compute_median_font_size(pdf, sample_pages: int = 10) -> float:
    """
    Compute the median (most common) font size across sampled pages.

    Collects font sizes from all words on up to ``sample_pages`` pages
    using pdfplumber's ``extra_attrs`` API. Returns the most frequent
    rounded size, or 10.0 if no size information is available.

    Args:
        pdf:
            An open pdfplumber PDF object.
        sample_pages:
            Maximum number of pages to sample.

    Returns:
        Most common font size as a float rounded to one decimal place,
        or 10.0 as a default fallback.
    """
    size_counts: Counter[float] = Counter()

    pages_to_check = min(sample_pages, len(pdf.pages))
    for page in pdf.pages[:pages_to_check]:
        try:
            words = page.extract_words(extra_attrs=["size"])
        except Exception:
            continue
        for w in words:
            size = w.get("size")
            if size is not None:
                size_counts[round(float(size), 1)] += 1

    if not size_counts:
        return 10.0

    return size_counts.most_common(1)[0][0]


def detect_headings_by_font(
    page,
    median_font_size: float,
    heading_ratio: float = 1.3,
) -> list[dict]:
    """
    Detect headings on a page by font size.

    Words whose font size exceeds ``median_font_size * heading_ratio``
    are treated as heading candidates. Adjacent large-font words sharing
    the same Y-coordinate (within 5 pt) are grouped into heading lines
    and sorted left-to-right.

    Args:
        page:
            A pdfplumber page object.
        median_font_size:
            Body-text baseline font size as returned by
            ``compute_median_font_size``.
        heading_ratio:
            Multiplier applied to ``median_font_size`` to set the minimum
            heading font size.

    Returns:
        List of heading dicts, each with keys ``text`` (str),
        ``font_size`` (float), and ``top`` (float). Returns an empty list
        if no large-font words are found or extraction fails.
    """
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
    except Exception:
        return []

    threshold = median_font_size * heading_ratio
    large_words = []
    for w in words:
        size = w.get("size")
        if size is not None and float(size) >= threshold:
            large_words.append({
                "text": str(w.get("text", "")).strip(),
                "size": float(size),
                "top": round(float(w.get("top", 0)), 1),
                "x0": float(w.get("x0", 0)),
            })

    if not large_words:
        return []

    # Group by Y position (same line)
    lines: list[list[dict]] = []
    current_line = [large_words[0]]
    current_top = large_words[0]["top"]

    for w in large_words[1:]:
        if abs(w["top"] - current_top) <= 5:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
            current_top = w["top"]
    lines.append(current_line)

    headings = []
    for line in lines:
        line.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line if w["text"])
        if text and len(text) >= 2:
            headings.append({
                "text": text,
                "font_size": max(w["size"] for w in line),
                "top": line[0]["top"],
            })

    return headings


# ── TOC detection ────────────────────────────────────────────────────────────


def detect_toc_by_structure(lines: list[str]) -> bool:
    """
    Detect whether a page is a table-of-contents page using
    language-agnostic structural patterns.

    Checks five independent signals in order of specificity:

    - Presence of a multilingual TOC keyword in the first 12 lines.
    - Four or more dot-leader lines (``...``).
    - Three or more multilingual chapter heading matches.
    - Eight or more page references in the first 20 lines.
    - Three or more numbered section TOC entries (``1.2 Title ... 45``).
    - Six or more lines ending with a trailing standalone page number.

    Returns True if any single signal threshold is met.

    Args:
        lines:
            Lines extracted from a candidate page.

    Returns:
        True if the page is identified as a table-of-contents page.
    """
    if not lines:
        return False

    head = " ".join(line.lower() for line in lines[:12] if line)

    # Check multilingual TOC keywords
    for keyword in TOC_KEYWORDS:
        if keyword in head:
            return True

    # Structural signals
    dot_lines = sum(1 for line in lines if DOTS_RE.search(line))
    page_refs = sum(len(PAGE_REF_RE.findall(line)) for line in lines[:20])

    # Count lines that look like TOC entries: "1.2.3 Title ........... 45"
    toc_entry_pattern = re.compile(
        r"^\s*\d+(?:\.\d+)*\s+.+?(?:\.{3,}\s*\d+|\s{3,}\d+)\s*$"
    )
    toc_entries = sum(1 for line in lines if toc_entry_pattern.match(line))

    # Count lines ending with a standalone page number
    trailing_number = re.compile(r".+\s{2,}\d{1,4}\s*$")
    trailing_nums = sum(1 for line in lines if trailing_number.match(line))

    heading_re_matches = sum(1 for line in lines if HEADING_RE.match(line.strip()))

    return (
        dot_lines >= 4
        or heading_re_matches >= 3
        or page_refs >= 8
        or toc_entries >= 3
        or trailing_nums >= 6
    )


# ── Chapter heading detection ────────────────────────────────────────────────


def detect_chapter_heading(
    lines: list[str],
    page_headings: list[dict] | None = None,
) -> dict | None:
    """
    Detect a chapter or section heading on a page.

    Applies two strategies in sequence:

    1. Text-pattern matching against ``HEADING_RE`` — if a match is found,
       the following non-heading lines are scanned for a title string.
    2. Font-based heading fallback via ``page_headings`` — looks for a
       large-font line matching ``NUMBERED_SECTION_RE`` at the top level
       (no dots in the number beyond the first character).

    Args:
        lines:
            Lines extracted from the page.
        page_headings:
            Optional list of font-based heading dicts as returned by
            ``detect_headings_by_font``. Used as a fallback when
            text-pattern matching finds no match.

    Returns:
        Dict with keys ``number`` (str) and ``title`` (str) if a heading
        is detected, or None otherwise.
    """
    # Try text-pattern matching with multilingual heading regex
    for i, line in enumerate(lines):
        stripped = line.strip()
        match = HEADING_RE.match(stripped)
        if match:
            number = match.group("number")
            # Look for the title on the next line
            for next_i in range(i + 1, min(i + 3, len(lines))):
                title = lines[next_i].strip()
                if title and not HEADING_RE.match(title):
                    title = clean_heading_title(title)
                    if title:
                        return {"number": number, "title": title}
            return {"number": number, "title": f"Section {number}"}

    # Try font-based headings: look for large-font numbered text
    if page_headings:
        for heading in page_headings:
            text = heading["text"].strip()
            match = NUMBERED_SECTION_RE.match(text)
            if match:
                number = match.group("number")
                title = match.group("title").strip()
                if title and "." not in number[1:]:  # top-level section
                    return {"number": number, "title": clean_heading_title(title)}

    return None


def clean_heading_title(text: str) -> str:
    """
    Remove dot leaders and trailing page numbers from a heading title.

    Splits on the first dot-leader sequence (three or more dots), takes
    the left fragment, strips trailing standalone page numbers, and
    trims surrounding punctuation and whitespace.

    Args:
        text:
            Raw heading title string, potentially containing dot leaders
            or a trailing page reference.

    Returns:
        Cleaned title string, or an empty string if nothing remains.
    """
    text = DOTS_RE.split(text)[0]
    text = re.sub(r"\s+\d+(?:-\d+)?\s*$", "", text)
    return text.strip(" .:-|")


# ── Full document structure analysis ─────────────────────────────────────────


def analyze_document(pdf) -> DocumentStructure:
    """
    Pre-compute structural information about a PDF for use during extraction.

    Runs margin detection, repeated-line detection, and median font size
    computation in sequence and returns the results as a single
    ``DocumentStructure`` instance. This function is intended to be called
    once when the PDF is opened; its output is then passed to all
    downstream extraction and cleaning helpers.

    Args:
        pdf:
            An open pdfplumber PDF object.

    Returns:
        ``DocumentStructure`` carrying adaptive margins, repeated header
        and footer line sets, and the median body font size.
    """
    margins = detect_margins(pdf)
    headers, footers = detect_repeated_lines(pdf)
    median_size = compute_median_font_size(pdf)

    return DocumentStructure(
        margins=margins,
        repeated_header_lines=headers,
        repeated_footer_lines=footers,
        median_font_size=median_size,
    )
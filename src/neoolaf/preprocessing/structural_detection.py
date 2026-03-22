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
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),          # "3 / 10"
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),             # "- 5 -"
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),  # "Page 12"
    re.compile(r"^\s*\d+\s*$"),                     # bare page number
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
    """Adaptive top/bottom margins computed from word positions."""

    top: float = 50.0
    bottom_offset: float = 40.0


@dataclass
class DocumentStructure:
    """Pre-computed structural information about a PDF."""

    margins: PageMargins = field(default_factory=PageMargins)
    repeated_header_lines: set[str] = field(default_factory=set)
    repeated_footer_lines: set[str] = field(default_factory=set)
    median_font_size: float = 10.0


# ── Margin detection ─────────────────────────────────────────────────────────


def detect_margins(pdf, sample_pages: int = 10) -> PageMargins:
    """
    Compute adaptive top/bottom margins from word position distribution.

    Looks at where words actually appear on the page and uses percentiles
    to determine where body content starts and ends.
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
    """Normalize text for repetition comparison."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


def detect_repeated_lines(
    pdf,
    sample_pages: int = 15,
    min_ratio: float = 0.4,
) -> tuple[set[str], set[str]]:
    """
    Find header/footer text by checking which lines repeat across pages
    in the top/bottom zones.

    Returns:
        (repeated_header_lines, repeated_footer_lines) — sets of normalized text.
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
    """Compute the median (most common) font size across sampled pages."""
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

    Words with font size > median * heading_ratio are heading candidates.
    Adjacent large-font words are grouped into heading lines.

    Returns:
        List of {"text": str, "font_size": float, "top": float}
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
    Detect a chapter/section heading on a page.

    First tries font-based headings (if available), then falls back
    to text-pattern matching with multilingual keywords.

    Returns:
        {"number": str, "title": str} or None
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
    """Remove dot leaders and trailing page numbers from a title."""
    text = DOTS_RE.split(text)[0]
    text = re.sub(r"\s+\d+(?:-\d+)?\s*$", "", text)
    return text.strip(" .:-|")


# ── Full document structure analysis ─────────────────────────────────────────


def analyze_document(pdf) -> DocumentStructure:
    """
    Pre-compute structural information about a PDF for use during extraction.

    This runs once when the PDF is opened and provides:
    - Adaptive margins
    - Repeated header/footer lines
    - Median font size
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

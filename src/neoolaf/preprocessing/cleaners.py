from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict

from bs4 import BeautifulSoup


from neoolaf.preprocessing.structural_detection import (
    GENERIC_FOOTER_PATTERNS as HEADER_PATTERNS,
    PAGE_REF_RE,
)

TOC_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+)+(?:\.?[^\n]*)?\s*\d+(?:-\d+)?\s*$")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
EMPTY_TABLE_HTML = {
    "<table><tr><td></td></tr></table>",
}
SOFT_HYPHENS = ("\u00ad", "\u200b", "\ufeff")


# ── Normalization helpers ────────────────────────────────────────────────────


def normalize_compare(text: str) -> str:
    """
    Normalize text before fuzzy comparisons.

    Lowercases the input, strips all non-alphanumeric characters, and
    collapses consecutive whitespace to a single space.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_line(text: str) -> str:
    """
    Normalize spacing inside one reconstructed line.

    Removes spurious spaces before punctuation, inside brackets, around
    slashes and hyphens, and collapses multiple consecutive spaces.
    """
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+([/%])", r"\1", text)
    text = re.sub(r"([/\-])\s+", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode variants into one consistent form.

    Applies NFKC normalization to unify compatibility equivalents such as
    ligatures, full-width characters, and typographic variants.
    """
    return unicodedata.normalize("NFKC", text)


# ── Structural line filters ──────────────────────────────────────────────────


def strip_chapter_lines(lines: list[str], chapter_re) -> list[str]:
    """
    Remove chapter heading lines from page content.

    Drops any line matched by ``chapter_re`` and the line immediately
    following it, which typically contains a redundant subtitle or blank.

    Args:
        lines:
            Raw lines extracted from a single page.
        chapter_re:
            Compiled regular expression matching chapter heading patterns.

    Returns:
        Filtered list of stripped lines with chapter headings removed.
    """
    cleaned = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        stripped = line.strip()
        if chapter_re.match(stripped):
            skip_next = True
            continue
        cleaned.append(stripped)
    return cleaned


def strip_repeated_title(
    lines: list[str], section_key: str, section_title: str, similarity_fn
) -> list[str]:
    """
    Remove a repeated section title at the top of a page.

    Pops leading lines that match or closely resemble the section title
    (plain or prefixed with ``section_key``), using both exact and
    fuzzy comparison via ``similarity_fn``.

    Args:
        lines:
            Lines from the top of a page.
        section_key:
            Numeric or alphanumeric section identifier (e.g. ``"3.2"``).
        section_title:
            Human-readable section title to match against.
        similarity_fn:
            Callable ``(str, str) -> float`` returning a similarity score
            in [0, 1]; lines scoring ≥ 0.95 are considered duplicates.

    Returns:
        Lines with the leading repeated title stripped.
    """
    cleaned = list(lines)
    target = normalize_compare(section_title)
    numbered_target = normalize_compare(f"{section_key} {section_title}")
    while cleaned:
        candidate = normalize_compare(cleaned[0])
        if not candidate:
            cleaned.pop(0)
            continue
        if (
            candidate == target
            or candidate == numbered_target
            or similarity_fn(candidate, numbered_target) >= 0.95
        ):
            cleaned.pop(0)
            continue
        break
    return cleaned


# ── Encoding and artifact repair ─────────────────────────────────────────────


def fix_mojibake(text: str) -> str:
    """
    Repair common UTF-8 decoded-as-Latin-1 mojibake when it is detected.

    Only attempts re-encoding when known mojibake markers are present, so
    correctly-encoded text is never touched.
    """
    markers = ("Ã", "â€™", "â€", "Â", "â€¢")
    if not any(marker in text for marker in markers):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired


def fix_whitespace(text: str) -> str:
    """
    Trim lines and collapse repeated blank lines.

    Strips leading/trailing horizontal whitespace from each line and reduces
    runs of two or more consecutive blank lines to a single blank line.
    """
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    result = []
    blanks = 0
    for line in lines:
        if not line:
            blanks += 1
            if blanks <= 1:
                result.append("")
            continue
        blanks = 0
        result.append(line)
    return "\n".join(result).strip()


def remove_soft_hyphens(text: str) -> str:
    """
    Remove soft hyphens and zero-width separators introduced by PDFs.

    Strips U+00AD (soft hyphen), U+200B (zero-width space), and U+FEFF
    (byte-order mark / zero-width no-break space).
    """
    for marker in SOFT_HYPHENS:
        text = text.replace(marker, "")
    return text


def fix_broken_words(text: str) -> str:
    """
    Join words split by line-break hyphenation.

    Detects patterns where a word character is followed by a hyphen and a
    newline before another word character, and merges them into one token.
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


# ── Glyph-level normalization ────────────────────────────────────────────────


def collapse_spaced_letters(text: str) -> str:
    """
    Collapse lines made of letter-by-letter spaced text.

    Detects lines where at least 70 % of alphabetic tokens are single
    letters (a PDF copy artefact) and joins them into a single word.
    """

    def collapse_line(line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return line
        tokens = stripped.split()
        alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
        single_alpha = [t for t in alpha_tokens if len(t) == 1 and t.isalpha()]
        if alpha_tokens and len(single_alpha) / len(alpha_tokens) >= 0.7 and len(single_alpha) >= 4:
            collapsed = "".join(tokens)
            return re.sub(r"([A-Za-zÀ-ÿ])([''])([A-Za-zÀ-ÿ])", r"\1\2\3", collapsed)
        return line

    return "\n".join(collapse_line(line) for line in text.splitlines())


def collapse_doubled_glyph_words(text: str) -> str:
    """
    Collapse words where most alphabetic glyphs are duplicated consecutively.

    Identifies tokens where at least 35 % of character positions form
    consecutive identical pairs (e.g. ``"iinnppuutt"`` → ``"input"``) and
    halves them in a single left-to-right pass.
    """

    def fix_token(token: str) -> str:
        letters = [ch for ch in token if ch.isalpha()]
        if len(letters) < 4:
            return token

        repeated_pairs = 0
        i = 0
        while i < len(token) - 1:
            if token[i].isalpha() and token[i] == token[i + 1]:
                repeated_pairs += 1
                i += 2
            else:
                i += 1

        if repeated_pairs / len(letters) < 0.35:
            return token

        collapsed = []
        i = 0
        while i < len(token):
            collapsed.append(token[i])
            if (
                i < len(token) - 1
                and token[i].isalpha()
                and token[i] == token[i + 1]
            ):
                i += 2
            else:
                i += 1
        return "".join(collapsed)

    return " ".join(fix_token(token) for token in text.split(" "))


def collapse_alternating_doubled_letters(text: str) -> str:
    """
    Collapse alternating doubled letters such as ``'MMaaiinn'`` to ``'Main'``.

    Targets purely alphabetic tokens of even length where at least 75 % of
    consecutive character pairs are identical, then takes every other character.
    """

    def fix_token(token: str) -> str:
        stripped = re.sub(r"[^A-Za-zÀ-ÿ]", "", token)
        if len(stripped) < 6 or len(stripped) % 2 != 0:
            return token

        pair_matches = sum(
            1 for i in range(0, len(stripped) - 1, 2) if stripped[i] == stripped[i + 1]
        )
        if pair_matches / max(len(stripped) // 2, 1) < 0.75:
            return token

        if re.fullmatch(r"[A-Za-zÀ-ÿ]+", token):
            return "".join(token[i] for i in range(0, len(token), 2))
        return token

    return " ".join(fix_token(token) for token in text.split(" "))


def repair_fused_words(text: str) -> str:
    """
    Detect and repair fused words generically.

    Looks for camelCase-like patterns in the middle of text that are
    likely two words stuck together (e.g. ``"ManualInstruction"`` →
    ``"Manual Instruction"``).
    """
    # Split camelCase fused words: "ManualInstruction" → "Manual Instruction"
    text = re.sub(r"([a-zà-ÿ])([A-ZÀ-Ý])", r"\1 \2", text)
    return text


# ── Header / footer / noise removal ─────────────────────────────────────────


def remove_header_footer_lines(text: str) -> str:
    """
    Remove common running headers and footers.

    Drops lines that match known header/footer patterns, standalone page
    numbers (up to three digits), and dense dot-leader page-reference lines
    typical of generated table-of-contents entries.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if any(pattern.match(stripped) for pattern in HEADER_PATTERNS):
            continue
        if stripped.isdigit() and len(stripped) <= 3:
            continue
        if PAGE_REF_RE.findall(stripped) and (
            stripped.count(".") >= 3 or len(PAGE_REF_RE.findall(stripped)) >= 2
        ):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def remove_toc_noise(text: str) -> str:
    """
    Remove table-of-contents fragments from normal text.

    Detects and drops dot-leader lines, numbered section lines matching
    ``TOC_LINE_RE``, lines with multiple page references, and null-byte
    artefacts left by some PDF extractors.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.upper() == "INDEX":
            continue
        if stripped.count(".") >= 8 and PAGE_REF_RE.search(stripped):
            continue
        if TOC_LINE_RE.match(stripped):
            continue
        if len(PAGE_REF_RE.findall(stripped)) >= 2:
            continue
        if PAGE_REF_RE.search(stripped) and stripped.count(".") >= 2:
            continue
        if "\x01" in stripped:
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def dedupe_adjacent_lines(text: str) -> str:
    """
    Remove directly repeated neighboring lines.

    Keeps the first occurrence of any non-empty line and silently drops
    the next line if it is identical after stripping.
    """
    output = []
    previous = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped == previous:
            continue
        output.append(line)
        if stripped:
            previous = stripped
    return "\n".join(output)


def remove_garbage_lines(text: str, min_alpha_ratio: float = 0.3) -> str:
    """
    Drop short low-signal lines dominated by symbols.

    Removes lines shorter than four characters or whose alphabetic character
    ratio falls below ``min_alpha_ratio``.

    Args:
        text:
            Multi-line input string.
        min_alpha_ratio:
            Minimum fraction of alphabetic characters required to keep a line.

    Returns:
        Cleaned multi-line string with garbage lines removed.
    """
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        alpha_count = sum(char.isalpha() for char in stripped)
        if len(stripped) < 4 or alpha_count / len(stripped) >= min_alpha_ratio:
            cleaned.append(line)
    return "\n".join(cleaned)


# ── Markup removal ───────────────────────────────────────────────────────────


def fix_text_artifacts(text: str) -> str:
    """
    Fix common text-layer extraction artifacts.

    Repairs OCR character confusions and normalizes em-dash and en-dash
    sequences that appear mid-word due to PDF glyph mapping issues.
    """
    text = fix_ocr_char_confusions(text)
    text = re.sub(r"(?<=\w)\s*---\s*(?=\w)", "-", text)
    text = re.sub(r"(?<=\w)\s*--\s*(?=\w)", "-", text)
    return text


def remove_markup(text: str) -> str:
    """
    Remove HTML, Markdown, and LaTeX-like fragments.

    Delegates to the three specialized removal helpers and returns the
    stripped result.
    """
    text = remove_html_artifacts(text)
    text = remove_latex_noise(text)
    text = remove_markdown_formatting(text)
    return text.strip()


# ── Plain-text cleanup pipeline ──────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Run the full cleanup pipeline for plain text.

    Applies all cleaning steps in order:
    control-character removal → soft-hyphen stripping → markup removal →
    mojibake repair → unicode normalization → glyph normalization →
    word repair → header/footer removal → TOC noise removal →
    artifact fixing → deduplication → garbage-line removal →
    whitespace normalization.
    """
    text = CONTROL_CHAR_RE.sub("", text)
    text = remove_soft_hyphens(text)
    text = remove_markup(text)
    text = fix_mojibake(text)
    text = normalize_unicode(text)
    text = collapse_doubled_glyph_words(text)
    text = collapse_alternating_doubled_letters(text)
    text = repair_fused_words(text)
    text = fix_broken_words(text)
    text = collapse_spaced_letters(text)
    text = remove_header_footer_lines(text)
    text = remove_toc_noise(text)
    text = fix_text_artifacts(text)
    text = dedupe_adjacent_lines(text)
    text = remove_garbage_lines(text)
    return fix_whitespace(text)


def clean_plain_text(text: str) -> str:
    """Public high-level entrypoint for plain-text cleanup."""
    return clean_text(text)


# ── HTML table cleanup ───────────────────────────────────────────────────────


def clean_html(html: str) -> str:
    """
    Clean text content inside extracted HTML tables.

    Parses the HTML with BeautifulSoup, applies ``clean_plain_text`` to
    every ``<th>`` and ``<td>`` cell, and serializes the result back to a
    string.
    """
    soup = BeautifulSoup(html, "html.parser")
    for cell in soup.find_all(["th", "td"]):
        cell.string = clean_plain_text(cell.get_text("\n"))
    return str(soup).strip()


def clean_table_html(html: str) -> str:
    """Public high-level entrypoint for extracted HTML table cleanup."""
    if html is None:
        return None
    return clean_html(html)


def table_html_to_text(html: str) -> str:
    """
    Flatten cleaned HTML table content into one plain-text string.

    Extracts cell text from every ``<th>`` and ``<td>``, normalizes
    internal whitespace, and joins non-empty cells with a single space.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    parts = []
    for cell in soup.find_all(["th", "td"]):
        text = cell.get_text(" ", strip=True)
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


# ── Document-tree cleanup ────────────────────────────────────────────────────


def clean_value(value):
    """
    Recursively clean all string fields in a document tree.

    Applies ``clean_table_html`` to keys named ``table_html`` or ``html``,
    and ``clean_plain_text`` to all other string values. Dicts and lists
    are traversed recursively; all other types are returned unchanged.
    """
    if isinstance(value, dict):
        cleaned = OrderedDict()
        for key, item in value.items():
            if key in {"table_html", "html"}:
                cleaned[key] = clean_table_html(item)
            else:
                cleaned[key] = clean_value(item)
        return cleaned
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, str):
        return clean_plain_text(value)
    return value


def drop_empty_tables(value):
    """
    Remove subsection entries that contain only empty table HTML.

    Traverses the document tree and drops any ``sous_sections`` entry
    whose ``table_html`` value is present in ``EMPTY_TABLE_HTML``.
    """
    if isinstance(value, dict):
        cleaned = OrderedDict()
        for key, item in value.items():
            if key == "sous_sections":
                kept = OrderedDict()
                for sub_key, sub_value in item.items():
                    html = sub_value.get("table_html", "") if isinstance(sub_value, dict) else ""
                    if html.strip() in EMPTY_TABLE_HTML:
                        continue
                    kept[sub_key] = drop_empty_tables(sub_value)
                cleaned[key] = kept
            else:
                cleaned[key] = drop_empty_tables(item)
        return cleaned
    if isinstance(value, list):
        return [drop_empty_tables(item) for item in value]
    return value


def finalize(value):
    """
    Apply all final postprocessing to the extracted document.

    Runs ``clean_value`` followed by ``drop_empty_tables`` on the full
    document tree.
    """
    return drop_empty_tables(clean_value(value))


def finalize_extracted_document(value):
    """Public high-level entrypoint for final document-tree cleanup."""
    return finalize(value)


# ── OCR-specific postprocessing ──────────────────────────────────────────────


def remove_html_artifacts(text: str) -> str:
    """
    Remove raw HTML and SVG tags from text.

    Strips ``<div>``, ``<svg>``, and any remaining HTML tags using
    successive regex passes.
    """
    text = re.sub(r"<div[^>]*>.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def remove_latex_noise(text: str) -> str:
    """
    Remove LaTeX math artifacts from text.

    Strips inline math expressions of the forms ``$\\cmd{arg}$``,
    ``$\\cmd$``, and short arbitrary ``$…$`` spans (up to 30 characters).
    """
    text = re.sub(r"\$\\[a-zA-Z]+\{[^}]*\}\$", "", text)
    text = re.sub(r"\$\\[a-zA-Z]+\$", "", text)
    text = re.sub(r"\$[^$]{1,30}\$", "", text)
    return text.strip()


def remove_markdown_formatting(text: str) -> str:
    """
    Remove Markdown formatting from text.

    Strips headings, horizontal rules, bold/italic markers, images, links,
    inline code, blockquotes, and unordered/ordered list prefixes.
    """
    text = re.sub(r"^#{1,6}[ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*---\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^[ \t]*>[ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*[-*+][ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\d+\.[ \t]+", "", text, flags=re.MULTILINE)
    return text


def fix_ocr_char_confusions(text: str) -> str:
    """
    Fix OCR character confusions.

    Corrects three common substitutions: digit ``0`` before a letter
    (→ ``O``), digit ``1`` before a lowercase letter (→ ``l``), and
    the sequence ``rn`` between lowercase letters (→ ``m``).
    """
    text = re.sub(r"\b0([a-zA-Z])", r"O\1", text)
    text = re.sub(r"\b1([a-z])", r"l\1", text)
    text = re.sub(r"(?<=[a-z])rn(?=[a-z])", "m", text)
    return text


def fix_common_ocr_errors(text: str) -> str:
    """
    Fix OCR character confusions and remove Markdown formatting artifacts.

    Combines ``fix_ocr_char_confusions`` and ``remove_markdown_formatting``
    into a single convenience call.
    """
    text = fix_ocr_char_confusions(text)
    text = remove_markdown_formatting(text)
    return text


def is_html_table(text: str) -> bool:
    """
    Return True if the text block is an HTML table.

    Checks whether the stripped text starts with the ``<table`` opening tag.
    """
    return text.strip().startswith("<table")


def postprocess_ocr_text(text: str) -> str:
    """
    Clean plain text blocks from OCR output while preserving HTML tables.

    Splits the input on newlines and processes each block independently:
    HTML table blocks are passed through unchanged; all other blocks go
    through artifact removal, soft-hyphen stripping, OCR error correction,
    and the full plain-text cleanup pipeline.
    """
    lines_out = []
    for block in text.split("\n"):
        if is_html_table(block):
            lines_out.append(block)
        else:
            cleaned = remove_html_artifacts(block)
            cleaned = remove_latex_noise(cleaned)
            cleaned = remove_soft_hyphens(cleaned)
            cleaned = fix_common_ocr_errors(cleaned)
            if cleaned:
                lines_out.append(clean_plain_text(cleaned))
    result = "\n".join(lines_out)
    result = remove_garbage_lines(result)
    return result


def clean_ocr_text_output(text: str) -> str:
    """Public high-level entrypoint for OCR plain-text cleanup."""
    return postprocess_ocr_text(text)


def postprocess_ocr_tables(tables: list) -> list:
    """
    Apply cell-level cleaning to every table in the OCR output.

    For each table dict, parses its ``html`` field with BeautifulSoup,
    cleans every ``<th>`` and ``<td>`` cell through the artifact-removal
    and OCR-correction helpers, then re-serializes via ``clean_table_html``.

    Args:
        tables:
            List of table dicts, each expected to contain an ``html`` key.

    Returns:
        New list of table dicts with cleaned ``html`` values.
    """
    cleaned = []
    for table in tables:
        html = table.get("html", "")
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for cell in soup.find_all(["th", "td"]):
                raw = cell.get_text()
                cell_text = remove_html_artifacts(raw)
                cell_text = remove_latex_noise(cell_text)
                cell_text = remove_soft_hyphens(cell_text)
                cell_text = fix_common_ocr_errors(cell_text)
                cell.string = cell_text
            html = clean_table_html(str(soup).strip())
        cleaned.append({**table, "html": html})
    return cleaned


def clean_ocr_table_output(tables: list) -> list:
    """Public high-level entrypoint for OCR table cleanup."""
    return postprocess_ocr_tables(tables)
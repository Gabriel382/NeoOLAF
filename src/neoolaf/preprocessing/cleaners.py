from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict

from bs4 import BeautifulSoup


PAGE_REF_RE = re.compile(r"\b\d+-\d+\b")
TOC_LINE_RE = re.compile(r"^\s*\d+(?:\.\d+)+(?:\.?[^\n]*)?\s*\d+(?:-\d+)?\s*$")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
HEADER_PATTERNS = [
    re.compile(r"^\s*CAP\.?\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*CHAPITRE\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*Page\s+\d+(?:-\d+)?\s+SPRINT.*$", re.IGNORECASE),
    re.compile(r"^\s*SPRINT.*\s+Page\s+\d+(?:-\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Manuel d['']emploi\s+SPRINT.*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*MANUEL D['']INSTRUCTIONS.*\d+(?:-\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Documentazione Meccanica\s*\d*\s*$", re.IGNORECASE),
    re.compile(r"^\s*[IVXLCDM]+\s+MANUEL.*SPRINT.*$", re.IGNORECASE),
    re.compile(r"^\s*Manuel d[?'']instruction\s+SPRINT.*\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*TABLEAU DES REVISIONS\s*$", re.IGNORECASE),
    re.compile(r"^\s*VER/\s*PARAGRAPHE.*$", re.IGNORECASE),
    re.compile(r"^\s*DATE\s*$", re.IGNORECASE),
    re.compile(r"^\s*SPRINT[\w.-]+.*$", re.IGNORECASE),
]
EMPTY_TABLE_HTML = {
    "<table><tr><td></td></tr></table>",
    "<table><tr><td>Aucune donnee</td></tr></table>",
}
SOFT_HYPHENS = ("\u00ad", "\u200b", "\ufeff")
BANNER_LINE_RE = re.compile(
    r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-–—|]{10,}(?:maintenance|utilisation|installation|sécurité|security)$",
    re.IGNORECASE,
)
FUSED_WORD_FIXES = {
    "Manueldeprogrammation": "Manuel de programmation",
    "withCNC-aveccommandenumÃ©rique-decontrolnumÃ©rico": "with CNC - avec commande numÃ©rique - de control numÃ©rico",
    "aveccommandenumérique": "avec commande numérique",
    "DocumentazioneMeccanica": "Documentazione Meccanica",
    "MECHANISCHEDOKUMENTATION": "MECHANISCHE DOKUMENTATION",
    "MECHANICALDOCUMENTATION": "MECHANICAL DOCUMENTATION",
    "DOCUMENTATIONMÉCANIQUE": "DOCUMENTATION MÉCANIQUE",
    "DOCUMENTACIÓNMECÁNICA": "DOCUMENTACIÓN MECÁNICA",
    "Qualificationdel": "Qualification de l",
    "Phasedelavietechniquedelamachine": "Phase de la vie technique de la machine",
}


def normalize_compare(text: str) -> str:
    """Normalize text before fuzzy comparisons."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_line(text: str) -> str:
    """Normalize spacing inside one reconstructed line."""
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+([/%])", r"\1", text)
    text = re.sub(r"([/\-])\s+", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def strip_chapter_lines(lines: list[str], chapter_re) -> list[str]:
    """Remove chapter heading lines from page content."""
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
    """Remove a repeated section title at the top of a page."""
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


def normalize_unicode(text: str) -> str:
    """Normalize unicode variants into one consistent form."""
    return unicodedata.normalize("NFKC", text)


def fix_mojibake(text: str) -> str:
    """Repair common UTF-8 decoded-as-Latin-1 mojibake when it is detected."""
    markers = ("Ã", "â€™", "â€", "Â", "â€¢")
    if not any(marker in text for marker in markers):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired


def fix_whitespace(text: str) -> str:
    """Trim lines and collapse repeated blank lines."""
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
    """Remove soft hyphens and zero-width separators introduced by PDFs."""
    for marker in SOFT_HYPHENS:
        text = text.replace(marker, "")
    return text


def fix_broken_words(text: str) -> str:
    """Join words split by line-break hyphenation."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_spaced_letters(text: str) -> str:
    """Collapse lines made of letter-by-letter spaced text."""

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
    """Collapse words where most alphabetic glyphs are duplicated consecutively."""

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
    """Collapse alternating doubled letters such as 'MMaaiinn' to 'Main'."""

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


def repair_known_fused_words(text: str) -> str:
    """Repair common fused words seen across the exported manuals."""
    for wrong, right in FUSED_WORD_FIXES.items():
        text = text.replace(wrong, right)
    return text


def remove_banner_lines(text: str) -> str:
    """Remove decorative cover-banner lines that are not useful for extraction."""
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        normalized = collapse_alternating_doubled_letters(stripped.replace("------", " "))
        lowered = normalized.lower()
        if "------" in stripped and (
            BANNER_LINE_RE.match(normalized)
            or (
                "installation" in lowered
                and "utilisation" in lowered
                and ("maintenance" in lowered or "sécurité" in lowered or "securite" in lowered)
            )
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def normalize_bullets(text: str) -> str:
    """Convert legacy bullet markers to one bullet style."""
    return re.sub(r"(?m)^D\s+", "• ", text)


def remove_header_footer_lines(text: str) -> str:
    """Remove common running headers and footers."""
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
    """Remove table-of-contents fragments from normal text."""
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
    """Remove directly repeated neighboring lines."""
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
    """Drop short low-signal lines dominated by symbols."""
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


def fix_text_artifacts(text: str) -> str:
    """Fix common text-layer extraction artifacts."""
    text = re.sub(r"\b0([a-zA-Z])", r"O\1", text)
    text = re.sub(r"\b1([a-z])", r"l\1", text)
    text = re.sub(r"(?<=\w)\s*---\s*(?=\w)", "-", text)
    text = re.sub(r"(?<=\w)\s*--\s*(?=\w)", "-", text)
    return text


def remove_markup(text: str) -> str:
    """Remove HTML, markdown, and latex-like fragments."""
    text = re.sub(r"<div[^>]*>.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\$\\[a-zA-Z]+\{[^}]*\}\$", "", text)
    text = re.sub(r"\$\\[a-zA-Z]+\$", "", text)
    text = re.sub(r"\$[^$]{1,30}\$", "", text)
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
    return text.strip()


def clean_text(text: str) -> str:
    """Run the full cleanup pipeline for plain text."""
    text = CONTROL_CHAR_RE.sub("", text)
    text = remove_soft_hyphens(text)
    text = remove_markup(text)
    text = fix_mojibake(text)
    text = normalize_unicode(text)
    text = collapse_doubled_glyph_words(text)
    text = collapse_alternating_doubled_letters(text)
    text = repair_known_fused_words(text)
    text = fix_broken_words(text)
    text = collapse_spaced_letters(text)
    text = normalize_bullets(text)
    text = remove_banner_lines(text)
    text = remove_header_footer_lines(text)
    text = remove_toc_noise(text)
    text = fix_text_artifacts(text)
    text = dedupe_adjacent_lines(text)
    text = remove_garbage_lines(text)
    return fix_whitespace(text)


def clean_plain_text(text: str) -> str:
    """Public high-level entrypoint for plain-text cleanup."""
    return clean_text(text)


def clean_html(html: str) -> str:
    """Clean text content inside extracted HTML tables."""
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
    """Flatten cleaned HTML table content into one plain-text string."""
    soup = BeautifulSoup(html or "", "html.parser")
    parts = []
    for cell in soup.find_all(["th", "td"]):
        text = cell.get_text(" ", strip=True)
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def clean_value(value):
    """Recursively clean all string fields in a document tree."""
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
    """Remove subsection entries that contain only empty table HTML."""
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
    """Apply all final postprocessing to the extracted document."""
    return drop_empty_tables(clean_value(value))


def finalize_extracted_document(value):
    """Public high-level entrypoint for final document-tree cleanup."""
    return finalize(value)


# ── OCR-specific postprocessing ──────────────────────────────────────────────


def remove_html_artifacts(text: str) -> str:
    """Remove raw HTML/SVG tags hallucinated by OCR models into plain text."""
    text = re.sub(r"<div[^>]*>.*?</div>", "", text, flags=re.DOTALL)
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def remove_latex_noise(text: str) -> str:
    """Remove LaTeX math artifacts injected by OCR models into plain text."""
    text = re.sub(r"\$\\[a-zA-Z]+\{[^}]*\}\$", "", text)
    text = re.sub(r"\$\\[a-zA-Z]+\$", "", text)
    text = re.sub(r"\$[^$]{1,30}\$", "", text)
    return text.strip()


def fix_common_ocr_errors(text: str) -> str:
    """Fix OCR character confusions and remove Markdown formatting artifacts."""
    text = re.sub(r"\b0([a-zA-Z])", r"O\1", text)
    text = re.sub(r"\b1([a-z])", r"l\1", text)
    text = re.sub(r"(?<=[a-z])rn(?=[a-z])", "m", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*---\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^\s*>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text


def is_html_table(text: str) -> bool:
    """Return True if the text block is an HTML table."""
    return text.strip().startswith("<table")


def postprocess_ocr_text(text: str) -> str:
    """Clean plain text blocks from OCR output while preserving HTML tables."""
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
    """Apply cell-level cleaning to every table in the OCR output."""
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

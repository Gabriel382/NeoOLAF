"""Text normalization utilities used by all matching code."""

from __future__ import annotations

import re
import unicodedata

_CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def split_camel_case(text: str) -> str:
    """Split CamelCase labels while preserving normal text."""
    text = _CAMEL_RE_1.sub(r"\1 \2", str(text))
    text = _CAMEL_RE_2.sub(r"\1 \2", text)
    return text


def strip_accents(text: str) -> str:
    """Remove accents without changing base letters."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: object, *, split_camel: bool = True) -> str:
    """Normalize text for robust fuzzy matching.

    The function is intentionally shared across all evaluation profiles so that
    strictness is controlled by thresholds and profile switches, not by multiple
    incompatible normalizers.
    """
    if text is None:
        return ""

    out = str(text)
    out = (
        out.replace("â", '"')
        .replace("â", '"')
        .replace("â", "'")
        .replace("â", " ")
        .replace("â", " ")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("—", " ")
        .replace("–", " ")
    )
    if split_camel:
        out = split_camel_case(out)
    out = strip_accents(out)
    out = out.lower().strip()
    out = re.sub(r"[_/\\:;|]+", " ", out)
    out = out.replace("-", " ")
    out = re.sub(r"[^\w\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def token_overlap_score(a: object, b: object) -> float:
    """Return token overlap score in [0, 100]."""
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return 100.0 * len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))

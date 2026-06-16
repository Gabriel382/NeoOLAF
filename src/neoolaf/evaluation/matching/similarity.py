"""Similarity functions with rapidfuzz fallback."""

from __future__ import annotations

from difflib import SequenceMatcher

from neoolaf.evaluation.matching.normalization import normalize_text

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback for minimal environments.
    fuzz = None


def ratio(a: object, b: object) -> float:
    """Character-level similarity in [0, 100]."""
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    if fuzz is not None:
        return float(fuzz.ratio(a_n, b_n))
    return 100.0 * SequenceMatcher(None, a_n, b_n).ratio()


def token_sort_ratio(a: object, b: object) -> float:
    """Token-sort similarity in [0, 100]."""
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_sort_ratio(a_n, b_n))
    return ratio(" ".join(sorted(a_n.split())), " ".join(sorted(b_n.split())))


def token_set_ratio(a: object, b: object) -> float:
    """Token-set similarity in [0, 100]."""
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(a_n, b_n))

    a_set = set(a_n.split())
    b_set = set(b_n.split())
    if not a_set or not b_set:
        return 0.0
    inter = " ".join(sorted(a_set & b_set))
    a_diff = " ".join(sorted(a_set - b_set))
    b_diff = " ".join(sorted(b_set - a_set))
    return max(ratio(inter, f"{inter} {a_diff}"), ratio(inter, f"{inter} {b_diff}"))


def loose_similarity(a: object, b: object) -> float:
    """Default loose similarity used by general evaluation."""
    return token_set_ratio(a, b)


def strict_similarity(a: object, b: object) -> float:
    """Conservative similarity used by strict XQuality matching."""
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 100.0
    return min(ratio(a_n, b_n), token_sort_ratio(a_n, b_n), token_set_ratio(a_n, b_n))

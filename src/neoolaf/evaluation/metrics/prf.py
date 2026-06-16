"""Precision / recall / F1 helpers."""

from __future__ import annotations

from neoolaf.evaluation.schema.metrics import PRF


def safe_div(a: float, b: float) -> float:
    """Safely divide two numbers."""
    return a / b if b != 0 else 0.0


def harmonic_f1(precision: float, recall: float) -> float:
    """Compute harmonic F1."""
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def make_prf(tp: int, fp: int, fn: int) -> PRF:
    """Build PRF from counts."""
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return PRF(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=harmonic_f1(precision, recall))

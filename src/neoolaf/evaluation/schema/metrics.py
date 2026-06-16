"""Metric dataclasses and helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PRF:
    """Precision, recall, F1 with counts."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MatchResult:
    """Greedy matching result."""

    prf: PRF
    matches: list[dict[str, Any]] = field(default_factory=list)
    unmatched_pred: list[Any] = field(default_factory=list)
    unmatched_gold: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

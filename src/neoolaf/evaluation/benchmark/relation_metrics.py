from __future__ import annotations

# Standard library imports
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple

# Third-party imports
from rapidfuzz import fuzz

# Local imports
from neoolaf.core.pipeline_state import PipelineState


@dataclass
class PRF:
    """Precision / Recall / F1 scores."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class RelationMetricsReport:
    """
    Benchmark evaluation of relation/triple extraction against a gold standard.
    """

    total_predicted: int = 0
    total_gold: int = 0

    # P/R/F1 for each matching mode
    strict: PRF = field(default_factory=PRF)
    fuzzy: PRF = field(default_factory=PRF)
    relaxed: PRF = field(default_factory=PRF)
    predicate_only: PRF = field(default_factory=PRF)

    # Detail lists (strict mode)
    matched_triples: List[List[str]] = field(default_factory=list)
    missed_triples: List[List[str]] = field(default_factory=list)
    extra_triples: List[List[str]] = field(default_factory=list)


@dataclass
class GoldTriple:
    """One gold triple from the annotation file."""
    subject: str
    predicate: str
    object: str


# ------------------------------------------------------------------
# Gold loading
# ------------------------------------------------------------------

def load_gold_triples(path: str) -> List[GoldTriple]:
    """
    Load gold triples from a JSON annotation file.

    Expected format:
    {
        "triples": [
            {"subject": "BearingFailure", "predicate": "causes", "object": "Overheating"},
            ...
        ]
    }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldTriple(subject=t["subject"], predicate=t["predicate"], object=t["object"])
        for t in data.get("triples", [])
    ]


# ------------------------------------------------------------------
# Normalization and matching
# ------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip().replace("_", " ").replace("-", " ")


def _fuzzy_eq(a: str, b: str, threshold: float) -> bool:
    """Check if two strings fuzzy-match above threshold (0-100)."""
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    score = max(
        fuzz.ratio(na, nb),
        fuzz.partial_ratio(na, nb),
        fuzz.token_sort_ratio(na, nb),
    )
    return score >= threshold


def _compute_prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return PRF(precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4))


# ------------------------------------------------------------------
# Matching helpers
# ------------------------------------------------------------------

def _strict_match(pred_s: str, pred_p: str, pred_o: str, gold: GoldTriple) -> bool:
    return (
        _normalize(pred_s) == _normalize(gold.subject)
        and _normalize(pred_p) == _normalize(gold.predicate)
        and _normalize(pred_o) == _normalize(gold.object)
    )


def _fuzzy_triple_match(
    pred_s: str, pred_p: str, pred_o: str, gold: GoldTriple, threshold: float,
) -> bool:
    return (
        _fuzzy_eq(pred_s, gold.subject, threshold)
        and _fuzzy_eq(pred_p, gold.predicate, threshold)
        and _fuzzy_eq(pred_o, gold.object, threshold)
    )


def _relaxed_match(
    pred_s: str, pred_p: str, pred_o: str, gold: GoldTriple, threshold: float,
) -> bool:
    """Fuzzy match ignoring direction (subject/object can be swapped)."""
    if _fuzzy_triple_match(pred_s, pred_p, pred_o, gold, threshold):
        return True
    # Try swapped direction
    return (
        _fuzzy_eq(pred_o, gold.subject, threshold)
        and _fuzzy_eq(pred_p, gold.predicate, threshold)
        and _fuzzy_eq(pred_s, gold.object, threshold)
    )


# ------------------------------------------------------------------
# Main computation
# ------------------------------------------------------------------

def compute_relation_metrics(
    state: PipelineState,
    gold_triples: List[GoldTriple],
    fuzzy_threshold: float = 85.0,
) -> RelationMetricsReport:
    """
    Compute relation/triple extraction metrics against a gold standard.

    Args:
        state:            PipelineState after Layer 5.
        gold_triples:     List of GoldTriple from annotation file.
        fuzzy_threshold:  Minimum rapidfuzz score (0-100) for fuzzy match.

    Returns:
        RelationMetricsReport with all metrics populated.
    """
    report = RelationMetricsReport()

    pred_triples = [
        (t.subject_label, t.predicate_label, t.object_label)
        for t in state.candidate_triples
    ]
    report.total_predicted = len(pred_triples)
    report.total_gold = len(gold_triples)

    # --- Strict ---
    tp_s, fp_s, fn_s = _match_triples(
        pred_triples, gold_triples,
        lambda s, p, o, g: _strict_match(s, p, o, g),
    )
    report.strict = _compute_prf(tp_s, fp_s, fn_s)

    # Build detail lists from strict matching
    matched_gold_indices: Set[int] = set()
    for s, p, o in pred_triples:
        for i, g in enumerate(gold_triples):
            if i not in matched_gold_indices and _strict_match(s, p, o, g):
                report.matched_triples.append([s, p, o])
                matched_gold_indices.add(i)
                break
        else:
            report.extra_triples.append([s, p, o])
    for i, g in enumerate(gold_triples):
        if i not in matched_gold_indices:
            report.missed_triples.append([g.subject, g.predicate, g.object])

    # --- Fuzzy ---
    tp_f, fp_f, fn_f = _match_triples(
        pred_triples, gold_triples,
        lambda s, p, o, g: _fuzzy_triple_match(s, p, o, g, fuzzy_threshold),
    )
    report.fuzzy = _compute_prf(tp_f, fp_f, fn_f)

    # --- Relaxed ---
    tp_r, fp_r, fn_r = _match_triples(
        pred_triples, gold_triples,
        lambda s, p, o, g: _relaxed_match(s, p, o, g, fuzzy_threshold),
    )
    report.relaxed = _compute_prf(tp_r, fp_r, fn_r)

    # --- Predicate only ---
    pred_predicates = [_normalize(p) for _, p, _ in pred_triples]
    gold_predicates = [_normalize(g.predicate) for g in gold_triples]
    pred_set = set(pred_predicates)
    gold_set = set(gold_predicates)
    tp_p = len(pred_set & gold_set)
    fp_p = len(pred_set - gold_set)
    fn_p = len(gold_set - pred_set)
    report.predicate_only = _compute_prf(tp_p, fp_p, fn_p)

    return report


def _match_triples(
    predicted: List[Tuple[str, str, str]],
    golds: List[GoldTriple],
    match_fn,
) -> Tuple[int, int, int]:
    """
    Match predicted triples against gold triples.

    Returns (tp_count, fp_count, fn_count).
    """
    matched_gold_indices: Set[int] = set()
    tp = 0

    for s, p, o in predicted:
        for i, gold in enumerate(golds):
            if i in matched_gold_indices:
                continue
            if match_fn(s, p, o, gold):
                tp += 1
                matched_gold_indices.add(i)
                break

    fp = len(predicted) - tp
    fn = len(golds) - len(matched_gold_indices)
    return tp, fp, fn


def relation_metrics_to_dict(report: RelationMetricsReport) -> dict:
    """Serialize RelationMetricsReport to a JSON-compatible dictionary."""
    return {
        "total_predicted": report.total_predicted,
        "total_gold": report.total_gold,
        "strict": {"precision": report.strict.precision, "recall": report.strict.recall, "f1": report.strict.f1},
        "fuzzy": {"precision": report.fuzzy.precision, "recall": report.fuzzy.recall, "f1": report.fuzzy.f1},
        "relaxed": {"precision": report.relaxed.precision, "recall": report.relaxed.recall, "f1": report.relaxed.f1},
        "predicate_only": {
            "precision": report.predicate_only.precision,
            "recall": report.predicate_only.recall,
            "f1": report.predicate_only.f1,
        },
        "matched_triples": report.matched_triples,
        "missed_triples": report.missed_triples,
        "extra_triples": report.extra_triples,
    }

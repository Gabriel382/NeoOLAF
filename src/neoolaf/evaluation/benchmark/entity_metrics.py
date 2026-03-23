from __future__ import annotations

# Standard library imports
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
class EntityMetricsReport:
    """
    Benchmark evaluation of entity extraction against a gold standard.
    """

    total_predicted: int = 0
    total_gold: int = 0

    # P/R/F1 for each matching mode
    exact: PRF = field(default_factory=PRF)
    fuzzy: PRF = field(default_factory=PRF)
    partial: PRF = field(default_factory=PRF)

    # Detail lists
    true_positives_exact: List[str] = field(default_factory=list)
    false_positives_exact: List[str] = field(default_factory=list)
    false_negatives_exact: List[str] = field(default_factory=list)

    # Per-type breakdown
    per_type: Dict[str, PRF] = field(default_factory=dict)


@dataclass
class GoldEntity:
    """One gold entity from the annotation file."""
    label: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Gold loading
# ------------------------------------------------------------------

def load_gold_entities(path: str) -> List[GoldEntity]:
    """
    Load gold entities from a JSON annotation file.

    Expected format:
    {
        "entities": [
            {"label": "BearingFailure", "type": "event", "aliases": [...]},
            ...
        ]
    }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entities = []
    for e in data.get("entities", []):
        entities.append(GoldEntity(
            label=e["label"],
            entity_type=e.get("type", "entity"),
            aliases=e.get("aliases", []),
        ))
    return entities


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip().replace("_", " ").replace("-", " ")


# ------------------------------------------------------------------
# Matching functions
# ------------------------------------------------------------------

def _exact_match(pred_label: str, gold: GoldEntity) -> bool:
    """Check if predicted label matches gold label or any alias exactly."""
    norm_pred = _normalize(pred_label)
    candidates = [gold.label] + gold.aliases
    return any(_normalize(c) == norm_pred for c in candidates)


def _fuzzy_match(pred_label: str, gold: GoldEntity, threshold: float) -> bool:
    """Check if predicted label fuzzy-matches gold label or any alias."""
    norm_pred = _normalize(pred_label)
    candidates = [gold.label] + gold.aliases
    for c in candidates:
        score = max(
            fuzz.ratio(norm_pred, _normalize(c)),
            fuzz.partial_ratio(norm_pred, _normalize(c)),
            fuzz.token_sort_ratio(norm_pred, _normalize(c)),
        )
        if score >= threshold:
            return True
    return False


def _partial_match(pred_label: str, gold: GoldEntity) -> bool:
    """Check if predicted label contains gold label or vice versa."""
    norm_pred = _normalize(pred_label)
    candidates = [gold.label] + gold.aliases
    for c in candidates:
        norm_c = _normalize(c)
        if norm_pred in norm_c or norm_c in norm_pred:
            return True
    return False


def _compute_prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return PRF(precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4))


# ------------------------------------------------------------------
# Main computation
# ------------------------------------------------------------------

def compute_entity_metrics(
    state: PipelineState,
    gold_entities: List[GoldEntity],
    fuzzy_threshold: float = 85.0,
) -> EntityMetricsReport:
    """
    Compute entity extraction metrics against a gold standard.

    Args:
        state:            PipelineState after Layer 3.
        gold_entities:    List of GoldEntity from annotation file.
        fuzzy_threshold:  Minimum rapidfuzz score (0-100) for fuzzy match.

    Returns:
        EntityMetricsReport with all metrics populated.
    """
    report = EntityMetricsReport()

    predicted_labels = [c.canonical_label for c in state.entity_candidates]
    report.total_predicted = len(predicted_labels)
    report.total_gold = len(gold_entities)

    # --- Exact ---
    tp_exact, fp_exact, fn_exact = _match_sets(predicted_labels, gold_entities, _exact_match)
    report.exact = _compute_prf(len(tp_exact), len(fp_exact), len(fn_exact))
    report.true_positives_exact = tp_exact
    report.false_positives_exact = fp_exact
    report.false_negatives_exact = fn_exact

    # --- Fuzzy ---
    tp_fuzzy, fp_fuzzy, fn_fuzzy = _match_sets(
        predicted_labels, gold_entities,
        lambda p, g: _fuzzy_match(p, g, fuzzy_threshold),
    )
    report.fuzzy = _compute_prf(len(tp_fuzzy), len(fp_fuzzy), len(fn_fuzzy))

    # --- Partial ---
    tp_partial, fp_partial, fn_partial = _match_sets(
        predicted_labels, gold_entities, _partial_match,
    )
    report.partial = _compute_prf(len(tp_partial), len(fp_partial), len(fn_partial))

    # --- Per-type breakdown (exact match) ---
    types = {g.entity_type for g in gold_entities}
    for etype in types:
        type_preds = [
            c.canonical_label for c in state.entity_candidates
            if c.candidate_type == etype
        ]
        type_golds = [g for g in gold_entities if g.entity_type == etype]
        tp_t, fp_t, fn_t = _match_sets(type_preds, type_golds, _exact_match)
        report.per_type[etype] = _compute_prf(len(tp_t), len(fp_t), len(fn_t))

    return report


def _match_sets(
    predicted: List[str],
    golds: List[GoldEntity],
    match_fn,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Match predicted labels against gold entities using a given match function.

    Returns (true_positives, false_positives, false_negatives) as label lists.
    """
    matched_gold_indices: Set[int] = set()
    tp_labels: List[str] = []
    fp_labels: List[str] = []

    for pred in predicted:
        found = False
        for i, gold in enumerate(golds):
            if i in matched_gold_indices:
                continue
            if match_fn(pred, gold):
                tp_labels.append(pred)
                matched_gold_indices.add(i)
                found = True
                break
        if not found:
            fp_labels.append(pred)

    fn_labels = [
        golds[i].label for i in range(len(golds))
        if i not in matched_gold_indices
    ]

    return tp_labels, fp_labels, fn_labels


def entity_metrics_to_dict(report: EntityMetricsReport) -> dict:
    """Serialize EntityMetricsReport to a JSON-compatible dictionary."""
    return {
        "total_predicted": report.total_predicted,
        "total_gold": report.total_gold,
        "exact": {"precision": report.exact.precision, "recall": report.exact.recall, "f1": report.exact.f1},
        "fuzzy": {"precision": report.fuzzy.precision, "recall": report.fuzzy.recall, "f1": report.fuzzy.f1},
        "partial": {"precision": report.partial.precision, "recall": report.partial.recall, "f1": report.partial.f1},
        "true_positives": report.true_positives_exact,
        "false_positives": report.false_positives_exact,
        "false_negatives": report.false_negatives_exact,
        "per_type": {
            t: {"precision": prf.precision, "recall": prf.recall, "f1": prf.f1}
            for t, prf in report.per_type.items()
        },
    }

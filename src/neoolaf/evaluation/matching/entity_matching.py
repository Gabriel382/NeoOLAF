"""Entity matching implementations."""

from __future__ import annotations

from typing import Iterable
from dataclasses import asdict

from neoolaf.evaluation.matching.normalization import normalize_text
from neoolaf.evaluation.matching.similarity import loose_similarity, strict_similarity
from neoolaf.evaluation.metrics.prf import make_prf
from neoolaf.evaluation.schema.config import EvaluationProfile
from neoolaf.evaluation.schema.metrics import MatchResult

GENERIC_ENTITY_BLOCKLIST = {
    "operator",
    "maintenance",
    "maintenance technician",
    "technician",
    "officer",
    "person",
    "people",
    "device",
    "machine",
    "alarm",
    "check",
    "manual",
    "reference",
    "page",
    "diagram",
    "input",
    "cause",
    "intervention",
    "responsible",
    "actor",
    "failure",
    "part",
}


def is_generic_entity(text: object) -> bool:
    """Return True when an entity label is too generic to be useful."""
    return normalize_text(text) in GENERIC_ENTITY_BLOCKLIST


def entity_score(pred: str, gold: str, profile: EvaluationProfile) -> float:
    """Profile-aware entity similarity."""
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    if not pred_n or not gold_n:
        return 0.0
    if pred_n == gold_n:
        return 100.0
    if profile.reject_generic_entities and (is_generic_entity(pred) or is_generic_entity(gold)):
        return 0.0

    if profile.name == "xquality_strict_extraction":
        # Conservative filters copied from the previous strict evaluator.
        len_ratio = min(len(pred_n), len(gold_n)) / max(len(pred_n), len(gold_n))
        if len_ratio < 0.75:
            return 0.0
        pred_tokens = pred_n.split()
        gold_tokens = gold_n.split()
        tok_ratio = min(len(pred_tokens), len(gold_tokens)) / max(len(pred_tokens), len(gold_tokens))
        if tok_ratio < 0.60:
            return 0.0
        return strict_similarity(pred_n, gold_n)

    return loose_similarity(pred_n, gold_n)


def greedy_entity_matching(
    pred_entities: Iterable[str],
    gold_entities: Iterable[str],
    profile: EvaluationProfile,
) -> MatchResult:
    """Greedy one-to-one matching between predicted and gold entity labels."""
    pred_list = sorted({str(x).strip() for x in pred_entities if str(x).strip()})
    gold_list = sorted({str(x).strip() for x in gold_entities if str(x).strip()})

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(pred_list):
        if profile.reject_generic_entities and is_generic_entity(pred):
            continue
        for gold_idx, gold in enumerate(gold_list):
            if profile.reject_generic_entities and is_generic_entity(gold):
                continue
            score = entity_score(pred, gold, profile)
            if score >= profile.entity_threshold:
                candidates.append((score, pred_idx, gold_idx))

    candidates.sort(reverse=True, key=lambda item: (item[0], -item[1], -item[2]))

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches: list[dict] = []

    for score, pred_idx, gold_idx in candidates:
        if pred_idx in used_pred or gold_idx in used_gold:
            continue
        used_pred.add(pred_idx)
        used_gold.add(gold_idx)
        matches.append({"pred": pred_list[pred_idx], "gold": gold_list[gold_idx], "score": score})

    unmatched_pred = [pred_list[i] for i in range(len(pred_list)) if i not in used_pred]
    unmatched_gold = [gold_list[i] for i in range(len(gold_list)) if i not in used_gold]
    prf = make_prf(tp=len(matches), fp=len(unmatched_pred), fn=len(unmatched_gold))
    return MatchResult(prf=prf, matches=matches, unmatched_pred=unmatched_pred, unmatched_gold=unmatched_gold)

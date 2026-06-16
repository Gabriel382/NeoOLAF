"""Relation/triple matching implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable
from dataclasses import asdict

from neoolaf.evaluation.matching.normalization import normalize_text
from neoolaf.evaluation.matching.similarity import loose_similarity, strict_similarity
from neoolaf.evaluation.metrics.prf import make_prf
from neoolaf.evaluation.schema.artifact import EvalRelation
from neoolaf.evaluation.schema.config import EvaluationProfile
from neoolaf.evaluation.schema.metrics import MatchResult


def relation_endpoint_score(a: str, b: str, profile: EvaluationProfile) -> float:
    """Profile-aware endpoint similarity."""
    if profile.name == "xquality_strict_extraction":
        return strict_similarity(a, b)
    return loose_similarity(a, b)


def triple_pair_score(pred: EvalRelation, gold: EvalRelation, profile: EvaluationProfile) -> dict | None:
    """Return a match score dictionary or None if the triple pair does not match."""
    pred_rel = normalize_text(pred.relation).upper()
    gold_rel = normalize_text(gold.relation).upper()
    if pred_rel != gold_rel:
        return None

    head_score = relation_endpoint_score(pred.head, gold.head, profile)
    tail_score = relation_endpoint_score(pred.tail, gold.tail, profile)

    if head_score < profile.relation_head_threshold:
        return None
    if tail_score < profile.relation_tail_threshold(pred.relation):
        return None

    total_score = 0.5 * head_score + 0.5 * tail_score
    return {
        "score": total_score,
        "head_score": head_score,
        "tail_score": tail_score,
        "direction": "direct",
    }


def greedy_relation_matching(
    pred_relations: Iterable[EvalRelation],
    gold_relations: Iterable[EvalRelation],
    profile: EvaluationProfile,
) -> MatchResult:
    """Greedy one-to-one matching between predicted and gold triples."""
    pred_list = list(pred_relations)
    gold_list = list(gold_relations)

    candidates: list[tuple[float, float, float, int, int, dict]] = []

    for pred_idx, pred in enumerate(pred_list):
        for gold_idx, gold in enumerate(gold_list):
            score = triple_pair_score(pred, gold, profile)

            if score is None:
                continue

            candidates.append(
                (
                    score["score"],
                    score["head_score"],
                    score["tail_score"],
                    pred_idx,
                    gold_idx,
                    score,
                )
            )

    candidates.sort(
        reverse=True,
        key=lambda item: (item[0], item[1], item[2], -item[3], -item[4]),
    )

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matches: list[dict] = []

    for total_score, head_score, tail_score, pred_idx, gold_idx, score_info in candidates:
        if pred_idx in used_pred or gold_idx in used_gold:
            continue

        used_pred.add(pred_idx)
        used_gold.add(gold_idx)

        matches.append(
            {
                "pred_idx": pred_idx,
                "gold_idx": gold_idx,
                "score": total_score,
                "head_score": head_score,
                "tail_score": tail_score,
                "direction": score_info.get("direction", "direct"),
                "pred": asdict(pred_list[pred_idx]),
                "gold": asdict(gold_list[gold_idx]),
            }
        )

    unmatched_pred = [
        asdict(pred_list[i])
        for i in range(len(pred_list))
        if i not in used_pred
    ]

    unmatched_gold = [
        asdict(gold_list[j])
        for j in range(len(gold_list))
        if j not in used_gold
    ]

    prf = make_prf(
        tp=len(matches),
        fp=len(unmatched_pred),
        fn=len(unmatched_gold),
    )

    return MatchResult(
        prf=prf,
        matches=matches,
        unmatched_pred=unmatched_pred,
        unmatched_gold=unmatched_gold,
    )


def per_relation_metrics(
    pred_relations: list[EvalRelation],
    gold_relations: list[EvalRelation],
    profile: EvaluationProfile,
) -> list[dict]:
    """Compute per-relation PRF rows."""
    labels = sorted({r.relation for r in pred_relations} | {r.relation for r in gold_relations})
    rows: list[dict] = []
    for label in labels:
        pred_subset = [r for r in pred_relations if r.relation == label]
        gold_subset = [r for r in gold_relations if r.relation == label]
        result = greedy_relation_matching(pred_subset, gold_subset, profile)
        row = {
            "relation": label,
            "pred_count": len(pred_subset),
            "gold_count": len(gold_subset),
            **result.prf.to_dict(),
        }
        rows.append(row)
    return rows

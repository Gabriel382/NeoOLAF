from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Third-party imports
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Local imports
from neoolaf.core.pipeline_state import PipelineState


@dataclass
class BleuReport:
    """
    Aggregated BLEU score metrics for a pipeline run.
    """

    # --- Global stats ---
    scores_count: int = 0
    avg_bleu: Optional[float] = None
    median_bleu: Optional[float] = None
    min_bleu: Optional[float] = None
    max_bleu: Optional[float] = None

    # --- Flagged items ---
    low_bleu_threshold: float = 0.2
    low_bleu_ids: List[str] = field(default_factory=list)

    # --- Per-item detail ---
    per_item: List[Dict] = field(default_factory=list)


# Smoothing to handle short sentences where higher n-gram counts are zero
_smoothing = SmoothingFunction().method1


def compute_sentence_bleu(
    reference: str,
    hypothesis: str,
) -> float:
    """
    Compute sentence-level BLEU score between a reference and hypothesis
    using NLTK.

    Uses word_tokenize for proper tokenization and smoothing (method1)
    to handle short texts where higher-order n-grams may be zero.

    Args:
        reference:  The source text (ground truth).
        hypothesis: The generated text to evaluate.

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())

    if not ref_tokens or not hyp_tokens:
        return 0.0

    return sentence_bleu(
        [ref_tokens],
        hyp_tokens,
        smoothing_function=_smoothing,
    )


# ------------------------------------------------------------------
# Pipeline-level computation
# ------------------------------------------------------------------

def _collect_pairs(state: PipelineState) -> List[Tuple[str, str, str]]:
    """
    Collect (item_id, reference_text, hypothesis_text) pairs from state.

    Sources:
    - CandidateTriple: reference = provenance snippets, hypothesis = justification
    - GeneralAxiomCandidate (description): reference = evidence snippets,
      hypothesis = literal_value
    - GeneralAxiomCandidate (other): reference = evidence snippets,
      hypothesis = justification
    """
    pairs: List[Tuple[str, str, str]] = []

    # Triples: provenance snippets vs justification
    for triple in state.candidate_triples:
        snippets = [ev.snippet for ev in triple.provenance if ev.snippet]
        if not snippets or not triple.justification:
            continue
        reference = " ".join(snippets)
        pairs.append((triple.triple_id, reference, triple.justification))

    # General axioms
    for axiom in state.general_axiom_candidates:
        snippets = [ev.snippet for ev in axiom.evidence if ev.snippet]
        if not snippets:
            continue

        reference = " ".join(snippets)

        if axiom.axiom_type == "description" and axiom.literal_value:
            pairs.append((axiom.axiom_id, reference, axiom.literal_value))
        elif axiom.justification:
            pairs.append((axiom.axiom_id, reference, axiom.justification))

    return pairs


def compute_bleu_scores(
    state: PipelineState,
    low_bleu_threshold: float = 0.2,
) -> BleuReport:
    """
    Compute BLEU scores for all reference/hypothesis pairs in the pipeline.

    Args:
        state: PipelineState after at least Layer 5 and Layer 9 have run.
        low_bleu_threshold: Items below this score are flagged.

    Returns:
        BleuReport with all metrics populated.
    """
    report = BleuReport(low_bleu_threshold=low_bleu_threshold)

    pairs = _collect_pairs(state)
    if not pairs:
        return report

    scores: List[float] = []

    for item_id, reference, hypothesis in pairs:
        score = compute_sentence_bleu(reference, hypothesis)
        scores.append(score)

        report.per_item.append({
            "item_id": item_id,
            "bleu": round(score, 4),
            "reference_preview": reference[:120],
            "hypothesis_preview": hypothesis[:120],
        })

        if score < low_bleu_threshold:
            report.low_bleu_ids.append(item_id)

    report.scores_count = len(scores)
    report.avg_bleu = sum(scores) / len(scores)
    report.min_bleu = min(scores)
    report.max_bleu = max(scores)

    sorted_scores = sorted(scores)
    mid = len(sorted_scores) // 2
    if len(sorted_scores) % 2 == 0:
        report.median_bleu = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
    else:
        report.median_bleu = sorted_scores[mid]

    return report


def bleu_to_dict(report: BleuReport) -> dict:
    """
    Serialize BleuReport to a JSON-compatible dictionary.
    """
    return {
        "scores_count": report.scores_count,
        "avg_bleu": report.avg_bleu,
        "median_bleu": report.median_bleu,
        "min_bleu": report.min_bleu,
        "max_bleu": report.max_bleu,
        "low_bleu_threshold": report.low_bleu_threshold,
        "low_bleu_ids": report.low_bleu_ids,
    }

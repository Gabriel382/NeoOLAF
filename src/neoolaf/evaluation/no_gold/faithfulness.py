from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Local imports
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.candidate_triple import CandidateTriple


@dataclass
class FaithfulnessReport:
    """
    Aggregated faithfulness metrics for a pipeline run.
    """

    # --- Provenance coverage ---
    total_triples: int = 0
    triples_with_provenance: int = 0
    provenance_coverage: Optional[float] = None

    # --- Textual grounding ---
    triples_grounded: int = 0
    textual_grounding_rate: Optional[float] = None
    ungrounded_triple_ids: List[str] = field(default_factory=list)

    # --- Contradiction detection ---
    contradiction_pairs: List[Tuple[str, str]] = field(default_factory=list)
    contradiction_rate: Optional[float] = None

    # --- Per-triple detail ---
    per_triple: List[Dict] = field(default_factory=list)


def _normalize(text: str) -> str:
    """Lowercase and strip for matching."""
    return text.lower().strip()


def _label_found_in_snippets(label: str, snippets: List[str]) -> bool:
    """
    Check if a label appears in at least one evidence snippet.

    Uses normalized substring matching.
    """
    norm_label = _normalize(label)
    if not norm_label:
        return False
    for snippet in snippets:
        if norm_label in _normalize(snippet):
            return True
    return False


def _check_triple_grounding(triple: CandidateTriple) -> Tuple[bool, bool, bool]:
    """
    Check provenance coverage and textual grounding for a single triple.

    Returns:
        (has_provenance, subject_grounded, object_grounded)
    """
    snippets = [ev.snippet for ev in triple.provenance if ev.snippet]

    has_provenance = len(snippets) > 0

    subject_grounded = _label_found_in_snippets(triple.subject_label, snippets)
    object_grounded = _label_found_in_snippets(triple.object_label, snippets)

    return has_provenance, subject_grounded, object_grounded


def _detect_contradictions(
    triples: List[CandidateTriple],
) -> List[Tuple[str, str]]:
    """
    Detect potential contradictions: triples sharing the same
    (subject, predicate) but with different objects.

    Returns list of (triple_id_a, triple_id_b) pairs.
    """
    groups: Dict[Tuple[str, str], List[CandidateTriple]] = {}
    for triple in triples:
        key = (_normalize(triple.subject_label), _normalize(triple.predicate_label))
        groups.setdefault(key, []).append(triple)

    pairs: List[Tuple[str, str]] = []
    for group_triples in groups.values():
        if len(group_triples) < 2:
            continue

        seen_objects: Dict[str, str] = {}
        for triple in group_triples:
            norm_obj = _normalize(triple.object_label)
            if norm_obj in seen_objects:
                continue
            for other_obj, other_id in seen_objects.items():
                if norm_obj != other_obj:
                    pairs.append((other_id, triple.triple_id))
            seen_objects[norm_obj] = triple.triple_id

    return pairs


def compute_faithfulness(state: PipelineState) -> FaithfulnessReport:
    """
    Compute faithfulness metrics from a completed pipeline state.

    Args:
        state: PipelineState after at least Layer 5 has run.

    Returns:
        FaithfulnessReport with all metrics populated.
    """
    report = FaithfulnessReport()
    triples = state.candidate_triples
    report.total_triples = len(triples)

    if not triples:
        return report

    # ------------------------------------------------------------------
    # 1. Provenance coverage and textual grounding
    # ------------------------------------------------------------------
    for triple in triples:
        has_prov, subj_ok, obj_ok = _check_triple_grounding(triple)

        if has_prov:
            report.triples_with_provenance += 1

        is_grounded = has_prov and subj_ok and obj_ok
        if is_grounded:
            report.triples_grounded += 1
        else:
            report.ungrounded_triple_ids.append(triple.triple_id)

        report.per_triple.append({
            "triple_id": triple.triple_id,
            "subject_label": triple.subject_label,
            "predicate_label": triple.predicate_label,
            "object_label": triple.object_label,
            "has_provenance": has_prov,
            "subject_grounded": subj_ok,
            "object_grounded": obj_ok,
            "is_grounded": is_grounded,
        })

    report.provenance_coverage = report.triples_with_provenance / report.total_triples
    report.textual_grounding_rate = report.triples_grounded / report.total_triples

    # ------------------------------------------------------------------
    # 2. Contradiction detection
    # ------------------------------------------------------------------
    report.contradiction_pairs = _detect_contradictions(triples)
    report.contradiction_rate = len(report.contradiction_pairs) / report.total_triples

    return report


def faithfulness_to_dict(report: FaithfulnessReport) -> dict:
    """
    Serialize FaithfulnessReport to a JSON-compatible dictionary.
    """
    return {
        "total_triples": report.total_triples,
        "provenance_coverage": report.provenance_coverage,
        "textual_grounding_rate": report.textual_grounding_rate,
        "contradiction_rate": report.contradiction_rate,
        "triples_with_provenance": report.triples_with_provenance,
        "triples_grounded": report.triples_grounded,
        "ungrounded_triple_ids": report.ungrounded_triple_ids,
        "contradiction_pairs": [
            {"triple_a": a, "triple_b": b}
            for a, b in report.contradiction_pairs
        ],
    }

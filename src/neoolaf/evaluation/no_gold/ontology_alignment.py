from __future__ import annotations

# Standard library imports
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Third-party imports
import rdflib
from rapidfuzz import fuzz

# Local imports
from neoolaf.core.pipeline_state import PipelineState


@dataclass
class AlignmentPair:
    """One aligned pair between a NeoOLAF label and a reference label."""

    neoolaf_label: str
    reference_label: str
    score: float

@dataclass
class OntologyAlignmentReport:
    """
    Aggregated ontology alignment metrics for a pipeline run.
    """

    # --- Concept alignment ---
    total_concepts: int = 0
    aligned_concepts: int = 0
    concept_alignment_rate: Optional[float] = None
    unaligned_concepts: List[str] = field(default_factory=list)
    concept_pairs: List[AlignmentPair] = field(default_factory=list)

    # --- Relation alignment ---
    total_relations: int = 0
    aligned_relations: int = 0
    relation_alignment_rate: Optional[float] = None
    unaligned_relations: List[str] = field(default_factory=list)
    relation_pairs: List[AlignmentPair] = field(default_factory=list)

    # --- Hierarchy alignment ---
    total_hierarchy_links: int = 0
    aligned_hierarchy_links: int = 0
    hierarchy_alignment_rate: Optional[float] = None


@dataclass
class ReferenceOntology:
    """
    A reference ontology loaded from an external source.

    Expected JSON format:
    {
        "concepts": ["BearingDefect", "MechanicalFailure", ...],
        "relations": ["causes", "partOf", ...],
        "hierarchy": [
            {"child": "BearingDefect", "parent": "MechanicalFailure"},
            ...
        ]
    }
    """

    concepts: Set[str] = field(default_factory=set)
    relations: Set[str] = field(default_factory=set)
    hierarchy: List[Tuple[str, str]] = field(default_factory=list)

    # Normalized lookup sets (built on load)
    _norm_concepts: Set[str] = field(default_factory=set, repr=False)
    _norm_relations: Set[str] = field(default_factory=set, repr=False)


# ------------------------------------------------------------------
# Normalization and matching
# ------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip, replace underscores/hyphens with spaces."""
    return text.lower().strip().replace("_", " ").replace("-", " ")


def _tokenize(text: str) -> List[str]:
    """Split a normalized label into tokens."""
    return _normalize(text).split()


def _fuzzy_score(a: str, b: str) -> float:
    """
    Compute similarity between two strings using rapidfuzz.

    Takes the maximum of three strategies:
    - fuzz.ratio: overall character similarity
    - fuzz.partial_ratio: best partial substring match
    - fuzz.token_sort_ratio: token-order-independent similarity

    Returns a score between 0.0 and 1.0.
    """
    norm_a = _normalize(a)
    norm_b = _normalize(b)

    ratio = fuzz.ratio(norm_a, norm_b) / 100.0
    partial = fuzz.partial_ratio(norm_a, norm_b) / 100.0
    token_sort = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0

    return max(ratio, partial, token_sort)


def _token_overlap_score(a: str, b: str) -> float:
    """Compute Jaccard similarity on token sets."""
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _best_match(
    label: str,
    reference_labels: Set[str],
    threshold: float,
) -> Optional[Tuple[str, float]]:
    """
    Find the best matching reference label for a given NeoOLAF label.

    Uses the maximum of rapidfuzz score and token Jaccard overlap.
    Returns (best_label, best_score) if above threshold, else None.
    """
    best_label = None
    best_score = 0.0

    norm_label = _normalize(label)

    for ref_label in reference_labels:
        norm_ref = _normalize(ref_label)

        # Exact match shortcut
        if norm_label == norm_ref:
            return ref_label, 1.0

        # Combined score: max of rapidfuzz and token overlap
        fuzzy = _fuzzy_score(label, ref_label)
        jaccard = _token_overlap_score(label, ref_label)
        score = max(fuzzy, jaccard)

        if score > best_score:
            best_score = score
            best_label = ref_label

    if best_label is not None and best_score >= threshold:
        return best_label, best_score

    return None


# ------------------------------------------------------------------
# Reference ontology loading
# ------------------------------------------------------------------

def load_reference_from_json(path: str) -> ReferenceOntology:
    """
    Load a reference ontology from a JSON file.

    Expected format:
    {
        "concepts": ["Label1", "Label2", ...],
        "relations": ["rel1", "rel2", ...],
        "hierarchy": [{"child": "Label1", "parent": "Label2"}, ...]
    }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    ref = ReferenceOntology()
    ref.concepts = set(data.get("concepts", []))
    ref.relations = set(data.get("relations", []))
    ref.hierarchy = [
        (h["child"], h["parent"])
        for h in data.get("hierarchy", [])
    ]
    ref._norm_concepts = {_normalize(c) for c in ref.concepts}
    ref._norm_relations = {_normalize(r) for r in ref.relations}

    return ref


def _get_label(g: rdflib.Graph, node) -> Optional[str]:
    """Extract rdfs:label or fragment from a URI node."""
    for label in g.objects(node, rdflib.RDFS.label):
        return str(label)

    # Fallback: use URI fragment
    uri = str(node)
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.rsplit("/", 1)[-1]

    return None


def load_reference_from_rdf(path: str) -> ReferenceOntology:
    """
    Load a reference ontology from an OWL/TTL/RDF file using rdflib.

    Supports all formats rdflib can parse: Turtle (.ttl), RDF/XML (.rdf, .owl),
    N-Triples (.nt), JSON-LD (.jsonld), etc.
    """
    g = rdflib.Graph()
    g.parse(path)

    ref = ReferenceOntology()

    # Extract OWL classes
    for s in g.subjects(rdflib.RDF.type, rdflib.OWL.Class):
        label = _get_label(g, s)
        if label:
            ref.concepts.add(label)

    # Extract RDFS classes (some ontologies use rdfs:Class)
    for s in g.subjects(rdflib.RDF.type, rdflib.RDFS.Class):
        label = _get_label(g, s)
        if label:
            ref.concepts.add(label)

    # Extract object properties
    for s in g.subjects(rdflib.RDF.type, rdflib.OWL.ObjectProperty):
        label = _get_label(g, s)
        if label:
            ref.relations.add(label)

    # Extract RDF properties (broader coverage)
    for s in g.subjects(rdflib.RDF.type, rdflib.RDF.Property):
        label = _get_label(g, s)
        if label:
            ref.relations.add(label)

    # Extract subClassOf hierarchy
    for s, o in g.subject_objects(rdflib.RDFS.subClassOf):
        child_label = _get_label(g, s)
        parent_label = _get_label(g, o)
        if child_label and parent_label:
            ref.hierarchy.append((child_label, parent_label))

    ref._norm_concepts = {_normalize(c) for c in ref.concepts}
    ref._norm_relations = {_normalize(r) for r in ref.relations}

    return ref


# ------------------------------------------------------------------
# Pipeline-level computation
# ------------------------------------------------------------------

def compute_ontology_alignment(
    state: PipelineState,
    reference: ReferenceOntology,
    threshold: float = 0.75,
) -> OntologyAlignmentReport:
    """
    Compute ontology alignment metrics between pipeline output and a reference.

    Args:
        state:     PipelineState after at least Layer 6 and Layer 7 have run.
        reference: A ReferenceOntology loaded from JSON or OWL.
        threshold: Minimum similarity score to consider a match (default 0.75).

    Returns:
        OntologyAlignmentReport with all metrics populated.
    """
    report = OntologyAlignmentReport()

    # ------------------------------------------------------------------
    # 1. Concept alignment
    # ------------------------------------------------------------------
    report.total_concepts = len(state.concept_candidates)

    for concept in state.concept_candidates:
        match = _best_match(concept.label, reference.concepts, threshold)
        if match is not None:
            ref_label, score = match
            report.aligned_concepts += 1
            report.concept_pairs.append(
                AlignmentPair(
                    neoolaf_label=concept.label,
                    reference_label=ref_label,
                    score=round(score, 4),
                )
            )
        else:
            report.unaligned_concepts.append(concept.label)

    if report.total_concepts > 0:
        report.concept_alignment_rate = report.aligned_concepts / report.total_concepts

    # ------------------------------------------------------------------
    # 2. Relation alignment
    # ------------------------------------------------------------------
    report.total_relations = len(state.ontology_relation_candidates)

    for relation in state.ontology_relation_candidates:
        match = _best_match(relation.label, reference.relations, threshold)
        if match is not None:
            ref_label, score = match
            report.aligned_relations += 1
            report.relation_pairs.append(
                AlignmentPair(
                    neoolaf_label=relation.label,
                    reference_label=ref_label,
                    score=round(score, 4),
                )
            )
        else:
            report.unaligned_relations.append(relation.label)

    if report.total_relations > 0:
        report.relation_alignment_rate = report.aligned_relations / report.total_relations

    # ------------------------------------------------------------------
    # 3. Hierarchy alignment
    # ------------------------------------------------------------------
    report.total_hierarchy_links = len(state.concept_hierarchy_links)

    # Build normalized reference hierarchy set for lookup
    ref_hierarchy_set: Set[Tuple[str, str]] = {
        (_normalize(child), _normalize(parent))
        for child, parent in reference.hierarchy
    }

    for link in state.concept_hierarchy_links:
        # Try exact normalized match first
        norm_pair = (_normalize(link.child_label), _normalize(link.parent_label))
        if norm_pair in ref_hierarchy_set:
            report.aligned_hierarchy_links += 1
            continue

        # Try fuzzy match: find if any ref pair matches both child and parent
        for ref_child, ref_parent in reference.hierarchy:
            child_score = _fuzzy_score(link.child_label, ref_child)
            parent_score = _fuzzy_score(link.parent_label, ref_parent)
            if child_score >= threshold and parent_score >= threshold:
                report.aligned_hierarchy_links += 1
                break

    if report.total_hierarchy_links > 0:
        report.hierarchy_alignment_rate = (
            report.aligned_hierarchy_links / report.total_hierarchy_links
        )

    return report


def alignment_to_dict(report: OntologyAlignmentReport) -> dict:
    """
    Serialize OntologyAlignmentReport to a JSON-compatible dictionary.
    """
    return {
        "concepts": {
            "total": report.total_concepts,
            "aligned": report.aligned_concepts,
            "alignment_rate": report.concept_alignment_rate,
            "unaligned": report.unaligned_concepts,
            "pairs": [
                {
                    "neoolaf": p.neoolaf_label,
                    "reference": p.reference_label,
                    "score": p.score,
                }
                for p in report.concept_pairs
            ],
        },
        "relations": {
            "total": report.total_relations,
            "aligned": report.aligned_relations,
            "alignment_rate": report.relation_alignment_rate,
            "unaligned": report.unaligned_relations,
            "pairs": [
                {
                    "neoolaf": p.neoolaf_label,
                    "reference": p.reference_label,
                    "score": p.score,
                }
                for p in report.relation_pairs
            ],
        },
        "hierarchy": {
            "total": report.total_hierarchy_links,
            "aligned": report.aligned_hierarchy_links,
            "alignment_rate": report.hierarchy_alignment_rate,
        },
    }

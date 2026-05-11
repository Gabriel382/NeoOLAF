from __future__ import annotations

# Standard library imports
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Third-party imports
import rdflib
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
class DepthComparison:
    """Structural depth comparison."""
    predicted: float = 0.0
    gold: float = 0.0


@dataclass
class RatioComparison:
    """Ratio comparison between predicted and gold."""
    predicted: float = 0.0
    gold: float = 0.0


@dataclass
class OntologyConformanceReport:
    """
    Benchmark evaluation of ontology structure against a gold standard.
    """

    # Concept coverage
    total_predicted_concepts: int = 0
    total_gold_concepts: int = 0
    concept_coverage: PRF = field(default_factory=PRF)

    # Hierarchy accuracy
    total_predicted_links: int = 0
    total_gold_links: int = 0
    hierarchy: PRF = field(default_factory=PRF)

    # Domain/Range accuracy
    total_predicted_dr: int = 0
    total_gold_dr: int = 0
    domain_range: PRF = field(default_factory=PRF)

    # Structural metrics
    structural_depth: DepthComparison = field(default_factory=DepthComparison)
    orphan_ratio: RatioComparison = field(default_factory=RatioComparison)

    # Detail
    matched_concepts: List[str] = field(default_factory=list)
    unmatched_concepts: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)


@dataclass
class GoldOntology:
    """Gold ontology from the annotation file."""
    concepts: Set[str] = field(default_factory=set)
    relations: Set[str] = field(default_factory=set)
    hierarchy: List[Tuple[str, str]] = field(default_factory=list)
    domain_range: List[Tuple[str, str, str]] = field(default_factory=list)


# ------------------------------------------------------------------
# Gold loading
# ------------------------------------------------------------------

def load_gold_ontology_from_json(path: str) -> GoldOntology:
    """
    Load gold ontology from a JSON annotation file.

    Expected format:
    {
        "ontology": {
            "concepts": ["FailureEvent", ...],
            "relations": ["causes", ...],
            "hierarchy": [{"child": "A", "parent": "B"}, ...],
            "domain_range": [{"relation": "causes", "domain": "X", "range": "Y"}, ...]
        }
    }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    onto = data.get("ontology", data)

    gold = GoldOntology()
    gold.concepts = set(onto.get("concepts", []))
    gold.relations = set(onto.get("relations", []))
    gold.hierarchy = [
        (h["child"], h["parent"]) for h in onto.get("hierarchy", [])
    ]
    gold.domain_range = [
        (dr["relation"], dr["domain"], dr["range"])
        for dr in onto.get("domain_range", [])
    ]
    return gold


def load_gold_ontology_from_rdf(path: str) -> GoldOntology:
    """
    Load gold ontology from an OWL/TTL file using rdflib.
    """
    g = rdflib.Graph()
    g.parse(path)

    gold = GoldOntology()

    for s in g.subjects(rdflib.RDF.type, rdflib.OWL.Class):
        label = _get_rdf_label(g, s)
        if label:
            gold.concepts.add(label)

    for s in g.subjects(rdflib.RDF.type, rdflib.RDFS.Class):
        label = _get_rdf_label(g, s)
        if label:
            gold.concepts.add(label)

    for s in g.subjects(rdflib.RDF.type, rdflib.OWL.ObjectProperty):
        label = _get_rdf_label(g, s)
        if label:
            gold.relations.add(label)

    for s, o in g.subject_objects(rdflib.RDFS.subClassOf):
        child = _get_rdf_label(g, s)
        parent = _get_rdf_label(g, o)
        if child and parent:
            gold.hierarchy.append((child, parent))

    for s in g.subjects(rdflib.RDF.type, rdflib.OWL.ObjectProperty):
        rel_label = _get_rdf_label(g, s)
        if not rel_label:
            continue
        for domain in g.objects(s, rdflib.RDFS.domain):
            domain_label = _get_rdf_label(g, domain)
            for range_obj in g.objects(s, rdflib.RDFS.range):
                range_label = _get_rdf_label(g, range_obj)
                if domain_label and range_label:
                    gold.domain_range.append((rel_label, domain_label, range_label))

    return gold


def _get_rdf_label(g: rdflib.Graph, node) -> Optional[str]:
    for label in g.objects(node, rdflib.RDFS.label):
        return str(label)
    uri = str(node)
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.rsplit("/", 1)[-1]
    return None


# ------------------------------------------------------------------
# Normalization and matching
# ------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.lower().strip().replace("_", " ").replace("-", " ")


def _fuzzy_eq(a: str, b: str, threshold: float) -> bool:
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
# Structural analysis helpers
# ------------------------------------------------------------------

def _compute_depth(hierarchy: List[Tuple[str, str]], concepts: Set[str]) -> float:
    """Compute average depth of a concept hierarchy."""
    if not hierarchy or not concepts:
        return 0.0

    parent_map: Dict[str, str] = {}
    for child, parent in hierarchy:
        nc = _normalize(child)
        np = _normalize(parent)
        parent_map[nc] = np

    depths: List[int] = []
    for concept in concepts:
        nc = _normalize(concept)
        depth = 0
        current = nc
        visited = set()
        while current in parent_map and current not in visited:
            visited.add(current)
            current = parent_map[current]
            depth += 1
        depths.append(depth)

    return sum(depths) / len(depths) if depths else 0.0


def _compute_orphan_ratio(
    hierarchy: List[Tuple[str, str]], concepts: Set[str],
) -> float:
    """Compute ratio of concepts that have no parent."""
    if not concepts:
        return 0.0
    children = {_normalize(child) for child, _ in hierarchy}
    norm_concepts = {_normalize(c) for c in concepts}
    orphans = norm_concepts - children
    return len(orphans) / len(norm_concepts)


# ------------------------------------------------------------------
# Main computation
# ------------------------------------------------------------------

def compute_ontology_conformance(
    state: PipelineState,
    gold: GoldOntology,
    fuzzy_threshold: float = 85.0,
) -> OntologyConformanceReport:
    """
    Compute ontology conformance metrics against a gold standard.

    Args:
        state:            PipelineState after Layer 7 and Layer 9.
        gold:             GoldOntology from annotation file.
        fuzzy_threshold:  Minimum rapidfuzz score (0-100) for fuzzy match.

    Returns:
        OntologyConformanceReport with all metrics populated.
    """
    report = OntologyConformanceReport()

    # ------------------------------------------------------------------
    # 1. Concept coverage
    # ------------------------------------------------------------------
    pred_concepts = {c.label for c in state.concept_candidates}
    report.total_predicted_concepts = len(pred_concepts)
    report.total_gold_concepts = len(gold.concepts)

    matched_gold: Set[str] = set()
    for pc in pred_concepts:
        for gc in gold.concepts:
            if gc in matched_gold:
                continue
            if _fuzzy_eq(pc, gc, fuzzy_threshold):
                report.matched_concepts.append(pc)
                matched_gold.add(gc)
                break
        else:
            report.unmatched_concepts.append(pc)

    report.missing_concepts = [gc for gc in gold.concepts if gc not in matched_gold]

    tp_c = len(report.matched_concepts)
    fp_c = len(report.unmatched_concepts)
    fn_c = len(report.missing_concepts)
    report.concept_coverage = _compute_prf(tp_c, fp_c, fn_c)

    # ------------------------------------------------------------------
    # 2. Hierarchy accuracy
    # ------------------------------------------------------------------
    pred_links = [
        (link.child_label, link.parent_label)
        for link in state.concept_hierarchy_links
    ]
    report.total_predicted_links = len(pred_links)
    report.total_gold_links = len(gold.hierarchy)

    matched_gold_links: Set[int] = set()
    tp_h = 0
    for pc, pp in pred_links:
        for i, (gc, gp) in enumerate(gold.hierarchy):
            if i in matched_gold_links:
                continue
            if _fuzzy_eq(pc, gc, fuzzy_threshold) and _fuzzy_eq(pp, gp, fuzzy_threshold):
                tp_h += 1
                matched_gold_links.add(i)
                break

    fp_h = len(pred_links) - tp_h
    fn_h = len(gold.hierarchy) - tp_h
    report.hierarchy = _compute_prf(tp_h, fp_h, fn_h)

    # ------------------------------------------------------------------
    # 3. Domain/Range accuracy
    # ------------------------------------------------------------------
    pred_dr: List[Tuple[str, str, str]] = []
    for axiom in state.general_axiom_candidates:
        if axiom.axiom_type == "relation_domain" and axiom.object_label:
            # Find matching range axiom for same relation
            for other in state.general_axiom_candidates:
                if (
                    other.axiom_type == "relation_range"
                    and other.subject_id == axiom.subject_id
                    and other.object_label
                ):
                    pred_dr.append((axiom.subject_label, axiom.object_label, other.object_label))

    report.total_predicted_dr = len(pred_dr)
    report.total_gold_dr = len(gold.domain_range)

    matched_gold_dr: Set[int] = set()
    tp_dr = 0
    for pr, pd, prng in pred_dr:
        for i, (gr, gd, grng) in enumerate(gold.domain_range):
            if i in matched_gold_dr:
                continue
            if (
                _fuzzy_eq(pr, gr, fuzzy_threshold)
                and _fuzzy_eq(pd, gd, fuzzy_threshold)
                and _fuzzy_eq(prng, grng, fuzzy_threshold)
            ):
                tp_dr += 1
                matched_gold_dr.add(i)
                break

    fp_dr = len(pred_dr) - tp_dr
    fn_dr = len(gold.domain_range) - tp_dr
    report.domain_range = _compute_prf(tp_dr, fp_dr, fn_dr)

    # ------------------------------------------------------------------
    # 4. Structural metrics
    # ------------------------------------------------------------------
    report.structural_depth = DepthComparison(
        predicted=round(_compute_depth(pred_links, pred_concepts), 2),
        gold=round(_compute_depth(gold.hierarchy, gold.concepts), 2),
    )

    report.orphan_ratio = RatioComparison(
        predicted=round(_compute_orphan_ratio(pred_links, pred_concepts), 4),
        gold=round(_compute_orphan_ratio(gold.hierarchy, gold.concepts), 4),
    )

    return report


def conformance_to_dict(report: OntologyConformanceReport) -> dict:
    """Serialize OntologyConformanceReport to a JSON-compatible dictionary."""
    return {
        "concept_coverage": {
            "total_predicted": report.total_predicted_concepts,
            "total_gold": report.total_gold_concepts,
            "precision": report.concept_coverage.precision,
            "recall": report.concept_coverage.recall,
            "f1": report.concept_coverage.f1,
            "matched": report.matched_concepts,
            "unmatched": report.unmatched_concepts,
            "missing": report.missing_concepts,
        },
        "hierarchy": {
            "total_predicted": report.total_predicted_links,
            "total_gold": report.total_gold_links,
            "precision": report.hierarchy.precision,
            "recall": report.hierarchy.recall,
            "f1": report.hierarchy.f1,
        },
        "domain_range": {
            "total_predicted": report.total_predicted_dr,
            "total_gold": report.total_gold_dr,
            "precision": report.domain_range.precision,
            "recall": report.domain_range.recall,
            "f1": report.domain_range.f1,
        },
        "structural_depth": {
            "predicted": report.structural_depth.predicted,
            "gold": report.structural_depth.gold,
        },
        "orphan_ratio": {
            "predicted": report.orphan_ratio.predicted,
            "gold": report.orphan_ratio.gold,
        },
    }

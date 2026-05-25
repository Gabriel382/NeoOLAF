"""Ontology structural metrics."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from neoolaf.evaluation.matching.normalization import normalize_text
from neoolaf.evaluation.metrics.prf import safe_div
from neoolaf.evaluation.schema.artifact import EvalOntology


def _duplicate_count(labels: list[str]) -> int:
    counts = Counter(normalize_text(label) for label in labels if normalize_text(label))
    return sum(count - 1 for count in counts.values() if count > 1)


def _max_hierarchy_depth(ontology: EvalOntology) -> int:
    children_by_parent = defaultdict(list)
    all_children = set()
    for link in ontology.hierarchy_links:
        children_by_parent[normalize_text(link.parent)].append(normalize_text(link.child))
        all_children.add(normalize_text(link.child))
    roots = [parent for parent in children_by_parent if parent not in all_children]
    if not roots:
        return 0
    max_depth = 0
    queue = deque((root, 0) for root in roots)
    seen = set()
    while queue:
        node, depth = queue.popleft()
        if (node, depth) in seen:
            continue
        seen.add((node, depth))
        max_depth = max(max_depth, depth)
        for child in children_by_parent.get(node, []):
            queue.append((child, depth + 1))
    return max_depth


def _cycle_count(ontology: EvalOntology) -> int:
    graph = defaultdict(list)
    for link in ontology.hierarchy_links:
        graph[normalize_text(link.parent)].append(normalize_text(link.child))

    visiting = set()
    visited = set()
    cycles = 0

    def dfs(node: str) -> None:
        nonlocal cycles
        if node in visiting:
            cycles += 1
            return
        if node in visited:
            return
        visiting.add(node)
        for child in graph.get(node, []):
            dfs(child)
        visiting.remove(node)
        visited.add(node)

    for node in list(graph):
        dfs(node)
    return cycles


def evaluate_ontology(ontology: EvalOntology | None, seed_ontology: EvalOntology | None = None) -> dict:
    """Compute ontology structural and evolution metrics."""
    if ontology is None:
        return {
            "available": False,
            "class_count": 0,
            "property_count": 0,
            "hierarchy_link_count": 0,
            "axiom_count": 0,
        }

    class_labels = [item.label for item in ontology.classes]
    property_labels = [item.label for item in ontology.properties]
    described_items = [item for item in ontology.classes + ontology.properties if item.description]
    domain_covered = [prop for prop in ontology.properties if prop.domain]
    range_covered = [prop for prop in ontology.properties if prop.range]

    seed_class_count = len(seed_ontology.classes) if seed_ontology else 0
    seed_property_count = len(seed_ontology.properties) if seed_ontology else 0
    delta_size = max(0, len(ontology.classes) + len(ontology.properties) - seed_class_count - seed_property_count)
    seed_total = seed_class_count + seed_property_count
    ontology_total = len(ontology.classes) + len(ontology.properties)

    return {
        "available": True,
        "class_count": len(ontology.classes),
        "property_count": len(ontology.properties),
        "hierarchy_link_count": len(ontology.hierarchy_links),
        "axiom_count": len(ontology.axioms),
        "description_coverage": safe_div(len(described_items), ontology_total),
        "domain_coverage": safe_div(len(domain_covered), len(ontology.properties)),
        "range_coverage": safe_div(len(range_covered), len(ontology.properties)),
        "duplicate_class_count": _duplicate_count(class_labels),
        "duplicate_property_count": _duplicate_count(property_labels),
        "hierarchy_depth": _max_hierarchy_depth(ontology),
        "cycle_count": _cycle_count(ontology),
        "ontology_delta_size": delta_size,
        "promoted_concept_count": max(0, len(ontology.classes) - seed_class_count),
        "promoted_relation_count": max(0, len(ontology.properties) - seed_property_count),
        "ontology_growth_rate": safe_div(delta_size, seed_total) if seed_total else float(delta_size > 0),
    }

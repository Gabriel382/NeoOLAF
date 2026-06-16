"""RDF/TTL adapter helpers."""

from __future__ import annotations

from pathlib import Path

from neoolaf.evaluation.schema.artifact import (
    EvalHierarchyLink,
    EvalOntology,
    EvalOntologyClass,
    EvalOntologyProperty,
)
from neoolaf.evaluation.matching.normalization import normalize_text

try:
    from rdflib import Graph, Literal, RDF, RDFS, OWL
except Exception:  # pragma: no cover
    Graph = None
    Literal = None
    RDF = None
    RDFS = None
    OWL = None


def uri_local_name(value: object) -> str:
    """Extract a readable local name from a URIRef, Literal, or string."""
    if Literal is not None and isinstance(value, Literal):
        return str(value)
    text = str(value)
    if "#" in text:
        text = text.split("#")[-1]
    else:
        text = text.rstrip("/").split("/")[-1]
    return text.replace("%20", " ")


def load_graph(path: str | Path):
    """Load an RDF graph from Turtle/XML with a small fallback loop."""
    if Graph is None:
        raise RuntimeError("rdflib is required for TTL/XML evaluation. Install rdflib.")
    graph = Graph()
    errors = []
    for fmt in ("turtle", "xml", "nt"):
        try:
            graph.parse(str(path), format=fmt)
            return graph
        except Exception as exc:
            errors.append(f"{fmt}: {exc}")
    raise ValueError(f"Could not parse RDF file {path}. Errors: {' | '.join(errors)}")


def ontology_from_ttl(path: str | Path | None) -> EvalOntology:
    """Extract ontology classes/properties/hierarchy from a TTL/XML file."""
    if not path or not Path(path).exists():
        return EvalOntology(ttl_path=str(path) if path else None)

    graph = load_graph(path)
    classes: dict[str, EvalOntologyClass] = {}
    properties: dict[str, EvalOntologyProperty] = {}
    hierarchy_links: list[EvalHierarchyLink] = []

    labels: dict[str, str] = {}
    comments: dict[str, str] = {}
    domains: dict[str, str] = {}
    ranges: dict[str, str] = {}

    for subject, predicate, obj in graph:
        pred_name = normalize_text(uri_local_name(predicate))
        subj_key = str(subject)
        if pred_name == "label":
            labels[subj_key] = str(obj)
        elif pred_name in {"comment", "description"}:
            comments[subj_key] = str(obj)
        elif pred_name == "domain":
            domains[subj_key] = uri_local_name(obj)
        elif pred_name == "range":
            ranges[subj_key] = uri_local_name(obj)

    for subject, predicate, obj in graph:
        pred_name = normalize_text(uri_local_name(predicate))
        obj_name = normalize_text(uri_local_name(obj))
        subj_key = str(subject)
        label = labels.get(subj_key) or uri_local_name(subject)

        if pred_name == "type" and obj_name in {"class", "owl class"}:
            classes[subj_key] = EvalOntologyClass(
                iri=subj_key,
                label=label,
                description=comments.get(subj_key),
            )
        elif pred_name == "type" and obj_name in {"objectproperty", "object property", "datatypeproperty", "datatype property", "rdf property"}:
            properties[subj_key] = EvalOntologyProperty(
                iri=subj_key,
                label=label,
                domain=domains.get(subj_key),
                range=ranges.get(subj_key),
                description=comments.get(subj_key),
            )
        elif pred_name == "subclassof":
            hierarchy_links.append(EvalHierarchyLink(child=label, parent=labels.get(str(obj)) or uri_local_name(obj)))

    # Some generated TTLs only contain labels/comments without explicit OWL typing.
    for subject_key, label in labels.items():
        if subject_key not in classes and subject_key not in properties:
            if normalize_text(label).startswith("has ") or normalize_text(label) in {"causes", "references", "requires"}:
                properties[subject_key] = EvalOntologyProperty(
                    iri=subject_key,
                    label=label,
                    description=comments.get(subject_key),
                    domain=domains.get(subject_key),
                    range=ranges.get(subject_key),
                )
            else:
                classes[subject_key] = EvalOntologyClass(
                    iri=subject_key,
                    label=label,
                    description=comments.get(subject_key),
                )

    return EvalOntology(
        classes=list(classes.values()),
        properties=list(properties.values()),
        hierarchy_links=hierarchy_links,
        axioms=[],
        ttl_path=str(path),
    )

"""Canonical evaluation artifacts used by all evaluation adapters.

The goal of this schema is to make SinglePass, TaxoDrivenKG, NeoOLAF,
and future methods comparable through the same metric code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalEntity:
    """Canonical entity representation."""

    label: str
    id: str | None = None
    type: str | None = None
    evidence: str | None = None
    provenance_present: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRelation:
    """Canonical relation/triple representation.

    For relation extraction datasets, this is the main prediction unit.
    For KG construction, it corresponds to a graph triple simplified to
    head / relation / tail labels.
    """

    head: str
    relation: str
    tail: str
    evidence: str | None = None
    confidence: float | None = None
    provenance_present: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalOntologyClass:
    """Canonical ontology class representation."""

    label: str
    iri: str | None = None
    description: str | None = None
    evidence: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalOntologyProperty:
    """Canonical ontology property representation."""

    label: str
    iri: str | None = None
    domain: str | None = None
    range: str | None = None
    description: str | None = None
    evidence: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalHierarchyLink:
    """Canonical hierarchy link representation."""

    child: str
    parent: str
    relation: str = "subClassOf"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalAxiom:
    """Canonical axiom representation."""

    label: str
    expression: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalOntology:
    """Canonical ontology artifact extracted from JSON or TTL."""

    classes: list[EvalOntologyClass] = field(default_factory=list)
    properties: list[EvalOntologyProperty] = field(default_factory=list)
    hierarchy_links: list[EvalHierarchyLink] = field(default_factory=list)
    axioms: list[EvalAxiom] = field(default_factory=list)
    ttl_path: str | None = None


@dataclass(slots=True)
class EvalDocument:
    """Canonical document metadata."""

    document_id: str
    text: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationArtifact:
    """Canonical run artifact consumed by metrics."""

    method: str
    dataset: str
    profile: str
    run_id: str
    documents: list[EvalDocument] = field(default_factory=list)
    entities_by_doc: dict[str, list[EvalEntity]] = field(default_factory=dict)
    relations_by_doc: dict[str, list[EvalRelation]] = field(default_factory=dict)
    ontology_by_doc: dict[str, EvalOntology] = field(default_factory=dict)
    global_ontology: EvalOntology | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)

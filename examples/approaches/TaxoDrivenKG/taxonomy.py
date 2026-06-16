"""Ontology loading and lexical retrieval utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS


@dataclass
class OntologyEntry:
    """One ontology entry harvested from the seed ontology."""
    uri: str
    label: str
    entry_type: str  # class / property / other


class OntologyRetriever:
    """Harvest labels from an OWL ontology and retrieve lexical matches."""

    def __init__(self, ontology_path: str | Path) -> None:
        """Load ontology and index entries."""
        self.ontology_path = str(ontology_path)
        self.graph = Graph()
        self.graph.parse(self.ontology_path)
        self.entries: List[OntologyEntry] = self._harvest_entries()

    def _local_name(self, uri: URIRef) -> str:
        """Extract a readable local name from a URI."""
        value = str(uri)
        if "#" in value:
            return value.split("#")[-1]
        return value.rstrip("/").split("/")[-1]

    def _harvest_entries(self) -> List[OntologyEntry]:
        """Extract ontology classes and properties with labels."""
        entries: List[OntologyEntry] = []
        seen = set()

        def add_entry(uri: URIRef, label: str, entry_type: str) -> None:
            key = (str(uri), label.strip().lower(), entry_type)
            if key in seen:
                return
            seen.add(key)
            entries.append(
                OntologyEntry(
                    uri=str(uri),
                    label=label.strip(),
                    entry_type=entry_type,
                )
            )

        for subject in set(self.graph.subjects(RDF.type, OWL.Class)):
            labels = list(self.graph.objects(subject, RDFS.label))
            if labels:
                for lbl in labels:
                    add_entry(subject, str(lbl), "class")
            else:
                add_entry(subject, self._local_name(subject), "class")

        for predicate_type in [OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property]:
            for subject in set(self.graph.subjects(RDF.type, predicate_type)):
                labels = list(self.graph.objects(subject, RDFS.label))
                if labels:
                    for lbl in labels:
                        add_entry(subject, str(lbl), "property")
                else:
                    add_entry(subject, self._local_name(subject), "property")

        return entries

    def retrieve(self, text: str, max_hits: int = 40) -> Dict[str, dict]:
        """Retrieve ontology entries whose labels appear in the text."""
        lowered = text.lower()
        hits: Dict[str, dict] = {}

        for entry in self.entries:
            if entry.label.lower() in lowered:
                hits[entry.label] = {
                    "uri": entry.uri,
                    "label": entry.label,
                    "type": entry.entry_type,
                    "score": 1.0,
                }
                if len(hits) >= max_hits:
                    break

        return hits
from __future__ import annotations

# Standard library imports
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Third-party imports
from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal
from rdflib.namespace import SKOS, DCTERMS, DC

# Local imports
from neoolaf.domain.seed_ontology import (
    SeedOntology,
    SeedOntologyClass,
    SeedOntologyProperty,
)




class SeedOntologyLoader:
    """
    Load a seed/source ontology from RDF/OWL/Turtle and convert it to SeedOntology.
    """

    def load(self, ontology_path: str) -> SeedOntology:
        """
        Load an ontology file into a SeedOntology structure.

        Args:
            ontology_path:
                Path to an ontology file supported by rdflib.

        Returns:
            Parsed SeedOntology.
        """
        path = Path(ontology_path)
        if not path.exists():
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

        graph = Graph()
        graph.parse(str(path))

        ontology = SeedOntology()

        # Try to recover ontology-level metadata
        for subj in graph.subjects(RDF.type, OWL.Ontology):
            ontology.ontology_uri = str(subj)
            ontology.ontology_label = self._get_best_label(graph, subj)
            ontology.ontology_description = self._get_best_description(graph, subj)
            break

        # Load classes
        for class_uri in graph.subjects(RDF.type, OWL.Class):
            self._register_class(graph, ontology, class_uri)

        # Also include RDFS classes if present
        for class_uri in graph.subjects(RDF.type, RDFS.Class):
            self._register_class(graph, ontology, class_uri)

        # Load properties
        for prop_uri in graph.subjects(RDF.type, OWL.ObjectProperty):
            self._register_property(graph, ontology, prop_uri, property_type="object_property")

        for prop_uri in graph.subjects(RDF.type, OWL.DatatypeProperty):
            self._register_property(graph, ontology, prop_uri, property_type="data_property")

        # Build child links for classes
        for class_uri, cls in ontology.classes_by_uri.items():
            for parent_uri in cls.parent_uris:
                if parent_uri in ontology.classes_by_uri:
                    ontology.classes_by_uri[parent_uri].child_uris.append(class_uri)

        # Build child links for properties
        for prop_uri, prop in ontology.properties_by_uri.items():
            for parent_uri in prop.parent_uris:
                if parent_uri in ontology.properties_by_uri:
                    ontology.properties_by_uri[parent_uri].child_uris.append(prop_uri)

        return ontology

    def _register_class(self, graph: Graph, ontology: SeedOntology, class_uri: URIRef) -> None:
        """
        Register one class in the seed ontology if not already present.
        """
        uri_str = str(class_uri)
        if uri_str in ontology.classes_by_uri:
            return

        # Get the main label, or fall back to the URI fragment
        label = self._get_best_label(graph, class_uri) or self._fallback_label(uri_str)

        # Get an optional textual description
        description = self._get_best_description(graph, class_uri)

        # Get optional alternative labels
        alt_labels = self._get_alt_labels(graph, class_uri)

        # Get parent class URIs
        parent_uris = [str(obj) for obj in graph.objects(class_uri, RDFS.subClassOf)]

        # Register the class object
        ontology.classes_by_uri[uri_str] = SeedOntologyClass(
            uri=uri_str,
            label=label,
            description=description,
            alt_labels=alt_labels,
            parent_uris=parent_uris,
            child_uris=[],
        )

        # Index the main label
        ontology.class_uris_by_label.setdefault(label.lower().strip(), []).append(uri_str)

        # Index optional alternative labels too
        for alt_label in alt_labels:
            ontology.class_uris_by_label.setdefault(alt_label.lower().strip(), []).append(uri_str)


    def _register_property(
        self,
        graph: Graph,
        ontology: SeedOntology,
        prop_uri: URIRef,
        property_type: str,
    ) -> None:
        """
        Register one property in the seed ontology if not already present.
        """
        uri_str = str(prop_uri)
        if uri_str in ontology.properties_by_uri:
            return

        # Get the main label, or fall back to the URI fragment
        label = self._get_best_label(graph, prop_uri) or self._fallback_label(uri_str)

        # Get an optional textual description
        description = self._get_best_description(graph, prop_uri)

        # Get optional alternative labels
        alt_labels = self._get_alt_labels(graph, prop_uri)

        # Get structural information
        domain_uris = [str(obj) for obj in graph.objects(prop_uri, RDFS.domain)]
        range_uris = [str(obj) for obj in graph.objects(prop_uri, RDFS.range)]
        parent_uris = [str(obj) for obj in graph.objects(prop_uri, RDFS.subPropertyOf)]

        # Register the property object
        ontology.properties_by_uri[uri_str] = SeedOntologyProperty(
            uri=uri_str,
            label=label,
            property_type=property_type,
            description=description,
            alt_labels=alt_labels,
            domain_uris=domain_uris,
            range_uris=range_uris,
            parent_uris=parent_uris,
            child_uris=[],
        )

        # Index the main label
        ontology.property_uris_by_label.setdefault(label.lower().strip(), []).append(uri_str)

        # Index optional alternative labels too
        for alt_label in alt_labels:
            ontology.property_uris_by_label.setdefault(alt_label.lower().strip(), []).append(uri_str)

    def _get_best_label(self, graph: Graph, uri: URIRef) -> Optional[str]:
        """
        Try several common label predicates.
        """
        predicates = [RDFS.label, SKOS.prefLabel]
        for predicate in predicates:
            for obj in graph.objects(uri, predicate):
                if isinstance(obj, Literal):
                    value = str(obj).strip()
                    if value:
                        return value
        return None

    def _get_best_description(self, graph: Graph, uri: URIRef) -> Optional[str]:
        """
        Try several common description predicates.
        """
        predicates = [
            RDFS.comment,
            SKOS.definition,
            DCTERMS.description,
            DC.description,
        ]

        for predicate in predicates:
            for obj in graph.objects(uri, predicate):
                if isinstance(obj, Literal):
                    value = str(obj).strip()
                    if value:
                        return value
        return None

    def _fallback_label(self, uri_str: str) -> str:
        """
        Fallback label extraction from URI fragment.
        """
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        return uri_str.rstrip("/").split("/")[-1]
    
    def _get_alt_labels(self, graph: Graph, uri: URIRef) -> list[str]:
        """
        Collect optional alternative labels.
        """
        values = []

        predicates = [
            SKOS.altLabel,
        ]

        for predicate in predicates:
            for obj in graph.objects(uri, predicate):
                if isinstance(obj, Literal):
                    value = str(obj).strip()
                    if value:
                        values.append(value)

        # Deduplicate while preserving order
        return list(dict.fromkeys(values))
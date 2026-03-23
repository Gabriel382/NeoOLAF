from __future__ import annotations

# Standard library imports
from difflib import SequenceMatcher
from typing import List, Tuple

# Local imports
from neoolaf.domain.seed_ontology import SeedOntology, SeedOntologyClass, SeedOntologyProperty


class SeedOntologyRetriever:
    """
    Retrieval utilities over a loaded seed ontology.
    """

    def __init__(self, ontology: SeedOntology) -> None:
        """
        Initialize the retriever.
        """
        self.ontology = ontology

    def nearest_classes(self, query: str, top_k: int = 5) -> List[SeedOntologyClass]:
        """
        Return the nearest class labels to a query string.
        """
        query_norm = query.lower().strip()
        scored: List[Tuple[float, SeedOntologyClass]] = []

        for cls in self.ontology.get_classes():
            score = SequenceMatcher(None, query_norm, cls.label.lower().strip()).ratio()
            scored.append((score, cls))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k] if item[0] > 0]

    def nearest_properties(self, query: str, top_k: int = 5) -> List[SeedOntologyProperty]:
        """
        Return the nearest property labels to a query string.
        """
        query_norm = query.lower().strip()
        scored: List[Tuple[float, SeedOntologyProperty]] = []

        for prop in self.ontology.get_properties():
            score = SequenceMatcher(None, query_norm, prop.label.lower().strip()).ratio()
            scored.append((score, prop))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k] if item[0] > 0]

    def class_context(self, class_uri: str) -> dict:
        """
        Return parent/child context for a class URI.
        """
        cls = self.ontology.classes_by_uri.get(class_uri)
        if cls is None:
            return {}

        return {
            "uri": cls.uri,
            "label": cls.label,
            "description": cls.description,
            "parents": [
                self.ontology.classes_by_uri[parent_uri].label
                for parent_uri in cls.parent_uris
                if parent_uri in self.ontology.classes_by_uri
            ],
            "children": [
                self.ontology.classes_by_uri[child_uri].label
                for child_uri in cls.child_uris
                if child_uri in self.ontology.classes_by_uri
            ],
        }

    def property_context(self, property_uri: str) -> dict:
        """
        Return parent/child/domain/range context for a property URI.
        """
        prop = self.ontology.properties_by_uri.get(property_uri)
        if prop is None:
            return {}

        return {
            "uri": prop.uri,
            "label": prop.label,
            "description": prop.description,
            "type": prop.property_type,
            "domain_labels": [
                self.ontology.classes_by_uri[uri].label
                for uri in prop.domain_uris
                if uri in self.ontology.classes_by_uri
            ],
            "range_labels": [
                self.ontology.classes_by_uri[uri].label
                for uri in prop.range_uris
                if uri in self.ontology.classes_by_uri
            ],
            "parents": [
                self.ontology.properties_by_uri[parent_uri].label
                for parent_uri in prop.parent_uris
                if parent_uri in self.ontology.properties_by_uri
            ],
            "children": [
                self.ontology.properties_by_uri[child_uri].label
                for child_uri in prop.child_uris
                if child_uri in self.ontology.properties_by_uri
            ],
        }
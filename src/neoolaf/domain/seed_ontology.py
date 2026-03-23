from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SeedOntologyClass:
    """
    One ontology class loaded from the source ontology.
    """

    # Stable URI of the class
    uri: str

    # Preferred label
    label: str

    # Optional textual description
    description: Optional[str] = None

    # Parent class URIs
    parent_uris: List[str] = field(default_factory=list)

    # Child class URIs
    child_uris: List[str] = field(default_factory=list)

    alt_labels: List[str] = field(default_factory=list)


@dataclass
class SeedOntologyProperty:
    """
    One ontology property loaded from the source ontology.
    """

    # Stable URI of the property
    uri: str

    # Preferred label
    label: str

    # Property kind: object_property or data_property
    property_type: str

    # Optional textual description
    description: Optional[str] = None

    # Optional domain/range URIs
    domain_uris: List[str] = field(default_factory=list)
    range_uris: List[str] = field(default_factory=list)

    # Parent property URIs
    parent_uris: List[str] = field(default_factory=list)

    # Child property URIs
    child_uris: List[str] = field(default_factory=list)

    alt_labels: List[str] = field(default_factory=list)


@dataclass
class SeedOntology:
    """
    In-memory representation of a seed/source ontology.

    This object is used as an input semantic constraint for NeoOLAF.
    """

    # Ontology metadata
    ontology_uri: Optional[str] = None
    ontology_label: Optional[str] = None
    ontology_description: Optional[str] = None

    # Indexed classes and properties
    classes_by_uri: Dict[str, SeedOntologyClass] = field(default_factory=dict)
    properties_by_uri: Dict[str, SeedOntologyProperty] = field(default_factory=dict)

    # Label indexes for retrieval
    class_uris_by_label: Dict[str, List[str]] = field(default_factory=dict)
    property_uris_by_label: Dict[str, List[str]] = field(default_factory=dict)

    def get_classes(self) -> List[SeedOntologyClass]:
        """
        Return all loaded classes.
        """
        return list(self.classes_by_uri.values())

    def get_properties(self) -> List[SeedOntologyProperty]:
        """
        Return all loaded properties.
        """
        return list(self.properties_by_uri.values())
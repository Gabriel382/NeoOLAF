from __future__ import annotations

# Local imports
from neoolaf.domain.seed_ontology import SeedOntology, SeedOntologyClass


def build_minimal_seed_ontology() -> SeedOntology:
    """
    Build a minimal fallback ontology containing only owl:Thing-like semantics.

    This is useful when no seed ontology is provided but the pipeline still
    expects a valid ontology object.
    """
    ontology = SeedOntology(
        ontology_uri="http://www.w3.org/2002/07/owl#",
        ontology_label="Minimal Seed Ontology",
        ontology_description="Fallback seed ontology containing a minimal Thing concept.",
    )

    thing_uri = "http://www.w3.org/2002/07/owl#Thing"

    thing_class = SeedOntologyClass(
        uri=thing_uri,
        label="Thing",
        description="The most general class in the minimal seed ontology.",
        parent_uris=[],
        child_uris=[],
    )

    ontology.classes_by_uri[thing_uri] = thing_class
    ontology.class_uris_by_label["thing"] = [thing_uri]

    return ontology
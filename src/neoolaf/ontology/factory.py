from __future__ import annotations

# Local imports
from neoolaf.domain.seed_ontology import SeedOntology
from neoolaf.ontology.loader import SeedOntologyLoader
from neoolaf.ontology.minimal import build_minimal_seed_ontology


def build_seed_ontology(seed_ontology_input: str | None) -> SeedOntology | None:
    """
    Build a seed ontology from one of three supported modes.

    Supported modes:
    - None:
        no ontology is used
    - "minimal":
        build a minimal fallback ontology containing only a Thing-like class
    - any other string:
        interpret it as a path to a real ontology file and load it

    Args:
        seed_ontology_input:
            Input mode or ontology file path.

    Returns:
        A SeedOntology object or None.
    """
    # No ontology mode
    if seed_ontology_input is None:
        return None

    # Minimal fallback ontology mode
    if seed_ontology_input == "minimal":
        return build_minimal_seed_ontology()

    # Real ontology file mode
    loader = SeedOntologyLoader()
    return loader.load(seed_ontology_input)
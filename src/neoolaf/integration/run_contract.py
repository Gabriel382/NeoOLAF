from __future__ import annotations

"""Descriptive contract for existing NeoOLAF run directories.

Nothing in this module participates in pipeline execution.  It only names the
artifacts already produced by the published L0--L12 implementation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerContract:
    index: int
    name: str


SCIENTIFIC_LAYERS: tuple[LayerContract, ...] = (
    LayerContract(0, "layer00_preprocessing"),
    LayerContract(1, "layer01_linguistic_expression_extraction"),
    LayerContract(2, "layer02_candidate_enrichment"),
    LayerContract(3, "layer03_candidate_typing_resolution"),
    LayerContract(4, "layer04_candidate_relation_extraction"),
    LayerContract(5, "layer05_candidate_triple_generation"),
    LayerContract(6, "layer06_concept_relation_induction"),
    LayerContract(7, "layer07_hierarchisation"),
    LayerContract(8, "layer08_axiom_schemata_extraction"),
    LayerContract(9, "layer09_general_axiom_extraction"),
    LayerContract(10, "layer10_validation_reasoning"),
    LayerContract(11, "layer11_inference_completion"),
    LayerContract(12, "layer12_serialization"),
)

SCIENTIFIC_LAYER_NAMES: tuple[str, ...] = tuple(layer.name for layer in SCIENTIFIC_LAYERS)
CANONICAL_LAYER_FILES: tuple[str, ...] = ("state.json", "output.json", "metadata.json")

EXPORT_NAMES: tuple[str, ...] = (
    "ontology_local.ttl",
    "ontology_inferred.ttl",
    "kg_local.ttl",
    "kg_inferred.ttl",
    "kg_local.json",
    "kg_inferred.json",
)

# Newer runs use data/exports.  The additional roots are read-only compatibility
# fallbacks for historical or externally copied runs.
EXPORT_SEARCH_ROOTS: tuple[str, ...] = (
    "data/exports",
    "exports",
    "layer12_serialization/data/exports",
)

RUN_CONFIG_FILENAMES: tuple[str, ...] = ("run_config.json",)
RUN_MANIFEST_FILENAMES: tuple[str, ...] = (
    "orchestration_manifest.json",
    "run_manifest.json",
)
CHECKPOINT_MANIFEST = "checkpoints/manifest.json"

# JSON files that are infrastructure metadata rather than legacy layer payloads.
RESERVED_LAYER_JSON_NAMES: frozenset[str] = frozenset(
    {
        *CANONICAL_LAYER_FILES,
        "prompt_stats.json",
        "failed_chunks.json",
        "failed_items.json",
        "failed_expressions.json",
    }
)

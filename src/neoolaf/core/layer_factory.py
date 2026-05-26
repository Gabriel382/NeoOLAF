from __future__ import annotations

"""Factory for the default NeoOLAF layer sequence used by CLI and tests."""

from typing import Any

from neoolaf.core.base_layer import BaseLayer


LLM_LAYER_INDEXES = {1, 2, 3, 4, 6, 7, 8, 9}


def build_default_layers(
    *,
    llm_backend: Any | None = None,
    rag_backend: Any | None = None,
    verbose: bool = False,
    save_intermediate: bool = True,
    chunk_size: int = 1500,
    overlap: int = 200,
    max_chunks_layer01: int | None = None,
    profile_config: dict | None = None,
    translate_preprocessing: bool = False,
    translator: Any | None = None,
    source_language: str | None = None,
    target_language: str = "en",
) -> list[BaseLayer]:
    """
    Build the standard layer list.

    Imports are lazy so `python -m neoolaf --help` does not fail when optional
    runtime dependencies such as langgraph are not installed yet.
    """
    from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
    from neoolaf.layers.layer01_linguistic_expression_extraction.component import LinguisticExpressionExtractionLayer
    from neoolaf.layers.layer02_candidate_enrichment.component import CandidateEnrichmentLayer
    from neoolaf.layers.layer03_candidate_typing_resolution.component import CandidateTypingResolutionLayer
    from neoolaf.layers.layer04_candidate_relation_extraction.component import CandidateRelationExtractionLayer
    from neoolaf.layers.layer05_candidate_triple_generation.component import CandidateTripleGenerationLayer
    from neoolaf.layers.layer06_concept_relation_induction.component import ConceptRelationInductionLayer
    from neoolaf.layers.layer07_hierarchisation.component import HierarchisationLayer
    from neoolaf.layers.layer08_axiom_schemata_extraction.component import AxiomSchemataExtractionLayer
    from neoolaf.layers.layer09_general_axiom_extraction.component import GeneralAxiomExtractionLayer
    from neoolaf.layers.layer10_validation_reasoning.component import ValidationReasoningLayer
    from neoolaf.layers.layer11_inference_completion.component import InferenceCompletionLayer
    from neoolaf.layers.layer12_serialization.component import SerializationLayer

    return [
        PreprocessingLayer(
            chunk_size=chunk_size,
            overlap=overlap,
            save_intermediate=save_intermediate,
            verbose=verbose,
            profile_config=profile_config,
            translate=translate_preprocessing,
            translator=translator,
            source_language=source_language,
            target_language=target_language,
        ),
        LinguisticExpressionExtractionLayer(
            ollama_backend=llm_backend,
            max_chunks=max_chunks_layer01,
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        CandidateEnrichmentLayer(
            ollama_backend=llm_backend,
            rag_adapter=rag_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        CandidateTypingResolutionLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        CandidateRelationExtractionLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        CandidateTripleGenerationLayer(
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        ConceptRelationInductionLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        HierarchisationLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        AxiomSchemataExtractionLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        GeneralAxiomExtractionLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        ValidationReasoningLayer(
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        InferenceCompletionLayer(
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
        SerializationLayer(
            save_intermediate=save_intermediate,
            verbose=verbose,
        ),
    ]

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
    max_concurrency_layer01: int = 1,
    max_concurrency_layer02: int = 1,
    max_concurrency_layer03: int = 1,
    max_concurrency_layer04: int = 1,
    max_concurrency_layer05: int = 1,
    max_concurrency_layer06: int = 1,
    max_concurrency_layer07: int = 1,
    max_concurrency_layer08: int = 1,
    max_concurrency_layer09: int = 1,
    max_concurrency_layer10: int = 1,
    max_concurrency_layer11: int = 1,
    max_concurrency_layer12: int = 1,
    retry_failed_calls: int = 0,
    retry_sleep_seconds: float = 2.0,
    rag_layer01_enabled: bool | None = None,
    rag_top_k_layer01: int = 0,
    rag_max_chars_layer01: int = 0,
    failed_chunks_file_layer01: str | None = None,
    failed_expressions_file_layer02: str | None = None,
    failed_items_file_layer03: str | None = None,
    max_expressions_layer02: int | None = None,
    max_expressions_layer03: int | None = None,
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
            rag_backend=rag_backend,
            max_concurrency=max_concurrency_layer01,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
            rag_enabled=rag_layer01_enabled,
            rag_top_k=rag_top_k_layer01,
            rag_max_chars=rag_max_chars_layer01,
            failed_chunks_file=failed_chunks_file_layer01,
        ),
        CandidateEnrichmentLayer(
            ollama_backend=llm_backend,
            rag_adapter=rag_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
            max_expressions=max_expressions_layer02,
            max_concurrency=max_concurrency_layer02,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
            failed_expressions_file=failed_expressions_file_layer02,
        ),
        CandidateTypingResolutionLayer(
            ollama_backend=llm_backend,
            rag_adapter=rag_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
            max_expressions=max_expressions_layer03,
            max_concurrency=max_concurrency_layer03,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
            failed_items_file=failed_items_file_layer03,
        ),
        CandidateRelationExtractionLayer(
            ollama_backend=llm_backend,
            rag_adapter=rag_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
            max_concurrency=max_concurrency_layer04,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        CandidateTripleGenerationLayer(
            save_intermediate=save_intermediate,
            verbose=verbose,
            max_concurrency=max_concurrency_layer05,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        ConceptRelationInductionLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
            rag_adapter=rag_backend,
            max_concurrency=max_concurrency_layer06,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        HierarchisationLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
            rag_adapter=rag_backend,
            max_concurrency=max_concurrency_layer07,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        AxiomSchemataExtractionLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
            rag_adapter=rag_backend,
            max_concurrency=max_concurrency_layer08,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        GeneralAxiomExtractionLayer(
            ollama_backend=llm_backend,
            save_intermediate=save_intermediate,
            verbose=verbose,
            max_concurrency=max_concurrency_layer09,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        ValidationReasoningLayer(
            save_intermediate=save_intermediate,
            verbose=verbose,
            max_concurrency=max_concurrency_layer10,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        InferenceCompletionLayer(
            save_intermediate=save_intermediate,
            verbose=verbose,
            max_concurrency=max_concurrency_layer11,
            retry_failed_calls=retry_failed_calls,
            retry_sleep_seconds=retry_sleep_seconds,
        ),
        SerializationLayer(
            save_intermediate=save_intermediate,
            verbose=verbose,
            max_concurrency=max_concurrency_layer12,
        ),
    ]

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import spacy

from olaf import Pipeline
from olaf.commons.prompts import (
    openai_prompt_concept_extraction,
    openai_prompt_relation_extraction,
)
from olaf.pipeline.pipeline_component.term_extraction import POSTermExtraction
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    LLMBasedConceptExtraction,
    LLMBasedRelationExtraction,
)

from openrouter_generator import OpenRouterGenerator


@dataclass
class OlafSmokeResult:
    elapsed_seconds: float
    concepts: list[dict[str, Any]]
    relations: list[dict[str, Any]]


def _occurrences(lrs) -> list[str]:
    out = []
    for lr in lrs:
        for co in lr.corpus_occurrences:
            out.append(co.text)
    return sorted(set(out))


def run_olaf_lite_document(
    text: str,
    spacy_model_name: str = "en_core_web_sm",
    model_name: str = "openai/gpt-oss-20b",
    reasoning_effort: str = "minimal",
    debug_log_path: str | None = None,
) -> OlafSmokeResult:
    """Cheap OLAF baseline: 2 LLM calls per document.

    1) POS noun/proper-noun/verb candidate terms -> LLM concept grouping.
    2) POS verb/aux/adposition candidate terms -> LLM relation grouping.

    Hierarchisation and OWL axiom generation are deliberately omitted because
    they do not improve the benchmark relation projection and would add cost.
    """
    nlp = spacy.load(spacy_model_name)
    doc = nlp(text)

    generator = OpenRouterGenerator(
        model_name=model_name,
        max_tokens=2048,
        reasoning_effort=reasoning_effort,
        debug_log_path=debug_log_path,
    )

    concept_terms = POSTermExtraction(
        pos_selection=["NOUN", "PROPN", "VERB"]
    )
    concept_extraction = LLMBasedConceptExtraction(
        prompt_template=openai_prompt_concept_extraction,
        llm_generator=generator,
        doc_context_max_len=4000,
    )

    relation_terms = POSTermExtraction(
        pos_selection=["VERB", "AUX", "ADP"]
    )
    relation_extraction = LLMBasedRelationExtraction(
        prompt_template=openai_prompt_relation_extraction,
        llm_generator=generator,
        doc_context_max_len=4000,
        concept_max_distance=8,
        scope="sent",
    )

    pipeline = Pipeline(
        spacy_model=nlp,
        corpus=[doc],
        pipeline_components=[
            concept_terms,
            concept_extraction,
            relation_terms,
            relation_extraction,
        ],
    )

    t0 = perf_counter()
    pipeline.run()
    elapsed = perf_counter() - t0

    concepts = []
    for c in sorted(pipeline.kr.concepts, key=lambda x: x.label):
        concepts.append(
            {
                "label": c.label,
                "linguistic_realisations": sorted(
                    {lr.label for lr in c.linguistic_realisations}
                ),
                "occurrences": _occurrences(c.linguistic_realisations),
            }
        )

    relations = []
    for r in sorted(
        pipeline.kr.relations,
        key=lambda x: (
            x.label,
            x.source_concept.label if x.source_concept else "",
            x.destination_concept.label if x.destination_concept else "",
        ),
    ):
        relations.append(
            {
                "source": r.source_concept.label if r.source_concept else None,
                "predicate": r.label,
                "target": r.destination_concept.label if r.destination_concept else None,
                "linguistic_realisations": sorted(
                    {lr.label for lr in r.linguistic_realisations}
                ),
                "occurrences": _occurrences(r.linguistic_realisations),
            }
        )

    return OlafSmokeResult(
        elapsed_seconds=elapsed,
        concepts=concepts,
        relations=relations,
    )

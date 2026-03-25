from __future__ import annotations

# Standard library imports
import json
from typing import List, Dict
# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.domain.enriched_expression import EnrichedExpression, EnrichmentEvidence
from neoolaf.layers.layer02_candidate_enrichment.graph import Layer02EnrichmentGraphFactory
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend
from neoolaf.resources.knowledge_sources.wordnet_source import WordNetSource
from neoolaf.resources.knowledge_sources.wikipedia_source import WikipediaSource
from neoolaf.resources.knowledge_sources.wikidata_source import WikidataSource
from neoolaf.resources.knowledge_sources.web_search_source import WebSearchSource


class CandidateEnrichmentLayer(BaseLayer):
    """
    Layer 2: candidate enrichment.

    Responsibilities:
    - enrich Layer 1 linguistic expressions
    - collect lexical and semantic evidence from multiple sources
    - synthesize a final enrichment with explicit lexical provenance
    """

    name = "layer02_candidate_enrichment"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        wikipedia_source: WikipediaSource | None = None,
        wikidata_source: WikidataSource | None = None,
        wordnet_source: WordNetSource | None = None,
        web_search_source: WebSearchSource | None = None,
        max_expressions: int | None = None,
        use_web_search: bool = True,
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize Layer 2.

        Args:
            ollama_backend:
                LLM backend used for synthesis.
            wikipedia_source:
                Wikipedia evidence source.
            wikidata_source:
                Wikidata evidence source.
            wordnet_source:
                WordNet lexical source.
            web_search_source:
                Optional web search source.
            max_expressions:
                Optional debug limit on how many expressions to enrich.
            use_web_search:
                Whether to include internet search.
            save_intermediate:
                Whether to save intermediate artifacts.
            verbose:
                Wheter to show logs or not.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.wikipedia_source = wikipedia_source or WikipediaSource()
        self.wikidata_source = wikidata_source or WikidataSource()
        self.wordnet_source = wordnet_source or WordNetSource()
        self.web_search_source = web_search_source or WebSearchSource()
        self.max_expressions = max_expressions
        self.use_web_search = use_web_search

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Enrich Layer 1 expressions and keep provenance for lexical items.
        """
        expressions = state.linguistic_expressions
        if self.max_expressions is not None:
            expressions = expressions[: self.max_expressions]

        enriched: List[EnrichedExpression] = []

        expression_iterator = expressions
        if self.verbose:
            expression_iterator = tqdm(expressions, desc="Layer 2 - expressions", leave=False)

        for expr in expression_iterator:
            graph = Layer02EnrichmentGraphFactory(
                ollama_backend=self.ollama_backend,
                model_name=state.llm_model,
                wikipedia_source=self.wikipedia_source,
                wikidata_source=self.wikidata_source,
                wordnet_source=self.wordnet_source,
                web_search_source=self.web_search_source,
                user_guidance=state.user_guidance,
                use_web_search=self.use_web_search,
            ).build()

            result = graph.invoke({"expression": expr})
            enrichment_result = result.get("enrichment_result", {})
            gathered_evidence = result.get("gathered_evidence", {})

            # Convert raw evidence into structured evidence objects
            evidence_objects = self._convert_evidence(
                gathered_evidence=gathered_evidence,
                enrichment_result=enrichment_result,
                llm_model=state.llm_model,
            )

            # Build provenance-aware lexical maps
            alias_sources: Dict[str, List[str]] = {}
            synonym_sources: Dict[str, List[str]] = {}
            lexical_variant_sources: Dict[str, List[str]] = {}

            # -------------------------
            # Source lexical material
            # -------------------------

            # WordNet
            self._add_items_with_source(
                alias_sources,
                gathered_evidence.get("wordnet", {}).get("aliases", []),
                "wordnet",
            )
            self._add_items_with_source(
                synonym_sources,
                gathered_evidence.get("wordnet", {}).get("synonyms", []),
                "wordnet",
            )
            self._add_items_with_source(
                lexical_variant_sources,
                gathered_evidence.get("wordnet", {}).get("lexical_variants", []),
                "wordnet",
            )

            # Wikipedia
            self._add_items_with_source(
                alias_sources,
                gathered_evidence.get("wikipedia", {}).get("aliases", []),
                "wikipedia",
            )

            # Wikidata
            self._add_items_with_source(
                alias_sources,
                gathered_evidence.get("wikidata", {}).get("aliases", []),
                "wikidata",
            )
            self._add_items_with_source(
                alias_sources,
                gathered_evidence.get("wikidata", {}).get("labels", []),
                "wikidata",
            )

            # -------------------------
            # LLM lexical material
            # -------------------------
            self._add_items_with_source(
                alias_sources,
                enrichment_result.get("aliases", []),
                "llm",
            )
            self._add_items_with_source(
                synonym_sources,
                enrichment_result.get("synonyms", []),
                "llm",
            )
            self._add_items_with_source(
                lexical_variant_sources,
                enrichment_result.get("lexical_variants", []),
                "llm",
            )

            # Final flat lists derived from the provenance maps
            aliases = list(alias_sources.keys())
            synonyms = list(synonym_sources.keys())
            lexical_variants = list(lexical_variant_sources.keys())

            enriched.append(
                EnrichedExpression(
                    base_expression=expr,
                    aliases=aliases,
                    synonyms=synonyms,
                    lexical_variants=lexical_variants,
                    alias_sources=alias_sources,
                    synonym_sources=synonym_sources,
                    lexical_variant_sources=lexical_variant_sources,
                    definition=enrichment_result.get("definition"),
                    ontology_hints=self._dedup(enrichment_result.get("ontology_hints", [])),
                    enrichment_evidence=evidence_objects,
                )
            )

        state.enriched_expressions = enriched
        state.log(
            f"[layer02_candidate_enrichment] enriched {len(state.enriched_expressions)} expressions"
        )
        return state

    def _add_items_with_source(
        self,
        mapping: Dict[str, List[str]],
        items: List[str],
        source: str,
    ) -> None:
        """
        Add lexical items to a provenance mapping.

        Example:
            "stop" -> ["wordnet", "llm"]
        """
        for item in items:
            normalized = item.strip()
            if not normalized:
                continue

            if normalized not in mapping:
                mapping[normalized] = []

            if source not in mapping[normalized]:
                mapping[normalized].append(source)

    def _dedup(self, items: List[str]) -> List[str]:
        """
        Deduplicate strings while preserving order.
        """
        cleaned = []
        for x in items:
            if isinstance(x, dict):
                x = str(x)
            if not isinstance(x, str):
                continue
            x = x.strip()
            if x:
                cleaned.append(x)
        return list(dict.fromkeys(cleaned))

    def _convert_evidence(
        self,
        gathered_evidence: dict,
        enrichment_result: dict,
        llm_model: str,
    ) -> List[EnrichmentEvidence]:
        """
        Convert raw gathered evidence and the final LLM synthesis into typed evidence objects.
        """
        evidences: List[EnrichmentEvidence] = []

        # WordNet evidence
        wordnet_data = gathered_evidence.get("wordnet", {})
        for definition in wordnet_data.get("definitions", []):
            evidences.append(
                EnrichmentEvidence(
                    source="wordnet",
                    content=definition,
                    reference=None,
                )
            )

        # Wikipedia evidence
        wikipedia_data = gathered_evidence.get("wikipedia", {})
        if wikipedia_data.get("found"):
            evidences.append(
                EnrichmentEvidence(
                    source="wikipedia",
                    content=wikipedia_data.get("summary", ""),
                    reference=wikipedia_data.get("url"),
                )
            )

        # Wikidata evidence
        wikidata_data = gathered_evidence.get("wikidata", {})
        for item in wikidata_data.get("results", []):
            text = f"{', '.join(item.get('labels', []))} -- {item.get('description', '')}".strip()
            evidences.append(
                EnrichmentEvidence(
                    source="wikidata",
                    content=text,
                    reference=item.get("url"),
                )
            )

        # Web evidence
        web_data = gathered_evidence.get("web", {})
        for item in web_data.get("results", []):
            text = f"{item.get('title', '')} -- {item.get('body', '')}".strip()
            evidences.append(
                EnrichmentEvidence(
                    source="web",
                    content=text,
                    reference=item.get("href"),
                )
            )

        # LLM evidence
        evidences.append(
            EnrichmentEvidence(
                source="llm",
                content=json.dumps(enrichment_result, ensure_ascii=False),
                reference=llm_model,
            )
        )

        return evidences

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize enriched expressions with lexical provenance.
        """
        return {
            "layer": self.name,
            "num_enriched_expressions": len(state.enriched_expressions),
            "enriched_expressions": [
                {
                    "base_expression": {
                        "expr_id": item.base_expression.expr_id,
                        "text": item.base_expression.text,
                        "label": item.base_expression.label,
                    },
                    "aliases": item.aliases,
                    "synonyms": item.synonyms,
                    "lexical_variants": item.lexical_variants,
                    "alias_sources": item.alias_sources,
                    "synonym_sources": item.synonym_sources,
                    "lexical_variant_sources": item.lexical_variant_sources,
                    "definition": item.definition,
                    "ontology_hints": item.ontology_hints,
                    "evidence": [
                        {
                            "source": ev.source,
                            "content": ev.content,
                            "reference": ev.reference,
                        }
                        for ev in item.enrichment_evidence
                    ],
                }
                for item in state.enriched_expressions
            ],
        }
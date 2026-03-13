from __future__ import annotations

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Local imports
from neoolaf.layers.layer02_candidate_enrichment.schema import EnrichmentGraphState
from neoolaf.layers.layer02_candidate_enrichment.prompt import (
    build_system_prompt,
    build_user_prompt,
)
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend
from neoolaf.resources.knowledge_sources.wordnet_source import WordNetSource
from neoolaf.resources.knowledge_sources.wikipedia_source import WikipediaSource
from neoolaf.resources.knowledge_sources.wikidata_source import WikidataSource
from neoolaf.resources.knowledge_sources.web_search_source import WebSearchSource


class Layer02EnrichmentGraphFactory:
    """
    Factory that builds the LangGraph workflow for Layer 2.
    """

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        model_name: str,
        wikipedia_source: WikipediaSource,
        wikidata_source: WikidataSource,
        wordnet_source: WordNetSource,
        web_search_source: WebSearchSource | None = None,
        user_guidance=None,
        use_web_search: bool = True,
    ) -> None:
        self.ollama_backend = ollama_backend
        self.model_name = model_name
        self.wikipedia_source = wikipedia_source
        self.wikidata_source = wikidata_source
        self.wordnet_source = wordnet_source
        self.web_search_source = web_search_source
        self.user_guidance = user_guidance
        self.use_web_search = use_web_search

    def build(self):
        """
        Build and compile the enrichment graph.
        """
        graph = StateGraph(EnrichmentGraphState)

        graph.add_node("select_sources", self.select_sources)
        graph.add_node("fetch_wordnet", self.fetch_wordnet)
        graph.add_node("fetch_wikipedia", self.fetch_wikipedia)
        graph.add_node("fetch_wikidata", self.fetch_wikidata)
        graph.add_node("fetch_web", self.fetch_web)
        graph.add_node("merge_evidence", self.merge_evidence)
        graph.add_node("synthesize_enrichment", self.synthesize_enrichment)

        graph.add_edge(START, "select_sources")
        graph.add_edge("select_sources", "fetch_wordnet")
        graph.add_edge("fetch_wordnet", "fetch_wikipedia")
        graph.add_edge("fetch_wikipedia", "fetch_wikidata")
        graph.add_edge("fetch_wikidata", "fetch_web")
        graph.add_edge("fetch_web", "merge_evidence")
        graph.add_edge("merge_evidence", "synthesize_enrichment")
        graph.add_edge("synthesize_enrichment", END)

        return graph.compile()

    def select_sources(self, state: EnrichmentGraphState) -> EnrichmentGraphState:
        """
        Select all default sources, with optional web search.
        """
        selected = ["wordnet", "wikipedia", "wikidata"]
        if self.use_web_search:
            selected.append("web")
        return {"selected_sources": selected}

    def fetch_wordnet(self, state: EnrichmentGraphState) -> EnrichmentGraphState:
        """
        Fetch WordNet lexical evidence.
        """
        expr = state["expression"]
        return {"wordnet_result": self.wordnet_source.search(expr.text)}

    def fetch_wikipedia(self, state: EnrichmentGraphState) -> EnrichmentGraphState:
        """
        Fetch Wikipedia evidence.
        """
        expr = state["expression"]
        return {"wikipedia_result": self.wikipedia_source.search(expr.text)}

    def fetch_wikidata(self, state: EnrichmentGraphState) -> EnrichmentGraphState:
        """
        Fetch Wikidata evidence.
        """
        expr = state["expression"]
        return {"wikidata_result": self.wikidata_source.search(expr.text)}

    def fetch_web(self, state: EnrichmentGraphState) -> EnrichmentGraphState:
        """
        Fetch web evidence if enabled.
        """
        if not self.use_web_search or self.web_search_source is None:
            return {"web_result": {"source": "web", "term": state["expression"].text, "results": []}}

        expr = state["expression"]
        return {"web_result": self.web_search_source.search(expr.text)}

    def merge_evidence(self, state: EnrichmentGraphState) -> EnrichmentGraphState:
        """
        Merge raw source outputs into one evidence payload.
        """
        gathered = {
            "wordnet": state.get("wordnet_result", {}),
            "wikipedia": state.get("wikipedia_result", {}),
            "wikidata": state.get("wikidata_result", {}),
            "web": state.get("web_result", {}),
        }
        return {"gathered_evidence": gathered}

    def synthesize_enrichment(self, state: EnrichmentGraphState) -> EnrichmentGraphState:
        """
        Use the LLM to synthesize final enrichment from the gathered evidence.
        """
        expr = state["expression"]
        evidence = state.get("gathered_evidence", {})

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(
                    expression=expr,
                    gathered_evidence=evidence,
                    guidance=self.user_guidance,
                ),
            },
        ]

        raw = self.ollama_backend.chat(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )
        parsed = self.ollama_backend.extract_json(raw)

        return {"enrichment_result": parsed}
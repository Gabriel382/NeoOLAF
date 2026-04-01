from __future__ import annotations
import json
# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Local imports
from neoolaf.grounding.rag.schema import GroundingGraphState
from neoolaf.grounding.rag.prompt import (
    build_source_selection_system_prompt,
    build_source_selection_user_prompt,
    build_grounding_summary_system_prompt,
)


class SemanticRAGGraphFactory:
    """
    LangGraph-based semantic grounding workflow.
    """

    def __init__(self, registry, ollama_backend, model_name: str) -> None:
        """
        Initialize the graph factory.
        """
        self.registry = registry
        self.ollama_backend = ollama_backend
        self.model_name = model_name

    def build(self):
        """
        Build and compile the grounding graph.
        """
        graph = StateGraph(GroundingGraphState)

        graph.add_node("select_sources", self.select_sources)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("merge_items", self.merge_items)
        graph.add_node("summarize_grounding", self.summarize_grounding)

        graph.add_edge(START, "select_sources")
        graph.add_edge("select_sources", "retrieve")
        graph.add_edge("retrieve", "merge_items")
        graph.add_edge("merge_items", "summarize_grounding")
        graph.add_edge("summarize_grounding", END)

        return graph.compile()

    def select_sources(self, state: GroundingGraphState) -> GroundingGraphState:
        """
        Select retrieval sources using the LLM.
        """
        request = state["request"]
        available_sources = state["available_sources"]

        # If preferred sources were explicitly requested, keep them
        if request.preferred_sources:
            selected = [src for src in request.preferred_sources if src in available_sources]
            return {"selected_sources": selected}

        messages = [
            {"role": "system", "content": build_source_selection_system_prompt()},
            {
                "role": "user",
                "content": build_source_selection_user_prompt(
                    request_payload={
                        "layer_name": request.layer_name,
                        "query": request.query,
                        "payload": request.payload,
                        "top_k": request.top_k,
                    },
                    available_sources=available_sources,
                ),
            },
        ]

        raw = self.ollama_backend.chat(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )
        parsed = self.ollama_backend.extract_json(raw)

        selected = parsed.get("selected_sources", [])
        selected = [src for src in selected if src in available_sources]

        # Fallback if LLM returns nothing
        if not selected:
            selected = available_sources[:3]

        return {"selected_sources": selected}

    def retrieve(self, state: GroundingGraphState) -> GroundingGraphState:
        """
        Retrieve evidence from all selected sources.
        """
        request = state["request"]
        selected_sources = state.get("selected_sources", [])

        retrieved_by_source = {}

        for source_name in selected_sources:
            space = self.registry.get(source_name)
            if space is None:
                continue

            try:
                retrieved_by_source[source_name] = space.retrieve(
                    query=request.query,
                    top_k=request.top_k,
                )
            except Exception:
                retrieved_by_source[source_name] = []

        return {"retrieved_by_source": retrieved_by_source}

    def merge_items(self, state: GroundingGraphState) -> GroundingGraphState:
        """
        Flatten all retrieved items into one list.
        """
        retrieved_by_source = state.get("retrieved_by_source", {})
        merged = []

        for items in retrieved_by_source.values():
            merged.extend(items)

        return {"retrieved_items": merged}

    def summarize_grounding(self, state: GroundingGraphState) -> GroundingGraphState:
        """
        Summarize retrieved evidence for downstream use.
        """
        request = state["request"]
        retrieved_items = state.get("retrieved_items", [])

        if not retrieved_items:
            return {
                "grounding_summary": "",
                "merged_context": {},
            }

        evidence_payload = [
            {
                "source": item.source,
                "content": item.content,
                "metadata": item.metadata,
                "reference": item.reference,
            }
            for item in retrieved_items[:20]
        ]

        messages = [
            {"role": "system", "content": build_grounding_summary_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "layer_name": request.layer_name,
                        "query": request.query,
                        "payload": request.payload,
                        "retrieved_items": evidence_payload,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            },
        ]

        raw = self.ollama_backend.chat(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )
        parsed = self.ollama_backend.extract_json(raw)

        return {
            "grounding_summary": parsed.get("grounding_summary", ""),
            "merged_context": parsed.get("merged_context", {}),
        }
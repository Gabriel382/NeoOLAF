from __future__ import annotations

# Standard library imports
from pathlib import Path

# Third-party imports
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef


class KGTTLSerialiser:
    """
    Serialize NeoOLAF KG outputs into Turtle.
    """

    def __init__(self, base_uri: str = "http://neoolaf.org/resource/") -> None:
        """
        Initialize namespaces.

        Args:
            base_uri:
                Base URI used to mint graph resources.
        """
        self.base_uri = base_uri.rstrip("/") + "/"
        self.NEO = Namespace(self.base_uri)


    @staticmethod
    def _safe_local_name(value: str | None, fallback: str = "unnamed") -> str:
        """Return a stable URI-safe local name while preserving readable labels."""
        import re
        raw = str(value or "").strip() or fallback
        raw = raw.replace("&", "and")
        raw = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
        raw = re.sub(r"_+", "_", raw).strip("_")
        return raw or fallback

    def _relation_uri(self, relation_id: str | None, relation_label: str | None) -> URIRef:
        """Mint semantic relation URIs from labels, falling back to IDs only if needed."""
        if relation_label:
            return self.NEO[f"relation/{self._safe_local_name(relation_label, relation_id or 'relation')}"]
        return self.NEO[f"relation/{self._safe_local_name(relation_id, 'relation')}"]

    def _node_uri(self, node_id: str | None) -> URIRef:
        """Mint stable node URI from candidate IDs."""
        return self.NEO[f"node/{self._safe_local_name(node_id, 'node')}"]

    def serialise_local(self, state, output_path: str) -> None:
        """
        Serialize the local KG from candidate triples.
        """
        graph = self._build_base_graph()

        for triple in state.candidate_triples:
            subject_uri = self._node_uri(triple.subject_id)
            predicate_uri = self._relation_uri(triple.predicate_id, triple.predicate_label)
            object_uri = self._node_uri(triple.object_id)

            graph.add((subject_uri, predicate_uri, object_uri))

            # Labels
            graph.add((subject_uri, self.NEO.label, Literal(triple.subject_label)))
            graph.add((object_uri, self.NEO.label, Literal(triple.object_label)))
            graph.add((predicate_uri, self.NEO.label, Literal(triple.predicate_label)))
            graph.add((predicate_uri, RDF.type, OWL.ObjectProperty))

            # Types
            graph.add((subject_uri, self.NEO.nodeType, Literal(triple.subject_type)))
            graph.add((object_uri, self.NEO.nodeType, Literal(triple.object_type)))

            # Provenance and confidence on reified triple node
            assertion_uri = self.NEO[f"triple/{triple.triple_id}"]
            graph.add((assertion_uri, RDF.type, self.NEO.CandidateTriple))
            graph.add((assertion_uri, self.NEO.subject, subject_uri))
            graph.add((assertion_uri, self.NEO.predicate, predicate_uri))
            graph.add((assertion_uri, self.NEO.object, object_uri))
            graph.add((assertion_uri, self.NEO.chunkId, Literal(triple.chunk_id)))

            if triple.justification:
                graph.add((assertion_uri, self.NEO.justification, Literal(triple.justification)))

            if triple.confidence is not None:
                graph.add((assertion_uri, self.NEO.confidence, Literal(triple.confidence)))

            for idx, ev in enumerate(triple.provenance):
                ev_uri = self.NEO[f"triple/{triple.triple_id}/prov/{idx}"]
                graph.add((ev_uri, RDF.type, self.NEO.Provenance))
                graph.add((ev_uri, self.NEO.chunkId, Literal(ev.chunk_id)))
                graph.add((ev_uri, self.NEO.snippet, Literal(ev.snippet)))
                graph.add((assertion_uri, self.NEO.hasProvenance, ev_uri))

        self._write_graph(graph, output_path)

    def serialise_inferred(self, state, output_path: str) -> None:
        """
        Serialize the inferred/completed KG.
        """
        graph = self._build_base_graph()

        inferred_triples = []
        if state.reasoning_report is not None:
            inferred_triples.extend(state.reasoning_report.inferred_triples)

        for completion in state.completion_candidates:
            if completion.completed_triple is not None:
                inferred_triples.append(completion.completed_triple)

        dedup = {}
        for triple in inferred_triples:
            key = (triple.subject_id, triple.predicate_id, triple.object_id, triple.chunk_id)
            if key not in dedup:
                dedup[key] = triple

        for triple in dedup.values():
            subject_uri = self._node_uri(triple.subject_id)
            predicate_uri = self._relation_uri(triple.predicate_id, triple.predicate_label)
            object_uri = self._node_uri(triple.object_id)

            graph.add((subject_uri, predicate_uri, object_uri))
            graph.add((subject_uri, self.NEO.label, Literal(triple.subject_label)))
            graph.add((object_uri, self.NEO.label, Literal(triple.object_label)))
            graph.add((predicate_uri, self.NEO.label, Literal(triple.predicate_label)))
            graph.add((predicate_uri, RDF.type, OWL.ObjectProperty))

        self._write_graph(graph, output_path)

    def _build_base_graph(self) -> Graph:
        """
        Create a graph with namespace binding.
        """
        graph = Graph()
        graph.bind("neo", self.NEO)
        graph.bind("rdfs", RDFS)
        graph.bind("owl", OWL)
        return graph

    def _write_graph(self, graph: Graph, output_path: str) -> None:
        """
        Write the graph to Turtle.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=str(path), format="turtle")
from __future__ import annotations

# Standard library imports
from pathlib import Path

# Third-party imports
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef


class OntologyTTLSerialiser:
    """
    Serialize NeoOLAF ontology outputs into Turtle.
    """

    def __init__(self, base_uri: str = "http://neoolaf.org/resource/") -> None:
        """
        Initialize namespaces.

        Args:
            base_uri:
                Base URI used to mint ontology resources.
        """
        self.base_uri = base_uri.rstrip("/") + "/"
        self.NEO = Namespace(self.base_uri)
        self.SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

    def serialise_local(self, state, output_path: str) -> None:
        """
        Serialize the local ontology from Layers 6–9.
        """
        graph = self._build_base_graph()

        # ---------------------------------------------------------
        # Concept candidates
        # ---------------------------------------------------------
        for concept in state.concept_candidates:
            concept_uri = self.NEO[f"concept/{concept.concept_id}"]
            graph.add((concept_uri, RDF.type, OWL.Class))
            graph.add((concept_uri, RDFS.label, Literal(concept.label)))

            if concept.description:
                graph.add((concept_uri, RDFS.comment, Literal(concept.description)))

            if concept.concept_kind:
                graph.add((concept_uri, self.NEO.conceptKind, Literal(concept.concept_kind)))

            if concept.parent_hint:
                graph.add((concept_uri, self.NEO.parentHint, Literal(concept.parent_hint)))

        # ---------------------------------------------------------
        # Ontology relation candidates
        # ---------------------------------------------------------
        for relation in state.ontology_relation_candidates:
            relation_uri = self.NEO[f"relation/{relation.relation_id}"]
            graph.add((relation_uri, RDF.type, OWL.ObjectProperty))
            graph.add((relation_uri, RDFS.label, Literal(relation.label)))

            if relation.description:
                graph.add((relation_uri, RDFS.comment, Literal(relation.description)))

            if relation.domain_hint:
                graph.add((relation_uri, self.NEO.domainHint, Literal(relation.domain_hint)))

            if relation.range_hint:
                graph.add((relation_uri, self.NEO.rangeHint, Literal(relation.range_hint)))

        # ---------------------------------------------------------
        # Concept hierarchy
        # ---------------------------------------------------------
        for link in state.concept_hierarchy_links:
            child_uri = self.NEO[f"concept/{link.child_concept_id}"]
            parent_uri = self.NEO[f"concept/{link.parent_concept_id}"]
            graph.add((child_uri, RDFS.subClassOf, parent_uri))

        # ---------------------------------------------------------
        # Relation hierarchy
        # ---------------------------------------------------------
        for link in state.relation_hierarchy_links:
            child_uri = self.NEO[f"relation/{link.child_relation_id}"]
            parent_uri = self.NEO[f"relation/{link.parent_relation_id}"]
            graph.add((child_uri, RDFS.subPropertyOf, parent_uri))

        # ---------------------------------------------------------
        # General axioms
        # ---------------------------------------------------------
        for axiom in state.general_axiom_candidates:
            subject_uri = self._resolve_axiom_subject_uri(axiom)

            if axiom.predicate == "subClassOf" and axiom.object_id is not None:
                object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                graph.add((subject_uri, RDFS.subClassOf, object_uri))

            elif axiom.predicate == "domain" and axiom.object_label is not None:
                graph.add((subject_uri, RDFS.domain, Literal(axiom.object_label)))

            elif axiom.predicate == "range" and axiom.object_label is not None:
                graph.add((subject_uri, RDFS.range, Literal(axiom.object_label)))

            elif axiom.predicate == "rdfs:description" and axiom.literal_value is not None:
                graph.add((subject_uri, RDFS.comment, Literal(axiom.literal_value)))

        self._write_graph(graph, output_path)

    def serialise_inferred(self, state, output_path: str) -> None:
        """
        Serialize the inferred/completed ontology from Layers 10–11.
        """
        graph = self._build_base_graph()

        # Inferred general axioms
        if state.reasoning_report is not None:
            for axiom in state.reasoning_report.inferred_general_axioms:
                subject_uri = self._resolve_axiom_subject_uri(axiom)

                if axiom.predicate == "subClassOf" and axiom.object_id is not None:
                    object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                    graph.add((subject_uri, RDFS.subClassOf, object_uri))

                elif axiom.predicate == "domain" and axiom.object_label is not None:
                    graph.add((subject_uri, RDFS.domain, Literal(axiom.object_label)))

                elif axiom.predicate == "range" and axiom.object_label is not None:
                    graph.add((subject_uri, RDFS.range, Literal(axiom.object_label)))

                elif axiom.predicate == "rdfs:description" and axiom.literal_value is not None:
                    graph.add((subject_uri, RDFS.comment, Literal(axiom.literal_value)))

        # Completed axioms
        for completion in state.completion_candidates:
            if completion.completed_axiom is None:
                continue

            axiom = completion.completed_axiom
            subject_uri = self._resolve_axiom_subject_uri(axiom)

            if axiom.predicate == "subClassOf" and axiom.object_id is not None:
                object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                graph.add((subject_uri, RDFS.subClassOf, object_uri))

            elif axiom.predicate == "domain" and axiom.object_label is not None:
                graph.add((subject_uri, RDFS.domain, Literal(axiom.object_label)))

            elif axiom.predicate == "range" and axiom.object_label is not None:
                graph.add((subject_uri, RDFS.range, Literal(axiom.object_label)))

            elif axiom.predicate == "rdfs:description" and axiom.literal_value is not None:
                graph.add((subject_uri, RDFS.comment, Literal(axiom.literal_value)))

        self._write_graph(graph, output_path)

    def _build_base_graph(self) -> Graph:
        """
        Create a graph with common namespace bindings.
        """
        graph = Graph()
        graph.bind("neo", self.NEO)
        graph.bind("rdfs", RDFS)
        graph.bind("owl", OWL)
        graph.bind("skos", self.SKOS)
        return graph

    def _resolve_axiom_subject_uri(self, axiom) -> URIRef:
        """
        Resolve the URI for the subject of a general axiom.
        """
        if str(axiom.subject_id).startswith("concept_"):
            return self.NEO[f"concept/{axiom.subject_id}"]
        if str(axiom.subject_id).startswith("ont_rel_"):
            return self.NEO[f"relation/{axiom.subject_id}"]
        if str(axiom.subject_id).startswith("cand_"):
            return self.NEO[f"candidate/{axiom.subject_id}"]
        return self.NEO[f"resource/{axiom.subject_id}"]

    def _resolve_object_uri(self, object_id: str, object_label: str | None) -> URIRef:
        """
        Resolve the URI for an axiom object.
        """
        if object_id.startswith("concept_"):
            return self.NEO[f"concept/{object_id}"]
        if object_id.startswith("ont_rel_"):
            return self.NEO[f"relation/{object_id}"]
        return self.NEO[f"schema/{object_label or object_id}"]

    def _write_graph(self, graph: Graph, output_path: str) -> None:
        """
        Write the graph to Turtle.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=str(path), format="turtle")
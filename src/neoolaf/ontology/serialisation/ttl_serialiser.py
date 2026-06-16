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
        Serialize the merged local ontology:
        source ontology + NeoOLAF local ontology.

        This export contains:
        - the seed/source ontology, when available
        - local concept candidates
        - local ontology relation candidates
        - local hierarchy links
        - local general axioms
        """
        # Create a fresh RDF graph with namespace bindings
        graph = self._build_base_graph()

        # 1. Add the source ontology first, if one was provided
        self._add_seed_ontology(graph, state.seed_ontology)

        # 2. Add the NeoOLAF local ontology content
        self._add_local_ontology(graph, state)

        # 3. Write the merged ontology to disk
        self._write_graph(graph, output_path)


    def serialise_inferred(self, state, output_path: str) -> None:
        """
        Serialize the merged inferred/completed ontology:
        source ontology + inferred/completed ontology content.

        This export contains:
        - the seed/source ontology, when available
        - inferred general axioms from Layer 10
        - completed ontology axioms from Layer 11
        """
        # Create a fresh RDF graph with namespace bindings
        graph = self._build_base_graph()

        # 1. Add the source ontology first, if one was provided
        self._add_seed_ontology(graph, state.seed_ontology)

        # 2. Add inferred general axioms from reasoning
        if state.reasoning_report is not None:
            for axiom in state.reasoning_report.inferred_general_axioms:
                subject_uri = self._resolve_axiom_subject_uri_with_state(axiom, state)

                if axiom.predicate == "subClassOf" and axiom.object_id is not None:
                    object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                    graph.add((subject_uri, RDFS.subClassOf, object_uri))

                elif axiom.predicate == "domain" and axiom.object_label is not None:
                    graph.add((subject_uri, RDFS.domain, Literal(axiom.object_label)))

                elif axiom.predicate == "range" and axiom.object_label is not None:
                    graph.add((subject_uri, RDFS.range, Literal(axiom.object_label)))

                elif axiom.predicate in {"rdfs:description", "rdfs:comment"} and axiom.literal_value is not None:
                    graph.add((subject_uri, RDFS.comment, Literal(axiom.literal_value)))

                elif axiom.predicate == "rdf:type" and axiom.object_id is not None:
                    object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                    graph.add((subject_uri, RDF.type, object_uri))

                elif axiom.predicate == "owl:sameAs" and axiom.object_id is not None:
                    object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                    graph.add((subject_uri, OWL.sameAs, object_uri))

                elif axiom.predicate == "skos:prefLabel" and axiom.literal_value is not None:
                    graph.add((subject_uri, self.SKOS.prefLabel, Literal(axiom.literal_value)))

                elif axiom.predicate == "skos:altLabel" and axiom.literal_value is not None:
                    graph.add((subject_uri, self.SKOS.altLabel, Literal(axiom.literal_value)))

        # 3. Add completed ontology axioms from Layer 11
        for completion in state.completion_candidates:
            if completion.completed_axiom is None:
                continue

            axiom = completion.completed_axiom
            subject_uri = self._resolve_axiom_subject_uri_with_state(axiom, state)

            if axiom.predicate == "subClassOf" and axiom.object_id is not None:
                object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                graph.add((subject_uri, RDFS.subClassOf, object_uri))

            elif axiom.predicate == "domain" and axiom.object_label is not None:
                graph.add((subject_uri, RDFS.domain, Literal(axiom.object_label)))

            elif axiom.predicate == "range" and axiom.object_label is not None:
                graph.add((subject_uri, RDFS.range, Literal(axiom.object_label)))

            elif axiom.predicate in {"rdfs:description", "rdfs:comment"} and axiom.literal_value is not None:
                graph.add((subject_uri, RDFS.comment, Literal(axiom.literal_value)))

            elif axiom.predicate == "rdf:type" and axiom.object_id is not None:
                object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                graph.add((subject_uri, RDF.type, object_uri))

            elif axiom.predicate == "owl:sameAs" and axiom.object_id is not None:
                object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                graph.add((subject_uri, OWL.sameAs, object_uri))

            elif axiom.predicate == "skos:prefLabel" and axiom.literal_value is not None:
                graph.add((subject_uri, self.SKOS.prefLabel, Literal(axiom.literal_value)))

            elif axiom.predicate == "skos:altLabel" and axiom.literal_value is not None:
                graph.add((subject_uri, self.SKOS.altLabel, Literal(axiom.literal_value)))

        # 4. Write the merged inferred ontology to disk
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


    @staticmethod
    def _safe_local_name(value: str | None, fallback: str = "unnamed") -> str:
        """Return a stable URI-safe local name for RDF resources."""
        import re
        raw = str(value or "").strip() or fallback
        raw = raw.replace("&", "and")
        raw = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
        raw = re.sub(r"_+", "_", raw).strip("_")
        return raw or fallback

    def _relation_uri(self, relation_id: str | None, relation_label: str | None = None) -> URIRef:
        """Mint semantic relation URIs from relation labels when available."""
        if relation_label:
            return self.NEO[f"relation/{self._safe_local_name(relation_label, relation_id or 'relation')}"]
        return self.NEO[f"relation/{self._safe_local_name(relation_id, 'relation')}"]

    def _concept_uri(self, concept_id: str | None) -> URIRef:
        """Mint stable concept URIs from concept IDs."""
        return self.NEO[f"concept/{self._safe_local_name(concept_id, 'concept')}"]

    def _candidate_uri(self, candidate_id: str | None) -> URIRef:
        """Mint stable candidate URIs from candidate IDs."""
        return self.NEO[f"candidate/{self._safe_local_name(candidate_id, 'candidate')}"]

    def _relation_label_by_id(self, state) -> dict[str, str]:
        """Build a relation-id to label index from ontology and candidate relations."""
        index: dict[str, str] = {}
        for relation in getattr(state, "ontology_relation_candidates", []) or []:
            index[str(relation.relation_id)] = str(relation.label)
        for relation in getattr(state, "relation_candidates", []) or []:
            rid = getattr(relation, "candidate_id", None) or getattr(relation, "relation_id", None)
            label = getattr(relation, "canonical_label", None) or getattr(relation, "label", None)
            if rid and label:
                index[str(rid)] = str(label)
        return index

    def _resolve_axiom_subject_uri_with_state(self, axiom, state) -> URIRef:
        """Resolve axiom subject URI while using semantic relation labels when possible."""
        subject_id = str(axiom.subject_id)
        if subject_id.startswith("ont_rel_") or subject_id.startswith("cand_r_"):
            label = self._relation_label_by_id(state).get(subject_id, getattr(axiom, "subject_label", None))
            return self._relation_uri(subject_id, label)
        if subject_id.startswith("concept_"):
            return self._concept_uri(subject_id)
        if subject_id.startswith("cand_"):
            return self._candidate_uri(subject_id)
        return self.NEO[f"resource/{self._safe_local_name(subject_id, 'resource')}"]

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

    def _add_seed_ontology(self, graph: Graph, seed_ontology) -> None:
        """
        Add the loaded seed/source ontology into the RDF graph.
        """
        if seed_ontology is None:
            return

        # Ontology metadata
        if seed_ontology.ontology_uri:
            ontology_uri = URIRef(seed_ontology.ontology_uri)
            graph.add((ontology_uri, RDF.type, OWL.Ontology))

            if seed_ontology.ontology_label:
                graph.add((ontology_uri, RDFS.label, Literal(seed_ontology.ontology_label)))

            if seed_ontology.ontology_description:
                graph.add((ontology_uri, RDFS.comment, Literal(seed_ontology.ontology_description)))

        # Classes
        for cls in seed_ontology.get_classes():
            class_uri = URIRef(cls.uri)
            graph.add((class_uri, RDF.type, OWL.Class))
            graph.add((class_uri, RDFS.label, Literal(cls.label)))

            if cls.description:
                graph.add((class_uri, RDFS.comment, Literal(cls.description)))

            for alt_label in getattr(cls, "alt_labels", []):
                graph.add((class_uri, self.SKOS.altLabel, Literal(alt_label)))

            for parent_uri in cls.parent_uris:
                graph.add((class_uri, RDFS.subClassOf, URIRef(parent_uri)))

        # Properties
        for prop in seed_ontology.get_properties():
            prop_uri = URIRef(prop.uri)

            if prop.property_type == "data_property":
                graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            else:
                graph.add((prop_uri, RDF.type, OWL.ObjectProperty))

            graph.add((prop_uri, RDFS.label, Literal(prop.label)))

            if prop.description:
                graph.add((prop_uri, RDFS.comment, Literal(prop.description)))

            for alt_label in getattr(prop, "alt_labels", []):
                graph.add((prop_uri, self.SKOS.altLabel, Literal(alt_label)))

            for domain_uri in prop.domain_uris:
                graph.add((prop_uri, RDFS.domain, URIRef(domain_uri)))

            for range_uri in prop.range_uris:
                graph.add((prop_uri, RDFS.range, URIRef(range_uri)))

            for parent_uri in prop.parent_uris:
                graph.add((prop_uri, RDFS.subPropertyOf, URIRef(parent_uri)))

    def _add_local_ontology(self, graph: Graph, state) -> None:
        """
        Add NeoOLAF local ontology content to the RDF graph.
        """
        # Concept candidates
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

        # Ontology relation candidates
        for relation in state.ontology_relation_candidates:
            relation_uri = self._relation_uri(relation.relation_id, relation.label)
            graph.add((relation_uri, RDF.type, OWL.ObjectProperty))
            graph.add((relation_uri, RDFS.label, Literal(relation.label)))

            if relation.description:
                graph.add((relation_uri, RDFS.comment, Literal(relation.description)))

            if relation.domain_hint:
                graph.add((relation_uri, self.NEO.domainHint, Literal(relation.domain_hint)))

            if relation.range_hint:
                graph.add((relation_uri, self.NEO.rangeHint, Literal(relation.range_hint)))

        # Concept hierarchy
        for link in state.concept_hierarchy_links:
            child_uri = self.NEO[f"concept/{link.child_concept_id}"]
            parent_uri = self.NEO[f"concept/{link.parent_concept_id}"]
            graph.add((child_uri, RDFS.subClassOf, parent_uri))

        # Relation hierarchy
        for link in state.relation_hierarchy_links:
            rel_index = self._relation_label_by_id(state)
            child_uri = self._relation_uri(link.child_relation_id, rel_index.get(str(link.child_relation_id)))
            parent_uri = self._relation_uri(link.parent_relation_id, rel_index.get(str(link.parent_relation_id)))
            graph.add((child_uri, RDFS.subPropertyOf, parent_uri))

        # General axioms
        for axiom in state.general_axiom_candidates:
            subject_uri = self._resolve_axiom_subject_uri_with_state(axiom, state)

            if axiom.predicate == "subClassOf" and axiom.object_id is not None:
                object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                graph.add((subject_uri, RDFS.subClassOf, object_uri))

            elif axiom.predicate == "domain" and axiom.object_label is not None:
                graph.add((subject_uri, RDFS.domain, Literal(axiom.object_label)))

            elif axiom.predicate == "range" and axiom.object_label is not None:
                graph.add((subject_uri, RDFS.range, Literal(axiom.object_label)))

            elif axiom.predicate in {"rdfs:description", "rdfs:comment"} and axiom.literal_value is not None:
                graph.add((subject_uri, RDFS.comment, Literal(axiom.literal_value)))

            elif axiom.predicate == "rdf:type" and axiom.object_id is not None:
                object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                graph.add((subject_uri, RDF.type, object_uri))

            elif axiom.predicate == "owl:sameAs" and axiom.object_id is not None:
                object_uri = self._resolve_object_uri(axiom.object_id, axiom.object_label)
                graph.add((subject_uri, OWL.sameAs, object_uri))

            elif axiom.predicate == "skos:prefLabel" and axiom.literal_value is not None:
                graph.add((subject_uri, self.SKOS.prefLabel, Literal(axiom.literal_value)))

            elif axiom.predicate == "skos:altLabel" and axiom.literal_value is not None:
                graph.add((subject_uri, self.SKOS.altLabel, Literal(axiom.literal_value)))
"""TTL serialization utilities for ontology and KG outputs."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import OWL


EX = Namespace("http://taxodrivenkg-xquality.org/resource/")
VOC = Namespace("http://taxodrivenkg-xquality.org/vocab/")


def _safe_name(text: str) -> str:
    """Turn arbitrary text into a URI-friendly local identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "Unnamed"


def _looks_like_relation(label: str) -> bool:
    """Heuristic to detect relation-like labels."""
    lowered = label.lower()
    relation_markers = [
        "cause",
        "causes",
        "caused by",
        "affects",
        "affect",
        "observed by",
        "has",
        "part of",
        "part-of",
        "located in",
        "leads to",
        "results in",
        "indicates",
        "signals",
    ]
    return any(marker in lowered for marker in relation_markers)


def build_result_ontology_graph(
    seed_ontology_path: str | Path,
    outputs: Dict[str, dict],
) -> Graph:
    """Build a resulting ontology graph, starting from the original ontology."""
    graph = Graph()
    graph.parse(str(seed_ontology_path))

    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)
    graph.bind("ex", EX)
    graph.bind("voc", VOC)

    added_classes = set()
    added_properties = set()

    for _, chunk_output in outputs.items():
        for entity in chunk_output.get("entities", []):
            label = entity.get("label", "").strip()
            if not label:
                continue

            uri = VOC[_safe_name(label)]
            if uri not in added_classes:
                graph.add((uri, RDF.type, OWL.Class))
                graph.add((uri, RDFS.label, Literal(label)))
                added_classes.add(uri)

        for relation in chunk_output.get("relationships", []):
            rel_label = relation.get("relation", "").strip()
            if not rel_label:
                continue

            uri = VOC[_safe_name(rel_label)]
            if uri not in added_properties:
                graph.add((uri, RDF.type, OWL.ObjectProperty))
                graph.add((uri, RDFS.label, Literal(rel_label)))
                added_properties.add(uri)

    return graph


def build_result_kg_graph(outputs: Dict[str, dict]) -> Graph:
    """Build a KG graph from extracted entities and relationships."""
    graph = Graph()
    graph.bind("ex", EX)
    graph.bind("voc", VOC)
    graph.bind("rdfs", RDFS)

    entity_uri_by_name: Dict[Tuple[str, str], URIRef] = {}

    for span_key, chunk_output in outputs.items():
        chunk_node = EX[f"chunk_{_safe_name(span_key)}"]

        for entity in chunk_output.get("entities", []):
            name = entity.get("name", "").strip()
            label = entity.get("label", "").strip()
            description = entity.get("description", "").strip()

            if not name:
                continue

            entity_uri = EX[_safe_name(name)]
            entity_uri_by_name[(span_key, name)] = entity_uri

            graph.add((entity_uri, RDF.type, VOC[_safe_name(label or "Entity")]))
            graph.add((entity_uri, RDFS.label, Literal(name)))
            if description:
                graph.add((entity_uri, RDFS.comment, Literal(description)))

            graph.add((entity_uri, VOC.extractedFromChunk, chunk_node))

        for relation in chunk_output.get("relationships", []):
            source_name = relation.get("source", "").strip()
            target_name = relation.get("target", "").strip()
            rel_label = relation.get("relation", "").strip()

            if not source_name or not target_name or not rel_label:
                continue

            source_uri = EX[_safe_name(source_name)]
            target_uri = EX[_safe_name(target_name)]
            rel_uri = VOC[_safe_name(rel_label)]

            graph.add((source_uri, rel_uri, target_uri))

    return graph


def save_ttl_outputs(
    outputs: Dict[str, dict],
    seed_ontology_path: str | Path,
    ttl_dir: str | Path,
    state_name: str,
) -> Tuple[Path, Path]:
    """Create and save ontology and KG TTL files."""
    ttl_dir = Path(ttl_dir)
    ttl_dir.mkdir(parents=True, exist_ok=True)

    ontology_graph = build_result_ontology_graph(seed_ontology_path, outputs)
    kg_graph = build_result_kg_graph(outputs)

    ontology_path = ttl_dir / f"{state_name}_result_ontology.ttl"
    kg_path = ttl_dir / f"{state_name}_result_kg.ttl"

    ontology_graph.serialize(destination=str(ontology_path), format="turtle")
    kg_graph.serialize(destination=str(kg_path), format="turtle")

    return ontology_path, kg_path
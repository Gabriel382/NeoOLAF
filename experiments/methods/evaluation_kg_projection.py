#!/usr/bin/env python3
"""Project NeoOLAF KG exports into a DocRED-compatible evaluation KG.

This script reads the native NeoOLAF KG exports, usually ``kg_local.json`` and
``kg_inferred.json``, then creates strict evaluation views whose entities and
relations are aligned with a source/gold document schema.

The important rule is that gold triples are never copied as predictions. The
source/gold input is used only as a controlled entity universe and relation
vocabulary.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ENTITY_TYPE_LABELS = {"PER", "PERSON", "ORG", "LOC", "LOCATION", "MISC", "TIME", "NUM"}


@dataclass(frozen=True)
class SourceEntity:
    """Canonical entity allowed for evaluation."""

    id: str
    label: str
    type: str = "entity"
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class AllowedRelation:
    """Canonical relation allowed for evaluation."""

    id: str
    label: str
    aliases: tuple[str, ...] = ()


@dataclass
class ProjectionReport:
    """Diagnostics emitted together with the projected evaluation KG."""

    source_name: str
    input_path: str
    output_path: str
    total_triples: int = 0
    accepted_triples: int = 0
    rejected_triples: int = 0
    repaired_entity_roles: int = 0
    duplicate_triples: int = 0
    projected_entities: int = 0
    projected_relations: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)


def normalize_text(value: Any) -> str:
    """Normalize labels/IDs for exact-but-robust matching."""

    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text


def uri_tail(uri: str) -> str:
    """Return the final URI/local-name component."""

    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    tail = uri.rstrip("/").rsplit("/", 1)[-1]
    if ":" in tail and "://" not in tail:
        return tail.rsplit(":", 1)[-1]
    return tail


def read_json(path: Path) -> Any:
    """Read a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    """Write a stable, UTF-8 JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield records from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_source_doc(path: Path, document_id: str | None = None) -> dict[str, Any]:
    """Load one source/gold document from JSON or JSONL."""

    if path.suffix.lower() == ".jsonl":
        records = list(iter_jsonl(path))
        if document_id is None:
            if len(records) != 1:
                raise ValueError("--document-id is required when the JSONL has multiple records")
            return records[0]
        for record in records:
            if str(record.get("document_id") or record.get("id") or record.get("title")) == document_id:
                return record
        raise ValueError(f"Document id not found in {path}: {document_id}")

    data = read_json(path)
    if isinstance(data, list):
        if document_id is None:
            if len(data) != 1:
                raise ValueError("--document-id is required when the JSON has multiple records")
            return data[0]
        for record in data:
            if str(record.get("document_id") or record.get("id") or record.get("title")) == document_id:
                return record
        raise ValueError(f"Document id not found in {path}: {document_id}")
    return data


def extract_source_entities(source_doc: dict[str, Any]) -> list[SourceEntity]:
    """Extract entities from normalized JSON, DocRED vertexSet, or simple schema."""

    entities: list[SourceEntity] = []

    if isinstance(source_doc.get("entities"), list):
        for index, entity in enumerate(source_doc["entities"]):
            label = entity.get("label") or entity.get("name") or entity.get("text")
            if not label:
                continue
            entity_id = str(entity.get("id") or entity.get("entity_id") or f"entity_{index}")
            entity_type = str(entity.get("type") or entity.get("entity_type") or "entity")
            aliases = _extract_aliases(entity)
            entities.append(SourceEntity(entity_id, str(label), entity_type, tuple(aliases)))
        return entities

    # Native DocRED uses vertexSet: one entity cluster, multiple mentions.
    if isinstance(source_doc.get("vertexSet"), list):
        for index, cluster in enumerate(source_doc["vertexSet"]):
            if not cluster:
                continue
            first = cluster[0]
            label = first.get("name") or first.get("label")
            if not label:
                continue
            entity_type = str(first.get("type") or "entity")
            aliases = sorted(
                {
                    str(mention.get("name") or mention.get("label"))
                    for mention in cluster
                    if mention.get("name") or mention.get("label")
                }
            )
            entities.append(SourceEntity(f"entity_{index}", str(label), entity_type, tuple(aliases)))
    return entities


def _extract_aliases(entity: dict[str, Any]) -> list[str]:
    """Collect common alias/mention fields from a source entity object."""

    aliases: set[str] = set()
    for key in ("aliases", "mentions", "names"):
        value = entity.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    aliases.add(item)
                elif isinstance(item, dict):
                    label = item.get("label") or item.get("name") or item.get("text")
                    if label:
                        aliases.add(str(label))
    return sorted(aliases)


def load_relation_vocabulary(
    ontology_path: Path | None,
    relation_vocab_path: Path | None,
    source_docs_path: Path | None,
    relation_filter_mode: str,
) -> tuple[list[AllowedRelation], dict[str, int]]:
    """Load and combine fixed evaluation vocabularies.

    The ontology passed here must be the dataset/reference ontology, never an
    ontology generated by the current NeoOLAF run. This preserves NeoOLAF's
    ability to discover relations while keeping evaluation constrained.
    """

    ontology_relations = extract_relations_from_ontology(ontology_path) if ontology_path else []
    listed_relations: list[AllowedRelation] = []
    if relation_vocab_path:
        listed_relations.extend(extract_relations_from_vocab_json(relation_vocab_path))
    if source_docs_path:
        listed_relations.extend(extract_relation_vocab_from_source(source_docs_path))

    ontology_relations = merge_relation_records(ontology_relations)
    listed_relations = merge_relation_records(listed_relations)
    relations = combine_relation_vocabularies(
        ontology_relations=ontology_relations,
        listed_relations=listed_relations,
        mode=relation_filter_mode,
    )
    counts = {
        "ontology_relation_count": len(ontology_relations),
        "listed_relation_count": len(listed_relations),
        "allowed_relation_count": len(relations),
    }
    return relations, counts


def relation_match_keys(relation: AllowedRelation) -> set[str]:
    """Return normalized keys used to match a relation across vocabularies."""

    return {
        normalized
        for value in (relation.id, relation.label, *relation.aliases)
        if (normalized := normalize_text(value))
    }


def merge_relation_records(relations: Iterable[AllowedRelation]) -> list[AllowedRelation]:
    """Deduplicate relation records while preserving their known aliases."""

    merged: list[AllowedRelation] = []
    for relation in relations:
        matching_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if relation_match_keys(existing) & relation_match_keys(relation)
            ),
            None,
        )
        if matching_index is None:
            merged.append(relation)
            continue

        existing = merged[matching_index]
        aliases = tuple(
            sorted(
                {
                    existing.id,
                    existing.label,
                    *existing.aliases,
                    relation.id,
                    relation.label,
                    *relation.aliases,
                }
            )
        )
        merged[matching_index] = AllowedRelation(existing.id, existing.label, aliases)
    return sorted(merged, key=lambda item: (item.id, item.label))


def combine_relation_vocabularies(
    ontology_relations: list[AllowedRelation],
    listed_relations: list[AllowedRelation],
    mode: str,
) -> list[AllowedRelation]:
    """Apply the requested relation-vocabulary filter mode."""

    if mode == "ontology-only":
        if not ontology_relations:
            raise ValueError(
                "relation-filter-mode=ontology-only requires an ontology containing RDF/OWL properties."
            )
        return ontology_relations

    if mode == "list-only":
        if not listed_relations:
            raise ValueError(
                "relation-filter-mode=list-only requires --relation-vocab-json "
                "or --source-relation-vocab-json."
            )
        return listed_relations

    if mode == "union":
        if not ontology_relations and not listed_relations:
            raise ValueError("relation-filter-mode=union requires at least one relation vocabulary.")
        return merge_relation_records([*ontology_relations, *listed_relations])

    if mode == "intersection":
        if not ontology_relations or not listed_relations:
            raise ValueError(
                "relation-filter-mode=intersection requires both an ontology and a relation list."
            )

        intersected: list[AllowedRelation] = []
        for listed in listed_relations:
            matches = [
                ontology
                for ontology in ontology_relations
                if relation_match_keys(listed) & relation_match_keys(ontology)
            ]
            if not matches:
                continue
            aliases = {
                listed.id,
                listed.label,
                *listed.aliases,
            }
            for match in matches:
                aliases.update({match.id, match.label, *match.aliases})
            # Prefer the explicit dataset relation ID and label in evaluation.
            intersected.append(AllowedRelation(listed.id, listed.label, tuple(sorted(aliases))))
        return merge_relation_records(intersected)

    raise ValueError(f"Unsupported relation filter mode: {mode}")


def extract_relations_from_ontology(path: Path) -> list[AllowedRelation]:
    """Extract RDF/OWL properties from an ontology.

    If rdflib is unavailable or parsing fails, a lightweight Turtle regex fallback
    is used. Classes are intentionally ignored because they are entity types, not
    relation predicates.
    """

    try:
        from rdflib import Graph, URIRef
        from rdflib.namespace import OWL, RDF, RDFS
    except Exception:
        return extract_relations_from_ttl_fallback(path)

    graph = Graph()
    graph.parse(str(path))
    property_types = {
        RDF.Property,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
    }
    relation_uris: set[URIRef] = set()
    for prop_type in property_types:
        relation_uris.update(subject for subject in graph.subjects(RDF.type, prop_type) if isinstance(subject, URIRef))

    relations: list[AllowedRelation] = []
    for uri in relation_uris:
        labels = [str(label) for label in graph.objects(uri, RDFS.label)]
        local_id = uri_tail(str(uri))
        label = labels[0] if labels else local_id
        aliases = tuple(sorted(set(labels + [local_id, str(uri)])))
        relations.append(AllowedRelation(local_id, label, aliases))
    return relations


def extract_relations_from_ttl_fallback(path: Path) -> list[AllowedRelation]:
    """Best-effort relation extraction from Turtle without rdflib."""

    statements: list[str] = []
    current_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("@prefix") or line.startswith("PREFIX"):
            continue
        current_lines.append(line)
        if line.endswith("."):
            statements.append(" ".join(current_lines))
            current_lines = []
    if current_lines:
        statements.append(" ".join(current_lines))

    relations: list[AllowedRelation] = []
    for statement in statements:
        declaration = re.match(
            r"^(<[^>]+>|[A-Za-z_][\w:-]*)\s+(?:a|rdf:type)\s+(.+?)(?:\s*;\s*|\s*\.\s*$)",
            statement,
        )
        if not declaration:
            continue
        subject, rdf_types = declaration.groups()
        if not any(
            name in rdf_types
            for name in ("owl:ObjectProperty", "owl:DatatypeProperty", "rdf:Property")
        ):
            continue

        uri = subject.strip("<>")
        local_id = uri_tail(uri)
        label_match = re.search(r"rdfs:label\s+\"([^\"]+)\"", statement)
        label = label_match.group(1) if label_match else local_id
        relations.append(AllowedRelation(local_id, label, (label, local_id, uri)))
    return relations


def extract_relations_from_vocab_json(path: Path) -> list[AllowedRelation]:
    """Extract allowed relations from a JSON mapping/list."""

    data = read_json(path)
    items: list[Any]
    if isinstance(data, dict):
        if isinstance(data.get("relations"), list):
            items = data["relations"]
        else:
            items = [{"id": key, "label": value} for key, value in data.items()]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported relation vocabulary JSON: {path}")

    relations: list[AllowedRelation] = []
    for item in items:
        if isinstance(item, str):
            relations.append(AllowedRelation(item, item, (item,)))
            continue
        relation_id = str(item.get("id") or item.get("relation_id") or item.get("property") or item.get("label"))
        label = str(item.get("label") or item.get("name") or item.get("relation") or relation_id)
        aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
        relations.append(AllowedRelation(relation_id, label, tuple(sorted(set([relation_id, label, *map(str, aliases)])))))
    return relations


def extract_relation_vocab_from_source(path: Path) -> list[AllowedRelation]:
    """Extract relation labels/IDs from source/gold files as vocabulary only."""

    records = list(iter_jsonl(path)) if path.suffix.lower() == ".jsonl" else read_json(path)
    if isinstance(records, dict):
        records = records.get("documents") or records.get("data") or [records]

    relations: dict[str, AllowedRelation] = {}
    for record in records:
        for triple in _iter_source_gold_relations(record):
            relation_id = str(triple.get("relation_id") or triple.get("r") or triple.get("relation") or triple.get("predicate"))
            label = str(triple.get("relation_label") or triple.get("label") or triple.get("relation") or relation_id)
            if relation_id and relation_id != "None":
                relations.setdefault(relation_id, AllowedRelation(relation_id, label, (relation_id, label)))
    return list(relations.values())


def _iter_source_gold_relations(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield gold relation records from common DocRED/normalized fields."""

    for key in ("relations", "labels", "triples"):
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item


def build_entity_indexes(entities: list[SourceEntity]) -> tuple[dict[str, SourceEntity], dict[str, SourceEntity]]:
    """Build ID and label indexes for source entities."""

    by_id: dict[str, SourceEntity] = {}
    by_label: dict[str, SourceEntity] = {}
    for entity in entities:
        by_id[str(entity.id)] = entity
        for label in (entity.label, *entity.aliases):
            norm = normalize_text(label)
            if norm and norm not in by_label:
                by_label[norm] = entity
    return by_id, by_label


def build_relation_index(relations: list[AllowedRelation]) -> dict[str, AllowedRelation]:
    """Build an ID/label/alias index for allowed relation predicates."""

    index: dict[str, AllowedRelation] = {}
    for relation in relations:
        for key in (relation.id, relation.label, *relation.aliases):
            norm = normalize_text(key)
            if norm:
                index[norm] = relation
    return index


def project_kg(
    input_path: Path,
    output_path: Path,
    source_name: str,
    entities: list[SourceEntity],
    allowed_relations: list[AllowedRelation],
) -> ProjectionReport:
    """Project one NeoOLAF KG into an evaluation-compatible KG."""

    kg = read_json(input_path)
    triples = kg.get("triples", kg) if isinstance(kg, dict) else kg
    if not isinstance(triples, list):
        raise ValueError(f"KG does not contain a triples list: {input_path}")

    entity_by_id, entity_by_label = build_entity_indexes(entities)
    relation_index = build_relation_index(allowed_relations)
    entity_label_index = {normalize_text(entity.label) for entity in entities}

    report = ProjectionReport(source_name=source_name, input_path=str(input_path), output_path=str(output_path))
    projected: list[dict[str, Any]] = []
    predicted_entities: dict[str, SourceEntity] = {}
    predicted_relations: dict[str, AllowedRelation] = {}
    seen: set[tuple[str, str, str]] = set()

    for index, triple in enumerate(triples):
        report.total_triples += 1
        subject = resolve_entity_node(triple.get("subject") or triple.get("head"), entity_by_id, entity_by_label)
        obj = resolve_entity_node(triple.get("object") or triple.get("tail"), entity_by_id, entity_by_label)
        predicate = resolve_relation_node(triple.get("predicate") or triple.get("relation"), relation_index)

        # Resolvable native KG nodes count as predicted entities even when their
        # relation is later rejected. This avoids copying the complete gold
        # entity universe while preserving entity-level evaluation.
        if subject is not None:
            predicted_entities[subject.id] = subject
        if obj is not None:
            predicted_entities[obj.id] = obj

        rejection_reasons: list[str] = []
        if subject is None:
            rejection_reasons.append("invalid_subject_entity")
        if obj is None:
            rejection_reasons.append("invalid_object_entity")
        if predicate is None:
            rejection_reasons.append("invalid_predicate_relation")
            raw_predicate = extract_node_label(triple.get("predicate") or triple.get("relation"))
            if normalize_text(raw_predicate) in entity_label_index:
                rejection_reasons.append("predicate_is_entity_label")
            if normalize_text(raw_predicate).upper() in ENTITY_TYPE_LABELS:
                rejection_reasons.append("predicate_is_entity_type_label")

        if rejection_reasons:
            report.rejected_triples += 1
            report.rejected.append(
                {
                    "triple_index": index,
                    "triple_id": triple.get("triple_id"),
                    "reasons": rejection_reasons,
                    "raw_subject": triple.get("subject") or triple.get("head"),
                    "raw_predicate": triple.get("predicate") or triple.get("relation"),
                    "raw_object": triple.get("object") or triple.get("tail"),
                }
            )
            continue

        assert subject is not None and obj is not None and predicate is not None
        if _node_declared_type(triple.get("subject") or triple.get("head")) == "relation":
            report.repaired_entity_roles += 1
        if _node_declared_type(triple.get("object") or triple.get("tail")) == "relation":
            report.repaired_entity_roles += 1

        key = (subject.id, predicate.id, obj.id)
        if key in seen:
            report.duplicate_triples += 1
            continue
        seen.add(key)
        predicted_relations[predicate.id] = predicate
        projected.append(
            {
                "triple_id": f"{source_name}_{len(projected):05d}",
                "subject": {"id": subject.id, "label": subject.label, "type": subject.type},
                "predicate": {"id": predicate.id, "label": predicate.label},
                "object": {"id": obj.id, "label": obj.label, "type": obj.type},
                "source": source_name,
                "evidence": triple.get("evidence") or triple.get("justification"),
                "confidence": triple.get("confidence"),
                "original_triple_id": triple.get("triple_id"),
            }
        )

    report.accepted_triples = len(projected)
    report.projected_entities = len(predicted_entities)
    report.projected_relations = len(predicted_relations)
    output = {
        "source": source_name,
        "entities": [
            entity.__dict__
            for entity in sorted(predicted_entities.values(), key=lambda item: item.id)
        ],
        "relations": [
            relation.__dict__
            for relation in sorted(predicted_relations.values(), key=lambda item: item.id)
        ],
        "triples": projected,
        "projection": {
            "input_path": str(input_path),
            "strict": True,
            "allowed_relation_count": len(allowed_relations),
            "projected_entities": report.projected_entities,
            "projected_relations": report.projected_relations,
            "accepted_triples": report.accepted_triples,
            "rejected_triples": report.rejected_triples,
            "duplicate_triples": report.duplicate_triples,
            "repaired_entity_roles": report.repaired_entity_roles,
        },
    }
    write_json(output_path, output)
    return report


def extract_node_label(node: Any) -> str:
    """Return the best label-like value from a KG node."""

    if isinstance(node, dict):
        return str(node.get("label") or node.get("name") or node.get("id") or "")
    return "" if node is None else str(node)


def _node_declared_type(node: Any) -> str:
    """Return the lower-case type declared by the native KG node, if any."""

    if isinstance(node, dict):
        return normalize_text(node.get("type"))
    return ""


def resolve_entity_node(
    node: Any,
    by_id: dict[str, SourceEntity],
    by_label: dict[str, SourceEntity],
) -> SourceEntity | None:
    """Resolve a KG subject/object node to a source entity."""

    if isinstance(node, dict):
        node_id = node.get("id")
        if node_id is not None and str(node_id) in by_id:
            return by_id[str(node_id)]
        label = node.get("label") or node.get("name")
    else:
        label = node

    norm = normalize_text(label)
    if norm in by_label:
        return by_label[norm]
    return None


def resolve_relation_node(node: Any, relation_index: dict[str, AllowedRelation]) -> AllowedRelation | None:
    """Resolve a KG predicate/relation node to an allowed relation."""

    if isinstance(node, dict):
        keys = [node.get("id"), node.get("label"), node.get("name")]
    else:
        keys = [node]

    for key in keys:
        norm = normalize_text(key)
        if norm in relation_index:
            return relation_index[norm]
    return None


def write_combined_kg(local_path: Path, inferred_path: Path, output_path: Path) -> None:
    """Write a deduplicated local+inferred evaluation KG."""

    local = read_json(local_path)
    inferred = read_json(inferred_path)
    triples: list[dict[str, Any]] = []
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str, str]] = set()
    for source in (local, inferred):
        for entity in source.get("entities", []):
            entities[str(entity["id"])] = entity
        for relation in source.get("relations", []):
            relations[str(relation["id"])] = relation
        for triple in source.get("triples", []):
            key = (
                triple["subject"]["id"],
                triple["predicate"]["id"],
                triple["object"]["id"],
            )
            if key in seen:
                continue
            seen.add(key)
            triples.append(triple)

    combined = {
        "source": "combined",
        "entities": [entities[key] for key in sorted(entities)],
        "relations": [relations[key] for key in sorted(relations)],
        "triples": triples,
        "projection": {
            "local_path": str(local_path),
            "inferred_path": str(inferred_path),
            "deduplicated_triples": len(triples),
        },
    }
    write_json(output_path, combined)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-local-json", type=Path, required=True)
    parser.add_argument("--kg-inferred-json", type=Path, required=True)
    parser.add_argument("--source-doc-json", type=Path, required=True)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--ontology-path", type=Path, default=None)
    parser.add_argument("--relation-vocab-json", type=Path, default=None)
    parser.add_argument(
        "--source-relation-vocab-json",
        type=Path,
        default=None,
        help="Optional full dataset JSON/JSONL used only to collect allowed relation labels/IDs.",
    )
    parser.add_argument(
        "--relation-filter-mode",
        choices=("ontology-only", "list-only", "union", "intersection"),
        default="union",
        help=(
            "Controls which fixed relations are evaluable. The ontology must be "
            "the reference ontology, not NeoOLAF's generated ontology."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    source_doc = load_source_doc(args.source_doc_json, args.document_id)
    entities = extract_source_entities(source_doc)
    if not entities:
        raise ValueError("No source entities were found. The evaluation KG cannot be projected.")

    relations, relation_counts = load_relation_vocabulary(
        ontology_path=args.ontology_path,
        relation_vocab_path=args.relation_vocab_json,
        source_docs_path=args.source_relation_vocab_json,
        relation_filter_mode=args.relation_filter_mode,
    )
    if not relations:
        raise ValueError(
            "No allowed relations were found. Provide --ontology-path with RDF properties "
            "or --relation-vocab-json / --source-relation-vocab-json."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_output = args.output_dir / "evaluation_kg_local.json"
    inferred_output = args.output_dir / "evaluation_kg_inferred.json"
    combined_output = args.output_dir / "evaluation_kg_combined.json"
    report_output = args.output_dir / "evaluation_projection_report.json"

    reports = [
        project_kg(args.kg_local_json, local_output, "local", entities, relations),
        project_kg(args.kg_inferred_json, inferred_output, "inferred", entities, relations),
    ]
    write_combined_kg(local_output, inferred_output, combined_output)

    report = {
        "document_id": args.document_id or source_doc.get("document_id") or source_doc.get("id") or source_doc.get("title"),
        "source_doc_json": str(args.source_doc_json),
        "ontology_path": str(args.ontology_path) if args.ontology_path else None,
        "relation_vocab_json": str(args.relation_vocab_json) if args.relation_vocab_json else None,
        "source_relation_vocab_json": str(args.source_relation_vocab_json) if args.source_relation_vocab_json else None,
        "relation_filter_mode": args.relation_filter_mode,
        "relation_vocabulary_counts": relation_counts,
        "entity_count": len(entities),
        "allowed_relation_count": len(relations),
        "outputs": {
            "evaluation_kg_local": str(local_output),
            "evaluation_kg_inferred": str(inferred_output),
            "evaluation_kg_combined": str(combined_output),
            "evaluation_projection_report": str(report_output),
        },
        "reports": [report_item.__dict__ for report_item in reports],
    }
    write_json(report_output, report)

    print(
        f"[evaluation-kg] entities={len(entities)} allowed_relations={len(relations)} "
        f"relation_filter_mode={args.relation_filter_mode}"
    )
    print(f"[evaluation-kg] local={local_output}")
    print(f"[evaluation-kg] inferred={inferred_output}")
    print(f"[evaluation-kg] combined={combined_output}")
    print(f"[evaluation-kg] report={report_output}")


if __name__ == "__main__":
    main()

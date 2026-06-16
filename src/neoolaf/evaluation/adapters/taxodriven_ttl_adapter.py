"""Adapter for TaxoDrivenKG TTL outputs."""

from __future__ import annotations

from pathlib import Path

from neoolaf.evaluation.adapters.ttl_adapter import load_graph, ontology_from_ttl, uri_local_name
from neoolaf.evaluation.matching.predicate_mapping import map_xquality_predicate, should_invert_xquality_predicate
from neoolaf.evaluation.schema.artifact import EvalDocument, EvalEntity, EvalRelation, EvaluationArtifact


def artifact_from_taxodriven_ttl(
    kg_ttl_path: str | Path,
    ontology_ttl_path: str | Path | None,
    dataset: str,
    profile: str,
    run_id: str = "taxodrivenkg",
) -> EvaluationArtifact:
    """Convert a TaxoDrivenKG KG TTL and ontology TTL to an EvaluationArtifact."""
    graph = load_graph(kg_ttl_path)
    doc_id = Path(kg_ttl_path).stem.replace("_result_kg", "")
    artifact = EvaluationArtifact(method="taxodrivenkg", dataset=dataset, profile=profile, run_id=run_id)
    artifact.documents.append(EvalDocument(document_id=doc_id, source_path=str(kg_ttl_path)))

    entities: set[str] = set()
    relations: list[EvalRelation] = []

    ignored_predicates = {
        "type", "label", "comment", "subclassof", "subpropertyof", "domain", "range", "sameas"
    }

    for subject, predicate, obj in graph:
        raw_predicate = uri_local_name(predicate)
        pred_lower = raw_predicate.replace("_", "").replace("-", "").lower()
        subject_label = uri_local_name(subject)
        object_label = uri_local_name(obj)

        if subject_label:
            entities.add(subject_label)
        if object_label:
            entities.add(object_label)

        if pred_lower in ignored_predicates:
            continue

        mapped = map_xquality_predicate(raw_predicate) if dataset == "xquality" else raw_predicate
        if mapped is None:
            continue

        head = subject_label
        tail = object_label
        if dataset == "xquality" and should_invert_xquality_predicate(raw_predicate):
            head, tail = tail, head

        relations.append(
            EvalRelation(
                head=head,
                relation=mapped,
                tail=tail,
                evidence="",
                provenance_present=False,
                raw={"subject": str(subject), "predicate": str(predicate), "object": str(obj), "raw_predicate": raw_predicate},
            )
        )

    artifact.entities_by_doc[doc_id] = [EvalEntity(label=label) for label in sorted(entities)]
    artifact.relations_by_doc[doc_id] = relations
    artifact.global_ontology = ontology_from_ttl(ontology_ttl_path)
    return artifact

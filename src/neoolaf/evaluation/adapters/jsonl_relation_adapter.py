"""Adapter for prediction JSONL files used by generic relation extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from neoolaf.evaluation.schema.artifact import EvalDocument, EvalEntity, EvalRelation, EvaluationArtifact


def iter_prediction_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Stream prediction records from JSONL."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSONL at {path}:{line_number}: {exc}") from exc


def artifact_from_prediction_jsonl(path: str | Path, dataset: str, profile: str, method: str = "jsonl", run_id: str = "jsonl") -> EvaluationArtifact:
    """Convert prediction JSONL to an EvaluationArtifact.

    Compatible with the previous `eval_relations.py` format:
    {document_id, parsed_ok, prediction: {entities, relations}}.
    """
    artifact = EvaluationArtifact(method=method, dataset=dataset, profile=profile, run_id=run_id)

    for record in iter_prediction_records(path):
        doc_id = str(record.get("document_id") or record.get("id") or "").strip()
        if not doc_id:
            continue
        artifact.documents.append(EvalDocument(document_id=doc_id, source_path=str(path), metadata={"parsed_ok": record.get("parsed_ok", True)}))

        if not record.get("parsed_ok", True):
            artifact.entities_by_doc[doc_id] = []
            artifact.relations_by_doc[doc_id] = []
            continue

        prediction = record.get("prediction") or record
        entities = []
        for ent in prediction.get("entities", []) or []:
            label = str(ent.get("label") or ent.get("text") or ent.get("name") or "").strip()
            if label:
                entities.append(EvalEntity(label=label, id=str(ent.get("id", "") or "") or None, type=ent.get("type"), raw=ent))
        artifact.entities_by_doc[doc_id] = entities

        relations = []
        for rel in prediction.get("relations", []) or []:
            head = str(rel.get("head") or rel.get("head_text") or rel.get("subject") or "").strip()
            relation = str(rel.get("relation") or rel.get("predicate") or rel.get("label") or "").strip()
            tail = str(rel.get("tail") or rel.get("tail_text") or rel.get("object") or "").strip()
            evidence = str(rel.get("evidence") or rel.get("justification") or "").strip()
            if head and relation and tail:
                relations.append(EvalRelation(head=head, relation=relation, tail=tail, evidence=evidence, provenance_present=bool(evidence), raw=rel))
        artifact.relations_by_doc[doc_id] = relations

    return artifact

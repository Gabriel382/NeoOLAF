"""Adapter for SinglePass parsed JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neoolaf.evaluation.schema.artifact import EvalDocument, EvalEntity, EvalRelation, EvaluationArtifact


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_singlepass_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(path.glob("**/*singlepass_parsed.json")) or sorted(path.glob("**/*.json"))


def artifact_from_singlepass(input_path: str | Path, dataset: str, profile: str, run_id: str = "singlepass") -> EvaluationArtifact:
    """Convert SinglePass parsed JSON outputs to an EvaluationArtifact."""
    artifact = EvaluationArtifact(method="singlepass", dataset=dataset, profile=profile, run_id=run_id)

    for file_path in _iter_singlepass_files(input_path):
        data = _load_json(file_path)
        doc_id = file_path.stem.replace("__singlepass_parsed", "")
        artifact.documents.append(EvalDocument(document_id=doc_id, source_path=str(file_path)))

        entities = []
        for ent in data.get("entities", []) or []:
            label = str(ent.get("label") or ent.get("text") or "").strip()
            if label:
                entities.append(EvalEntity(label=label, id=str(ent.get("id", "") or "") or None, type=ent.get("type"), raw=ent))
        artifact.entities_by_doc[doc_id] = entities

        id_to_label = {ent.id: ent.label for ent in entities if ent.id}
        relations = []
        for rel in data.get("relations", []) or []:
            head = str(rel.get("head", "")).strip()
            tail = str(rel.get("tail", "")).strip()
            if head in id_to_label:
                head = id_to_label[head]
            if tail in id_to_label:
                tail = id_to_label[tail]
            relation = str(rel.get("relation") or rel.get("predicate") or "").strip().upper()
            evidence = str(rel.get("evidence") or rel.get("justification") or "").strip()
            if head and relation and tail:
                relations.append(
                    EvalRelation(
                        head=head,
                        relation=relation,
                        tail=tail,
                        evidence=evidence,
                        provenance_present=bool(evidence),
                        raw=rel,
                    )
                )
        artifact.relations_by_doc[doc_id] = relations

    return artifact

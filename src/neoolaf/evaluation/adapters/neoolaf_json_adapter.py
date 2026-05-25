"""Adapter for NeoOLAF export folders and JSON KG files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neoolaf.evaluation.adapters.ttl_adapter import ontology_from_ttl
from neoolaf.evaluation.schema.artifact import EvalDocument, EvalEntity, EvalRelation, EvaluationArtifact


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_first_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def _extract_label(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("label") or value.get("text") or value.get("name") or value.get("id") or "").strip()
    return str(value or "").strip()


def artifact_from_neoolaf_exports(
    input_path: str | Path,
    dataset: str,
    profile: str,
    run_id: str = "neoolaf",
    modality: str | None = None,
) -> EvaluationArtifact:
    """Convert NeoOLAF exports to an EvaluationArtifact.

    Supported input layouts:
    - exports folder containing kg_inferred.json or kg_local.json
    - direct JSON file with {triples: [...]}.
    """
    path = Path(input_path)
    base = path if path.is_dir() else path.parent
    kg_path = path if path.is_file() else _find_first_existing(base, ["kg_inferred.json", "kg_local.json", "kg.json"])
    if kg_path is None:
        raise FileNotFoundError(f"Could not find kg_inferred.json, kg_local.json, or kg.json in {base}")

    data = _load_json(kg_path)
    triples = data.get("triples", []) if isinstance(data, dict) else []

    doc_id = data.get("document_id") if isinstance(data, dict) else None
    doc_id = str(doc_id or base.name or "neoolaf_document")

    artifact = EvaluationArtifact(
        method="neoolaf",
        dataset=dataset,
        profile=profile,
        run_id=run_id,
        metadata={"modality": modality or "unknown", "kg_path": str(kg_path)},
    )
    artifact.documents.append(EvalDocument(document_id=doc_id, source_path=str(kg_path)))

    entities: set[str] = set()
    relations: list[EvalRelation] = []

    for triple in triples:
        subject = triple.get("subject") or triple.get("head") or triple.get("s")
        predicate = triple.get("predicate") or triple.get("relation") or triple.get("p")
        obj = triple.get("object") or triple.get("tail") or triple.get("o")

        head = _extract_label(subject)
        relation = _extract_label(predicate)
        tail = _extract_label(obj)
        evidence = str(triple.get("justification") or triple.get("evidence") or triple.get("support_text") or "").strip()
        chunk_id = str(triple.get("chunk_id") or triple.get("chunkid") or "").strip()
        confidence = triple.get("confidence")

        if head:
            entities.add(head)
        if tail:
            entities.add(tail)
        if head and relation and tail:
            relations.append(
                EvalRelation(
                    head=head,
                    relation=relation,
                    tail=tail,
                    evidence=evidence,
                    confidence=confidence if isinstance(confidence, (float, int)) else None,
                    provenance_present=bool(evidence or chunk_id),
                    provenance={"chunk_id": chunk_id} if chunk_id else {},
                    raw=triple,
                )
            )

    artifact.entities_by_doc[doc_id] = [EvalEntity(label=label) for label in sorted(entities)]
    artifact.relations_by_doc[doc_id] = relations

    ontology_path = _find_first_existing(base, ["ontology_inferred.ttl", "ontology_local.ttl", "ontology.ttl"])
    artifact.global_ontology = ontology_from_ttl(ontology_path)
    return artifact

"""Generic JSONL dataset loading for relation extraction benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from neoolaf.evaluation.schema.artifact import EvalDocument, EvalEntity, EvalRelation


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield dictionaries from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def passes_type_filter(record: dict[str, Any], type_filter: str | list[str]) -> bool:
    """Return whether a document record passes a split/type filter."""
    if type_filter == "all":
        return True
    values = [type_filter] if isinstance(type_filter, str) else type_filter
    candidates = {
        str(record.get("type", "")),
        str(record.get("split", "")),
        str(record.get("source_type", "")),
    }
    return any(value in candidates for value in values)


def load_generic_gold_jsonl(path: str | Path, type_filter: str | list[str] = "all") -> tuple[list[EvalDocument], dict[str, list[EvalEntity]], dict[str, list[EvalRelation]]]:
    """Load a generic gold JSONL dataset.

    Expected normalized document fields are flexible. The loader supports the
    format used by the previous `eval_relations.py` script:
    `document_id`, `entities`, and `relations` with `head_text` / `tail_text`.
    """
    documents: list[EvalDocument] = []
    entities_by_doc: dict[str, list[EvalEntity]] = {}
    relations_by_doc: dict[str, list[EvalRelation]] = {}

    for record in iter_jsonl(path):
        if not passes_type_filter(record, type_filter):
            continue

        doc_id = str(record.get("document_id") or record.get("id") or "").strip()
        if not doc_id:
            continue

        documents.append(
            EvalDocument(
                document_id=doc_id,
                text=record.get("text") or record.get("document") or record.get("content"),
                metadata={k: v for k, v in record.items() if k not in {"entities", "relations"}},
            )
        )

        entities: list[EvalEntity] = []
        for ent in record.get("entities", []) or []:
            label = str(ent.get("text") or ent.get("label") or ent.get("name") or "").strip()
            if label:
                entities.append(EvalEntity(label=label, id=str(ent.get("id", "") or "") or None, type=ent.get("type"), raw=ent))
        entities_by_doc[doc_id] = entities

        relations: list[EvalRelation] = []
        for rel in record.get("relations", []) or []:
            head = str(rel.get("head_text") or rel.get("head") or rel.get("subject") or "").strip()
            relation = str(rel.get("relation") or rel.get("label") or rel.get("predicate") or "").strip()
            tail = str(rel.get("tail_text") or rel.get("tail") or rel.get("object") or "").strip()
            if head and relation and tail:
                relations.append(EvalRelation(head=head, relation=relation, tail=tail, raw=rel))
        relations_by_doc[doc_id] = relations

    return documents, entities_by_doc, relations_by_doc

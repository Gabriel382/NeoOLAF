# jsonl_adapter.py
# Shared streaming adapter for datasets stored as JSONL, one document per line.
# The expected schema is:
# {
#   "document_id": "...",
#   "title": "...",
#   "text": "...",
#   "type": "...",
#   "entities": {
#       "EntityID": {
#           "type": "...",
#           "mentions": [{"trigger_word": "...", ...}, ...]
#       },
#       ...
#   },
#   "relations": {
#       "RelationName": [["EntityID1", "EntityID2"], ...],
#       ...
#   }
# }

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union, Any
from tqdm.auto import tqdm

TypeFilter = Union[str, List[str], tuple, set, None]


def normalize_type_filter(type_filter: TypeFilter) -> Optional[set[str]]:
    """
    Normalize a type filter.

    Accepted values:
    - "all" or None: no filtering
    - "dev": only dev
    - ["dev", "test"]&#58; only dev or test
    """
    if type_filter is None:
        return None

    if isinstance(type_filter, str):
        if type_filter.lower() == "all":
            return None
        return {type_filter}

    normalized = {str(x) for x in type_filter}
    if "all" in {x.lower() for x in normalized}:
        return None
    return normalized


def should_keep_document(doc_type: str, normalized_filter: Optional[set[str]]) -> bool:
    """
    Return True if a document type passes the filter.
    """
    if normalized_filter is None:
        return True
    return doc_type in normalized_filter


def choose_entity_text(entity_id: str, entity_data: Dict[str, Any]) -> str:
    """
    Choose a representative text for one entity.

    Priority:
    1. First non-empty mention trigger_word
    2. Entity id
    """
    mentions = entity_data.get("mentions", []) or []

    for mention in mentions:
        trigger_word = str(mention.get("trigger_word", "")).strip()
        if trigger_word:
            return trigger_word

    return entity_id


def adapt_document(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one raw dataset document into a normalized in-memory format.
    """
    entities_dict = raw_doc.get("entities", {}) or {}
    relations_dict = raw_doc.get("relations", {}) or {}

    entities: List[Dict[str, Any]] = []
    entity_id_to_text: Dict[str, str] = {}

    for entity_id, entity_data in entities_dict.items():
        entity_text = choose_entity_text(entity_id, entity_data)
        mentions = entity_data.get("mentions", []) or []

        mention_texts: List[str] = []
        for mention in mentions:
            trigger_word = str(mention.get("trigger_word", "")).strip()
            if trigger_word:
                mention_texts.append(trigger_word)

        entity_record = {
            "id": entity_id,
            "text": entity_text,
            "type": str(entity_data.get("type", "")).strip(),
            "mentions": mention_texts,
        }
        entities.append(entity_record)
        entity_id_to_text[entity_id] = entity_text

    relations: List[Dict[str, Any]] = []

    for relation_name, pairs in relations_dict.items():
        if not isinstance(pairs, list):
            continue

        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue

            head_id, tail_id = pair
            head_id = str(head_id)
            tail_id = str(tail_id)

            relations.append(
                {
                    "head_id": head_id,
                    "tail_id": tail_id,
                    "head_text": entity_id_to_text.get(head_id, head_id),
                    "tail_text": entity_id_to_text.get(tail_id, tail_id),
                    "relation": str(relation_name).strip(),
                }
            )

    adapted = {
        "document_id": str(raw_doc.get("document_id", "")).strip(),
        "title": str(raw_doc.get("title", "")).strip(),
        "text": str(raw_doc.get("text", "")).strip(),
        "type": str(raw_doc.get("type", "")).strip(),
        "entities": entities,
        "relations": relations,
    }
    return adapted


def iter_documents(jsonl_path: Union[str, Path], type_filter: TypeFilter = "all") -> Iterator[Dict[str, Any]]:
    """
    Stream documents from a JSONL file one line at a time.

    Parameters
    ----------
    jsonl_path:
        Path to the dataset JSONL.
    type_filter:
        "all", one type string, or a list/set/tuple of types.
    """
    jsonl_path = Path(jsonl_path)
    normalized_filter = normalize_type_filter(type_filter)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw_doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {jsonl_path}: {e}"
                ) from e

            doc_type = str(raw_doc.get("type", "")).strip()
            if not should_keep_document(doc_type, normalized_filter):
                continue

            yield adapt_document(raw_doc)


def count_documents(jsonl_path: Union[str, Path], type_filter: TypeFilter = "all") -> int:
    """
    Count how many documents pass the filter.
    """
    count = 0
    for _ in iter_documents(jsonl_path=jsonl_path, type_filter=type_filter):
        count += 1
    return count
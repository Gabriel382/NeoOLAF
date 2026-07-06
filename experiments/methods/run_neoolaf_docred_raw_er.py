#!/usr/bin/env python3
"""Raw-text DocRED entity + relation extraction benchmark adapter for NeoOLAF/RAGTree.

This runner intentionally does NOT expose DocRED gold/source entity IDs to the
LLM. It uses only:
  - raw document text/title,
  - global DocRED relation vocabulary IDs/labels,
  - optional gold-free relation disambiguation hints.

The LLM must extract both entities and relations from text. Gold entities and
gold relation triples are used only later by the evaluation code/notebook.

The script is placed under experiments/methods so it does not modify NeoOLAF's
core package. It is designed for fast full-dataset runs with document-level
parallelism and OpenRouter retry/fallback behavior.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
# Basic IO / normalization
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def filter_records(records: Iterable[Dict[str, Any]], type_filter: str = "all") -> List[Dict[str, Any]]:
    if not type_filter or type_filter == "all":
        return list(records)
    return [r for r in records if r.get("type") == type_filter or r.get("split") == type_filter]


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    return text


def safe_filename(value: str, max_len: int = 96) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value[:max_len].strip("_") or "document"


def document_id_from_record(record: Dict[str, Any], index: int = 0) -> str:
    return str(record.get("document_id") or record.get("id") or record.get("doc_id") or record.get("title") or f"doc_{index:06d}")


def title_from_record(record: Dict[str, Any], doc_id: str) -> str:
    return str(record.get("title") or record.get("name") or doc_id)


def document_text_from_record(record: Dict[str, Any]) -> str:
    for key in ["text", "raw_text", "document", "content", "article"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    sents = record.get("sents") or record.get("sentences")
    if isinstance(sents, list):
        if sents and isinstance(sents[0], list):
            return "\n".join(" ".join(map(str, sent)) for sent in sents)
        return "\n".join(map(str, sents))
    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list):
        return "\n\n".join(map(str, paragraphs))
    return json.dumps(record, ensure_ascii=False)


def rel_id_from_label(label: object) -> str:
    label = str(label or "").strip()
    if " : " in label:
        return label.split(" : ", 1)[0].strip()
    m = re.match(r"^(P\d+)", label)
    if m:
        return m.group(1)
    return label


def split_relation_label(raw: object) -> Tuple[Optional[str], str, str]:
    text = str(raw or "").strip()
    if not text:
        return None, "", ""
    m = re.match(r"^(P\d+)\s*:\s*(.+)$", text)
    if m:
        rid, label = m.group(1).strip(), m.group(2).strip()
        return rid, label, f"{rid} : {label}"
    m = re.match(r"^(P\d+)$", text)
    if m:
        rid = m.group(1).strip()
        return rid, rid, rid
    return None, text, text


def make_relation_spec(raw: object) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        source = raw.get("canonical") or raw.get("relation") or raw.get("label") or raw.get("name") or raw.get("id")
        rid = raw.get("id") or raw.get("relation_id")
        label = raw.get("label") or raw.get("name") or raw.get("relation_label")
        if rid and label:
            canonical = f"{str(rid).strip()} : {str(label).strip()}"
        else:
            parsed_id, parsed_label, canonical = split_relation_label(source)
            rid = rid or parsed_id
            label = label or parsed_label
    else:
        rid, label, canonical = split_relation_label(raw)
    if not canonical:
        return None
    aliases = {str(canonical), str(label)}
    if rid:
        aliases.add(str(rid))
    return {
        "id": str(rid).strip() if rid else None,
        "label": str(label).strip(),
        "canonical": str(canonical).strip(),
        "aliases": sorted(a for a in aliases if str(a).strip()),
    }


def merge_relation_specs(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not item:
            continue
        key = item.get("id") or normalize_text(item.get("canonical"))
        if not key:
            continue
        if key not in merged:
            merged[key] = copy.deepcopy(item)
        else:
            aliases = set(merged[key].get("aliases") or []) | set(item.get("aliases") or [])
            merged[key]["aliases"] = sorted(a for a in aliases if str(a).strip())
            for field in ["id", "label", "canonical"]:
                if not merged[key].get(field) and item.get(field):
                    merged[key][field] = item[field]
    return sorted(merged.values(), key=lambda x: (str(x.get("id") or ""), str(x.get("canonical") or "")))


def extract_relation_vocab_from_dataset(path: str | Path, type_filter: str = "all") -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for record in filter_records(iter_jsonl(path), type_filter):
        value = record.get("relations") or record.get("gold_relations") or record.get("labels") or record.get("triples")
        if isinstance(value, dict):
            for key in value.keys():
                spec = make_relation_spec(key)
                if spec:
                    specs.append(spec)
        elif isinstance(value, list):
            for rel in value:
                raw = rel.get("relation") if isinstance(rel, dict) else rel
                spec = make_relation_spec(raw)
                if spec:
                    specs.append(spec)
    return merge_relation_specs(specs)


def relation_alias_index(relations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for rel in relations:
        for alias in [rel.get("id"), rel.get("label"), rel.get("canonical"), *(rel.get("aliases") or [])]:
            key = normalize_text(alias)
            if key and key not in idx:
                idx[key] = rel
    return idx


def relation_spec_for(raw: object, allowed_relations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    idx = relation_alias_index(allowed_relations)
    key = normalize_text(raw)
    if key in idx:
        return idx[key]
    rid = rel_id_from_label(raw)
    if normalize_text(rid) in idx:
        return idx[normalize_text(rid)]
    return None


# ---------------------------------------------------------------------------
# OpenRouter / OpenAI-compatible backend
# ---------------------------------------------------------------------------

class ChatBackend:
    def __init__(
        self,
        *,
        backend_name: str,
        host: str,
        api_key: str,
        timeout: int = 600,
        max_tokens: int = 8192,
        reasoning_effort: Optional[str] = "minimal",
        exclude_reasoning: bool = True,
    ) -> None:
        self.backend_name = backend_name
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.exclude_reasoning = exclude_reasoning

    def chat_url(self) -> str:
        if self.host.endswith("/chat/completions"):
            return self.host
        if self.host.endswith("/v1"):
            return f"{self.host}/chat/completions"
        if self.host.endswith("/api"):
            return f"{self.host}/v1/chat/completions"
        return f"{self.host}/v1/chat/completions"

    def chat(self, *, model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if self.backend_name.lower() == "openrouter":
            reasoning: Dict[str, Any] = {}
            if self.reasoning_effort:
                reasoning["effort"] = self.reasoning_effort
            if self.exclude_reasoning:
                reasoning["exclude"] = True
            if reasoning:
                payload["reasoning"] = reasoning
        response = requests.post(self.chat_url(), headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned by backend: {str(data)[:1000]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text")
        if isinstance(content, list):
            content = "".join(
                str(block.get("text") or block.get("content") or "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if content is None or not str(content).strip():
            raise RuntimeError("No final message.content returned by backend openrouter/provider.")
        return str(content).strip()


def extract_json(text: str) -> Any:
    text = text.strip()
    # Prefer fenced json if present.
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Try full text.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try object slice.
    start_obj, end_obj = text.find("{"), text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        try:
            return json.loads(text[start_obj : end_obj + 1])
        except Exception:
            pass
    # Try array slice.
    start_arr, end_arr = text.find("["), text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        try:
            return json.loads(text[start_arr : end_arr + 1])
        except Exception:
            pass
    raise ValueError(f"Could not extract JSON from model response: {text[:1000]}")


def chat_json_with_retries(
    backend: ChatBackend,
    *,
    model: str,
    messages: List[Dict[str, str]],
    fallback_messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    retries: int = 3,
    retry_sleep: float = 2.0,
) -> Tuple[Any, str, Dict[str, Any]]:
    errors: List[str] = []
    raw = ""
    for attempt in range(max(1, retries)):
        try:
            use_messages = messages if attempt == 0 or fallback_messages is None else fallback_messages
            raw = backend.chat(model=model, messages=use_messages, temperature=temperature)
            return extract_json(raw), raw, {"attempts": attempt + 1, "errors": errors}
        except Exception as e:
            errors.append(f"attempt={attempt+1}: {type(e).__name__}: {e}")
            if attempt + 1 < max(1, retries):
                time.sleep(retry_sleep)
    raise RuntimeError("Direct raw ER call failed after retries: " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

DOCRED_HINTS = """
Relation calibration hints, gold-free:
- P17 country: use when a place, organization, or work is in/associated with a country.
- P27 country of citizenship: use only for people and nationality/citizenship.
- P159 headquarters location: use for organizations based/headquartered in a city/place.
- P131 administrative territorial entity: use for city/state/county containment, not for a country tail when P17 fits.
- P127 owned by: use for ownership/control by a company/group.
- P749 parent organization: use for organization-to-parent organization.
- P355 subsidiary: inverse of subsidiary relation when explicitly supported.
- P361 part of: use generic part-whole only when P127/P749/P355 are not better.
- P175 performer: use for songs/albums performed by artists.
- P170 creator: use for creators/authors, not singer/rapper performance.
- P162 producer: use only when the text explicitly says produced/producer.
- P264 record label: use for music label.
- P577 publication date: release/publication date of a creative work.
- P19 place of birth, P569 date of birth, P570 date of death, P69 educated at: biography facts.
Do not add peripheral facts merely because they are plausible. Prefer relations explicitly supported by text.
""".strip()

ENTITY_SCHEMA = """
Entity types allowed: PER, ORG, LOC, MISC, DATE, NUM.
Merge aliases into one entity when they refer to the same real-world object.
For dates, keep both full dates and years as separate entities only if both are explicitly mentioned.
""".strip()


def relation_vocab_prompt(relations: List[Dict[str, Any]], focus_ids: Optional[str] = None, max_relations: Optional[int] = None) -> str:
    selected = relations
    if focus_ids:
        ids = {x.strip() for x in str(focus_ids).split(",") if x.strip()}
        selected = [r for r in relations if r.get("id") in ids]
    if max_relations:
        selected = selected[:max_relations]
    return "\n".join(f"- {r.get('id')} | {r.get('canonical')}" for r in selected)


def build_raw_er_messages(
    record: Dict[str, Any],
    allowed_relations: List[Dict[str, Any]],
    *,
    focus_relation_ids: Optional[str] = None,
    max_relations: Optional[int] = None,
    use_hints: bool = True,
    text_char_limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    doc_id = document_id_from_record(record)
    title = title_from_record(record, doc_id)
    text = document_text_from_record(record)
    if text_char_limit and len(text) > text_char_limit:
        text = text[:text_char_limit] + "\n[TRUNCATED]"
    hints = DOCRED_HINTS if use_hints else ""
    system = (
        "You are a strict raw-text entity and relation extraction system for DocRED-style evaluation. "
        "You must extract BOTH entities and relations from the raw document text. "
        "You are not given gold/source entity IDs. Do not assume any hidden entity inventory. "
        "Use only relation IDs from the allowed relation vocabulary. Return JSON only."
    )
    user = f"""
Task: extract entities and document-level relations from raw text only.

Important: You are NOT given the gold entities. You must discover entities yourself.

{ENTITY_SCHEMA}

Relation rules:
1. Relation heads and tails must refer to local entity IDs that you created in the entities list.
2. Use only the relation IDs from ALLOWED RELATIONS.
3. Evidence must be a short quote or paraphrase from the document.
4. Do not use outside knowledge.
5. Prefer high precision: skip merely plausible facts.
6. If no relation is supported, return an empty relations list, but still extract salient entities.

RELATION DISAMBIGUATION HINTS:
{hints}

Output JSON schema:
{{
  "entities": [
    {{"entity_id": "E1", "label": "canonical name", "type": "PER|ORG|LOC|MISC|DATE|NUM", "aliases": ["alias1"], "evidence": "short evidence"}}
  ],
  "relations": [
    {{"head_entity_id": "E1", "relation_id": "P17", "tail_entity_id": "E2", "evidence": "short evidence"}}
  ]
}}

ALLOWED RELATIONS:
{relation_vocab_prompt(allowed_relations, focus_relation_ids=focus_relation_ids, max_relations=max_relations)}

DOCUMENT ID: {doc_id}
TITLE: {title}

RAW DOCUMENT TEXT:
{text}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Model output normalization and calibration
# ---------------------------------------------------------------------------

TYPE_MAP = {
    "PERSON": "PER", "PER": "PER", "PEOPLE": "PER",
    "ORG": "ORG", "ORGANIZATION": "ORG", "COMPANY": "ORG", "INSTITUTION": "ORG",
    "LOC": "LOC", "LOCATION": "LOC", "PLACE": "LOC", "COUNTRY": "LOC", "CITY": "LOC", "STATE": "LOC",
    "DATE": "DATE", "TIME": "DATE",
    "NUM": "NUM", "NUMBER": "NUM",
    "MISC": "MISC", "WORK": "MISC", "SONG": "MISC", "CREATIVE_WORK": "MISC",
}
COUNTRY_NAMES = {
    "greece", "greek", "united states", "usa", "u.s.", "u.s.a.", "america", "american", "brazil", "brazilian",
    "canada", "france", "england", "united kingdom", "uk", "ireland", "china", "japan", "germany", "italy",
}


def normalize_entity_item(item: Any, idx: int) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        return {"entity_id": f"E{idx}", "label": item, "type": "MISC", "aliases": [], "evidence": ""}
    if isinstance(item, (list, tuple)) and item:
        label = str(item[0])
        typ = str(item[1]) if len(item) > 1 else "MISC"
        return {"entity_id": f"E{idx}", "label": label, "type": typ, "aliases": [], "evidence": ""}
    if not isinstance(item, dict):
        return None
    eid = item.get("entity_id") or item.get("id") or item.get("local_id") or f"E{idx}"
    label = item.get("label") or item.get("name") or item.get("text") or item.get("canonical")
    if not label:
        return None
    raw_type = str(item.get("type") or item.get("entity_type") or "MISC").upper().replace(" ", "_")
    typ = TYPE_MAP.get(raw_type, raw_type if raw_type in {"PER", "ORG", "LOC", "MISC", "DATE", "NUM"} else "MISC")
    aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
    aliases = [str(a).strip() for a in aliases if str(a).strip()]
    if str(label).strip() not in aliases:
        aliases.insert(0, str(label).strip())
    return {
        "entity_id": str(eid).strip(),
        "label": str(label).strip(),
        "type": typ,
        "aliases": sorted(dict.fromkeys(aliases)),
        "evidence": str(item.get("evidence") or item.get("justification") or "").strip(),
        "source": "raw_text_llm_extraction",
    }


def normalize_relation_item(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {"head_entity_id": item[0], "relation_id": item[1], "tail_entity_id": item[2], "evidence": item[3] if len(item) >= 4 else ""}
    if not isinstance(item, dict):
        return None
    return item


def entity_lookup(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for ent in entities:
        for key in [ent.get("entity_id"), ent.get("label"), *(ent.get("aliases") or [])]:
            n = normalize_text(key)
            if n and n not in idx:
                idx[n] = ent
    return idx


def calibrate_relation_id(rel_id: str, head: Dict[str, Any], tail: Dict[str, Any], evidence: str) -> Optional[str]:
    """Gold-free conservative relation correction/rejection."""
    htype, ttype = head.get("type"), tail.get("type")
    hlabel, tlabel = normalize_text(head.get("label")), normalize_text(tail.get("label"))
    ev = normalize_text(evidence)
    rid = rel_id_from_label(rel_id)

    tail_is_country = tlabel in COUNTRY_NAMES or any(alias in COUNTRY_NAMES for alias in [normalize_text(a) for a in tail.get("aliases") or []])

    # Country versus administrative/geographic confusions.
    if rid in {"P131", "P159", "P276"} and tail_is_country:
        if htype == "PER":
            return "P27"
        return "P17"
    if rid == "P131" and tail_is_country:
        return "P17"

    # Headquartered/based-in organizations.
    if rid in {"P276", "P131"} and htype == "ORG" and ("based" in ev or "headquarter" in ev):
        return "P159"

    # Creative work families.
    if rid in {"P170", "P162"} and ("song" in hlabel or "album" in hlabel or "single" in ev) and ttype in {"PER", "ORG"}:
        if rid == "P162" and not re.search(r"\b(produced|producer|production)\b", ev):
            return None
        if rid == "P170" and re.search(r"\bby\b|performed|artist|rapper|singer", ev):
            return "P175"

    # Producer is often peripheral; require explicit evidence.
    if rid == "P162" and not re.search(r"\b(produced|producer|production)\b", ev):
        return None

    # Organization parent/ownership family.
    if rid in {"P361", "P127"} and htype == "ORG" and ttype == "ORG":
        if "parent" in ev:
            return "P749"
        if "subsidiary" in ev:
            return "P749"
        if "owned" in ev or "part of" in ev or "group" in ev:
            return "P127"

    # Employer should be explicit.
    if rid == "P108" and not re.search(r"\b(worked|employed|employee|taught|teacher|professor|appointed|served)\b", ev):
        return None

    return rid


def normalize_prediction(data: Any, allowed_relations: List[Dict[str, Any]], *, scoring_calibration: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(data, dict):
        raw_entities = data.get("entities") or data.get("entity_mentions") or []
        raw_relations = data.get("relations") or data.get("triples") or []
    elif isinstance(data, list):
        raw_entities, raw_relations = [], data
    else:
        raw_entities, raw_relations = [], []

    entities: List[Dict[str, Any]] = []
    seen_ent_keys: set[str] = set()
    for i, item in enumerate(raw_entities, 1):
        ent = normalize_entity_item(item, i)
        if not ent:
            continue
        key = normalize_text(ent["label"])
        if not key or key in seen_ent_keys:
            continue
        seen_ent_keys.add(key)
        entities.append(ent)

    ent_idx = entity_lookup(entities)
    allowed_idx = relation_alias_index(allowed_relations)
    relations: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    relabelled = 0
    seen_rel_keys: set[Tuple[str, str, str]] = set()

    for item in raw_relations:
        rel = normalize_relation_item(item)
        if rel is None:
            rejected.append({"raw": item, "reason": "invalid_relation_item"})
            continue
        raw_h = rel.get("head_entity_id") or rel.get("head_id") or rel.get("head") or rel.get("subject") or rel.get("h")
        raw_t = rel.get("tail_entity_id") or rel.get("tail_id") or rel.get("tail") or rel.get("object") or rel.get("t")
        raw_r = rel.get("relation_id") or rel.get("relation") or rel.get("predicate") or rel.get("r")
        ev = str(rel.get("evidence") or rel.get("justification") or "").strip()
        head = ent_idx.get(normalize_text(raw_h))
        tail = ent_idx.get(normalize_text(raw_t))
        spec = relation_spec_for(raw_r, allowed_relations)
        if head is None or tail is None or spec is None:
            rejected.append({"raw": rel, "reason": "unmapped_head_tail_or_relation"})
            continue
        rid = spec.get("id") or rel_id_from_label(spec.get("canonical"))
        if scoring_calibration:
            new_rid = calibrate_relation_id(rid, head, tail, ev)
            if new_rid is None:
                rejected.append({"raw": rel, "reason": "calibration_rejected"})
                continue
            if new_rid != rid:
                relabelled += 1
                rid = new_rid
                spec = allowed_idx.get(normalize_text(rid), spec)
        if head.get("entity_id") == tail.get("entity_id"):
            rejected.append({"raw": rel, "reason": "self_relation"})
            continue
        key = (str(head["entity_id"]), str(rid), str(tail["entity_id"]))
        if key in seen_rel_keys:
            continue
        seen_rel_keys.add(key)
        relations.append({
            "head_entity_id": head["entity_id"],
            "head": head["label"],
            "head_type": head["type"],
            "relation_id": rid,
            "relation": (spec.get("canonical") if spec else rid),
            "relation_label": (spec.get("label") if spec else rid),
            "tail_entity_id": tail["entity_id"],
            "tail": tail["label"],
            "tail_type": tail["type"],
            "evidence": ev,
            "source": "raw_text_entity_relation_extraction",
        })

    pred = {"entities": entities, "relations": relations}
    diag = {
        "raw_entity_items": len(raw_entities),
        "raw_relation_items": len(raw_relations),
        "entities": len(entities),
        "relations": len(relations),
        "rejected_relations": len(rejected),
        "relabelled_relations": relabelled,
        "rejected_preview": rejected[:20],
        "mode": "raw_text_entity_relation_extraction_no_gold_entities_in_prompt",
    }
    pred["projection_diagnostics"] = diag
    return pred, diag


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(index: int, record: Dict[str, Any], args: argparse.Namespace, allowed_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    start = time.time()
    doc_id = document_id_from_record(record, index)
    title = title_from_record(record, doc_id)
    artifact_dir = Path(args.artifacts_root) / f"{index:06d}_{safe_filename(doc_id)}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    backend = ChatBackend(
        backend_name=args.backend_name,
        host=args.host,
        api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or args.api_key or "",
        timeout=args.request_timeout,
        max_tokens=args.max_tokens,
        reasoning_effort=args.openrouter_reasoning_effort,
        exclude_reasoning=args.openrouter_exclude_reasoning,
    )

    messages = build_raw_er_messages(
        record,
        allowed_relations,
        focus_relation_ids=args.docred_raw_focus_relation_ids,
        max_relations=args.docred_raw_max_relations,
        use_hints=not args.docred_raw_disable_hints,
        text_char_limit=args.text_char_limit,
    )
    fallback_messages = build_raw_er_messages(
        record,
        allowed_relations,
        focus_relation_ids=args.docred_raw_focus_relation_ids,
        max_relations=args.docred_raw_max_relations,
        use_hints=False,
        text_char_limit=args.text_char_limit,
    )

    try:
        parsed, raw_response, call_diag = chat_json_with_retries(
            backend,
            model=args.model_name,
            messages=messages,
            fallback_messages=fallback_messages,
            temperature=args.temperature,
            retries=args.docred_raw_retries,
            retry_sleep=args.docred_raw_retry_sleep,
        )
        prediction, diag = normalize_prediction(parsed, allowed_relations, scoring_calibration=args.docred_raw_scoring_calibration)
        diag.update(call_diag)
        prediction["projection_diagnostics"].update(call_diag)
        result = {
            "document_id": doc_id,
            "title": title,
            "type": record.get("type") or record.get("split"),
            "method": "neoolaf_docred_raw_text_entity_relation_extraction",
            "parsed_ok": True,
            "prediction": prediction,
            "raw_counts": {
                "raw_entities": len(prediction.get("entities") or []),
                "raw_relations": len(prediction.get("relations") or []),
                "raw_relation_items": diag.get("raw_relation_items", 0),
                "rejected_relations": diag.get("rejected_relations", 0),
                "relabelled_relations": diag.get("relabelled_relations", 0),
            },
            "artifact_dir": str(artifact_dir),
            "runtime_seconds": time.time() - start,
        }
        (artifact_dir / "raw_text_entity_relation_extraction.json").write_text(
            json.dumps({"messages": messages, "raw_response": raw_response, "result": result}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
    except Exception as e:
        tb = traceback.format_exc()
        error_result = {
            "document_id": doc_id,
            "title": title,
            "type": record.get("type") or record.get("split"),
            "method": "neoolaf_docred_raw_text_entity_relation_extraction",
            "parsed_ok": False,
            "prediction": {"entities": [], "relations": []},
            "raw_counts": {"raw_entities": 0, "raw_relations": 0},
            "artifact_dir": str(artifact_dir),
            "runtime_seconds": time.time() - start,
            "error": str(e),
            "error_type": type(e).__name__,
            "error_traceback": tb,
        }
        (artifact_dir / "raw_text_entity_relation_extraction.error.json").write_text(
            json.dumps(error_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return error_result


def append_jsonl_threadsafe(path: Path, row: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run raw-text DocRED entity+relation extraction with OpenRouter.")
    p.add_argument("--dataset-jsonl-path", required=True)
    p.add_argument("--output-jsonl-path", required=True)
    p.add_argument("--summary-output-path", required=True)
    p.add_argument("--error-log-jsonl-path", required=True)
    p.add_argument("--artifacts-root", required=True)
    p.add_argument("--type-filter", default="dev")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--document-workers", type=int, default=8)

    p.add_argument("--backend-name", default="openrouter")
    p.add_argument("--host", default="https://openrouter.ai/api")
    p.add_argument("--api-key", default="")
    p.add_argument("--model-name", default="openai/gpt-oss-20b")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--openrouter-reasoning-effort", default="minimal")
    p.add_argument("--openrouter-exclude-reasoning", action="store_true", default=True)

    p.add_argument("--relation-vocab-dataset-path", default=None)
    p.add_argument("--relation-vocab-output-path", default=None)
    p.add_argument("--docred-raw-focus-relation-ids", default=None)
    p.add_argument("--docred-raw-max-relations", type=int, default=None)
    p.add_argument("--docred-raw-disable-hints", action="store_true")
    p.add_argument("--docred-raw-retries", type=int, default=3)
    p.add_argument("--docred-raw-retry-sleep", type=float, default=2.0)
    p.add_argument("--docred-raw-scoring-calibration", action="store_true", default=True)
    p.add_argument("--text-char-limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.output_jsonl_path)
    err_path = Path(args.error_log_jsonl_path)
    summary_path = Path(args.summary_output_path)
    Path(args.artifacts_root).mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean previous outputs for this run.
    out_path.unlink(missing_ok=True)
    err_path.unlink(missing_ok=True)

    records = filter_records(iter_jsonl(args.dataset_jsonl_path), args.type_filter)
    if args.max_docs is not None:
        records = records[: args.max_docs]
    if not records:
        raise SystemExit("No records selected.")

    vocab_path = args.relation_vocab_dataset_path or args.dataset_jsonl_path
    allowed_relations = extract_relation_vocab_from_dataset(vocab_path, args.type_filter)
    if args.relation_vocab_output_path:
        Path(args.relation_vocab_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.relation_vocab_output_path).write_text(
            json.dumps({"count": len(allowed_relations), "relations": allowed_relations}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"[raw-er] docs={len(records)} allowed_relations={len(allowed_relations)} workers={args.document_workers}", flush=True)

    lock = threading.Lock()
    results: List[Dict[str, Any]] = []
    started = time.time()
    iterator = range(len(records))
    progress = tqdm(total=len(records), desc="DocRED raw ER documents") if tqdm else None

    def run_one(i: int) -> Dict[str, Any]:
        return process_record(i, records[i], args, allowed_relations)

    with cf.ThreadPoolExecutor(max_workers=max(1, args.document_workers)) as ex:
        future_to_i = {ex.submit(run_one, i): i for i in iterator}
        for fut in cf.as_completed(future_to_i):
            i = future_to_i[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "document_id": document_id_from_record(records[i], i),
                    "title": title_from_record(records[i], document_id_from_record(records[i], i)),
                    "parsed_ok": False,
                    "prediction": {"entities": [], "relations": []},
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_traceback": traceback.format_exc(),
                }
            results.append(row)
            append_jsonl_threadsafe(out_path, row, lock)
            if not row.get("parsed_ok"):
                append_jsonl_threadsafe(err_path, row, lock)
            print(
                f"[{len(results)}/{len(records)}] {row.get('document_id')} "
                f"ok={row.get('parsed_ok')} entities={len(row.get('prediction',{}).get('entities',[]))} "
                f"relations={len(row.get('prediction',{}).get('relations',[]))} "
                f"time={row.get('runtime_seconds', 0):.2f}s",
                flush=True,
            )
            if progress:
                progress.update(1)
    if progress:
        progress.close()

    parsed_ok = sum(1 for r in results if r.get("parsed_ok"))
    failed = len(results) - parsed_ok
    relations = sum(len(r.get("prediction", {}).get("relations", [])) for r in results)
    entities = sum(len(r.get("prediction", {}).get("entities", [])) for r in results)
    zero_rel_docs = sum(1 for r in results if r.get("parsed_ok") and not r.get("prediction", {}).get("relations"))
    error_type_counts = Counter(r.get("error_type") for r in results if not r.get("parsed_ok"))
    summary = {
        "dataset_jsonl_path": str(args.dataset_jsonl_path),
        "output_jsonl_path": str(out_path),
        "method": "neoolaf_docred_raw_text_entity_relation_extraction",
        "model_name": args.model_name,
        "type_filter": args.type_filter,
        "documents": len(results),
        "parsed_ok": parsed_ok,
        "failed": failed,
        "entities": entities,
        "relations": relations,
        "zero_relation_docs": zero_rel_docs,
        "elapsed_seconds": time.time() - started,
        "document_workers": args.document_workers,
        "error_type_counts": dict(error_type_counts),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[raw-er] finished parsed_ok={parsed_ok}/{len(results)} entities={entities} relations={relations} "
        f"failed={failed} zero_relation_docs={zero_rel_docs} elapsed_seconds={summary['elapsed_seconds']:.2f} "
        f"output={out_path} summary={summary_path}",
        flush=True,
    )
    if failed:
        print(f"[raw-er] error_log={err_path}", flush=True)
        print(f"[raw-er] error_type_counts={dict(error_type_counts)}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

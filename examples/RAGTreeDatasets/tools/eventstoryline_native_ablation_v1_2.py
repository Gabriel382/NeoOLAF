from __future__ import annotations

"""Native NeoOLAF one-document EventStoryLine experiment support, v1.2.

This module is experiment-only and changes no file under ``src/neoolaf``.
It keeps the complete Layer 0--12 pipeline while adding dataset-specific,
profile-driven two-phase orchestration for mention-level event extraction and the two
EventStoryLine relations PRECONDITION and FALLING_ACTION.

Gold event IDs and gold relation pairs are never available to NeoOLAF. Sentence
and token indices are deterministic source-document structure and are used only
to preserve event mention identity. Strict projection to gold IDs happens after
Layer 12 for evaluation.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import csv
import json
import re
import shutil
import sys
import threading
import time
import traceback

import docred_native_ablation as v2
import docred_native_ablation_v3 as v3
import docred_native_ablation_v4 as v4

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.documents import Document
from neoolaf.domain.enriched_expression import EnrichedExpression, EnrichmentEvidence
from neoolaf.domain.linguistic_expression import Evidence, LinguisticExpression
from neoolaf.domain.user_guidance import UserGuidance
from neoolaf.grounding.rag.formatting import build_grounding_context
from neoolaf.grounding.rag.types import GroundingRequest
from neoolaf.ontology.loader import SeedOntologyLoader
from neoolaf.profiles.profile_loader import load_document_profile
from neoolaf.layers.layer03_candidate_typing_resolution.component import CandidateTypingResolutionLayer

from experiments.methods.run_neoolaf import (
    OfflineWebSearchSource,
    OfflineWikipediaSource,
    OfflineWikidataSource,
    OpenAICompatibleBackend,
    load_user_guidance,
)

# Re-export notebook helpers.
read_json = v4.read_json
read_jsonl = v4.read_jsonl
write_json = v4.write_json
append_jsonl = v4.append_jsonl
load_layer_states = v4.load_layer_states
state_counts = v4.state_counts
safe_name = v4.safe_name
Tee = v4.Tee
LAYER_NAMES = v4.LAYER_NAMES


def layer_name(index: int) -> str:
    """Return a stable layer name whether upstream exposes a list or mapping."""
    names = LAYER_NAMES
    if isinstance(names, dict):
        return str(names.get(index, f"layer_{index:02d}"))
    if isinstance(names, (list, tuple)) and 0 <= index < len(names):
        return str(names[index])
    return f"layer_{index:02d}"


SharedCallLogger = v4.SharedCallLogger
TaggedLoggedBackend = v4.TaggedLoggedBackend


def seed_ontology_summary(path: str | Path) -> dict[str, Any]:
    ontology = SeedOntologyLoader().load(str(Path(path).resolve()))
    return {
        "path": str(Path(path).resolve()),
        "class_count": len(ontology.classes_by_uri),
        "property_count": len(ontology.properties_by_uri),
    }


RELATION_IDS = ("PRECONDITION", "FALLING_ACTION")
EVENT_KEY_RE = re.compile(
    r"^S(?P<sent>\d+)\[(?P<start>\d+):(?P<end>\d+)\]::(?P<trigger>.+)$",
    re.IGNORECASE,
)


def _dedup(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _norm(text: Any) -> str:
    value = str(text or "").lower().replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _parse_relation_instance(text: Any) -> tuple[str, str, str] | None:
    parts = [part.strip() for part in str(text or "").split("||")]
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def _json_block(value: Any, max_chars: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= max_chars else text[:max_chars] + "\n... [truncated]"


def normalize_relation_id(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if "PRECONDITION" in text:
        return "PRECONDITION"
    if "FALLING_ACTION" in text or "FALLINGACTION" in text:
        return "FALLING_ACTION"
    return None


def parse_event_key(value: Any) -> dict[str, Any] | None:
    match = EVENT_KEY_RE.match(str(value or "").strip())
    if not match:
        return None
    return {
        "sent_id": int(match.group("sent")),
        "start": int(match.group("start")),
        "end": int(match.group("end")),
        "trigger": match.group("trigger").strip(),
    }


def canonical_event_key(value: Any, tokens: list[list[str]]) -> str | None:
    parsed = parse_event_key(value)
    if parsed is None:
        return None
    sent_id = parsed["sent_id"]
    start = parsed["start"]
    end = parsed["end"]
    if sent_id < 0 or sent_id >= len(tokens):
        return None
    sentence_tokens = tokens[sent_id]
    if start < 0 or end <= start or end > len(sentence_tokens):
        return None
    trigger = " ".join(str(token) for token in sentence_tokens[start:end]).strip()
    if not trigger:
        return None
    return f"S{sent_id}[{start}:{end}]::{trigger}"


def indexed_token_table(sentences: list[str], tokens: list[list[str]]) -> str:
    rows: list[str] = []
    total = max(len(sentences), len(tokens))
    for sent_id in range(total):
        sentence = sentences[sent_id] if sent_id < len(sentences) else ""
        sentence_tokens = tokens[sent_id] if sent_id < len(tokens) else []
        indexed = " ".join(f"{index}={token}" for index, token in enumerate(sentence_tokens))
        rows.append(f"[S{sent_id}] {sentence}\nTOKENS {indexed}")
    return "\n\n".join(rows)


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key in fieldnames:
            value = row.get(key)
            if isinstance(value, (dict, list, tuple, set)):
                value = json.dumps(value, ensure_ascii=False)
            item[key] = value
        normalized.append(item)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized)


class EventStoryLineExtractionLayer(v4.LinguisticExpressionExtractionLayer):
    """Two-phase Layer 1: exhaustive validated events, then closed-inventory pairs.

    Phase A asks only for event mentions. The experiment code validates every
    sentence/token span, performs bounded source-token repair, and rejects
    function-word/connective-only candidates. Phase B receives the resulting
    closed event inventory and may only connect those event IDs. No gold event
    IDs, gold spans, or gold pairs are available in either phase.
    """

    _CONNECTIVE_ONLY = {
        "a", "an", "and", "as", "at", "after", "before", "because", "but",
        "by", "for", "from", "if", "in", "into", "of", "on", "or", "so",
        "than", "that", "the", "then", "there", "to", "when", "where",
        "while", "with", "without", "yet",
    }
    _AUXILIARY_ONLY = {
        "am", "are", "be", "been", "being", "can", "could", "did", "do",
        "does", "had", "has", "have", "is", "may", "might", "must", "shall",
        "should", "was", "were", "will", "would",
    }
    _PHRASAL_PARTICLES = {
        "away", "back", "down", "in", "into", "off", "on", "onto", "out",
        "over", "through", "up",
    }
    _MAX_REPAIR_SPAN = 8

    def __init__(
        self,
        *args: Any,
        decision_log_path: str | Path,
        inventory_log_path: str | Path | None = None,
        relation_generation_log_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.decision_log_path = Path(decision_log_path)
        base = self.decision_log_path.parent
        self.inventory_log_path = Path(inventory_log_path or base / "layer01_event_inventory.json")
        self.relation_generation_log_path = Path(
            relation_generation_log_path or base / "layer01_relation_generation.json"
        )

    @staticmethod
    def _item_int(item: dict[str, Any], *names: str) -> int | None:
        for name in names:
            value = item.get(name)
            if value is None or value == "":
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _valid_event_tokens(cls, values: list[str]) -> tuple[bool, str]:
        normalized = [_norm(value) for value in values]
        normalized = [value for value in normalized if value]
        if not normalized:
            return False, "empty_or_punctuation_only"
        lexical = [
            value for value in normalized
            if value not in cls._CONNECTIVE_ONLY and value not in cls._AUXILIARY_ONLY
        ]
        if not lexical:
            return False, "connective_or_auxiliary_only"
        return True, "valid_lexical_event_span"

    @classmethod
    def _find_trigger_spans(
        cls,
        sentence_tokens: list[str],
        trigger: Any,
        proposed_start: int | None,
    ) -> list[tuple[int, int]]:
        target = _norm(trigger)
        if not target:
            return []
        matches: list[tuple[int, int]] = []
        for start in range(len(sentence_tokens)):
            max_end = min(len(sentence_tokens), start + cls._MAX_REPAIR_SPAN)
            for end in range(start + 1, max_end + 1):
                if _norm(" ".join(sentence_tokens[start:end])) == target:
                    matches.append((start, end))
        if proposed_start is not None:
            matches.sort(key=lambda span: (abs(span[0] - proposed_start), span[1] - span[0], span[0]))
        return matches

    @classmethod
    def _extend_source_span(
        cls,
        sentence_tokens: list[str],
        start: int,
        end: int,
    ) -> tuple[int, int, list[str]]:
        reasons: list[str] = []
        repaired_end = end

        # Preserve tokenized hyphen compounds such as rear - ended.
        if repaired_end > start and sentence_tokens[repaired_end - 1] == "-" and repaired_end < len(sentence_tokens):
            repaired_end += 1
            reasons.append("completed_trailing_hyphen_compound")
        elif repaired_end < len(sentence_tokens) and sentence_tokens[repaired_end] == "-" and repaired_end + 1 < len(sentence_tokens):
            repaired_end += 2
            reasons.append("completed_following_hyphen_compound")

        # Preserve common phrasal-verb particles: checked into, showed up, etc.
        if repaired_end < len(sentence_tokens):
            next_norm = _norm(sentence_tokens[repaired_end])
            current_values = sentence_tokens[start:repaired_end]
            valid_current, _ = cls._valid_event_tokens(current_values)
            if valid_current and next_norm in cls._PHRASAL_PARTICLES and repaired_end - start <= 3:
                repaired_end += 1
                reasons.append("completed_phrasal_verb_particle")

        return start, repaired_end, reasons

    @classmethod
    def _repair_event_item(
        cls,
        item: dict[str, Any],
        tokens: list[list[str]],
    ) -> tuple[str | None, dict[str, Any]]:
        raw_key = str(item.get("event_key") or item.get("text") or "").strip()
        parsed = parse_event_key(raw_key) if raw_key else None
        sent_id = cls._item_int(item, "sentence_id", "sent_id")
        start = cls._item_int(item, "token_start", "start")
        end = cls._item_int(item, "token_end", "end")
        trigger = str(item.get("trigger") or item.get("trigger_text") or "").strip()

        if parsed is not None:
            sent_id = parsed["sent_id"] if sent_id is None else sent_id
            start = parsed["start"] if start is None else start
            end = parsed["end"] if end is None else end
            trigger = trigger or parsed["trigger"]

        audit: dict[str, Any] = {
            "raw_item": item,
            "proposed_event_key": raw_key or None,
            "proposed_sentence_id": sent_id,
            "proposed_token_start": start,
            "proposed_token_end": end,
            "proposed_trigger": trigger,
            "repair_steps": [],
        }
        if sent_id is None or sent_id < 0 or sent_id >= len(tokens):
            audit.update(status="rejected", reason="invalid_sentence_id")
            return None, audit
        sentence_tokens = [str(token) for token in tokens[sent_id]]

        # First preference: locate the complete trigger text in the authoritative
        # source-token sentence. This fixes offsets while preserving model intent.
        trigger_matches = cls._find_trigger_spans(sentence_tokens, trigger, start)
        if trigger_matches:
            chosen_start, chosen_end = trigger_matches[0]
            if start != chosen_start or end != chosen_end:
                audit["repair_steps"].append("aligned_to_complete_trigger_text")
            start, end = chosen_start, chosen_end

        if start is None or end is None or start < 0 or end <= start or end > len(sentence_tokens):
            audit.update(status="rejected", reason="invalid_token_span")
            return None, audit

        start, end, extension_steps = cls._extend_source_span(sentence_tokens, start, end)
        audit["repair_steps"].extend(extension_steps)
        if end > len(sentence_tokens):
            audit.update(status="rejected", reason="repaired_span_out_of_bounds")
            return None, audit

        span_tokens = sentence_tokens[start:end]
        valid, reason = cls._valid_event_tokens(span_tokens)
        if not valid:
            audit.update(status="rejected", reason=reason, repaired_tokens=span_tokens)
            return None, audit

        canonical = f"S{sent_id}[{start}:{end}]::{' '.join(span_tokens).strip()}"
        audit.update(
            status="accepted",
            reason="validated_source_token_span",
            canonical_event_key=canonical,
            sentence_id=sent_id,
            token_start=start,
            token_end=end,
            repaired_trigger=" ".join(span_tokens).strip(),
            repaired=bool(audit["repair_steps"]),
        )
        return canonical, audit

    def _event_inventory_prompt(self, state: PipelineState, chunk_text: str) -> list[dict[str, str]]:
        profile = state.profile_config or {}
        task = profile.get("_input_task_guidance", {}) or {}
        sentences = profile.get("_input_sentences", []) or []
        tokens = profile.get("_input_tokens", []) or []
        table = indexed_token_table(sentences, tokens)
        layer_guidance = (task.get("layer_guidance") or {}).get("layer01", {})
        negatives = task.get("negative_examples") or []
        span_examples = task.get("event_span_examples") or []

        system = """
You are NeoOLAF Layer 1A for exhaustive EventStoryLine event-mention extraction.

Extract every explicit mention-level event, action, process, occurrence, or
state that could participate in a storyline relation. Work sentence by sentence.
Do not extract relations in this phase.

For each event return structured zero-based, end-exclusive token offsets from the
authoritative indexed token table. The trigger MUST equal the complete token
slice. Preserve the smallest complete semantic trigger:
- include phrasal-verb particles: checked into, showed up, pulled up, going back;
- include all tokenized hyphen pieces: rear - ended;
- include necessary multi-token event expressions: court hearing, on her way;
- do not truncate a trigger to checked, rear -, or another incomplete fragment;
- never output a connective/function token by itself, including to, then, after,
  before, because, and, but, or a bare auxiliary;
- ignore URLs and publication/update metadata unless they describe a storyline event;
- do not output people, organizations, places, dates, concrete objects, or gold data.

Return JSON only:
{
  "event_mentions": [
    {
      "sentence_id": 3,
      "token_start": 24,
      "token_end": 26,
      "trigger": "checked into",
      "event_kind": "action",
      "justification": "Explicit treatment-entry event."
    }
  ],
  "sentence_coverage": [
    {"sentence_id": 3, "event_count": 1, "checked": true}
  ]
}
""".strip()
        user = f"""
Dataset-specific Layer 1 guidance:
{_json_block(layer_guidance, 6000)}

Span examples:
{_json_block(span_examples, 4500)}

Negative examples:
{_json_block(negatives, 3500)}

Raw document:
\"\"\"
{chunk_text}
\"\"\"

Indexed sentence/token table (authoritative):
\"\"\"
{table}
\"\"\"

Return an exhaustive event inventory with exact structured spans. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _relation_inventory_prompt(
        self,
        state: PipelineState,
        chunk_text: str,
        event_rows: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        profile = state.profile_config or {}
        task = profile.get("_input_task_guidance", {}) or {}
        relation_specs = task.get("relation_specs") or []
        examples = task.get("synthetic_relation_examples") or []
        negatives = task.get("negative_examples") or []
        inventory = [
            {
                "event_id": row["event_ref"],
                "event_key": row["event_key"],
                "sentence": row.get("sentence"),
                "justification": row.get("justification"),
            }
            for row in event_rows
        ]
        system = """
You are NeoOLAF Layer 1B for exhaustive directed EventStoryLine pair discovery.

You receive a CLOSED, validated event inventory. Return all supported directed
storyline pairs whose semantics are compatible with either:
1. source establishes/enables/motivates/requires/sets up target; or
2. target is a downstream consequence/continuation/aftermath/follow-up of source.

Rules:
- source_event_id and target_event_id MUST come from the supplied inventory;
- never create or alter an event span;
- preserve source-to-target direction;
- include cross-sentence pairs when the document supports them;
- sentence order alone and mere co-occurrence are insufficient;
- omit null/unsupported pairs and self-relations;
- relation_cue must be a concise lexical/semantic phrase such as enables,
  required for, followed by, leads to, continuation of; do not put a canonical
  benchmark relation ID in relation_cue;
- perform a final coverage pass over plausible event pairs.

Return JSON only:
{
  "relation_instances": [
    {
      "source_event_id": "E0001",
      "relation_cue": "enables",
      "target_event_id": "E0004",
      "justification": "The source event establishes the target event."
    }
  ],
  "coverage_check": {"inventory_events": 0, "relations": 0, "checked": true}
}
""".strip()
        user = f"""
Relation-family definitions and direction rules:
{_json_block(relation_specs, 6500)}

Synthetic examples:
{_json_block(examples, 6000)}

Negative examples:
{_json_block(negatives, 3500)}

Validated closed event inventory:
{_json_block(inventory, 15000)}

Raw document for semantic context:
\"\"\"
{chunk_text}
\"\"\"

Return all supported directed pairs using only event_id values above. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _run(self, state: PipelineState) -> PipelineState:
        chunks = list(state.document.chunks)
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        sentences = (state.profile_config or {}).get("_input_sentences", []) or []
        expressions: list[LinguisticExpression] = []
        combined_decisions: list[dict[str, Any]] = []
        inventory_audit: list[dict[str, Any]] = []
        relation_audit: list[dict[str, Any]] = []
        expr_counter = 0

        for chunk in chunks:
            # Phase A: event inventory only.
            event_messages = self._event_inventory_prompt(state, chunk.text)
            event_raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=event_messages,
                temperature=self.temperature,
            )
            event_parsed = self._safe_extract_json(
                raw_response=event_raw,
                state=state,
                chunk_id=f"{chunk.chunk_id}_event_inventory",
                messages=event_messages,
            )
            event_items: list[dict[str, Any]] = []
            if isinstance(event_parsed, dict):
                raw_items = event_parsed.get("event_mentions")
                if raw_items is None:
                    # Backward-compatible acceptance of legacy expression objects.
                    raw_items = [
                        item for item in (event_parsed.get("expressions") or [])
                        if isinstance(item, dict) and str(item.get("label") or "").lower() == "event_mention"
                    ]
                event_items = [item for item in (raw_items or []) if isinstance(item, dict)]

            accepted_by_key: dict[str, dict[str, Any]] = {}
            for item in event_items:
                canonical, audit = self._repair_event_item(item, tokens)
                audit["phase"] = "event_inventory"
                audit["justification"] = str(item.get("justification") or "").strip()
                inventory_audit.append(audit)
                if canonical is None:
                    combined_decisions.append({
                        "phase": "event_inventory",
                        "status": "rejected",
                        "reason": audit.get("reason"),
                        "text": item.get("event_key") or item.get("text") or item.get("trigger"),
                        "label": "event_mention",
                        "repair_steps": audit.get("repair_steps", []),
                    })
                    continue
                accepted_by_key.setdefault(canonical, {
                    "event_key": canonical,
                    "justification": str(item.get("justification") or "Explicit event mention.").strip(),
                    "repair_steps": audit.get("repair_steps", []),
                })

            event_rows: list[dict[str, Any]] = []
            for index, key in enumerate(sorted(accepted_by_key, key=lambda value: (
                parse_event_key(value)["sent_id"],
                parse_event_key(value)["start"],
                parse_event_key(value)["end"],
            ))):
                row = accepted_by_key[key]
                parsed_key = parse_event_key(key) or {}
                sent_id = int(parsed_key.get("sent_id", -1))
                row.update(
                    event_ref=f"E{index:04d}",
                    sentence=sentences[sent_id] if 0 <= sent_id < len(sentences) else "",
                )
                event_rows.append(row)

            # Phase B: relation pairs can only reference Phase-A event IDs.
            relation_items: list[dict[str, Any]] = []
            if event_rows:
                relation_messages = self._relation_inventory_prompt(state, chunk.text, event_rows)
                relation_raw = self.ollama_backend.chat(
                    model=state.llm_model,
                    messages=relation_messages,
                    temperature=self.temperature,
                )
                relation_parsed = self._safe_extract_json(
                    raw_response=relation_raw,
                    state=state,
                    chunk_id=f"{chunk.chunk_id}_relation_inventory",
                    messages=relation_messages,
                )
                if isinstance(relation_parsed, dict):
                    relation_items = [
                        item for item in (relation_parsed.get("relation_instances") or [])
                        if isinstance(item, dict)
                    ]

            by_ref = {row["event_ref"]: row for row in event_rows}
            relation_rows: list[dict[str, str]] = []
            relation_seen: set[tuple[str, str, str]] = set()
            for item in relation_items:
                source_ref = str(item.get("source_event_id") or item.get("source_id") or "").strip()
                target_ref = str(item.get("target_event_id") or item.get("target_id") or "").strip()
                cue = str(item.get("relation_cue") or item.get("predicate") or item.get("cue") or "").strip()
                justification = str(item.get("justification") or "").strip()
                audit = {
                    "phase": "relation_inventory",
                    "source_event_id": source_ref,
                    "target_event_id": target_ref,
                    "relation_cue": cue,
                    "justification": justification,
                }
                if source_ref not in by_ref or target_ref not in by_ref:
                    audit.update(status="rejected", reason="endpoint_not_in_closed_inventory")
                    relation_audit.append(audit)
                    combined_decisions.append({**audit, "label": "relation_instance"})
                    continue
                if source_ref == target_ref:
                    audit.update(status="rejected", reason="self_relation")
                    relation_audit.append(audit)
                    combined_decisions.append({**audit, "label": "relation_instance"})
                    continue
                if not cue or normalize_relation_id(cue) is not None:
                    audit.update(status="rejected", reason="missing_or_canonical_relation_cue")
                    relation_audit.append(audit)
                    combined_decisions.append({**audit, "label": "relation_instance"})
                    continue
                source_key = by_ref[source_ref]["event_key"]
                target_key = by_ref[target_ref]["event_key"]
                identity = (source_key, _norm(cue), target_key)
                if identity in relation_seen:
                    audit.update(status="rejected", reason="duplicate_relation_instance")
                    relation_audit.append(audit)
                    continue
                relation_seen.add(identity)
                text = f"{source_key} || {cue} || {target_key}"
                relation_rows.append({"text": text, "label": "relation_instance", "justification": justification})
                audit.update(status="accepted", reason="closed_inventory_relation", relation_instance=text)
                relation_audit.append(audit)

            accepted_rows: list[dict[str, str]] = [
                {
                    "text": row["event_key"],
                    "label": "event_mention",
                    "justification": row["justification"],
                }
                for row in event_rows
            ]
            accepted_rows.extend(relation_rows)

            for row in accepted_rows:
                text = row["text"]
                label = row["label"]
                justification = row["justification"]
                expr_id = f"expr_{expr_counter:05d}"
                expressions.append(LinguisticExpression(
                    expr_id=expr_id,
                    text=text,
                    label=label,
                    justification=justification,
                    evidence=[Evidence(
                        chunk_id=chunk.chunk_id,
                        chunk_start_char=-1,
                        chunk_end_char=-1,
                        doc_start_char=-1,
                        doc_end_char=-1,
                        snippet=chunk.text[:1600],
                    )],
                ))
                combined_decisions.append({
                    "phase": "materialization",
                    "status": "accepted",
                    "expr_id": expr_id,
                    "text": text,
                    "label": label,
                    "relation_instance": _parse_relation_instance(text),
                    "justification": justification,
                })
                expr_counter += 1

        dedup: dict[tuple[str, str], LinguisticExpression] = {}
        for expr in expressions:
            dedup.setdefault((expr.text, expr.label), expr)
        state.linguistic_expressions = list(dedup.values())

        for path, rows in [
            (self.inventory_log_path, inventory_audit),
            (self.relation_generation_log_path, relation_audit),
            (self.decision_log_path, combined_decisions),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, rows)

        state.log(
            f"[{self.name}] EventStoryLine two-phase indexed extraction; "
            f"events={sum(1 for x in state.linguistic_expressions if x.label == 'event_mention')}; "
            f"relation_instances={sum(1 for x in state.linguistic_expressions if x.label == 'relation_instance')}; "
            f"event_repairs={sum(1 for x in inventory_audit if x.get('status') == 'accepted' and x.get('repaired'))}"
        )
        return state


class EventStoryLineCandidateEnrichmentLayer(v4.SelectiveContrastiveCandidateEnrichmentLayer):
    """Deterministic event enrichment plus parallel two-class relation linking."""

    def __init__(self, *args: Any, compact_prompt_log_path: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compact_prompt_log_path = Path(compact_prompt_log_path)
        self._prompt_audit: list[dict[str, Any]] = []
        self._prompt_lock = threading.Lock()

    def _candidate_ids_from_grounding(self, grounding_text: str) -> list[str]:
        values: list[str] = []
        text = str(grounding_text or "").upper()
        for relation_id in RELATION_IDS:
            if relation_id in text or relation_id.replace("_", "") in text.replace("_", ""):
                values.append(relation_id)
        return values

    def _contrastive_prompt(self, *, expr: LinguisticExpression, state: PipelineState, grounding_text: str) -> list[dict[str, str]]:
        triple = _parse_relation_instance(expr.text)
        if triple is None:
            raise ValueError(f"Malformed EventStoryLine relation instance: {expr.text}")
        source, predicate, target = triple
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        candidates = [
            {
                "relation_id": relation_id,
                "label": self.catalog[relation_id]["label"],
                "definition": self.catalog[relation_id].get("comment") or "",
                "domain": "Event",
                "range": "Event",
            }
            for relation_id in RELATION_IDS
        ]
        messages = [
            {
                "role":"system",
                "content": """
You are NeoOLAF Layer 2 for EventStoryLine ontology linking.

Classify exactly one already-extracted directed event pair as PRECONDITION,
FALLING_ACTION, or found=false. Do not create events, change event spans, swap
direction, or create another pair.

PRECONDITION: the source establishes, enables, motivates, requires, or sets up
the target.
FALLING_ACTION: the target is a downstream consequence, continuation, aftermath,
follow-up, or narratively dependent event connected from the source.

Sentence order alone and topical co-occurrence are insufficient. Return JSON only:
{"found":true,"selected_relation_id":"PRECONDITION","decision":"brief contrastive reason"}
or {"found":false,"decision":"brief reason"}.
""".strip(),
            },
            {
                "role":"user",
                "content": f"""
INSTANCE: {source} || {predicate} || {target}
LAYER1_EVIDENCE: {expr.justification}
CANDIDATES: {_json_block(candidates, 3500)}
RULES: {_json_block((task.get('layer_guidance') or {}).get('layer02', {}), 3500)}
EXAMPLES: {_json_block(task.get('synthetic_relation_examples') or [], 5000)}
ONTOLOGY_EVIDENCE: {grounding_text}
Choose exactly one relation or found=false. JSON only.
""".strip(),
            },
        ]
        with self._prompt_lock:
            self._prompt_audit.append({
                "expr_id": expr.expr_id,
                "relation_instance": expr.text,
                "candidate_relation_ids": list(RELATION_IDS),
                "system_chars": len(messages[0]["content"]),
                "user_chars": len(messages[1]["content"]),
            })
        return messages

    def _process_relation_contrastive(self, expr: LinguisticExpression, state: PipelineState) -> EnrichedExpression:
        triple = _parse_relation_instance(expr.text)
        if triple is None:
            raise ValueError(f"Malformed relation instance: {expr.text}")
        source, predicate, target = triple
        grounding_text = ""
        if self.rag_adapter is not None:
            grounding = self.rag_adapter.ground(GroundingRequest(
                layer_name=self.name,
                query=predicate,
                payload={"relation_instance":expr.text,"source":source,"predicate":predicate,"target":target},
                preferred_sources=["ontology"],
                top_k=4,
            ))
            grounding_text = build_grounding_context(grounding)
        messages = self._contrastive_prompt(expr=expr, state=state, grounding_text=grounding_text)
        raw = self.ollama_backend.chat(model=state.llm_model, messages=messages, temperature=0.0)
        parsed = self.ollama_backend.extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Layer 2 response is not a JSON object")
        found = bool(parsed.get("found", False))
        selected_id = normalize_relation_id(parsed.get("selected_relation_id"))
        if found and selected_id not in self.catalog:
            raise ValueError(f"Invalid EventStoryLine relation ID: {parsed.get('selected_relation_id')}")

        if found:
            item = self.catalog[selected_id]
            canonical = selected_id
            hints = _dedup([
                f"controlled_relation:{canonical}",
                "promote_to_ontology:true",
                item.get("uri"),
                item.get("label"),
                f"source_label:{source}",
                f"target_label:{target}",
                f"lexical_predicate:{predicate}",
                "source_type:EVENT",
                "target_type:EVENT",
                f"domain:{', '.join(item.get('domain_uris') or [])}",
                f"range:{', '.join(item.get('range_uris') or [])}",
                f"contrastive_decision:{parsed.get('decision', '')}",
            ])
            definition = str(item.get("comment") or parsed.get("decision") or "").strip()
        else:
            canonical = None
            hints = _dedup([
                "promote_to_ontology:false",
                f"source_label:{source}",
                f"target_label:{target}",
                f"lexical_predicate:{predicate}",
                "source_type:EVENT",
                "target_type:EVENT",
                f"contrastive_decision:{parsed.get('decision', '')}",
            ])
            definition = str(parsed.get("decision") or "No supported relation selected.")

        self._record_decision({
            "expr_id":expr.expr_id,
            "relation_instance":expr.text,
            "source":source,
            "predicate":predicate,
            "target":target,
            "found":found,
            "selected_relation_id":selected_id,
            "canonical_relation":canonical,
            "decision":parsed.get("decision"),
            "ontology_hints":hints,
        })
        aliases = _dedup([expr.text, predicate, canonical])
        return EnrichedExpression(
            base_expression=expr,
            aliases=aliases,
            synonyms=[],
            lexical_variants=[],
            alias_sources={value:["source" if value in {expr.text,predicate} else "ontology"] for value in aliases},
            synonym_sources={},
            lexical_variant_sources={},
            definition=definition,
            ontology_hints=hints,
            enrichment_evidence=[EnrichmentEvidence(
                source="llm",
                content=json.dumps(parsed, ensure_ascii=False),
                reference=state.llm_model,
            )],
        )

    def _run(self, state: PipelineState) -> PipelineState:
        self._prompt_audit = []
        state = super()._run(state)
        self.compact_prompt_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.compact_prompt_log_path, sorted(self._prompt_audit, key=lambda x: x.get("expr_id", "")))
        return state


class EventStoryLineCanonicalizingLayer(CandidateTypingResolutionLayer):
    """Native Layer 3 followed by deterministic normalization of its own hints."""

    def __init__(self, *args: Any, relation_catalog_path: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        rows = read_json(relation_catalog_path)["relations"]
        self.catalog = {str(row["relation_id"]).upper(): row for row in rows}

    @staticmethod
    def _controlled_id(hints: Iterable[str]) -> str | None:
        for hint in hints or []:
            text = str(hint)
            if text.lower().startswith("controlled_relation:"):
                return normalize_relation_id(text.split(":", 1)[1])
        for hint in hints or []:
            relation_id = normalize_relation_id(hint)
            if relation_id:
                return relation_id
        return None

    def _run(self, state: PipelineState) -> PipelineState:
        state = super()._run(state)
        diagnostics: list[dict[str, Any]] = []
        for candidate in state.relation_candidates or []:
            original = candidate.canonical_label
            relation_id = self._controlled_id(candidate.ontology_hints)
            if relation_id in self.catalog:
                metadata = self.catalog[relation_id]
                candidate.aliases = _dedup([original, *list(candidate.aliases or [])])
                candidate.canonical_label = relation_id
                candidate.normalized_label = self._normalize_label(relation_id)
                candidate.ontology_hints = _dedup([
                    f"controlled_relation:{relation_id}",
                    "promote_to_ontology:true",
                    metadata.get("uri"),
                    metadata.get("label"),
                    f"domain:{', '.join(metadata.get('domain_uris') or [])}",
                    f"range:{', '.join(metadata.get('range_uris') or [])}",
                    *list(candidate.ontology_hints or []),
                ])
            diagnostics.append({
                "candidate_id":candidate.candidate_id,
                "original_label":original,
                "canonical_label":candidate.canonical_label,
                "relation_id":relation_id,
                "has_mentions":bool(candidate.mentions),
                "ontology_hints":list(candidate.ontology_hints or []),
            })
        if state.artifact_dir:
            write_json(Path(state.artifact_dir)/self.name/"relation_canonicalization.json", diagnostics)
        return state


class EventStoryLineEndpointLayer(v4.StructuredEndpointValidatedRelationLayer):
    """Exact event-key endpoint resolution; native LLM fallback remains available."""

    def _valid_types(self, relation_id: str | None, source: Any, target: Any) -> tuple[bool, str]:
        source_type = getattr(source, "candidate_type", None)
        target_type = getattr(target, "candidate_type", None)
        valid = source_type == "event" and target_type == "event"
        return valid, f"source_type={source_type}; target_type={target_type}; required=event->event"


# ---------------------------------------------------------------------------
# Pipeline construction and execution
# ---------------------------------------------------------------------------


def _layer_cfg(profile: dict[str, Any], layer_name: str) -> dict[str, Any]:
    return dict((profile.get("layers") or {}).get(layer_name) or {})


def _make_backend(
    *, logger: SharedCallLogger, layer_tag: str, model_host: str, api_key: str,
    cfg: dict[str, Any], fallback_max_tokens: int, fallback_timeout: int,
    reasoning_effort: str,
) -> TaggedLoggedBackend:
    core = OpenAICompatibleBackend(
        backend_name="openrouter",
        host=model_host,
        api_key=api_key,
        timeout=int(cfg.get("request_timeout_seconds", fallback_timeout)),
        max_tokens=int(cfg.get("max_output_tokens", fallback_max_tokens)),
        reasoning_effort=reasoning_effort,
        exclude_reasoning=True,
    )
    return TaggedLoggedBackend(
        core,
        logger,
        layer_tag=layer_tag,
        response_hard_cap_chars=cfg.get("response_hard_cap_chars"),
    )


def choose_chunk_size(text: str, max_safe_chars: int = 26000) -> int:
    return v4.choose_chunk_size(text, max_safe_chars)


def build_document(record: dict[str, Any], source_path: str | Path) -> Document:
    return v4.build_document(record, source_path)


def build_pipeline(
    *, backends: dict[str, TaggedLoggedBackend], rag_adapter: Any,
    profile_config: dict[str, Any], relation_catalog_path: str | Path,
    chunk_size: int, run_dir: str | Path, workers: int = 16,
    verbose: bool = True,
) -> Pipeline:
    pipeline = v4.build_pipeline(
        backends=backends,
        rag_adapter=rag_adapter,
        profile_config=profile_config,
        relation_catalog_path=relation_catalog_path,
        chunk_size=chunk_size,
        run_dir=run_dir,
        workers=workers,
        verbose=verbose,
    )
    run_dir = Path(run_dir)
    retry_default = int((profile_config.get("orchestration") or {}).get("retry_failed_calls", 1))
    sleep_default = float((profile_config.get("orchestration") or {}).get("retry_sleep_seconds", 1.0))
    l2_cfg = _layer_cfg(profile_config, "layer02_candidate_enrichment")
    l4_cfg = _layer_cfg(profile_config, "layer04_candidate_relation_extraction")

    pipeline.layers[1] = EventStoryLineExtractionLayer(
        backends["layer01"],
        decision_log_path=run_dir/"run_logs/layer01_event_relation_instances.json",
        inventory_log_path=run_dir/"run_logs/layer01_event_inventory.json",
        relation_generation_log_path=run_dir/"run_logs/layer01_relation_generation.json",
        max_chunks=1,
        temperature=0.0,
        save_intermediate=True,
        verbose=verbose,
        rag_backend=rag_adapter,
        max_concurrency=1,
        retry_failed_calls=0,
        retry_sleep_seconds=sleep_default,
        rag_enabled=False,
    )
    pipeline.layers[2] = EventStoryLineCandidateEnrichmentLayer(
        backends["layer02"],
        wikipedia_source=OfflineWikipediaSource(),
        wikidata_source=OfflineWikidataSource(),
        web_search_source=OfflineWebSearchSource(),
        relation_catalog_path=relation_catalog_path,
        decision_log_path=run_dir/"run_logs/layer02_relation_decisions.json",
        compact_prompt_log_path=run_dir/"run_logs/layer02_compact_prompt_audit.json",
        max_expressions=None,
        use_web_search=False,
        save_intermediate=True,
        verbose=verbose,
        rag_adapter=rag_adapter,
        max_concurrency=int(l2_cfg.get("max_concurrency", workers)),
        retry_failed_calls=int(l2_cfg.get("retry_failed_calls", retry_default)),
        retry_sleep_seconds=sleep_default,
    )
    pipeline.layers[3] = EventStoryLineCanonicalizingLayer(
        backends["other"],
        relation_catalog_path=relation_catalog_path,
        max_expressions=None,
        temperature=0.0,
        save_intermediate=True,
        verbose=verbose,
        rag_adapter=rag_adapter,
        max_concurrency=1,
        retry_failed_calls=retry_default,
        retry_sleep_seconds=sleep_default,
    )
    pipeline.layers[4] = EventStoryLineEndpointLayer(
        backends["layer04"],
        endpoint_log_path=run_dir/"run_logs/layer04_endpoint_assignment.json",
        rejection_log_path=run_dir/"run_logs/layer04_constraint_rejections.json",
        type_constraints={},
        max_relation_mentions=None,
        temperature=0.0,
        save_intermediate=True,
        verbose=verbose,
        rag_adapter=rag_adapter,
        max_concurrency=int(l4_cfg.get("max_concurrency", min(workers, 8))),
        retry_failed_calls=int(l4_cfg.get("retry_failed_calls", 0)),
        retry_sleep_seconds=sleep_default,
        max_attempts_per_relation=int(l4_cfg.get("max_attempts_per_relation", 1)),
        retry_wait_seconds=float(l4_cfg.get("retry_wait_seconds", 0.5)),
        failure_log_path=run_dir/"run_logs/layer04_relation_errors.jsonl",
    )
    return pipeline


def run_native_pipeline(
    *, project_root: str | Path, input_jsonl: str | Path, ontology_path: str | Path,
    profile_path: str | Path, guidance_path: str | Path, task_guidance_path: str | Path,
    relation_catalog_path: str | Path, relation_aliases_path: str | Path,
    run_dir: str | Path, model_name: str, api_key: str,
    host: str = "https://openrouter.ai/api/v1", workers: int = 16,
    max_tokens: int = 8192, request_timeout: int = 180,
    reasoning_effort: str = "minimal", verbose: bool = True,
    clean_run_dir: bool = True,
) -> PipelineState:
    project_root = Path(project_root).resolve()
    input_jsonl = Path(input_jsonl).resolve()
    ontology_path = Path(ontology_path).resolve()
    profile_path = Path(profile_path).resolve()
    guidance_path = Path(guidance_path).resolve()
    task_guidance_path = Path(task_guidance_path).resolve()
    relation_catalog_path = Path(relation_catalog_path).resolve()
    relation_aliases_path = Path(relation_aliases_path).resolve()
    run_dir = Path(run_dir).resolve()
    if clean_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir/"run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_jsonl)
    if len(records) != 1:
        raise ValueError(f"This one-document notebook expects exactly one record, found {len(records)}")
    record = records[0]
    if "entities" in record or "relations" in record:
        raise ValueError("Pipeline input must not contain gold entities or relations.")
    if not record.get("sentences") or not record.get("tokens"):
        raise ValueError("EventStoryLine input requires source sentences and tokens for mention identity.")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    profile = load_document_profile(profile_path=profile_path)
    profile_dict = profile.to_state_dict()
    task_guidance = read_json(task_guidance_path)
    profile_dict["_input_task_guidance"] = task_guidance
    profile_dict["_input_sentences"] = record.get("sentences") or []
    profile_dict["_input_tokens"] = record.get("tokens") or []
    guidance = load_user_guidance(str(guidance_path)) or UserGuidance()
    write_json(run_dir/"input_task_guidance.json", task_guidance)
    write_json(run_dir/"effective_user_guidance.json", asdict(guidance))

    seed_ontology = SeedOntologyLoader().load(str(ontology_path))
    ontology_class_count = len(seed_ontology.classes_by_uri)
    ontology_property_count = len(seed_ontology.properties_by_uri)
    if ontology_class_count == 0 and ontology_property_count == 0:
        raise RuntimeError(
            f"The seed ontology loaded no classes or properties: {ontology_path}"
        )

    chunk_size = choose_chunk_size(record["text"], int(profile.get("chunking.max_safe_chunk_chars", 26000)))
    logger = SharedCallLogger(logs_dir)
    backends = {
        "layer01": _make_backend(
            logger=logger, layer_tag="layer01_event_instances", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer01_linguistic_expression_extraction"),
            fallback_max_tokens=max_tokens, fallback_timeout=request_timeout,
            reasoning_effort=reasoning_effort,
        ),
        "layer02": _make_backend(
            logger=logger, layer_tag="layer02_event_relation_linking", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer02_candidate_enrichment"),
            fallback_max_tokens=384, fallback_timeout=60,
            reasoning_effort=reasoning_effort,
        ),
        "layer04": _make_backend(
            logger=logger, layer_tag="layer04_fallback_only", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer04_candidate_relation_extraction"),
            fallback_max_tokens=384, fallback_timeout=60,
            reasoning_effort=reasoning_effort,
        ),
        "other": _make_backend(
            logger=logger, layer_tag="other", model_host=host, api_key=api_key,
            cfg={}, fallback_max_tokens=768, fallback_timeout=90,
            reasoning_effort=reasoning_effort,
        ),
    }
    rag_adapter = v2.OntologyOnlyRAGAdapter(
        seed_ontology,
        log_path=logs_dir/"ontology_retrieval.jsonl",
        top_k=int(profile.get("rag.top_k", 4)),
        query_expansions=profile.get("rag.query_expansions", {}) or {},
    )
    pipeline = build_pipeline(
        backends=backends,
        rag_adapter=rag_adapter,
        profile_config=profile_dict,
        relation_catalog_path=relation_catalog_path,
        chunk_size=chunk_size,
        run_dir=run_dir,
        workers=workers,
        verbose=verbose,
    )
    state = PipelineState(
        document=build_document(record, input_jsonl),
        llm_model=model_name,
        user_guidance=guidance,
        seed_ontology=seed_ontology,
        artifact_dir=str(run_dir),
        profile_name=profile.name,
        profile_config=profile_dict,
    )
    runner = Runner(
        pipeline=pipeline,
        runs_root=str(run_dir.parent),
        verbose=verbose,
        max_workers=workers,
        enable_checkpoints=True,
        save_chunk_checkpoints=False,
    )
    manifest = {
        "dataset":"EventStoryLine",
        "document_id":record["document_id"],
        "title":record.get("title"),
        "model_name":model_name,
        "profile_name":profile.name,
        "profile_path":str(profile_path),
        "guidance_path":str(guidance_path),
        "task_guidance_path":str(task_guidance_path),
        "ontology_path":str(ontology_path),
        "ontology_classes":ontology_class_count,
        "ontology_properties":ontology_property_count,
        "task_relation_count":len(RELATION_IDS),
        "seed_ontology_role":"external temporal grounding resource",
        "task_relation_schema_role":"controlled benchmark mapping and ontology-evolution target",
        "input_has_gold":False,
        "source_sentence_count":len(record.get("sentences") or []),
        "source_token_count":sum(len(x) for x in record.get("tokens") or []),
        "chunk_size":chunk_size,
        "whole_document_single_chunk_expected":len(record["text"]) <= chunk_size,
        "workers":workers,
        "ignored_gold_relation_keys":["null"],
        "indexed_event_keys_from_source_tokens":True,
        "layer01_two_phase_closed_inventory":True,
        "layer01_bounded_span_repair":True,
        "gold_projection_after_execution_only":True,
        "anti_cheating":profile.get("anti_cheating", {}),
        "started_at":time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir/"run_manifest.json", manifest)

    console_log = logs_dir/"console.log"
    errors_path = logs_dir/"pipeline_errors.jsonl"
    started = time.time()
    with console_log.open("w", encoding="utf-8") as handle:
        tee_out = Tee(sys.stdout, handle)
        tee_err = Tee(sys.stderr, handle)
        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                final_state = runner.run(state, from_layer=0, to_layer=12, run_dir=run_dir)
        except Exception as exc:
            append_jsonl(errors_path, {
                "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_type":type(exc).__name__,
                "error":str(exc),
                "traceback":traceback.format_exc(),
            })
            raise

    manifest.update({
        "completed_at":time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds":round(time.time()-started, 3),
        "final_counts":state_counts(final_state),
    })
    write_json(run_dir/"run_manifest.json", manifest)
    return final_state


# ---------------------------------------------------------------------------
# Strict EventStoryLine evaluation
# ---------------------------------------------------------------------------


def gold_event_index(gold: dict[str, Any]) -> dict[str, Any]:
    by_span: dict[tuple[int, int, int], list[str]] = {}
    by_sentence_trigger: dict[tuple[int, str], list[str]] = {}
    by_trigger: dict[str, list[str]] = {}
    keys_by_id: dict[str, list[str]] = {}
    for event_id, entity in (gold.get("entities") or {}).items():
        keys: list[str] = []
        for mention in entity.get("mentions") or []:
            sent_id = int(mention["sent_id"])
            start, end = [int(x) for x in mention["offset"]]
            trigger = str(mention.get("trigger_word") or "").strip()
            key = f"S{sent_id}[{start}:{end}]::{trigger}"
            keys.append(key)
            by_span.setdefault((sent_id, start, end), []).append(event_id)
            by_sentence_trigger.setdefault((sent_id, _norm(trigger)), []).append(event_id)
            by_trigger.setdefault(_norm(trigger), []).append(event_id)
        keys_by_id[event_id] = keys
    return {"by_span":by_span,"by_sentence_trigger":by_sentence_trigger,"by_trigger":by_trigger,"keys_by_id":keys_by_id}


def project_event_label(label: Any, gold: dict[str, Any]) -> dict[str, Any]:
    index = gold_event_index(gold)
    parsed = parse_event_key(label)
    if parsed is not None:
        ids = _dedup(index["by_span"].get((parsed["sent_id"], parsed["start"], parsed["end"]), []))
        if len(ids) == 1:
            return {"event_id":ids[0],"method":"exact_sentence_token_span","label":str(label)}
        ids = _dedup(index["by_sentence_trigger"].get((parsed["sent_id"], _norm(parsed["trigger"])), []))
        if len(ids) == 1:
            return {"event_id":ids[0],"method":"sentence_plus_unique_trigger","label":str(label)}
    trigger_norm = _norm(parsed["trigger"] if parsed else label)
    ids = _dedup(index["by_trigger"].get(trigger_norm, []))
    if len(ids) == 1:
        return {"event_id":ids[0],"method":"globally_unique_trigger","label":str(label)}
    return {"event_id":None,"method":"unmapped_or_ambiguous","label":str(label),"candidate_event_ids":ids}


def gold_relation_set(gold: dict[str, Any]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for relation_id, pairs in (gold.get("relations") or {}).items():
        canonical = normalize_relation_id(relation_id)
        if canonical not in RELATION_IDS:
            continue
        for pair in pairs or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                result.add((str(pair[0]), canonical, str(pair[1])))
    return result


def _triples_from_state(state: PipelineState) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if state.candidate_triples:
        for triple in state.candidate_triples:
            rows.append((triple.subject_label, triple.predicate_label, triple.object_label))
    elif state.candidate_relation_assertions:
        for assertion in state.candidate_relation_assertions:
            rows.append((assertion.source_candidate_label, assertion.relation_label, assertion.target_candidate_label))
    return rows


def native_predictions(state: PipelineState, gold: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source_label, predicate_label, target_label in _triples_from_state(state):
        relation_id = normalize_relation_id(predicate_label)
        source_projection = project_event_label(source_label, gold)
        target_projection = project_event_label(target_label, gold)
        row = {
            "source_label":source_label,
            "predicate_label":predicate_label,
            "target_label":target_label,
            "relation_id":relation_id,
            "source_event_id":source_projection.get("event_id"),
            "target_event_id":target_projection.get("event_id"),
            "source_projection_method":source_projection.get("method"),
            "target_projection_method":target_projection.get("method"),
        }
        audit.append(row)
        if relation_id not in RELATION_IDS or not row["source_event_id"] or not row["target_event_id"]:
            continue
        key = (row["source_event_id"], relation_id, row["target_event_id"])
        if key in seen:
            continue
        seen.add(key)
        predictions.append(row)
    return predictions, audit


def metric_counts(predicted: set[Any], gold: set[Any]) -> dict[str, Any]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {
        "predicted":len(predicted),"gold":len(gold),"true_positive":tp,
        "false_positive":fp,"false_negative":fn,
        "precision":precision,"recall":recall,"f1":f1,
    }


def event_inventory_from_state(state: PipelineState, gold: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    labels: list[str] = []
    if state.event_candidates:
        labels.extend(candidate.canonical_label for candidate in state.event_candidates)
    elif state.enriched_expressions:
        labels.extend(item.base_expression.text for item in state.enriched_expressions if item.base_expression.label == "event_mention")
    elif state.linguistic_expressions:
        labels.extend(item.text for item in state.linguistic_expressions if item.label == "event_mention")
    projected: set[str] = set()
    audit: list[dict[str, Any]] = []
    for label in _dedup(labels):
        result = project_event_label(label, gold)
        audit.append(result)
        if result.get("event_id"):
            projected.add(result["event_id"])
    return projected, audit


def relation_endpoint_inventory(relations: set[tuple[str, str, str]]) -> set[str]:
    return {event_id for source, _, target in relations for event_id in (source, target)}


def per_relation_metrics(predicted: set[tuple[str, str, str]], gold: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation_id in RELATION_IDS:
        pred = {x for x in predicted if x[1] == relation_id}
        truth = {x for x in gold if x[1] == relation_id}
        row = {"relation_id":relation_id, **metric_counts(pred, truth)}
        rows.append(row)
    return rows


def relation_trace(run_dir: Path, gold: dict[str, Any], final_predictions: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    layer1_rows = read_json(run_dir/"run_logs/layer01_event_relation_instances.json")
    layer2_rows = read_json(run_dir/"run_logs/layer02_relation_decisions.json")
    endpoint_ids: set[str] = set()
    relation_instances: list[tuple[str, str, str]] = []
    for row in layer1_rows:
        if row.get("status") != "accepted":
            continue
        if row.get("label") == "event_mention":
            projection = project_event_label(row.get("text"), gold)
            if projection.get("event_id"):
                endpoint_ids.add(projection["event_id"])
        elif row.get("label") == "relation_instance":
            triple = _parse_relation_instance(row.get("text"))
            if triple:
                s = project_event_label(triple[0], gold).get("event_id")
                t = project_event_label(triple[2], gold).get("event_id")
                if s and t:
                    relation_instances.append((s, triple[1], t))
    linked: list[tuple[str, str, str]] = []
    for row in layer2_rows:
        if not row.get("found"):
            continue
        s = project_event_label(row.get("source"), gold).get("event_id")
        t = project_event_label(row.get("target"), gold).get("event_id")
        relation_id = normalize_relation_id(row.get("selected_relation_id"))
        if s and t and relation_id:
            linked.append((s, relation_id, t))

    rows: list[dict[str, Any]] = []
    for source, relation_id, target in sorted(gold_relation_set(gold)):
        if source not in endpoint_ids:
            failure = "layer01_source_event_missing"
        elif target not in endpoint_ids:
            failure = "layer01_target_event_missing"
        elif not any(s == source and t == target for s, _, t in relation_instances):
            failure = "layer01_relation_instance_missing"
        elif (source, relation_id, target) not in linked:
            failure = "layer02_wrong_or_missing_controlled_relation"
        elif (source, relation_id, target) not in final_predictions:
            failure = "layer04_endpoint_assignment_or_materialization"
        else:
            failure = "survived_to_layer05"
        rows.append({
            "source_event_id":source,
            "relation_id":relation_id,
            "target_event_id":target,
            "source_keys":gold_event_index(gold)["keys_by_id"].get(source, []),
            "target_keys":gold_event_index(gold)["keys_by_id"].get(target, []),
            "first_failure":failure,
        })
    return rows


def analyze_run(
    *, run_dir: str | Path, gold_jsonl: str | Path,
    catalog_path: str | Path, aliases_path: str | Path,
) -> dict[str, Any]:
    del catalog_path, aliases_path  # retained for a consistent notebook API
    run_dir = Path(run_dir)
    gold_rows = read_jsonl(gold_jsonl)
    if len(gold_rows) != 1:
        raise ValueError(f"Expected exactly one gold record, found {len(gold_rows)}")
    gold = gold_rows[0]
    states = {index: state for index, _, state in load_layer_states(run_dir)}
    if not states:
        raise FileNotFoundError(f"No saved layer states under {run_dir}")
    final_state = states[max(states)]

    predictions, prediction_audit = native_predictions(final_state, gold)
    predicted_set = {(row["source_event_id"], row["relation_id"], row["target_event_id"]) for row in predictions}
    gold_set = gold_relation_set(gold)
    relation_metrics = metric_counts(predicted_set, gold_set)

    predicted_events, projection_audit = event_inventory_from_state(final_state, gold)
    gold_events = set((gold.get("entities") or {}).keys())
    entity_metrics = metric_counts(predicted_events, gold_events)
    endpoint_metrics = metric_counts(relation_endpoint_inventory(predicted_set), relation_endpoint_inventory(gold_set))

    cumulative: list[dict[str, Any]] = []
    layer_summary: list[dict[str, Any]] = []
    for index in sorted(states):
        state = states[index]
        layer_predictions, _ = native_predictions(state, gold)
        layer_set = {(row["source_event_id"], row["relation_id"], row["target_event_id"]) for row in layer_predictions}
        layer_events, _ = event_inventory_from_state(state, gold)
        cumulative.append({
            "layer":index,
            "layer_name":layer_name(index),
            **{f"relation_{k}":v for k,v in metric_counts(layer_set, gold_set).items()},
            **{f"event_{k}":v for k,v in metric_counts(layer_events, gold_events).items()},
        })
        layer_summary.append({"layer":index,"layer_name":layer_name(index),**state_counts(state)})

    trace = relation_trace(run_dir, gold, predicted_set)
    failure_counts: dict[str, int] = {}
    for row in trace:
        failure_counts[row["first_failure"]] = failure_counts.get(row["first_failure"], 0) + 1
    relation_rows = per_relation_metrics(predicted_set, gold_set)

    analysis_dir = run_dir/"analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json(analysis_dir/"strict_relation_predictions.json", predictions)
    write_json(analysis_dir/"strict_relation_evaluation.json", relation_metrics)
    write_json(analysis_dir/"event_entity_evaluation.json", entity_metrics)
    write_json(analysis_dir/"relation_endpoint_evaluation.json", endpoint_metrics)
    write_json(analysis_dir/"prediction_projection_audit.json", prediction_audit)
    write_csv_rows(analysis_dir/"event_projection_audit.csv", projection_audit)
    write_csv_rows(analysis_dir/"cumulative_evaluation.csv", cumulative)
    write_csv_rows(analysis_dir/"gold_relation_trace.csv", trace)
    write_csv_rows(analysis_dir/"per_relation_metrics.csv", relation_rows)
    write_csv_rows(analysis_dir/"layer_summary.csv", layer_summary)

    summary = {
        "document_id":gold.get("document_id"),
        "title":gold.get("title"),
        "ignored_gold_relation_keys":["null"],
        "strict_relation_evaluation":relation_metrics,
        "event_entity_evaluation":entity_metrics,
        "relation_endpoint_evaluation":endpoint_metrics,
        "per_relation_metrics":relation_rows,
        "failure_counts":failure_counts,
        "cumulative_evaluation":cumulative,
        "layer_summary":layer_summary,
        "strict_relation_predictions":predictions,
        "gold_relation_trace":trace,
    }
    write_json(analysis_dir/"analysis_summary.json", summary)
    return summary

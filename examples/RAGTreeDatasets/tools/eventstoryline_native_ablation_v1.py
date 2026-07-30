from __future__ import annotations

"""Native NeoOLAF one-document EventStoryLine experiment support, v1.

This module is experiment-only and changes no file under ``src/neoolaf``.
It keeps the complete Layer 0--12 pipeline while adding dataset-specific,
profile-driven orchestration for mention-level event extraction and the two
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
SharedCallLogger = v4.SharedCallLogger
TaggedLoggedBackend = v4.TaggedLoggedBackend

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
    """One whole-document Layer 1 call for indexed events and event pairs."""

    def __init__(self, *args: Any, decision_log_path: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decision_log_path = Path(decision_log_path)

    def _prompt(self, state: PipelineState, chunk_text: str) -> list[dict[str, str]]:
        profile = state.profile_config or {}
        task = profile.get("_input_task_guidance", {}) or {}
        sentences = profile.get("_input_sentences", []) or []
        tokens = profile.get("_input_tokens", []) or []
        table = indexed_token_table(sentences, tokens)
        layer_guidance = (task.get("layer_guidance") or {}).get("layer01", {})
        relation_specs = task.get("relation_specs") or []
        examples = task.get("synthetic_relation_examples") or []
        negatives = task.get("negative_examples") or []

        system = """
You are NeoOLAF Layer 1 for EventStoryLine mention-level event extraction.

Extract only:
1. exact event mentions;
2. directed relation instances between exact event mentions.

Every event mention MUST have label event_mention and exact text:
S{sentence_id}[{token_start}:{token_end}]::{trigger tokens}

Token indices are zero-based and the end offset is exclusive. Copy the trigger
from the indexed token table. Repeated trigger words are different events when
their sentence or span differs.

Every relation instance MUST have label relation_instance and exact text:
SOURCE_EVENT_KEY || lexical relation cue || TARGET_EVENT_KEY

Rules:
- every relation endpoint must also be returned as event_mention;
- extract only event/action/process/state triggers, never people, places, dates,
  organizations, objects, or generic topics;
- identify PRECONDITION-like and FALLING_ACTION-like pairs, but do not output
  either canonical relation ID in Layer 1;
- preserve source-to-target direction and do not swap endpoints;
- sentence order alone is not sufficient evidence;
- do not output mere co-occurrence, null relations, or gold annotations;
- perform a final self-check for valid sentence/token spans and missing endpoint
  expressions.

Return JSON only:
{"expressions":[
 {"text":"S2[4:5]::order","label":"event_mention","justification":"Explicit event trigger."},
 {"text":"S2[4:5]::order || required || S2[9:11]::enter treatment","label":"relation_instance","justification":"source_type=EVENT; target_type=EVENT; semantic evidence..."}
],"coverage_check":{"event_mentions":0,"relation_instances":0}}
""".strip()

        user = f"""
Layer-specific guidance:
{_json_block(layer_guidance, 5000)}

Relation definitions and direction rules:
{_json_block(relation_specs, 6500)}

Synthetic examples:
{_json_block(examples, 6000)}

Negative examples:
{_json_block(negatives, 3500)}

Raw document:
\"\"\"
{chunk_text}
\"\"\"

Indexed sentence/token table (authoritative for event keys):
\"\"\"
{table}
\"\"\"

Extract exact event mentions and supported directed event relation instances.
Return JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _run(self, state: PipelineState) -> PipelineState:
        chunks = list(state.document.chunks)
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        expressions: list[LinguisticExpression] = []
        decisions: list[dict[str, Any]] = []
        expr_counter = 0

        for chunk in chunks:
            messages = self._prompt(state, chunk.text)
            raw = self.ollama_backend.chat(
                model=state.llm_model,
                messages=messages,
                temperature=self.temperature,
            )
            parsed = self._safe_extract_json(
                raw_response=raw,
                state=state,
                chunk_id=chunk.chunk_id,
                messages=messages,
            )
            if not isinstance(parsed, dict):
                continue

            accepted_rows: list[dict[str, str]] = []
            endpoint_keys: set[str] = set()
            relation_rows: list[dict[str, str]] = []

            for item in parsed.get("expressions", []) or []:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                label = str(item.get("label") or "").strip().lower()
                justification = str(item.get("justification") or "").strip()
                if not text:
                    continue

                if label == "event_mention":
                    canonical = canonical_event_key(text, tokens)
                    if canonical is None:
                        decisions.append({"status":"rejected","reason":"invalid_event_key","text":text,"label":label})
                        continue
                    endpoint_keys.add(canonical)
                    accepted_rows.append({"text":canonical,"label":"event_mention","justification":justification})
                elif label == "relation_instance":
                    triple = _parse_relation_instance(text)
                    if triple is None:
                        decisions.append({"status":"rejected","reason":"invalid_relation_instance_format","text":text,"label":label})
                        continue
                    source, predicate, target = triple
                    source_key = canonical_event_key(source, tokens)
                    target_key = canonical_event_key(target, tokens)
                    if source_key is None or target_key is None:
                        decisions.append({"status":"rejected","reason":"invalid_relation_endpoint_key","text":text,"label":label})
                        continue
                    if source_key == target_key:
                        decisions.append({"status":"rejected","reason":"self_relation","text":text,"label":label})
                        continue
                    normalized_text = f"{source_key} || {predicate.strip()} || {target_key}"
                    relation_rows.append({"text":normalized_text,"label":"relation_instance","justification":justification})
                    endpoint_keys.update([source_key, target_key])
                else:
                    decisions.append({"status":"rejected","reason":"unsupported_layer01_label","text":text,"label":label})

            # Endpoint completion remains inside Layer 1 and derives only from the
            # same Layer 1 relation-instance output. It does not invent relations.
            already = {row["text"] for row in accepted_rows}
            for key in sorted(endpoint_keys):
                if key not in already:
                    accepted_rows.append({
                        "text": key,
                        "label": "event_mention",
                        "justification": "Endpoint deterministically recovered from a Layer 1 relation instance.",
                    })
            accepted_rows.extend(relation_rows)

            for row in accepted_rows:
                text = row["text"]
                label = row["label"]
                justification = row["justification"]
                expressions.append(LinguisticExpression(
                    expr_id=f"expr_{expr_counter:05d}",
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
                decisions.append({
                    "status":"accepted",
                    "expr_id":f"expr_{expr_counter:05d}",
                    "text":text,
                    "label":label,
                    "relation_instance":_parse_relation_instance(text),
                    "justification":justification,
                })
                expr_counter += 1

        dedup: dict[tuple[str, str], LinguisticExpression] = {}
        for expr in expressions:
            dedup.setdefault((expr.text, expr.label), expr)
        state.linguistic_expressions = list(dedup.values())
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.decision_log_path, decisions)
        state.log(
            f"[{self.name}] EventStoryLine indexed extraction; "
            f"events={sum(1 for x in state.linguistic_expressions if x.label == 'event_mention')}; "
            f"relation_instances={sum(1 for x in state.linguistic_expressions if x.label == 'relation_instance')}"
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
    if len(seed_ontology.properties_by_uri) != 2:
        raise RuntimeError(f"Expected exactly 2 ontology properties, loaded {len(seed_ontology.properties_by_uri)}")

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
        "ontology_classes":len(seed_ontology.classes_by_uri),
        "ontology_properties":len(seed_ontology.properties_by_uri),
        "input_has_gold":False,
        "source_sentence_count":len(record.get("sentences") or []),
        "source_token_count":sum(len(x) for x in record.get("tokens") or []),
        "chunk_size":chunk_size,
        "whole_document_single_chunk_expected":len(record["text"]) <= chunk_size,
        "workers":workers,
        "ignored_gold_relation_keys":["null"],
        "indexed_event_keys_from_source_tokens":True,
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
            "layer_name":LAYER_NAMES.get(index, f"layer_{index:02d}"),
            **{f"relation_{k}":v for k,v in metric_counts(layer_set, gold_set).items()},
            **{f"event_{k}":v for k,v in metric_counts(layer_events, gold_events).items()},
        })
        layer_summary.append({"layer":index,"layer_name":LAYER_NAMES.get(index, f"layer_{index:02d}"),**state_counts(state)})

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

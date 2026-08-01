from __future__ import annotations

"""Native NeoOLAF one-document EventStoryLine experiment support, v1.5.

This experiment-only module changes no file under ``src/neoolaf``. It preserves
NeoOLAF Layers 0--12 and replaces only dataset-specific orchestration under
``examples/RAGTreeDatasets``.

v1.5 focuses on relation recall, direction and evaluation integrity:

* Layer 1 keeps v1.3 sentence-level event extraction and adds a third global
  mention-coverage review.
* Relation candidate coverage is deterministic: unordered event pairs are
  enumerated exhaustively for normal documents and by a broad hybrid policy for
  very large event inventories. No gold relation is consulted.
* Layer 2 classifies pairs in bounded batches with a five-way direction-aware
  schema: A_PRECONDITION_B, A_FALLING_ACTION_B, B_PRECONDITION_A,
  B_FALLING_ACTION_A, or NONE.
* Pair prompts contain exact source/target context, schema-exact positive,
  negative, and reversed-direction examples, evidence requirements, and
  optional OWL-Time retrieval as secondary support only.
* NONE, malformed, unsupported, and conflicting outputs are removed before
  native Layer 3. Missing/invalid decisions receive one recovery pass.
* For large hybrid inventories, accepted initial links may open two-hop closure
  pairs, which are classified rather than automatically predicted.
* Evaluation uses exact mention spans only. It reports projected benchmark
  metrics, strict native span metrics, candidate-pool recall, direction/class
  failures, and a relation confusion matrix.

Gold event IDs, gold spans, event counts and gold relation pairs are never
available to the pipeline. Gold is loaded only after Layer 12.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
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

import eventstoryline_native_ablation_v1_3 as v13

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

from experiments.methods.run_neoolaf import (
    OfflineWebSearchSource,
    OfflineWikipediaSource,
    OfflineWikidataSource,
    OpenAICompatibleBackend,
    load_user_guidance,
)

# Stable re-exports used by the notebook.
read_json = v13.read_json
read_jsonl = v13.read_jsonl
write_json = v13.write_json
append_jsonl = v13.append_jsonl
load_layer_states = v13.load_layer_states
state_counts = v13.state_counts
safe_name = v13.safe_name
Tee = v13.Tee
LAYER_NAMES = v13.LAYER_NAMES
SharedCallLogger = v13.SharedCallLogger
TaggedLoggedBackend = v13.TaggedLoggedBackend
seed_ontology_summary = v13.seed_ontology_summary
indexed_token_table = v13.indexed_token_table
parse_event_key = v13.parse_event_key
canonical_event_key = v13.canonical_event_key
gold_event_index = v13.gold_event_index
gold_relation_set = v13.gold_relation_set
metric_counts = v13.metric_counts
per_relation_metrics = v13.per_relation_metrics
normalize_relation_id = v13.normalize_relation_id
RELATION_IDS = v13.RELATION_IDS
_norm = v13._norm
_dedup = v13._dedup
_parse_relation_instance = v13._parse_relation_instance
_json_block = v13._json_block
write_csv_rows = v13.write_csv_rows


FIVE_WAY_DECISIONS = (
    "A_PRECONDITION_B",
    "A_FALLING_ACTION_B",
    "B_PRECONDITION_A",
    "B_FALLING_ACTION_A",
    "NONE",
)


def layer_name(index: int) -> str:
    return v13.layer_name(index)


def _event_sort_key(label: str) -> tuple[int, int, int, str]:
    parsed = parse_event_key(label) or {}
    return (
        int(parsed.get("sent_id", 10**9)),
        int(parsed.get("start", 10**9)),
        int(parsed.get("end", 10**9)),
        str(label),
    )


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b)), key=_event_sort_key))  # type: ignore[return-value]


def _extract_pair_id(justification: str, fallback: str) -> str:
    match = re.search(r"\bpair_id\s*=\s*([A-Za-z0-9_.:-]+)", justification or "")
    return match.group(1) if match else fallback


def _normalize_five_way(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "A_PRECONDITION_B": "A_PRECONDITION_B",
        "A_FALLING_ACTION_B": "A_FALLING_ACTION_B",
        "B_PRECONDITION_A": "B_PRECONDITION_A",
        "B_FALLING_ACTION_A": "B_FALLING_ACTION_A",
        "PRECONDITION": "A_PRECONDITION_B",
        "FALLING_ACTION": "A_FALLING_ACTION_B",
        "NO_RELATION": "NONE",
        "FALSE": "NONE",
        "NULL": "NONE",
        "NONE": "NONE",
    }
    return aliases.get(text)


def _decision_to_directed(
    decision: str,
    event_a: str,
    event_b: str,
) -> tuple[str, str, str] | None:
    if decision == "A_PRECONDITION_B":
        return event_a, "PRECONDITION", event_b
    if decision == "A_FALLING_ACTION_B":
        return event_a, "FALLING_ACTION", event_b
    if decision == "B_PRECONDITION_A":
        return event_b, "PRECONDITION", event_a
    if decision == "B_FALLING_ACTION_A":
        return event_b, "FALLING_ACTION", event_a
    return None


class EventStoryLineExtractionLayer(v13.EventStoryLineExtractionLayer):
    """High-recall mention extraction plus deterministic unordered pair pool."""

    _DISCOURSE_CUES = {
        "after", "before", "because", "if", "when", "while", "then", "prior",
        "following", "followed", "result", "resulted", "led", "caused", "triggered",
        "enabled", "required", "allowed", "continued", "subsequently", "therefore",
        "consequently", "upon", "during", "once", "so",
    }

    def __init__(
        self,
        *args: Any,
        pair_pool_log_path: str | Path | None = None,
        pair_strategy: str = "adaptive",
        exhaustive_event_threshold: int = 25,
        hybrid_sentence_distance: int = 2,
        include_headline_body_pairs: bool = True,
        **kwargs: Any,
    ) -> None:
        # v1.3 constructor fields remain useful for event extraction. Relation
        # worker values are ignored because v1.5 constructs the pair pool without
        # a Layer-1 relation-generation LLM call.
        super().__init__(*args, **kwargs)
        base = self.decision_log_path.parent
        self.pair_pool_log_path = Path(pair_pool_log_path or base / "layer01_pair_pool.json")
        self.pair_strategy = str(pair_strategy or "adaptive").lower()
        self.exhaustive_event_threshold = max(2, int(exhaustive_event_threshold))
        self.hybrid_sentence_distance = max(0, int(hybrid_sentence_distance))
        self.include_headline_body_pairs = bool(include_headline_body_pairs)

    def _coverage_review_prompt(
        self,
        state: PipelineState,
        chunk_text: str,
        accepted_events: list[dict[str, Any]],
        pass_index: int,
    ) -> list[dict[str, str]]:
        profile = state.profile_config or {}
        task = profile.get("_input_task_guidance", {}) or {}
        sentences = profile.get("_input_sentences", []) or []
        tokens = profile.get("_input_tokens", []) or []
        focus_by_pass = {
            1: "all missing explicit event mentions, including embedded predicates and repeated mentions",
            2: "eventive nouns and nominalizations, legal/administrative/medical processes, explicit states, participles and multi-token triggers",
            3: "a final mention-by-mention sweep for repeated occurrences, short event nouns, predicative adjectives, gerunds, and events in quotations or headlines",
        }
        focus = focus_by_pass.get(
            pass_index,
            "any remaining explicit event mention that is absent from the closed inventory",
        )
        existing = [
            {"event_key": row["event_key"], "sentence": row.get("sentence", "")}
            for row in accepted_events
        ]
        system = f"""
You are NeoOLAF Layer 1A2 coverage reviewer for EventStoryLine.

The validated event inventory is shown below. Find ONLY missing explicit
mention-level events. This pass focuses on {focus}.

EventStoryLine event endpoints may be lexical verbs, participles, infinitives,
phrasal verbs, eventive nouns, nominalizations, legal or administrative events,
occurrences, processes and explicit states. Repeated words at different indexed
spans are separate mentions and must not be merged. Use the smallest COMPLETE
semantic trigger and exact zero-based end-exclusive offsets from the token table.
Do not output entities, objects, dates, discourse connectives, auxiliaries alone,
or inferred events absent from the document. Do not create relations.

Schema-exact small example:
Tokens: 0=The 1=collision 2=led 3=to 4=an 5=investigation 6=.
Already present: S0[1:2]::collision
Correct JSON:
{{"missing_event_mentions":[{{"sentence_id":0,"token_start":5,"token_end":6,
"trigger":"investigation","event_kind":"eventive_nominal",
"justification":"Explicit downstream event noun."}}],
"coverage_check":{{"pass":{pass_index},"checked":true}}}}

Return JSON only with keys missing_event_mentions and coverage_check.
""".strip()
        user = f"""
Dataset guidance:
{_json_block((task.get('layer_guidance') or {}).get('layer01', {}), 6000)}

Already validated events (do not repeat):
{_json_block(existing, 15000)}

Raw document:
\"\"\"
{chunk_text}
\"\"\"

Authoritative indexed sentence/token table:
\"\"\"
{indexed_token_table(sentences, tokens)}
\"\"\"

Return only missing explicit event mentions. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _event_metadata(row: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_event_key(row["event_key"]) or {}
        return {
            **row,
            "sent_id": int(parsed.get("sent_id", -1)),
            "start": int(parsed.get("start", -1)),
            "end": int(parsed.get("end", -1)),
            "trigger": str(parsed.get("trigger", "")),
            "trigger_norm": _norm(parsed.get("trigger", "")),
        }

    def _hybrid_pair_reasons(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
        sentences: list[str],
        tokens: list[list[str]],
    ) -> list[str]:
        reasons: list[str] = []
        distance = abs(int(a["sent_id"]) - int(b["sent_id"]))
        if distance == 0:
            reasons.append("same_sentence")
        if distance <= self.hybrid_sentence_distance:
            reasons.append(f"sentence_distance_le_{self.hybrid_sentence_distance}")
        if a["trigger_norm"] and a["trigger_norm"] == b["trigger_norm"]:
            reasons.append("repeated_trigger")
        # In this normalization, sentence 1 is commonly the headline. Headline
        # links are broad candidates, but classification still decides NONE.
        if self.include_headline_body_pairs and (
            (a["sent_id"] == 1 and b["sent_id"] >= 2)
            or (b["sent_id"] == 1 and a["sent_id"] >= 2)
        ):
            reasons.append("headline_body")
        lo, hi = sorted((int(a["sent_id"]), int(b["sent_id"])))
        context_tokens: list[str] = []
        for sent_id in range(max(0, lo), min(len(tokens), hi + 1)):
            context_tokens.extend(_norm(x) for x in tokens[sent_id])
        if self._DISCOURSE_CUES.intersection(context_tokens):
            if distance <= max(4, self.hybrid_sentence_distance):
                reasons.append("discourse_or_causal_cue_window")
        return _dedup(reasons)

    def _construct_pair_pool(
        self,
        event_rows: list[dict[str, Any]],
        sentences: list[str],
        tokens: list[list[str]],
    ) -> tuple[str, list[dict[str, Any]]]:
        enriched = [self._event_metadata(row) for row in event_rows]
        if self.pair_strategy == "exhaustive":
            strategy = "exhaustive"
        elif self.pair_strategy == "hybrid":
            strategy = "hybrid"
        else:
            strategy = "exhaustive" if len(enriched) <= self.exhaustive_event_threshold else "hybrid"

        pool: list[dict[str, Any]] = []
        for a, b in combinations(enriched, 2):
            reasons = ["exhaustive_unordered_pair"] if strategy == "exhaustive" else self._hybrid_pair_reasons(a, b, sentences, tokens)
            if not reasons:
                continue
            event_a, event_b = sorted((a, b), key=lambda row: _event_sort_key(row["event_key"]))
            pool.append({
                "event_a_id": event_a["event_ref"],
                "event_b_id": event_b["event_ref"],
                "event_a_key": event_a["event_key"],
                "event_b_key": event_b["event_key"],
                "event_a_sentence": event_a.get("sentence", ""),
                "event_b_sentence": event_b.get("sentence", ""),
                "candidate_reasons": reasons,
                "strategy": strategy,
                "phase": "initial",
            })
        pool.sort(key=lambda row: (_event_sort_key(row["event_a_key"]), _event_sort_key(row["event_b_key"])))
        for index, row in enumerate(pool):
            row["pair_id"] = f"P{index:05d}"
        return strategy, pool

    def _run(self, state: PipelineState) -> PipelineState:
        chunks = list(state.document.chunks)
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        sentences = (state.profile_config or {}).get("_input_sentences", []) or []
        expressions: list[LinguisticExpression] = []
        combined_decisions: list[dict[str, Any]] = []
        inventory_audit: list[dict[str, Any]] = []
        pair_pool_audit: list[dict[str, Any]] = []
        call_audit: list[dict[str, Any]] = []
        expr_counter = 0

        for chunk in chunks:
            accepted_by_key: dict[str, dict[str, Any]] = {}
            story_sentence_ids: list[int] = []
            for sent_id in range(max(len(sentences), len(tokens))):
                sentence = sentences[sent_id] if sent_id < len(sentences) else ""
                sentence_tokens = [str(x) for x in (tokens[sent_id] if sent_id < len(tokens) else [])]
                skip, reason = self._sentence_is_non_story_metadata(sentence, sentence_tokens)
                if skip:
                    call_audit.append({"phase": "sentence_inventory", "sentence_id": sent_id, "status": "skipped", "reason": reason})
                else:
                    story_sentence_ids.append(sent_id)

            sentence_results: dict[int, tuple[dict[str, Any] | None, str | None]] = {}
            if story_sentence_ids:
                with ThreadPoolExecutor(max_workers=min(self.sentence_workers, len(story_sentence_ids))) as executor:
                    futures = {}
                    for sent_id in story_sentence_ids:
                        sentence = sentences[sent_id] if sent_id < len(sentences) else ""
                        sentence_tokens = [str(x) for x in tokens[sent_id]]
                        messages = self._sentence_event_prompt(state, sent_id, sentence, sentence_tokens)
                        futures[executor.submit(self._parse_chat, messages, state)] = sent_id
                    for future in as_completed(futures):
                        sent_id = futures[future]
                        try:
                            sentence_results[sent_id] = (future.result(), None)
                        except Exception as exc:
                            sentence_results[sent_id] = (None, f"{type(exc).__name__}: {exc}")

            for sent_id in sorted(story_sentence_ids):
                parsed, error = sentence_results.get(sent_id, (None, "missing_future_result"))
                if error:
                    call_audit.append({"phase": "sentence_inventory", "sentence_id": sent_id, "status": "error", "error": error})
                    continue
                items = self._event_items(parsed)
                added = self._ingest_events(
                    items=items,
                    phase=f"sentence_inventory_{sent_id}",
                    tokens=tokens,
                    sentences=sentences,
                    accepted_by_key=accepted_by_key,
                    inventory_audit=inventory_audit,
                    combined_decisions=combined_decisions,
                    expected_sent_id=sent_id,
                )
                call_audit.append({
                    "phase": "sentence_inventory",
                    "sentence_id": sent_id,
                    "status": "ok",
                    "proposals": len(items),
                    "new_validated_events": added,
                })

            for pass_index in range(1, self.coverage_review_passes + 1):
                current_rows = [accepted_by_key[key] for key in sorted(accepted_by_key, key=_event_sort_key)]
                messages = self._coverage_review_prompt(state, chunk.text, current_rows, pass_index)
                try:
                    parsed = self._parse_chat(messages, state)
                    items = self._event_items(parsed)
                    added = self._ingest_events(
                        items=items,
                        phase=f"coverage_review_{pass_index}",
                        tokens=tokens,
                        sentences=sentences,
                        accepted_by_key=accepted_by_key,
                        inventory_audit=inventory_audit,
                        combined_decisions=combined_decisions,
                    )
                    call_audit.append({
                        "phase": "coverage_review",
                        "pass_index": pass_index,
                        "status": "ok",
                        "proposals": len(items),
                        "new_validated_events": added,
                        "inventory_size_after": len(accepted_by_key),
                    })
                except Exception as exc:
                    call_audit.append({
                        "phase": "coverage_review",
                        "pass_index": pass_index,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })

            event_rows: list[dict[str, Any]] = []
            for index, key in enumerate(sorted(accepted_by_key, key=_event_sort_key)):
                row = dict(accepted_by_key[key])
                row["event_ref"] = f"E{index:04d}"
                row["source_phases"] = _dedup(row.get("source_phases", []))
                event_rows.append(row)

            strategy, pair_rows = self._construct_pair_pool(event_rows, sentences, tokens)
            pair_pool_audit.extend(pair_rows)
            call_audit.append({
                "phase": "deterministic_pair_pool",
                "status": "ok",
                "strategy": strategy,
                "event_count": len(event_rows),
                "pair_count": len(pair_rows),
                "exhaustive_event_threshold": self.exhaustive_event_threshold,
            })

            for row in event_rows:
                expr_id = f"expr_{expr_counter:05d}"
                expr_counter += 1
                expr = LinguisticExpression(
                    expr_id=expr_id,
                    text=row["event_key"],
                    label="event_mention",
                    justification=row.get("justification") or "Validated explicit event mention.",
                    evidence=[Evidence(
                        chunk_id=chunk.chunk_id,
                        chunk_start_char=-1,
                        chunk_end_char=-1,
                        doc_start_char=-1,
                        doc_end_char=-1,
                        snippet=chunk.text[:1600],
                    )],
                )
                expressions.append(expr)
                combined_decisions.append({
                    "phase": "materialization",
                    "status": "accepted",
                    "expr_id": expr_id,
                    "text": expr.text,
                    "label": expr.label,
                    "justification": expr.justification,
                })

            for row in pair_rows:
                expr_id = f"expr_{expr_counter:05d}"
                expr_counter += 1
                relation_text = f"{row['event_a_key']} || potentially related || {row['event_b_key']}"
                justification = (
                    f"pair_id={row['pair_id']}; pair_phase=initial; "
                    f"pair_strategy={row['strategy']}; candidate_reasons={','.join(row['candidate_reasons'])}; "
                    "source_type=EVENT; target_type=EVENT"
                )
                expr = LinguisticExpression(
                    expr_id=expr_id,
                    text=relation_text,
                    label="relation_instance",
                    justification=justification,
                    evidence=[Evidence(
                        chunk_id=chunk.chunk_id,
                        chunk_start_char=-1,
                        chunk_end_char=-1,
                        doc_start_char=-1,
                        doc_end_char=-1,
                        snippet=chunk.text[:1600],
                    )],
                )
                expressions.append(expr)
                combined_decisions.append({
                    "phase": "pair_pool_materialization",
                    "status": "accepted",
                    "expr_id": expr_id,
                    "pair_id": row["pair_id"],
                    "text": relation_text,
                    "label": "relation_instance",
                    "candidate_reasons": row["candidate_reasons"],
                    "strategy": row["strategy"],
                })

        dedup: dict[tuple[str, str], LinguisticExpression] = {}
        for expr in expressions:
            dedup.setdefault((expr.text, expr.label), expr)
        state.linguistic_expressions = list(dedup.values())

        for path, rows in [
            (self.inventory_log_path, inventory_audit),
            (self.pair_pool_log_path, pair_pool_audit),
            (self.relation_generation_log_path, pair_pool_audit),
            (self.call_audit_path, call_audit),
            (self.decision_log_path, combined_decisions),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, rows)

        state.log(
            f"[{self.name}] EventStoryLine v1.5 extraction; "
            f"events={sum(1 for x in state.linguistic_expressions if x.label == 'event_mention')}; "
            f"unordered_pair_candidates={sum(1 for x in state.linguistic_expressions if x.label == 'relation_instance')}; "
            f"coverage_reviews={sum(1 for x in call_audit if x.get('phase') == 'coverage_review' and x.get('status') == 'ok')}; "
            "relation_generation_llm_calls=0"
        )
        return state


class EventStoryLineCandidateEnrichmentLayer(v13.EventStoryLineCandidateEnrichmentLayer):
    """Batched five-way direction-aware pair classification.

    Event nodes are enriched deterministically. Pair decisions are batched, cached,
    validated and materialized only when accepted. NONE and invalid responses never
    enter Layer 3.
    """

    def __init__(
        self,
        *args: Any,
        pair_batch_size: int = 16,
        pair_batch_workers: int = 8,
        context_window_sentences: int = 1,
        use_ontology_evidence: bool = True,
        closure_enabled: bool = True,
        closure_max_pairs: int = 128,
        batch_cache_dir: str | Path | None = None,
        closure_pair_log_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        base = self.compact_prompt_log_path.parent
        self.pair_batch_size = max(1, int(pair_batch_size))
        self.pair_batch_workers = max(1, int(pair_batch_workers))
        self.context_window_sentences = max(0, int(context_window_sentences))
        self.use_ontology_evidence = bool(use_ontology_evidence)
        self.closure_enabled = bool(closure_enabled)
        self.closure_max_pairs = max(0, int(closure_max_pairs))
        self.batch_cache_dir = Path(batch_cache_dir or base / "layer02_batch_cache")
        self.closure_pair_log_path = Path(closure_pair_log_path or base / "layer02_closure_pair_pool.json")
        self._cache_lock = threading.Lock()
        self._batch_audit: list[dict[str, Any]] = []

    @staticmethod
    def _pair_record(expr: LinguisticExpression, fallback_index: int) -> dict[str, Any]:
        triple = _parse_relation_instance(expr.text)
        if triple is None:
            raise ValueError(f"Malformed pair expression: {expr.text}")
        a, _, b = triple
        return {
            "pair_id": _extract_pair_id(expr.justification, f"PX{fallback_index:05d}"),
            "event_a_key": a,
            "event_b_key": b,
            "expr": expr,
            "phase": "closure" if "pair_phase=closure" in (expr.justification or "") else "initial",
        }

    def _context_for_pair(self, pair: dict[str, Any], state: PipelineState) -> dict[str, Any]:
        profile = state.profile_config or {}
        sentences = profile.get("_input_sentences", []) or []
        tokens = profile.get("_input_tokens", []) or []
        parsed_a = parse_event_key(pair["event_a_key"]) or {}
        parsed_b = parse_event_key(pair["event_b_key"]) or {}
        sent_a = int(parsed_a.get("sent_id", -1))
        sent_b = int(parsed_b.get("sent_id", -1))
        valid_ids = [x for x in (sent_a, sent_b) if 0 <= x < len(sentences)]
        if valid_ids:
            start = max(0, min(valid_ids) - self.context_window_sentences)
            end = min(len(sentences), max(valid_ids) + self.context_window_sentences + 1)
        else:
            start, end = 0, min(len(sentences), 1)
        context = [
            {
                "sentence_id": sent_id,
                "sentence": sentences[sent_id],
                "tokens": [str(x) for x in (tokens[sent_id] if sent_id < len(tokens) else [])],
            }
            for sent_id in range(start, end)
        ]
        return {
            "pair_id": pair["pair_id"],
            "event_a": pair["event_a_key"],
            "event_b": pair["event_b_key"],
            "event_a_sentence_id": sent_a,
            "event_b_sentence_id": sent_b,
            "context": context,
            "layer1_candidate_reason": pair["expr"].justification,
        }

    def _ontology_context(self, pairs: list[dict[str, Any]], state: PipelineState) -> str:
        if not self.use_ontology_evidence or self.rag_adapter is None:
            return ""
        labels: list[str] = []
        for pair in pairs[:8]:
            for key in (pair["event_a_key"], pair["event_b_key"]):
                parsed = parse_event_key(key)
                if parsed:
                    labels.append(parsed["trigger"])
        query = "EventStoryLine temporal event sequence PRECONDITION FALLING_ACTION " + " ".join(labels)
        grounding = self.rag_adapter.ground(GroundingRequest(
            layer_name=self.name,
            query=query,
            payload={"pair_ids": [pair["pair_id"] for pair in pairs]},
            preferred_sources=["ontology"],
            top_k=4,
        ))
        return build_grounding_context(grounding)

    def _batch_prompt(
        self,
        *,
        pairs: list[dict[str, Any]],
        state: PipelineState,
        batch_index: int,
        phase: str,
        recovery: bool = False,
    ) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        pair_payload: list[dict[str, Any]] = []
        shared_context_by_id: dict[int, dict[str, Any]] = {}
        for pair in pairs:
            context_row = self._context_for_pair(pair, state)
            context_items = list(context_row.pop("context", []))
            context_row["context_sentence_ids"] = [item["sentence_id"] for item in context_items]
            pair_payload.append(context_row)
            for item in context_items:
                shared_context_by_id[int(item["sentence_id"])] = item
        shared_context = [shared_context_by_id[key] for key in sorted(shared_context_by_id)]
        ontology_text = self._ontology_context(pairs, state)
        recovery_note = (
            "This is a recovery/adjudication pass. Return exactly one valid decision for every listed pair_id."
            if recovery else
            "Return exactly one decision for every listed pair_id."
        )
        system = f"""
You are NeoOLAF Layer 2 for EventStoryLine PLOT_LINK classification.

Classify each UNORDERED pair by jointly deciding relation existence, direction and
class. Allowed decisions are exactly:
- A_PRECONDITION_B: A establishes, enables, motivates, requires, prepares or sets up B.
- A_FALLING_ACTION_B: B is a consequence, continuation, aftermath, follow-up,
  elaboration or downstream narrative development from A.
- B_PRECONDITION_A: B establishes/enables/sets up A.
- B_FALLING_ACTION_A: A is the downstream development from B.
- NONE: no document-supported PLOT_LINK in either direction.

EventStoryLine PLOT_LINK is broader than direct physical causality, but sentence
order, shared topic and co-occurrence are insufficient. Compare both directions
before deciding. Use the text, not world knowledge. OWL-Time evidence is secondary
and must never override the document or turn mere before/after order into a link.

Every non-NONE answer MUST include at least one valid evidence_sentence_id and a
non-empty evidence_text grounded in those sentences. Never alter event IDs or
spans. {recovery_note}

Schema-exact examples:

Example 1 — PRECONDITION
A = S0[1:3]::permit approval
B = S1[2:4]::construction began
Context: The permit approval allowed construction to begin.
{{"pair_id":"EX1","decision":"A_PRECONDITION_B","evidence_sentence_ids":[0,1],
"evidence_text":"permit approval allowed construction to begin",
"reason":"A enables B.","confidence":0.95}}

Example 2 — FALLING_ACTION
A = S0[1:2]::bridge collapse
B = S1[3:5]::emergency inspection
Context: The bridge collapsed, followed by an emergency inspection.
{{"pair_id":"EX2","decision":"A_FALLING_ACTION_B","evidence_sentence_ids":[0,1],
"evidence_text":"followed by an emergency inspection",
"reason":"B is a downstream follow-up to A.","confidence":0.94}}

Example 3 — NONE
A = S0[1:3]::press conference
B = S2[4:5]::rainfall
Context: Both occur in the article; no dependency is stated.
{{"pair_id":"EX3","decision":"NONE","evidence_sentence_ids":[],
"evidence_text":"","reason":"Topical co-occurrence is insufficient.","confidence":0.91}}

Example 4 — reversed direction
A = S1[2:4]::construction began
B = S0[1:3]::permit approval
Context: The permit approval allowed construction to begin.
{{"pair_id":"EX4","decision":"B_PRECONDITION_A","evidence_sentence_ids":[0,1],
"evidence_text":"permit approval allowed construction",
"reason":"The supplied A-to-B order is reversed.","confidence":0.96}}

Return JSON only:
{{"decisions":[{{"pair_id":"EX_PRIMARY","decision":"NONE",
"evidence_sentence_ids":[],"evidence_text":"","reason":"...","confidence":0.8}}]}}
""".strip()
        user = f"""
Controlled relation definitions:
{_json_block(task.get('relation_specs') or [], 8000)}

Decision rules:
{_json_block((task.get('layer_guidance') or {}).get('layer02', {}), 7000)}

Pairs and the sentence IDs that may be cited as evidence:
{_json_block(pair_payload, 30000)}

Shared authoritative sentence/token context for this batch:
{_json_block(shared_context, 50000)}

Optional OWL-Time retrieval (secondary evidence only):
{ontology_text}

Batch index: {batch_index}; phase: {phase}; recovery: {str(recovery).lower()}.
Return one decision for every pair_id. JSON only.
""".strip()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        with self._prompt_lock:
            self._prompt_audit.append({
                "batch_index": batch_index,
                "phase": phase,
                "recovery": recovery,
                "pair_ids": [pair["pair_id"] for pair in pairs],
                "pair_count": len(pairs),
                "candidate_decisions": list(FIVE_WAY_DECISIONS),
                "system_chars": len(system),
                "user_chars": len(user),
            })
        return messages

    def _cache_path(self, messages: list[dict[str, str]], state: PipelineState) -> Path:
        payload = json.dumps({"model": state.llm_model, "messages": messages}, ensure_ascii=False, sort_keys=True)
        digest = sha256(payload.encode("utf-8")).hexdigest()
        return self.batch_cache_dir / f"{digest}.json"

    def _chat_batch(
        self,
        *,
        pairs: list[dict[str, Any]],
        state: PipelineState,
        batch_index: int,
        phase: str,
        recovery: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        messages = self._batch_prompt(
            pairs=pairs,
            state=state,
            batch_index=batch_index,
            phase=phase,
            recovery=recovery,
        )
        cache_path = self._cache_path(messages, state)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            parsed = read_json(cache_path)
            return parsed, {"status": "cache_hit", "cache_path": str(cache_path)}

        last_exc: Exception | None = None
        for attempt in range(self.retry_failed_calls + 1):
            try:
                raw = self.ollama_backend.chat(model=state.llm_model, messages=messages, temperature=0.0)
                parsed = self.ollama_backend.extract_json(raw)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Layer 2 batch response must be a JSON object, got {type(parsed).__name__}")
                with self._cache_lock:
                    write_json(cache_path, parsed)
                return parsed, {
                    "status": "ok",
                    "attempt": attempt,
                    "cache_path": str(cache_path),
                }
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry_failed_calls and self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _valid_evidence_ids(values: Any, state: PipelineState) -> list[int]:
        sentences = (state.profile_config or {}).get("_input_sentences", []) or []
        result: list[int] = []
        for value in values or []:
            try:
                sent_id = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= sent_id < len(sentences) and sent_id not in result:
                result.append(sent_id)
        return result

    @staticmethod
    def _evidence_text_is_grounded(
        evidence_text: str,
        evidence_sentence_ids: list[int],
        state: PipelineState,
    ) -> bool:
        if not evidence_text or not evidence_sentence_ids:
            return False
        sentences = (state.profile_config or {}).get("_input_sentences", []) or []
        context = " ".join(
            str(sentences[sent_id])
            for sent_id in evidence_sentence_ids
            if 0 <= sent_id < len(sentences)
        )
        evidence_norm = _norm(evidence_text)
        context_norm = _norm(context)
        if not evidence_norm or not context_norm:
            return False
        if evidence_norm in context_norm:
            return True
        evidence_words = [word for word in evidence_norm.split() if len(word) > 2]
        context_words = set(context_norm.split())
        if not evidence_words:
            return False
        overlap = sum(1 for word in evidence_words if word in context_words)
        minimum = 1 if len(evidence_words) <= 2 else 2
        return overlap >= minimum

    def _parse_decisions(
        self,
        parsed: dict[str, Any],
        pairs: list[dict[str, Any]],
        state: PipelineState,
        *,
        phase: str,
        recovery: bool,
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        expected = {pair["pair_id"]: pair for pair in pairs}
        grouped: dict[str, list[dict[str, Any]]] = {pair_id: [] for pair_id in expected}
        raw_rows = parsed.get("decisions")
        if not isinstance(raw_rows, list):
            raw_rows = []
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            pair_id = str(raw.get("pair_id") or "").strip()
            if pair_id in grouped:
                grouped[pair_id].append(raw)
            else:
                self._decisions.append({
                    "phase": phase,
                    "recovery": recovery,
                    "pair_id": pair_id or None,
                    "status": "ignored_unknown_pair_id",
                    "raw_decision": raw,
                })

        accepted: dict[str, dict[str, Any]] = {}
        unresolved: list[str] = []
        for pair_id, rows in grouped.items():
            if not rows:
                unresolved.append(pair_id)
                self._decisions.append({
                    "phase": phase,
                    "recovery": recovery,
                    "pair_id": pair_id,
                    "status": "missing_decision",
                })
                continue
            normalized = _dedup(_normalize_five_way(row.get("decision")) for row in rows)
            normalized = [x for x in normalized if x]
            if len(normalized) != 1:
                unresolved.append(pair_id)
                self._decisions.append({
                    "phase": phase,
                    "recovery": recovery,
                    "pair_id": pair_id,
                    "status": "conflicting_or_invalid_decisions",
                    "raw_decisions": rows,
                    "normalized_decisions": normalized,
                })
                continue
            decision = normalized[0]
            chosen = next((row for row in rows if _normalize_five_way(row.get("decision")) == decision), rows[0])
            evidence_ids = self._valid_evidence_ids(chosen.get("evidence_sentence_ids"), state)
            allowed_context_ids = {
                int(item["sentence_id"])
                for item in self._context_for_pair(expected[pair_id], state).get("context", [])
            }
            evidence_ids = [sent_id for sent_id in evidence_ids if sent_id in allowed_context_ids]
            evidence_text = str(chosen.get("evidence_text") or "").strip()
            reason = str(chosen.get("reason") or chosen.get("decision_reason") or "").strip()
            try:
                confidence = float(chosen.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            evidence_grounded = self._evidence_text_is_grounded(evidence_text, evidence_ids, state) if decision != "NONE" else True
            if decision != "NONE" and (not evidence_ids or not evidence_text or not evidence_grounded):
                unresolved.append(pair_id)
                self._decisions.append({
                    "phase": phase,
                    "recovery": recovery,
                    "pair_id": pair_id,
                    "status": "invalid_missing_or_ungrounded_evidence",
                    "decision": decision,
                    "evidence_sentence_ids": evidence_ids,
                    "evidence_text": evidence_text,
                    "evidence_grounded": evidence_grounded,
                    "allowed_context_sentence_ids": sorted(allowed_context_ids),
                    "reason": reason,
                })
                continue
            accepted[pair_id] = {
                "pair_id": pair_id,
                "decision": decision,
                "evidence_sentence_ids": evidence_ids,
                "evidence_text": evidence_text,
                "reason": reason,
                "confidence": confidence,
                "raw_decision": chosen,
            }
        return accepted, unresolved

    @staticmethod
    def _batches(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
        return [rows[i:i + size] for i in range(0, len(rows), size)]

    def _classify_pairs(
        self,
        pairs: list[dict[str, Any]],
        state: PipelineState,
        *,
        phase: str,
        batch_index_offset: int = 0,
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        batches = self._batches(pairs, self.pair_batch_size)
        unresolved_pairs: dict[str, dict[str, Any]] = {}

        def run_one(batch_index: int, batch: list[dict[str, Any]], recovery: bool = False):
            parsed, meta = self._chat_batch(
                pairs=batch,
                state=state,
                batch_index=batch_index,
                phase=phase,
                recovery=recovery,
            )
            accepted, unresolved = self._parse_decisions(
                parsed,
                batch,
                state,
                phase=phase,
                recovery=recovery,
            )
            return accepted, unresolved, meta

        if batches:
            with ThreadPoolExecutor(max_workers=min(self.pair_batch_workers, len(batches))) as executor:
                futures = {
                    executor.submit(run_one, batch_index_offset + index, batch, False): (batch_index_offset + index, batch)
                    for index, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    batch_index, batch = futures[future]
                    try:
                        accepted, unresolved, meta = future.result()
                        results.update(accepted)
                        for pair_id in unresolved:
                            unresolved_pairs[pair_id] = next(pair for pair in batch if pair["pair_id"] == pair_id)
                        self._batch_audit.append({
                            "phase": phase,
                            "batch_index": batch_index,
                            "recovery": False,
                            "pair_count": len(batch),
                            "accepted_decisions": len(accepted),
                            "unresolved_decisions": len(unresolved),
                            **meta,
                        })
                    except Exception as exc:
                        for pair in batch:
                            unresolved_pairs[pair["pair_id"]] = pair
                        self._batch_audit.append({
                            "phase": phase,
                            "batch_index": batch_index,
                            "recovery": False,
                            "pair_count": len(batch),
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        })

        recovery_batches = self._batches(list(unresolved_pairs.values()), self.pair_batch_size)
        for recovery_index, batch in enumerate(recovery_batches):
            batch_index = batch_index_offset + len(batches) + recovery_index
            try:
                accepted, unresolved, meta = run_one(batch_index, batch, True)
                results.update(accepted)
                self._batch_audit.append({
                    "phase": phase,
                    "batch_index": batch_index,
                    "recovery": True,
                    "pair_count": len(batch),
                    "accepted_decisions": len(accepted),
                    "unresolved_decisions": len(unresolved),
                    **meta,
                })
                for pair_id in unresolved:
                    pair = unresolved_pairs[pair_id]
                    results[pair_id] = {
                        "pair_id": pair_id,
                        "decision": "NONE",
                        "evidence_sentence_ids": [],
                        "evidence_text": "",
                        "reason": "Filtered after invalid, missing or conflicting outputs in both primary and recovery passes.",
                        "confidence": 0.0,
                        "filtered_invalid_output": True,
                    }
            except Exception as exc:
                self._batch_audit.append({
                    "phase": phase,
                    "batch_index": batch_index,
                    "recovery": True,
                    "pair_count": len(batch),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                for pair in batch:
                    results[pair["pair_id"]] = {
                        "pair_id": pair["pair_id"],
                        "decision": "NONE",
                        "evidence_sentence_ids": [],
                        "evidence_text": "",
                        "reason": "Filtered after batch and recovery failure.",
                        "confidence": 0.0,
                        "filtered_invalid_output": True,
                    }
        return results

    def _make_enriched_relation(
        self,
        pair: dict[str, Any],
        decision_row: dict[str, Any],
        state: PipelineState,
    ) -> EnrichedExpression | None:
        decision = decision_row["decision"]
        directed = _decision_to_directed(decision, pair["event_a_key"], pair["event_b_key"])
        if directed is None:
            self._record_decision({
                "pair_id": pair["pair_id"],
                "phase": pair.get("phase", "initial"),
                "event_a": pair["event_a_key"],
                "event_b": pair["event_b_key"],
                "decision": decision,
                "found": False,
                "status": "filtered_none" if not decision_row.get("filtered_invalid_output") else "filtered_invalid_output",
                "evidence_sentence_ids": decision_row.get("evidence_sentence_ids", []),
                "evidence_text": decision_row.get("evidence_text", ""),
                "reason": decision_row.get("reason", ""),
                "confidence": decision_row.get("confidence", 0.0),
            })
            return None

        source, relation_id, target = directed
        metadata = self.catalog[relation_id]
        relation_text = f"{source} || {relation_id} || {target}"
        base = LinguisticExpression(
            expr_id=pair["expr"].expr_id,
            text=relation_text,
            label="relation_instance",
            justification=(
                f"pair_id={pair['pair_id']}; pair_phase={pair.get('phase', 'initial')}; "
                f"decision={decision}; evidence_sentence_ids={decision_row.get('evidence_sentence_ids', [])}; "
                f"evidence={decision_row.get('evidence_text', '')}; reason={decision_row.get('reason', '')}; "
                "source_type=EVENT; target_type=EVENT"
            ),
            evidence=list(pair["expr"].evidence or []),
        )
        hints = _dedup([
            f"controlled_relation:{relation_id}",
            "promote_to_ontology:true",
            metadata.get("uri"),
            metadata.get("label"),
            f"source_label:{source}",
            f"target_label:{target}",
            f"lexical_predicate:{relation_id}",
            "source_type:EVENT",
            "target_type:EVENT",
            f"domain:{', '.join(metadata.get('domain_uris') or [])}",
            f"range:{', '.join(metadata.get('range_uris') or [])}",
            f"five_way_decision:{decision}",
            f"evidence_sentence_ids:{decision_row.get('evidence_sentence_ids', [])}",
            f"evidence_text:{decision_row.get('evidence_text', '')}",
            f"decision_reason:{decision_row.get('reason', '')}",
        ])
        self._record_decision({
            "expr_id": base.expr_id,
            "pair_id": pair["pair_id"],
            "phase": pair.get("phase", "initial"),
            "event_a": pair["event_a_key"],
            "event_b": pair["event_b_key"],
            "source": source,
            "target": target,
            "selected_relation_id": relation_id,
            "decision": decision,
            "found": True,
            "status": "accepted",
            "evidence_sentence_ids": decision_row.get("evidence_sentence_ids", []),
            "evidence_text": decision_row.get("evidence_text", ""),
            "reason": decision_row.get("reason", ""),
            "confidence": decision_row.get("confidence", 0.0),
            "ontology_hints": hints,
        })
        return EnrichedExpression(
            base_expression=base,
            aliases=_dedup([relation_text, relation_id, metadata.get("label")]),
            synonyms=[],
            lexical_variants=[],
            alias_sources={
                value: ["source" if value == relation_text else "ontology"]
                for value in _dedup([relation_text, relation_id, metadata.get("label")])
            },
            synonym_sources={},
            lexical_variant_sources={},
            definition=str(metadata.get("comment") or decision_row.get("reason") or ""),
            ontology_hints=hints,
            enrichment_evidence=[EnrichmentEvidence(
                source="llm",
                content=json.dumps(decision_row, ensure_ascii=False),
                reference=state.llm_model,
            )],
        )

    def _closure_pairs(
        self,
        initial_pairs: list[dict[str, Any]],
        initial_decisions: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.closure_enabled or self.closure_max_pairs <= 0:
            return []
        by_id = {pair["pair_id"]: pair for pair in initial_pairs}
        adjacency: dict[str, set[str]] = {}
        existing = {_pair_key(pair["event_a_key"], pair["event_b_key"]) for pair in initial_pairs}
        for pair_id, decision in initial_decisions.items():
            pair = by_id.get(pair_id)
            if not pair or decision.get("decision") == "NONE":
                continue
            directed = _decision_to_directed(decision["decision"], pair["event_a_key"], pair["event_b_key"])
            if directed is None:
                continue
            source, _, target = directed
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)

        candidates: set[tuple[str, str]] = set()
        for middle, neighbors in adjacency.items():
            for a, b in combinations(sorted(neighbors, key=_event_sort_key), 2):
                key = _pair_key(a, b)
                if key not in existing:
                    candidates.add(key)
        rows: list[dict[str, Any]] = []
        for index, (a, b) in enumerate(sorted(candidates, key=lambda x: (_event_sort_key(x[0]), _event_sort_key(x[1])))):
            if index >= self.closure_max_pairs:
                break
            expr = LinguisticExpression(
                expr_id=f"expr_closure_{index:05d}",
                text=f"{a} || potentially related || {b}",
                label="relation_instance",
                justification=(
                    f"pair_id=C{index:05d}; pair_phase=closure; candidate_reasons=two_hop_candidate_closure; "
                    "source_type=EVENT; target_type=EVENT"
                ),
                evidence=[],
            )
            rows.append({
                "pair_id": f"C{index:05d}",
                "event_a_key": a,
                "event_b_key": b,
                "expr": expr,
                "phase": "closure",
                "candidate_reasons": ["two_hop_candidate_closure"],
            })
        return rows

    def _run(self, state: PipelineState) -> PipelineState:
        expressions = list(state.linguistic_expressions)
        if self.max_expressions is not None:
            expressions = expressions[: self.max_expressions]
        self._failed_details = []
        self._decisions = []
        self._prompt_audit = []
        self._batch_audit = []

        enriched_events: list[EnrichedExpression] = []
        pairs: list[dict[str, Any]] = []
        for index, expr in enumerate(expressions):
            if expr.label == "event_mention":
                enriched_events.append(self._process_expression_conservative(expr, state))
            elif self._is_relation(expr):
                pairs.append(self._pair_record(expr, index))

        initial_decisions = self._classify_pairs(pairs, state, phase="initial")
        enriched_relations: list[EnrichedExpression] = []
        for pair in pairs:
            decision = initial_decisions.get(pair["pair_id"], {
                "decision": "NONE",
                "reason": "No valid decision after classification.",
                "evidence_sentence_ids": [],
                "evidence_text": "",
                "confidence": 0.0,
                "filtered_invalid_output": True,
            })
            enriched = self._make_enriched_relation(pair, decision, state)
            if enriched is not None:
                enriched_relations.append(enriched)

        closure_pairs = self._closure_pairs(pairs, initial_decisions)
        write_json(self.closure_pair_log_path, [
            {
                "pair_id": pair["pair_id"],
                "event_a_key": pair["event_a_key"],
                "event_b_key": pair["event_b_key"],
                "phase": pair["phase"],
                "candidate_reasons": pair.get("candidate_reasons", []),
            }
            for pair in closure_pairs
        ])
        if closure_pairs:
            closure_decisions = self._classify_pairs(
                closure_pairs,
                state,
                phase="closure",
                batch_index_offset=len(self._batches(pairs, self.pair_batch_size)) * 2 + 1000,
            )
            for pair in closure_pairs:
                decision = closure_decisions.get(pair["pair_id"], {
                    "decision": "NONE",
                    "reason": "No valid closure decision.",
                    "evidence_sentence_ids": [],
                    "evidence_text": "",
                    "confidence": 0.0,
                    "filtered_invalid_output": True,
                })
                enriched = self._make_enriched_relation(pair, decision, state)
                if enriched is not None:
                    enriched_relations.append(enriched)

        # Deterministic conflict resolution: one accepted relation per unordered
        # pair. The five-way decision should already ensure this; this guard also
        # protects against duplicate initial/closure materialization.
        dedup_relations: dict[tuple[str, str], EnrichedExpression] = {}
        for enriched in enriched_relations:
            triple = _parse_relation_instance(enriched.base_expression.text)
            if triple is None:
                continue
            source, _, target = triple
            key = _pair_key(source, target)
            dedup_relations.setdefault(key, enriched)

        state.enriched_expressions = [*enriched_events, *dedup_relations.values()]
        self._save_failed_expressions(state)
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.decision_log_path, sorted(self._decisions, key=lambda row: (str(row.get("pair_id", "")), str(row.get("phase", "")))))
        write_json(self.compact_prompt_log_path, sorted(self._prompt_audit, key=lambda row: (int(row.get("batch_index", 0)), str(row.get("phase", "")))))
        write_json(self.compact_prompt_log_path.parent / "layer02_batch_audit.json", self._batch_audit)
        state.log(
            f"[{self.name}] EventStoryLine v1.5 batched five-way enrichment; "
            f"events={len(enriched_events)}; initial_pairs={len(pairs)}; closure_pairs={len(closure_pairs)}; "
            f"accepted_relations={len(dedup_relations)}; filtered_pairs={len(pairs)+len(closure_pairs)-len(dedup_relations)}; "
            f"batch_calls={len(self._batch_audit)}"
        )
        return state


# ---------------------------------------------------------------------------
# Pipeline construction and execution
# ---------------------------------------------------------------------------


def _layer_cfg(profile: dict[str, Any], layer_name_value: str) -> dict[str, Any]:
    return dict((profile.get("layers") or {}).get(layer_name_value) or {})


def _make_backend(
    *,
    logger: SharedCallLogger,
    layer_tag: str,
    model_host: str,
    api_key: str,
    cfg: dict[str, Any],
    fallback_max_tokens: int,
    fallback_timeout: int,
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
    return v13.choose_chunk_size(text, max_safe_chars)


def build_document(record: dict[str, Any], source_path: str | Path) -> Document:
    return v13.build_document(record, source_path)


def build_pipeline(
    *,
    backends: dict[str, TaggedLoggedBackend],
    rag_adapter: Any,
    profile_config: dict[str, Any],
    relation_catalog_path: str | Path,
    chunk_size: int,
    run_dir: str | Path,
    workers: int = 16,
    verbose: bool = True,
) -> Pipeline:
    # Reuse the proven v1.3 native L0--L12 construction, then replace only the
    # two experiment-specific orchestration layers.
    pipeline = v13.build_pipeline(
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
    l1_cfg = _layer_cfg(profile_config, "layer01_linguistic_expression_extraction")
    l2_cfg = _layer_cfg(profile_config, "layer02_candidate_enrichment")

    pipeline.layers[1] = EventStoryLineExtractionLayer(
        backends["layer01"],
        decision_log_path=run_dir / "run_logs/layer01_event_relation_instances.json",
        inventory_log_path=run_dir / "run_logs/layer01_event_inventory.json",
        relation_generation_log_path=run_dir / "run_logs/layer01_relation_generation.json",
        pair_pool_log_path=run_dir / "run_logs/layer01_pair_pool.json",
        call_audit_path=run_dir / "run_logs/layer01_call_audit.json",
        sentence_workers=int(l1_cfg.get("sentence_workers", min(workers, 8))),
        coverage_review_passes=int(l1_cfg.get("coverage_review_passes", 3)),
        relation_workers=1,
        relation_source_batch_size=1,
        pair_strategy=str(l1_cfg.get("pair_strategy", "adaptive")),
        exhaustive_event_threshold=int(l1_cfg.get("exhaustive_event_threshold", 25)),
        hybrid_sentence_distance=int(l1_cfg.get("hybrid_sentence_distance", 2)),
        include_headline_body_pairs=bool(l1_cfg.get("include_headline_body_pairs", True)),
        atomic_review_enabled=bool(l1_cfg.get("atomic_review_enabled", True)),
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
        decision_log_path=run_dir / "run_logs/layer02_relation_decisions.json",
        compact_prompt_log_path=run_dir / "run_logs/layer02_compact_prompt_audit.json",
        batch_cache_dir=run_dir / "run_logs/layer02_batch_cache",
        closure_pair_log_path=run_dir / "run_logs/layer02_closure_pair_pool.json",
        pair_batch_size=int(l2_cfg.get("pair_batch_size", 16)),
        pair_batch_workers=int(l2_cfg.get("pair_batch_workers", min(workers, 8))),
        context_window_sentences=int(l2_cfg.get("context_window_sentences", 1)),
        use_ontology_evidence=bool(l2_cfg.get("use_ontology_evidence", True)),
        closure_enabled=bool(l2_cfg.get("closure_enabled", True)),
        closure_max_pairs=int(l2_cfg.get("closure_max_pairs", 0)),
        positive_verifier_enabled=bool(l2_cfg.get("positive_verifier_enabled", True)),
        verifier_batch_size=int(l2_cfg.get("verifier_batch_size", 8)),
        verifier_batch_workers=int(l2_cfg.get("verifier_batch_workers", min(workers, 4))),
        none_review_enabled=bool(l2_cfg.get("none_review_enabled", True)),
        none_review_batch_size=int(l2_cfg.get("none_review_batch_size", 8)),
        none_review_batch_workers=int(l2_cfg.get("none_review_batch_workers", min(workers, 4))),
        none_review_max_pairs=int(l2_cfg.get("none_review_max_pairs", 64)),
        max_expressions=None,
        use_web_search=False,
        save_intermediate=True,
        verbose=verbose,
        rag_adapter=rag_adapter,
        max_concurrency=int(l2_cfg.get("max_concurrency", workers)),
        retry_failed_calls=int(l2_cfg.get("retry_failed_calls", retry_default)),
        retry_sleep_seconds=sleep_default,
    )
    return pipeline


def run_native_pipeline(
    *,
    project_root: str | Path,
    input_jsonl: str | Path,
    ontology_path: str | Path,
    profile_path: str | Path,
    guidance_path: str | Path,
    task_guidance_path: str | Path,
    relation_catalog_path: str | Path,
    relation_aliases_path: str | Path,
    run_dir: str | Path,
    model_name: str,
    api_key: str,
    host: str = "https://openrouter.ai/api/v1",
    workers: int = 16,
    max_tokens: int = 8192,
    request_timeout: int = 180,
    reasoning_effort: str = "minimal",
    verbose: bool = True,
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
    logs_dir = run_dir / "run_logs"
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
    write_json(run_dir / "input_task_guidance.json", task_guidance)
    write_json(run_dir / "effective_user_guidance.json", asdict(guidance))

    seed_ontology = SeedOntologyLoader().load(str(ontology_path))
    ontology_class_count = len(seed_ontology.classes_by_uri)
    ontology_property_count = len(seed_ontology.properties_by_uri)
    if ontology_class_count == 0 and ontology_property_count == 0:
        raise RuntimeError(f"The seed ontology loaded no classes or properties: {ontology_path}")

    chunk_size = choose_chunk_size(record["text"], int(profile.get("chunking.max_safe_chunk_chars", 26000)))
    logger = SharedCallLogger(logs_dir)
    backends = {
        "layer01": _make_backend(
            logger=logger,
            layer_tag="layer01_event_inventory_v1_5",
            model_host=host,
            api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer01_linguistic_expression_extraction"),
            fallback_max_tokens=max_tokens,
            fallback_timeout=request_timeout,
            reasoning_effort=reasoning_effort,
        ),
        "layer02": _make_backend(
            logger=logger,
            layer_tag="layer02_batched_five_way_relations_v1_5",
            model_host=host,
            api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer02_candidate_enrichment"),
            fallback_max_tokens=4096,
            fallback_timeout=180,
            reasoning_effort=reasoning_effort,
        ),
        "layer04": _make_backend(
            logger=logger,
            layer_tag="layer04_fallback_only",
            model_host=host,
            api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer04_candidate_relation_extraction"),
            fallback_max_tokens=384,
            fallback_timeout=60,
            reasoning_effort=reasoning_effort,
        ),
        "other": _make_backend(
            logger=logger,
            layer_tag="other",
            model_host=host,
            api_key=api_key,
            cfg={},
            fallback_max_tokens=768,
            fallback_timeout=90,
            reasoning_effort=reasoning_effort,
        ),
    }
    rag_adapter = v13.v2.OntologyOnlyRAGAdapter(
        seed_ontology,
        log_path=logs_dir / "ontology_retrieval.jsonl",
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

    l1_cfg = _layer_cfg(profile_dict, "layer01_linguistic_expression_extraction")
    l2_cfg = _layer_cfg(profile_dict, "layer02_candidate_enrichment")
    manifest = {
        "dataset": "EventStoryLine",
        "document_id": record["document_id"],
        "title": record.get("title"),
        "experiment_version": "1.5",
        "model_name": model_name,
        "profile_name": profile.name,
        "profile_path": str(profile_path),
        "guidance_path": str(guidance_path),
        "task_guidance_path": str(task_guidance_path),
        "ontology_path": str(ontology_path),
        "ontology_classes": ontology_class_count,
        "ontology_properties": ontology_property_count,
        "seed_ontology_role": "external OWL-Time temporal grounding; secondary in Layer 2",
        "task_relation_count": len(RELATION_IDS),
        "input_has_gold": False,
        "source_sentence_count": len(record.get("sentences") or []),
        "source_token_count": sum(len(x) for x in record.get("tokens") or []),
        "chunk_size": chunk_size,
        "whole_document_single_chunk_expected": len(record["text"]) <= chunk_size,
        "workers": workers,
        "ignored_gold_relation_keys": ["null"],
        "indexed_event_keys_from_source_tokens": True,
        "layer01_sentence_parallelism": int(l1_cfg.get("sentence_workers", min(workers, 8))),
        "layer01_coverage_review_passes": int(l1_cfg.get("coverage_review_passes", 3)),
        "relation_pair_strategy": str(l1_cfg.get("pair_strategy", "adaptive")),
        "relation_exhaustive_event_threshold": int(l1_cfg.get("exhaustive_event_threshold", 25)),
        "relation_pair_generation_llm_calls": 0,
        "layer02_five_way_direction_aware": True,
        "layer02_pair_batch_size": int(l2_cfg.get("pair_batch_size", 16)),
        "layer02_pair_batch_workers": int(l2_cfg.get("pair_batch_workers", min(workers, 8))),
        "layer02_evidence_required": True,
        "layer02_none_filtered_before_layer03": True,
        "layer02_recovery_pass": True,
        "layer02_batch_cache": True,
        "layer02_candidate_closure": bool(l2_cfg.get("closure_enabled", True)),
        "global_trigger_projection_disabled": True,
        "strict_native_span_metrics": True,
        "candidate_pool_recall_metrics": True,
        "gold_projection_after_execution_only": True,
        "anti_cheating": profile.get("anti_cheating", {}),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir / "run_manifest.json", manifest)

    console_log = logs_dir / "console.log"
    errors_path = logs_dir / "pipeline_errors.jsonl"
    started = time.time()
    with console_log.open("w", encoding="utf-8") as handle:
        tee_out = Tee(sys.stdout, handle)
        tee_err = Tee(sys.stderr, handle)
        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                final_state = runner.run(state)
        except Exception as exc:
            append_jsonl(errors_path, {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            raise

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["elapsed_seconds"] = time.time() - started
    manifest["final_state_counts"] = state_counts(final_state)
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "analysis_input_fingerprint.json", {
        "document_id": record["document_id"],
        "input_jsonl": str(input_jsonl),
        "ontology_path": str(ontology_path),
        "profile_path": str(profile_path),
        "guidance_path": str(guidance_path),
        "task_guidance_path": str(task_guidance_path),
        "relation_catalog_path": str(relation_catalog_path),
        "relation_aliases_path": str(relation_aliases_path),
        "experiment_version": "1.5",
    })
    return final_state


# ---------------------------------------------------------------------------
# Exact-span evaluation and relation diagnostics
# ---------------------------------------------------------------------------


def project_event_label(label: Any, gold: dict[str, Any]) -> dict[str, Any]:
    """Exact sentence/token projection only; no trigger-only fallback."""
    index = gold_event_index(gold)
    parsed = parse_event_key(label)
    if parsed is None:
        return {
            "event_id": None,
            "method": "invalid_event_key",
            "label": str(label),
            "candidate_event_ids": [],
        }
    ids = _dedup(index["by_span"].get((parsed["sent_id"], parsed["start"], parsed["end"]), []))
    if len(ids) == 1:
        return {
            "event_id": ids[0],
            "method": "exact_sentence_token_span",
            "label": str(label),
        }
    return {
        "event_id": None,
        "method": "unmapped_or_ambiguous_exact_span",
        "label": str(label),
        "candidate_event_ids": ids,
    }


def _triples_from_state(state: PipelineState) -> list[tuple[str, str, str]]:
    return v13._triples_from_state(state)


def native_predictions(state: PipelineState, gold: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source_label, predicate_label, target_label in _triples_from_state(state):
        relation_id = normalize_relation_id(predicate_label)
        source_projection = project_event_label(source_label, gold)
        target_projection = project_event_label(target_label, gold)
        row = {
            "source_label": source_label,
            "predicate_label": predicate_label,
            "target_label": target_label,
            "relation_id": relation_id,
            "source_event_id": source_projection.get("event_id"),
            "target_event_id": target_projection.get("event_id"),
            "source_projection_method": source_projection.get("method"),
            "target_projection_method": target_projection.get("method"),
        }
        audit.append(row)
        if relation_id not in RELATION_IDS or not row["source_event_id"] or not row["target_event_id"]:
            continue
        key = (row["source_event_id"], relation_id, row["target_event_id"])
        if key not in seen:
            seen.add(key)
            predictions.append(row)
    return predictions, audit


def _event_labels_from_state(state: PipelineState) -> list[str]:
    labels: list[str] = []
    if state.event_candidates:
        labels.extend(str(candidate.canonical_label) for candidate in state.event_candidates)
    elif state.enriched_expressions:
        labels.extend(
            str(item.base_expression.text)
            for item in state.enriched_expressions
            if item.base_expression.label == "event_mention"
        )
    elif state.linguistic_expressions:
        labels.extend(str(item.text) for item in state.linguistic_expressions if item.label == "event_mention")
    return _dedup(labels)


def event_inventory_from_state(state: PipelineState, gold: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    projected: set[str] = set()
    audit: list[dict[str, Any]] = []
    for label in _event_labels_from_state(state):
        result = project_event_label(label, gold)
        audit.append(result)
        if result.get("event_id"):
            projected.add(result["event_id"])
    return projected, audit


def gold_event_span_set(gold: dict[str, Any]) -> set[str]:
    return {
        key
        for keys in gold_event_index(gold)["keys_by_id"].values()
        for key in keys
    }


def native_event_span_set(state: PipelineState) -> set[str]:
    return set(_event_labels_from_state(state))


def gold_span_relation_set(gold: dict[str, Any]) -> set[tuple[str, str, str]]:
    keys_by_id = gold_event_index(gold)["keys_by_id"]
    result: set[tuple[str, str, str]] = set()
    for source_id, relation_id, target_id in gold_relation_set(gold):
        for source_key in keys_by_id.get(source_id, []):
            for target_key in keys_by_id.get(target_id, []):
                result.add((source_key, relation_id, target_key))
    return result


def native_span_relation_set(state: PipelineState) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for source, predicate, target in _triples_from_state(state):
        relation_id = normalize_relation_id(predicate)
        if relation_id in RELATION_IDS:
            result.add((str(source), relation_id, str(target)))
    return result


def relation_endpoint_inventory(relations: set[tuple[str, str, str]]) -> set[str]:
    return {event_id for source, _, target in relations for event_id in (source, target)}


def _pair_pool_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [
        run_dir / "run_logs/layer01_pair_pool.json",
        run_dir / "run_logs/layer02_closure_pair_pool.json",
    ]:
        if path.is_file():
            rows.extend(read_json(path) or [])
    return [row for row in rows if isinstance(row, dict)]


def _pair_pool_set(run_dir: Path) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for row in _pair_pool_rows(run_dir):
        a = row.get("event_a_key")
        b = row.get("event_b_key")
        if a and b:
            result.add(_pair_key(str(a), str(b)))
    return result


def _decision_index(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = run_dir / "run_logs/layer02_relation_decisions.json"
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in (read_json(path) if path.is_file() else []):
        if not isinstance(row, dict):
            continue
        a = row.get("event_a")
        b = row.get("event_b")
        if not a or not b:
            continue
        key = _pair_key(str(a), str(b))
        # Accepted rows take precedence over filtered audit rows.
        if key not in result or row.get("status") == "accepted":
            result[key] = row
    return result


def candidate_pool_evaluation(run_dir: Path, gold: dict[str, Any], state: PipelineState) -> dict[str, Any]:
    pool = _pair_pool_set(run_dir)
    events = native_event_span_set(state)
    gold_relations = gold_span_relation_set(gold)
    endpoint_available = {
        triple for triple in gold_relations if triple[0] in events and triple[2] in events
    }
    covered = {
        triple for triple in endpoint_available if _pair_key(triple[0], triple[2]) in pool
    }
    return {
        "gold_relations": len(gold_relations),
        "gold_relations_with_both_endpoints": len(endpoint_available),
        "gold_relations_in_pair_pool": len(covered),
        "recall_over_all_gold": len(covered) / len(gold_relations) if gold_relations else 0.0,
        "recall_given_endpoints": len(covered) / len(endpoint_available) if endpoint_available else 0.0,
        "pair_pool_size": len(pool),
    }


def relation_trace(
    run_dir: Path,
    gold: dict[str, Any],
    final_span_predictions: set[tuple[str, str, str]],
    state: PipelineState,
) -> list[dict[str, Any]]:
    events = native_event_span_set(state)
    pool = _pair_pool_set(run_dir)
    decisions = _decision_index(run_dir)
    rows: list[dict[str, Any]] = []
    for source_key, relation_id, target_key in sorted(gold_span_relation_set(gold)):
        pair_key = _pair_key(source_key, target_key)
        decision = decisions.get(pair_key)
        if source_key not in events:
            failure = "source_event_missing"
        elif target_key not in events:
            failure = "target_event_missing"
        elif pair_key not in pool:
            failure = "pair_not_in_candidate_pool"
        elif decision is None:
            failure = "missing_layer02_decision"
        elif decision.get("status") == "filtered_invalid_output":
            failure = "filtered_invalid_output"
        elif decision.get("status") != "accepted" or not decision.get("found"):
            failure = "classified_none"
        else:
            predicted = (
                str(decision.get("source")),
                normalize_relation_id(decision.get("selected_relation_id")),
                str(decision.get("target")),
            )
            if predicted == (source_key, relation_id, target_key):
                failure = "survived_to_layer05" if predicted in final_span_predictions else "layer04_endpoint_assignment_or_materialization"
            elif predicted[0] == target_key and predicted[2] == source_key:
                failure = "wrong_direction"
            elif predicted[0] == source_key and predicted[2] == target_key:
                failure = "wrong_relation_class"
            else:
                failure = "wrong_pair_materialization"
        rows.append({
            "source_key": source_key,
            "relation_id": relation_id,
            "target_key": target_key,
            "first_failure": failure,
            "pair_in_pool": pair_key in pool,
            "decision_status": decision.get("status") if decision else None,
            "predicted_decision": decision.get("decision") if decision else None,
            "predicted_source": decision.get("source") if decision else None,
            "predicted_relation": decision.get("selected_relation_id") if decision else None,
            "predicted_target": decision.get("target") if decision else None,
        })
    return rows


def relation_confusion_matrix(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [
        "PRECONDITION",
        "FALLING_ACTION",
        "NONE",
        "REVERSED_PRECONDITION",
        "REVERSED_FALLING_ACTION",
        "PAIR_MISSING",
        "INVALID",
    ]
    counts: dict[str, dict[str, int]] = {
        relation_id: {column: 0 for column in columns}
        for relation_id in RELATION_IDS
    }
    for row in trace:
        gold_id = row["relation_id"]
        failure = row["first_failure"]
        if failure in {"source_event_missing", "target_event_missing", "pair_not_in_candidate_pool"}:
            column = "PAIR_MISSING"
        elif failure in {"filtered_invalid_output", "missing_layer02_decision", "wrong_pair_materialization"}:
            column = "INVALID"
        elif failure == "classified_none":
            column = "NONE"
        elif failure == "wrong_direction":
            pred = normalize_relation_id(row.get("predicted_relation")) or "PRECONDITION"
            column = f"REVERSED_{pred}"
        else:
            column = normalize_relation_id(row.get("predicted_relation")) or (
                gold_id if failure in {"survived_to_layer05", "layer04_endpoint_assignment_or_materialization"} else "NONE"
            )
        if column not in counts[gold_id]:
            column = "INVALID"
        counts[gold_id][column] += 1
    return [{"gold_relation": relation_id, **counts[relation_id]} for relation_id in RELATION_IDS]


def analyze_run(
    *,
    run_dir: str | Path,
    gold_jsonl: str | Path,
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> dict[str, Any]:
    del catalog_path, aliases_path
    run_dir = Path(run_dir)
    gold_rows = read_jsonl(gold_jsonl)
    if len(gold_rows) != 1:
        raise ValueError(f"Expected exactly one gold record, found {len(gold_rows)}")
    gold = gold_rows[0]
    states = {index: state for index, _, state in load_layer_states(run_dir)}
    if not states:
        raise FileNotFoundError(f"No saved layer states under {run_dir}")
    final_state = states[max(states)]

    projected_predictions, prediction_audit = native_predictions(final_state, gold)
    projected_set = {
        (row["source_event_id"], row["relation_id"], row["target_event_id"])
        for row in projected_predictions
    }
    gold_id_set = gold_relation_set(gold)
    projected_relation_metrics = metric_counts(projected_set, gold_id_set)

    final_span_predictions = native_span_relation_set(final_state)
    gold_span_set = gold_span_relation_set(gold)
    native_span_relation_metrics = metric_counts(final_span_predictions, gold_span_set)

    projected_events, projection_audit = event_inventory_from_state(final_state, gold)
    gold_event_ids = set((gold.get("entities") or {}).keys())
    projected_event_metrics = metric_counts(projected_events, gold_event_ids)
    native_span_event_metrics = metric_counts(native_event_span_set(final_state), gold_event_span_set(gold))
    endpoint_metrics = metric_counts(
        relation_endpoint_inventory(projected_set),
        relation_endpoint_inventory(gold_id_set),
    )
    pool_metrics = candidate_pool_evaluation(run_dir, gold, final_state)

    cumulative: list[dict[str, Any]] = []
    layer_summary: list[dict[str, Any]] = []
    for index in sorted(states):
        state = states[index]
        layer_projected, _ = native_predictions(state, gold)
        layer_projected_set = {
            (row["source_event_id"], row["relation_id"], row["target_event_id"])
            for row in layer_projected
        }
        layer_span_set = native_span_relation_set(state)
        layer_projected_events, _ = event_inventory_from_state(state, gold)
        layer_native_events = native_event_span_set(state)
        cumulative.append({
            "layer": index,
            "layer_name": layer_name(index),
            **{f"projected_relation_{k}": v for k, v in metric_counts(layer_projected_set, gold_id_set).items()},
            **{f"native_span_relation_{k}": v for k, v in metric_counts(layer_span_set, gold_span_set).items()},
            **{f"projected_event_{k}": v for k, v in metric_counts(layer_projected_events, gold_event_ids).items()},
            **{f"native_span_event_{k}": v for k, v in metric_counts(layer_native_events, gold_event_span_set(gold)).items()},
        })
        layer_summary.append({"layer": index, "layer_name": layer_name(index), **state_counts(state)})

    trace = relation_trace(run_dir, gold, final_span_predictions, final_state)
    failure_counts: dict[str, int] = {}
    for row in trace:
        failure_counts[row["first_failure"]] = failure_counts.get(row["first_failure"], 0) + 1
    relation_rows = per_relation_metrics(projected_set, gold_id_set)
    confusion = relation_confusion_matrix(trace)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json(analysis_dir / "strict_relation_predictions.json", projected_predictions)
    write_json(analysis_dir / "strict_relation_evaluation.json", projected_relation_metrics)
    write_json(analysis_dir / "native_span_relation_evaluation.json", native_span_relation_metrics)
    write_json(analysis_dir / "event_entity_evaluation.json", projected_event_metrics)
    write_json(analysis_dir / "native_span_event_evaluation.json", native_span_event_metrics)
    write_json(analysis_dir / "relation_endpoint_evaluation.json", endpoint_metrics)
    write_json(analysis_dir / "candidate_pool_evaluation.json", pool_metrics)
    write_json(analysis_dir / "prediction_projection_audit.json", prediction_audit)
    write_csv_rows(analysis_dir / "event_projection_audit.csv", projection_audit)
    write_csv_rows(analysis_dir / "cumulative_evaluation.csv", cumulative)
    write_csv_rows(analysis_dir / "gold_relation_trace.csv", trace)
    write_csv_rows(analysis_dir / "per_relation_metrics.csv", relation_rows)
    write_csv_rows(analysis_dir / "relation_confusion_matrix.csv", confusion)
    write_csv_rows(analysis_dir / "layer_summary.csv", layer_summary)

    summary = {
        "document_id": gold.get("document_id"),
        "title": gold.get("title"),
        "ignored_gold_relation_keys": ["null"],
        "strict_relation_evaluation": projected_relation_metrics,
        "native_span_relation_evaluation": native_span_relation_metrics,
        "event_entity_evaluation": projected_event_metrics,
        "native_span_event_evaluation": native_span_event_metrics,
        "relation_endpoint_evaluation": endpoint_metrics,
        "candidate_pool_evaluation": pool_metrics,
        "per_relation_metrics": relation_rows,
        "relation_confusion_matrix": confusion,
        "failure_counts": failure_counts,
        "cumulative_evaluation": cumulative,
        "layer_summary": layer_summary,
        "strict_relation_predictions": projected_predictions,
        "gold_relation_trace": trace,
    }
    write_json(analysis_dir / "analysis_summary.json", summary)
    return summary

# ---------------------------------------------------------------------------
# v1.5 overrides: atomic event triggers and verified pair-local relations
# ---------------------------------------------------------------------------

_V14ExtractionLayer = EventStoryLineExtractionLayer
_V14CandidateEnrichmentLayer = EventStoryLineCandidateEnrichmentLayer


class EventStoryLineExtractionLayer(_V14ExtractionLayer):
    """v1.5 event extraction with atomic trigger refinement.

    The proven v1.4 sentence/review extraction is retained. A final source-only
    atomicity audit then removes auxiliaries/negation/arguments, splits broad
    event phrases when the text explicitly evokes multiple trigger mentions,
    recovers missed atomic mentions, and rebuilds the deterministic pair pool.
    """

    _ATOMIC_LEADING_STRIP = {
        "a", "an", "the", "to", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "not", "never", "no", "will",
        "would", "can", "could", "may", "might", "must", "shall", "should",
    }

    def __init__(
        self,
        *args: Any,
        atomic_review_enabled: bool = True,
        atomic_review_log_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.atomic_review_enabled = bool(atomic_review_enabled)
        base = self.decision_log_path.parent
        self.atomic_review_log_path = Path(
            atomic_review_log_path or base / "layer01_atomic_span_review.json"
        )

    def _sentence_event_prompt(
        self,
        state: PipelineState,
        sent_id: int,
        sentence: str,
        sentence_tokens: list[str],
    ) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        indexed = " ".join(f"{index}={token}" for index, token in enumerate(sentence_tokens))
        system = """
You are NeoOLAF Layer 1A1 for EventStoryLine mention extraction.

Extract EVERY explicit event mention, but use the SHORTEST COMPLETE ATOMIC trigger
span used by EventStoryLine. The endpoint is the lexical trigger, not the entire
clause or noun phrase.

Atomic-span rules:
- remove auxiliaries, negation, subjects, objects, determiners and complements;
- keep a particle when it is part of a phrasal verb ("checked into");
- keep every token of a hyphenated lexical trigger ("rear - ended");
- an eventive noun normally uses only its trigger head ("conviction", "case");
- when one broad phrase explicitly contains two event triggers, output two mentions;
- repeated triggers at different spans are separate events;
- adjectives or participles may be events when they explicitly evoke a state/action;
- do not output connectives, dates, entities, publication metadata or inferred events.

Schema-exact small examples:
1. Tokens: 0=She 1=was 2=not 3=operating 4=the 5=vehicle
   Correct: {"sentence_id":0,"token_start":3,"token_end":4,"trigger":"operating"}
   Incorrect: "was not operating".
2. Tokens: 0=He 1=has 2=checked 3=into 4=treatment
   Correct: [2:4] "checked into"; keep the phrasal particle.
3. Tokens: 0=the 1=reckless 2=driving 3=conviction
   Correct trigger: [3:4] "conviction", not the full noun phrase.
4. Tokens: 0=a 1=fraud 2=investigation 3=case
   If both are explicitly eventive, output [2:3] "investigation" and [3:4] "case".

Return JSON only:
{"event_mentions":[{"sentence_id":0,"token_start":3,"token_end":4,
"trigger":"operating","event_kind":"action","justification":"Atomic lexical trigger."}],
"coverage_check":{"sentence_id":0,"checked":true}}
""".strip()
        user = f"""
Dataset guidance:
{_json_block((task.get('layer_guidance') or {}).get('layer01', {}), 7000)}

Sentence ID: {sent_id}
Sentence: {sentence}
Authoritative tokens: {indexed}

Extract every explicit atomic event trigger. Use exact zero-based end-exclusive
source-token offsets. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _coverage_review_prompt(
        self,
        state: PipelineState,
        chunk_text: str,
        accepted_events: list[dict[str, Any]],
        pass_index: int,
    ) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        sentences = (state.profile_config or {}).get("_input_sentences", []) or []
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        focus = {
            1: "missed lexical verbs, phrasal verbs and repeated mentions",
            2: "missed atomic event nouns, nominalizations, legal/medical processes and explicit states",
            3: "overlooked short triggers embedded inside broad phrases, quotations, headlines and negated clauses",
        }.get(pass_index, "any remaining explicit atomic event trigger")
        existing = [{"event_key": row["event_key"]} for row in accepted_events]
        system = f"""
You are NeoOLAF Layer 1A2 atomic coverage reviewer for EventStoryLine.

Find ONLY missing explicit mention-level events. This pass focuses on {focus}.
Use the shortest complete atomic trigger span. Never copy auxiliaries, negation,
arguments or the surrounding noun phrase into the span. Keep phrasal particles and
hyphen components only when lexically required. Preserve repeated mentions.

Small examples:
- "was not driving" -> trigger "driving" only;
- "reckless driving conviction" -> trigger "conviction" only;
- "three-month stay" -> trigger "stay" only;
- "checked into treatment" -> trigger "checked into";
- a phrase containing distinct eventive heads may yield multiple atomic mentions.

Return JSON only:
{{"missing_event_mentions":[{{"sentence_id":2,"token_start":5,"token_end":6,
"trigger":"conviction","event_kind":"eventive_nominal",
"justification":"Missing atomic trigger."}}],
"coverage_check":{{"pass":{pass_index},"checked":true}}}}
""".strip()
        user = f"""
Dataset guidance:
{_json_block((task.get('layer_guidance') or {}).get('layer01', {}), 7000)}

Already validated events (do not repeat):
{_json_block(existing, 18000)}

Raw document:
\"\"\"
{chunk_text}
\"\"\"

Authoritative indexed sentence/token table:
\"\"\"
{indexed_token_table(sentences, tokens)}
\"\"\"

Return only missing atomic triggers. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @classmethod
    def _deterministic_minimize_key(
        cls,
        key: str,
        tokens: list[list[str]],
    ) -> tuple[str, list[str]]:
        parsed = parse_event_key(key)
        if parsed is None:
            return key, []
        sent_id = int(parsed["sent_id"])
        start = int(parsed["start"])
        end = int(parsed["end"])
        if not (0 <= sent_id < len(tokens)):
            return key, []
        sent_tokens = [str(x) for x in tokens[sent_id]]
        reasons: list[str] = []
        while start < end - 1 and _norm(sent_tokens[start]) in cls._ATOMIC_LEADING_STRIP:
            start += 1
            reasons.append("removed_leading_auxiliary_determiner_or_negation")
        while end > start and not _norm(sent_tokens[end - 1]):
            end -= 1
            reasons.append("removed_trailing_punctuation")
        if end <= start:
            return key, []
        candidate_raw = f"S{sent_id}[{start}:{end}]::{' '.join(sent_tokens[start:end]).strip()}"
        candidate = canonical_event_key(candidate_raw, tokens)
        valid, _ = cls._valid_event_tokens(sent_tokens[start:end])
        return (candidate or key, reasons) if valid else (key, [])

    def _atomic_review_prompt(
        self,
        state: PipelineState,
        event_keys: list[str],
    ) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        sentences = (state.profile_config or {}).get("_input_sentences", []) or []
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        system = """
You are the final source-only atomic span auditor for EventStoryLine.

Audit the proposed event inventory against the authoritative token table. For each
proposed event choose KEEP, DROP, REPLACE or SPLIT. REPLACE/SPLIT must provide exact
replacement_mentions. You may also return additional_missing_mentions.

Use only the source text. Do not use or infer gold annotations. The goal is the
shortest complete lexical trigger:
- "was not operating" -> REPLACE with "operating";
- "reckless driving conviction" -> REPLACE with "conviction";
- "fraud investigation case" may SPLIT into "investigation" and "case" when both
  explicitly evoke events;
- "checked into" stays intact as a phrasal verb;
- "rear - ended" stays intact as a hyphenated trigger;
- do not split a fixed phrasal verb or create every content word as an event.

Return JSON only:
{"decisions":[{"event_key":"S0[1:4]::was not operating","action":"REPLACE",
"replacement_mentions":[{"sentence_id":0,"token_start":3,"token_end":4,
"trigger":"operating"}],"reason":"Remove auxiliary and negation."}],
"additional_missing_mentions":[]}
""".strip()
        user = f"""
Dataset guidance:
{_json_block((task.get('layer_guidance') or {}).get('layer01', {}), 8000)}

Proposed event inventory:
{_json_block([{"event_key": key} for key in event_keys], 24000)}

Authoritative indexed sentence/token table:
\"\"\"
{indexed_token_table(sentences, tokens)}
\"\"\"

Audit every proposed event_key exactly once. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _run(self, state: PipelineState) -> PipelineState:
        state = super()._run(state)
        if not self.atomic_review_enabled:
            return state

        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        sentences = (state.profile_config or {}).get("_input_sentences", []) or []
        chunks = list(state.document.chunks)
        chunk = chunks[0] if chunks else None
        original_events = [expr for expr in state.linguistic_expressions if expr.label == "event_mention"]
        justification_by_key = {expr.text: expr.justification for expr in original_events}

        minimized_keys: list[str] = []
        atomic_audit: list[dict[str, Any]] = []
        for expr in original_events:
            minimized, reasons = self._deterministic_minimize_key(expr.text, tokens)
            minimized_keys.append(minimized)
            atomic_audit.append({
                "phase": "deterministic_atomic_minimization",
                "original_event_key": expr.text,
                "result_event_key": minimized,
                "action": "REPLACE" if minimized != expr.text else "KEEP",
                "reasons": reasons,
            })
        minimized_keys = sorted(set(minimized_keys), key=_event_sort_key)
        final_keys: set[str] = set(minimized_keys)

        call_status: dict[str, Any] = {"phase": "atomic_span_review", "status": "skipped"}
        try:
            parsed = self._parse_chat(self._atomic_review_prompt(state, minimized_keys), state)
            rows = parsed.get("decisions") if isinstance(parsed, dict) else []
            rows = rows if isinstance(rows, list) else []
            seen: set[str] = set()
            replacements: set[str] = set()
            drops: set[str] = set()
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                event_key = str(raw.get("event_key") or "").strip()
                if event_key not in final_keys:
                    atomic_audit.append({"phase": "llm_atomic_review", "status": "ignored_unknown_event", "raw": raw})
                    continue
                seen.add(event_key)
                action = str(raw.get("action") or "KEEP").strip().upper()
                if action == "DROP":
                    drops.add(event_key)
                elif action in {"REPLACE", "SPLIT"}:
                    drops.add(event_key)
                    for item in raw.get("replacement_mentions") or []:
                        if not isinstance(item, dict):
                            continue
                        canonical, audit = self._repair_event_item(item, tokens)
                        audit.update(phase="llm_atomic_review", parent_event_key=event_key, requested_action=action)
                        atomic_audit.append(audit)
                        if canonical:
                            replacements.add(canonical)
                atomic_audit.append({
                    "phase": "llm_atomic_review",
                    "event_key": event_key,
                    "action": action,
                    "reason": raw.get("reason", ""),
                    "status": "processed",
                })
            for item in parsed.get("additional_missing_mentions") or []:
                if not isinstance(item, dict):
                    continue
                canonical, audit = self._repair_event_item(item, tokens)
                audit.update(phase="llm_atomic_additional_missing")
                atomic_audit.append(audit)
                if canonical:
                    replacements.add(canonical)
            final_keys.difference_update(drops)
            final_keys.update(replacements)
            call_status = {
                "phase": "atomic_span_review",
                "status": "ok",
                "input_events": len(minimized_keys),
                "reviewed_events": len(seen),
                "dropped_events": len(drops),
                "replacement_or_added_events": len(replacements),
                "final_events": len(final_keys),
            }
        except Exception as exc:
            call_status = {
                "phase": "atomic_span_review",
                "status": "error_fallback_to_deterministic_inventory",
                "error": f"{type(exc).__name__}: {exc}",
                "final_events": len(final_keys),
            }

        event_rows: list[dict[str, Any]] = []
        for index, key in enumerate(sorted(final_keys, key=_event_sort_key)):
            parsed_key = parse_event_key(key) or {}
            sent_id = int(parsed_key.get("sent_id", -1))
            event_rows.append({
                "event_ref": f"E{index:04d}",
                "event_key": key,
                "justification": justification_by_key.get(key, "Validated atomic EventStoryLine trigger."),
                "sentence": sentences[sent_id] if 0 <= sent_id < len(sentences) else "",
                "source_phases": ["atomic_v1_5"],
            })

        strategy, pair_rows = self._construct_pair_pool(event_rows, sentences, tokens)
        expressions: list[LinguisticExpression] = []
        materialization: list[dict[str, Any]] = []
        expr_counter = 0
        snippet = chunk.text[:1600] if chunk is not None else ""
        chunk_id = chunk.chunk_id if chunk is not None else "document"
        for row in event_rows:
            expr_id = f"expr_{expr_counter:05d}"
            expr_counter += 1
            expressions.append(LinguisticExpression(
                expr_id=expr_id,
                text=row["event_key"],
                label="event_mention",
                justification=row["justification"],
                evidence=[Evidence(chunk_id=chunk_id, chunk_start_char=-1, chunk_end_char=-1,
                                   doc_start_char=-1, doc_end_char=-1, snippet=snippet)],
            ))
            materialization.append({"phase": "atomic_materialization", "status": "accepted", "expr_id": expr_id,
                                    "text": row["event_key"], "label": "event_mention"})
        for row in pair_rows:
            expr_id = f"expr_{expr_counter:05d}"
            expr_counter += 1
            relation_text = f"{row['event_a_key']} || potentially related || {row['event_b_key']}"
            justification = (
                f"pair_id={row['pair_id']}; pair_phase=initial; pair_strategy={row['strategy']}; "
                f"candidate_reasons={','.join(row['candidate_reasons'])}; source_type=EVENT; target_type=EVENT"
            )
            expressions.append(LinguisticExpression(
                expr_id=expr_id,
                text=relation_text,
                label="relation_instance",
                justification=justification,
                evidence=[Evidence(chunk_id=chunk_id, chunk_start_char=-1, chunk_end_char=-1,
                                   doc_start_char=-1, doc_end_char=-1, snippet=snippet)],
            ))
            materialization.append({"phase": "atomic_pair_pool_materialization", "status": "accepted",
                                    "expr_id": expr_id, "pair_id": row["pair_id"], "text": relation_text,
                                    "label": "relation_instance", "candidate_reasons": row["candidate_reasons"]})

        state.linguistic_expressions = expressions
        write_json(self.atomic_review_log_path, atomic_audit)
        write_json(self.inventory_log_path, atomic_audit)
        write_json(self.pair_pool_log_path, pair_rows)
        write_json(self.relation_generation_log_path, pair_rows)
        write_json(self.decision_log_path, materialization)
        existing_calls = read_json(self.call_audit_path) if self.call_audit_path.is_file() else []
        write_json(self.call_audit_path, [*(existing_calls if isinstance(existing_calls, list) else []), call_status,
                                         {"phase": "deterministic_pair_pool_after_atomic_review", "status": "ok",
                                          "strategy": strategy, "event_count": len(event_rows), "pair_count": len(pair_rows)}])
        state.log(
            f"[{self.name}] EventStoryLine v1.5 atomic inventory; events={len(event_rows)}; "
            f"unordered_pair_candidates={len(pair_rows)}; strategy={strategy}"
        )
        return state


class EventStoryLineCandidateEnrichmentLayer(_V14CandidateEnrichmentLayer):
    """v1.5 pair-local classifier with NONE review and positive verification."""

    _DIRECT_CUES = {
        "because", "after", "before", "when", "if", "unless", "following", "followed",
        "resulted", "led", "caused", "triggered", "enabled", "required", "allowed",
        "continued", "subsequently", "therefore", "consequently", "prior", "upon",
    }

    def __init__(
        self,
        *args: Any,
        positive_verifier_enabled: bool = True,
        verifier_batch_size: int = 8,
        verifier_batch_workers: int = 4,
        none_review_enabled: bool = True,
        none_review_batch_size: int = 8,
        none_review_batch_workers: int = 4,
        none_review_max_pairs: int = 64,
        verification_log_path: str | Path | None = None,
        none_review_log_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        base = self.compact_prompt_log_path.parent
        self.positive_verifier_enabled = bool(positive_verifier_enabled)
        self.verifier_batch_size = max(1, int(verifier_batch_size))
        self.verifier_batch_workers = max(1, int(verifier_batch_workers))
        self.none_review_enabled = bool(none_review_enabled)
        self.none_review_batch_size = max(1, int(none_review_batch_size))
        self.none_review_batch_workers = max(1, int(none_review_batch_workers))
        self.none_review_max_pairs = max(0, int(none_review_max_pairs))
        self.verification_log_path = Path(verification_log_path or base / "layer02_positive_verification.json")
        self.none_review_log_path = Path(none_review_log_path or base / "layer02_none_review.json")
        # Candidate closure easily turns a direct-link benchmark into transitive
        # prediction. v1.5 deliberately disables it.
        self.closure_enabled = False

    def _context_for_pair(self, pair: dict[str, Any], state: PipelineState) -> dict[str, Any]:
        row = super()._context_for_pair(pair, state)
        row["context"] = list(row.get("context") or [])
        return row

    def _pair_payload(self, pairs: list[dict[str, Any]], state: PipelineState) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for pair in pairs:
            row = self._context_for_pair(pair, state)
            row["candidate_reasons"] = self._candidate_reasons(pair)
            payload.append(row)
        return payload

    @staticmethod
    def _candidate_reasons(pair: dict[str, Any]) -> list[str]:
        text = str(pair.get("expr").justification if pair.get("expr") is not None else "")
        match = re.search(r"candidate_reasons=([^;]+)", text)
        return [x.strip() for x in (match.group(1).split(",") if match else []) if x.strip()]

    def _batch_prompt(
        self,
        *,
        pairs: list[dict[str, Any]],
        state: PipelineState,
        batch_index: int,
        phase: str,
        recovery: bool = False,
    ) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        pair_payload = self._pair_payload(pairs, state)
        recovery_note = (
            "This is a recovery pass. Return exactly one valid decision for every pair_id."
            if recovery else "Return exactly one decision for every pair_id."
        )
        system = f"""
You are NeoOLAF Layer 2 for EventStoryLine direct PLOT_LINK classification.

For each UNORDERED pair choose exactly one:
A_PRECONDITION_B, A_FALLING_ACTION_B, B_PRECONDITION_A,
B_FALLING_ACTION_A, or NONE.

Direct-link rules:
- PRECONDITION: the source directly establishes, enables, motivates, requires,
  prepares or sets up the target.
- FALLING_ACTION: the target is directly presented as a consequence,
  continuation, aftermath, follow-up, elaboration or downstream development.
- A->B and B->C NEVER justify A->C. Do not construct transitive causal chains.
- chronology, shared topic, semantic plausibility and world knowledge are not enough;
- headline/body reformulation and repeated event descriptions may be direct plot
  links when the document presents one as the narrative development of the other;
- compare both directions and use pair-local context only.

Every positive answer must cite evidence_sentence_ids covering BOTH endpoint
sentences (one ID when both events share a sentence) and grounded evidence_text.
{recovery_note}

Annotation-aligned schema examples:
1. Headline/body elaboration: headline "Agency announces an investigation";
   body "Officials began examining the incident" -> headline event
   A_FALLING_ACTION_B body event.
2. Required setup: "Officials threatened arrest unless the person entered
   treatment" -> arrest threat A_PRECONDITION_B entering treatment.
3. Narrative aftermath: "The vehicle crashed. The driver later gave a false
   account" -> crash A_FALLING_ACTION_B false account.
4. No transitive closure: "A enabled B, and B caused C"; for pair A/C output NONE
   unless the text directly links A and C.
5. Reversed direction: if B allowed A, choose B_PRECONDITION_A.

Return JSON only:
{{"decisions":[{{"pair_id":"EX_PRIMARY","decision":"NONE",
"evidence_sentence_ids":[],"evidence_text":"","reason":"...","confidence":0.8}}]}}
""".strip()
        user = f"""
Controlled relation definitions:
{_json_block(task.get('relation_specs') or [], 8000)}

Decision rules:
{_json_block((task.get('layer_guidance') or {}).get('layer02', {}), 9000)}

Each pair below contains its own authoritative local sentence/token context.
Do not borrow evidence from another pair in this batch:
{_json_block(pair_payload, 60000)}

Batch index: {batch_index}; phase: {phase}; recovery: {str(recovery).lower()}.
JSON only.
""".strip()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        with self._prompt_lock:
            self._prompt_audit.append({
                "batch_index": batch_index, "phase": phase, "recovery": recovery,
                "pair_ids": [pair["pair_id"] for pair in pairs], "pair_count": len(pairs),
                "prompt_kind": "pair_local_direct_primary", "candidate_decisions": list(FIVE_WAY_DECISIONS),
                "system_chars": len(system), "user_chars": len(user),
            })
        return messages

    def _chat_custom(
        self,
        messages: list[dict[str, str]],
        state: PipelineState,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_path = self._cache_path(messages, state)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            return read_json(cache_path), {"status": "cache_hit", "cache_path": str(cache_path)}
        last_exc: Exception | None = None
        for attempt in range(self.retry_failed_calls + 1):
            try:
                raw = self.ollama_backend.chat(model=state.llm_model, messages=messages, temperature=0.0)
                parsed = self.ollama_backend.extract_json(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("Custom Layer 2 response must be a JSON object")
                with self._cache_lock:
                    write_json(cache_path, parsed)
                return parsed, {"status": "ok", "attempt": attempt, "cache_path": str(cache_path)}
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry_failed_calls and self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        assert last_exc is not None
        raise last_exc

    def _none_review_score(self, pair: dict[str, Any], state: PipelineState) -> int:
        context = self._context_for_pair(pair, state)
        a = parse_event_key(pair["event_a_key"]) or {}
        b = parse_event_key(pair["event_b_key"]) or {}
        sa, sb = int(a.get("sent_id", -99)), int(b.get("sent_id", -99))
        score = 0
        if sa == sb:
            score += 5
        elif abs(sa - sb) == 1:
            score += 4
        elif abs(sa - sb) == 2:
            score += 2
        if (sa == 1 and sb >= 2) or (sb == 1 and sa >= 2):
            score += 4
        if _norm(a.get("trigger", "")) == _norm(b.get("trigger", "")):
            score += 4
        context_words = {_norm(tok) for item in context.get("context", []) for tok in item.get("tokens", [])}
        if self._DIRECT_CUES.intersection(context_words):
            score += 3
        return score

    def _none_review_prompt(
        self,
        pairs: list[dict[str, Any]],
        state: PipelineState,
        batch_index: int,
    ) -> list[dict[str, str]]:
        payload = self._pair_payload(pairs, state)
        system = """
You are the EventStoryLine false-NONE reviewer.

The primary classifier returned NONE for these structurally plausible pairs.
Reconsider only annotation patterns often missed by ordinary causal reasoning:
headline/body elaboration, repeated event description, direct setup, immediate
narrative continuation, consequence or aftermath. Do NOT infer transitive links,
use world knowledge, or accept mere chronology/co-occurrence.

Choose one of the same five decisions or keep NONE. Every positive answer must
cite both endpoint sentences and direct grounded evidence.

Small examples:
- headline announcement -> body realization: FALLING_ACTION when the body is the
  narrative development of the headline;
- arrest threat -> entering treatment: PRECONDITION when the text states the threat
  motivates/requires entry;
- crash -> false account: FALLING_ACTION when the account is explicitly presented
  as aftermath;
- A enabled B and B caused C: A/C remains NONE without a direct textual link.

Return JSON only with key decisions.
""".strip()
        user = f"""
Pairs with isolated local context:
{_json_block(payload, 60000)}
Batch index: {batch_index}. Return one decision per pair_id. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _review_none_decisions(
        self,
        pairs: list[dict[str, Any]],
        decisions: dict[str, dict[str, Any]],
        state: PipelineState,
    ) -> dict[str, dict[str, Any]]:
        if not self.none_review_enabled or self.none_review_max_pairs <= 0:
            return decisions
        candidates = [
            (self._none_review_score(pair, state), pair)
            for pair in pairs
            if decisions.get(pair["pair_id"], {}).get("decision") == "NONE"
        ]
        candidates = [item for item in candidates if item[0] >= 3]
        candidates.sort(key=lambda item: (-item[0], _event_sort_key(item[1]["event_a_key"]), _event_sort_key(item[1]["event_b_key"])))
        selected = [pair for _, pair in candidates[: self.none_review_max_pairs]]
        review_log: list[dict[str, Any]] = []

        def run_batch(index: int, batch: list[dict[str, Any]]):
            messages = self._none_review_prompt(batch, state, index)
            parsed, meta = self._chat_custom(messages, state)
            accepted, unresolved = self._parse_decisions(parsed, batch, state, phase="none_review", recovery=False)
            return accepted, unresolved, meta

        batches = self._batches(selected, self.none_review_batch_size)
        if batches:
            with ThreadPoolExecutor(max_workers=min(self.none_review_batch_workers, len(batches))) as executor:
                futures = {executor.submit(run_batch, i, batch): (i, batch) for i, batch in enumerate(batches)}
                for future in as_completed(futures):
                    i, batch = futures[future]
                    try:
                        accepted, unresolved, meta = future.result()
                        for pair_id, row in accepted.items():
                            if row.get("decision") != "NONE":
                                row["reason"] = f"NONE review: {row.get('reason', '')}"
                                decisions[pair_id] = row
                        review_log.append({"batch_index": i, "pair_count": len(batch),
                                           "positive_revisions": sum(1 for x in accepted.values() if x.get('decision') != 'NONE'),
                                           "unresolved": unresolved, **meta})
                    except Exception as exc:
                        review_log.append({"batch_index": i, "pair_count": len(batch), "status": "error",
                                           "error": f"{type(exc).__name__}: {exc}"})
        write_json(self.none_review_log_path, {"selected_pairs": [p["pair_id"] for p in selected], "batches": review_log})
        return decisions

    def _verification_prompt(
        self,
        pairs: list[dict[str, Any]],
        current: dict[str, dict[str, Any]],
        state: PipelineState,
        batch_index: int,
    ) -> list[dict[str, str]]:
        payload = []
        for pair in pairs:
            row = self._context_for_pair(pair, state)
            row["current_decision"] = current[pair["pair_id"]]
            payload.append(row)
        system = """
You are the conservative final verifier for positive EventStoryLine PLOT_LINKs.

For each proposed positive pair choose the FINAL five-way decision. You may KEEP,
CHANGE_DIRECTION, CHANGE_CLASS or REJECT by returning NONE. Retain a relation only
when the local text directly supports it. Reject transitive chains, general story
plausibility, chronology-only links and evidence borrowed from unrelated pairs.

Every retained positive must cite BOTH endpoint sentences (one sentence ID when
both events share a sentence) and grounded evidence_text. Check direction and the
PRECONDITION/FALLING_ACTION distinction independently.

Examples:
- permit directly allows construction -> PRECONDITION;
- crash explicitly followed by investigation -> FALLING_ACTION;
- A enabled B and B caused C -> reject A/C without direct evidence;
- events merely occurring in sequence -> NONE.

Return JSON only:
{"decisions":[{"pair_id":"EX_VERIFY","decision":"NONE",
"verification_action":"REJECT","evidence_sentence_ids":[],"evidence_text":"",
"reason":"No direct pair-local link.","confidence":0.9}]}
""".strip()
        user = f"""
Proposed positives and isolated local context:
{_json_block(payload, 60000)}
Batch index: {batch_index}. Return one final decision per pair_id. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _verify_positive_decisions(
        self,
        pairs: list[dict[str, Any]],
        decisions: dict[str, dict[str, Any]],
        state: PipelineState,
    ) -> dict[str, dict[str, Any]]:
        if not self.positive_verifier_enabled:
            return decisions
        by_id = {pair["pair_id"]: pair for pair in pairs}
        positive_pairs = [by_id[pair_id] for pair_id, row in decisions.items()
                          if row.get("decision") != "NONE" and pair_id in by_id]
        verification_log: list[dict[str, Any]] = []

        def run_batch(index: int, batch: list[dict[str, Any]]):
            messages = self._verification_prompt(batch, decisions, state, index)
            parsed, meta = self._chat_custom(messages, state)
            accepted, unresolved = self._parse_decisions(parsed, batch, state, phase="positive_verifier", recovery=False)
            return accepted, unresolved, meta

        batches = self._batches(positive_pairs, self.verifier_batch_size)
        if batches:
            with ThreadPoolExecutor(max_workers=min(self.verifier_batch_workers, len(batches))) as executor:
                futures = {executor.submit(run_batch, i, batch): (i, batch) for i, batch in enumerate(batches)}
                for future in as_completed(futures):
                    i, batch = futures[future]
                    try:
                        accepted, unresolved, meta = future.result()
                        for pair in batch:
                            pair_id = pair["pair_id"]
                            final = accepted.get(pair_id)
                            if final is None:
                                decisions[pair_id] = {"pair_id": pair_id, "decision": "NONE",
                                                      "evidence_sentence_ids": [], "evidence_text": "",
                                                      "reason": "Rejected: verifier returned no valid decision.",
                                                      "confidence": 0.0, "filtered_invalid_output": True}
                                continue
                            if final.get("decision") != "NONE":
                                a = parse_event_key(pair["event_a_key"]) or {}
                                b = parse_event_key(pair["event_b_key"]) or {}
                                required = {int(a.get("sent_id", -1)), int(b.get("sent_id", -1))}
                                required.discard(-1)
                                cited = set(final.get("evidence_sentence_ids") or [])
                                if not required.issubset(cited):
                                    final = {"pair_id": pair_id, "decision": "NONE",
                                             "evidence_sentence_ids": [], "evidence_text": "",
                                             "reason": "Rejected: verifier evidence did not cover both endpoint sentences.",
                                             "confidence": 0.0}
                            final["reason"] = f"Positive verifier: {final.get('reason', '')}"
                            decisions[pair_id] = final
                        verification_log.append({"batch_index": i, "pair_count": len(batch),
                                                 "kept_positive": sum(1 for p in batch if decisions[p['pair_id']].get('decision') != 'NONE'),
                                                 "unresolved": unresolved, **meta})
                    except Exception as exc:
                        for pair in batch:
                            decisions[pair["pair_id"]] = {"pair_id": pair["pair_id"], "decision": "NONE",
                                                          "evidence_sentence_ids": [], "evidence_text": "",
                                                          "reason": "Rejected after verifier batch failure.", "confidence": 0.0}
                        verification_log.append({"batch_index": i, "pair_count": len(batch), "status": "error",
                                                 "error": f"{type(exc).__name__}: {exc}"})
        write_json(self.verification_log_path, {"input_positive_pairs": [p["pair_id"] for p in positive_pairs],
                                                "batches": verification_log})
        return decisions

    def _run(self, state: PipelineState) -> PipelineState:
        expressions = list(state.linguistic_expressions)
        if self.max_expressions is not None:
            expressions = expressions[: self.max_expressions]
        self._failed_details = []
        self._decisions = []
        self._prompt_audit = []
        self._batch_audit = []

        enriched_events: list[EnrichedExpression] = []
        pairs: list[dict[str, Any]] = []
        for index, expr in enumerate(expressions):
            if expr.label == "event_mention":
                enriched_events.append(self._process_expression_conservative(expr, state))
            elif self._is_relation(expr):
                pairs.append(self._pair_record(expr, index))

        decisions = self._classify_pairs(pairs, state, phase="primary_direct")
        decisions = self._review_none_decisions(pairs, decisions, state)
        decisions = self._verify_positive_decisions(pairs, decisions, state)

        enriched_relations: list[EnrichedExpression] = []
        for pair in pairs:
            decision = decisions.get(pair["pair_id"], {
                "pair_id": pair["pair_id"], "decision": "NONE", "evidence_sentence_ids": [],
                "evidence_text": "", "reason": "No final valid decision.", "confidence": 0.0,
                "filtered_invalid_output": True,
            })
            enriched = self._make_enriched_relation(pair, decision, state)
            if enriched is not None:
                enriched_relations.append(enriched)

        dedup_relations: dict[tuple[str, str], EnrichedExpression] = {}
        for enriched in enriched_relations:
            triple = _parse_relation_instance(enriched.base_expression.text)
            if triple is None:
                continue
            source, _, target = triple
            dedup_relations.setdefault(_pair_key(source, target), enriched)

        state.enriched_expressions = [*enriched_events, *dedup_relations.values()]
        self._save_failed_expressions(state)
        write_json(self.decision_log_path, sorted(self._decisions,
                                                  key=lambda row: (str(row.get("pair_id", "")), str(row.get("phase", "")))))
        write_json(self.compact_prompt_log_path, sorted(self._prompt_audit,
                                                        key=lambda row: (int(row.get("batch_index", 0)), str(row.get("phase", "")))))
        write_json(self.compact_prompt_log_path.parent / "layer02_batch_audit.json", self._batch_audit)
        write_json(self.closure_pair_log_path, [])
        state.log(
            f"[{self.name}] EventStoryLine v1.5 relation resolution; events={len(enriched_events)}; "
            f"pairs={len(pairs)}; accepted_relations={len(dedup_relations)}; "
            f"none_review={self.none_review_enabled}; positive_verifier={self.positive_verifier_enabled}; "
            "candidate_closure=false"
        )
        return state

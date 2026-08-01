"""EventStoryLine native NeoOLAF one-document experiment v1.6.

This module keeps the proven v1.5 atomic event inventory and deterministic pair
pool, but replaces the over-complex five-way/review/verifier relation stack with
a source-centric directed checklist:

* one fixed source event per primary request;
* every candidate target receives PRECONDITION, FALLING_ACTION or NONE;
* missing target IDs receive one compact recovery request;
* evidence formatting is repaired non-destructively rather than deleting an
  otherwise valid relation decision;
* only opposite-direction positive conflicts receive a compact adjudication;
* no broad false-NONE review and no destructive positive verifier;
* native NeoOLAF Layers 0--12 remain unchanged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import json
import math
import re
import time

import eventstoryline_native_ablation_v1_5 as v15

# Re-export the stable v1.5 public helpers used by the notebook/evaluator.
RELATION_IDS = v15.RELATION_IDS
analyze_run = v15.analyze_run
gold_event_index = v15.gold_event_index
indexed_token_table = v15.indexed_token_table
load_layer_states = v15.load_layer_states
project_event_label = v15.project_event_label
read_json = v15.read_json
read_jsonl = v15.read_jsonl
seed_ontology_summary = v15.seed_ontology_summary
write_json = v15.write_json
write_csv_rows = v15.write_csv_rows
state_counts = v15.state_counts
parse_event_key = v15.parse_event_key

_V15_BUILD_PIPELINE = v15.build_pipeline
_V15_MAKE_BACKEND = v15._make_backend

_ALLOWED_SOURCE_RELATIONS = {"PRECONDITION", "FALLING_ACTION", "NONE"}
_FIVE_WAY = set(v15.FIVE_WAY_DECISIONS)


def _clip_confidence(value: Any, default: float = 0.5) -> float:
    try:
        result = float(value)
    except Exception:
        result = default
    if math.isnan(result) or math.isinf(result):
        result = default
    return max(0.0, min(1.0, result))


def _normalize_source_relation(value: Any) -> str | None:
    raw = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "NO_RELATION": "NONE",
        "NO_LINK": "NONE",
        "NULL": "NONE",
        "RISING_ACTION": "PRECONDITION",
        "PREREQUISITE": "PRECONDITION",
        "FALLINGACTION": "FALLING_ACTION",
    }
    raw = aliases.get(raw, raw)
    normalized = v15.normalize_relation_id(raw)
    if normalized in RELATION_IDS:
        return normalized
    return raw if raw == "NONE" else None


def _source_to_five_way(
    relation_id: str,
    source_key: str,
    target_key: str,
    pair: dict[str, Any],
) -> str:
    if relation_id == "NONE":
        return "NONE"
    source_is_a = source_key == pair["event_a_key"] and target_key == pair["event_b_key"]
    source_is_b = source_key == pair["event_b_key"] and target_key == pair["event_a_key"]
    if not (source_is_a or source_is_b):
        raise ValueError(f"Directed decision does not belong to pair {pair['pair_id']}")
    prefix = "A" if source_is_a else "B"
    suffix = "B" if source_is_a else "A"
    return f"{prefix}_{relation_id}_{suffix}"


def _event_sentence_id(event_key: str) -> int:
    parsed = parse_event_key(event_key) or {}
    try:
        return int(parsed.get("sent_id", -1))
    except Exception:
        return -1


def _event_trigger(event_key: str) -> str:
    parsed = parse_event_key(event_key) or {}
    return str(parsed.get("trigger") or event_key)


class EventStoryLineSourceCentricCandidateEnrichmentLayer(v15.EventStoryLineCandidateEnrichmentLayer):
    """Directed source-centric relation classification for EventStoryLine."""

    def __init__(
        self,
        *args: Any,
        source_workers: int = 8,
        missing_target_retry: bool = True,
        conflict_adjudication_enabled: bool = True,
        conflict_batch_size: int = 8,
        conflict_confidence_margin: float = 0.15,
        source_decision_log_path: str | Path | None = None,
        source_call_audit_path: str | Path | None = None,
        conflict_log_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        # Disable the v1.5 false-NONE review and positive verifier entirely.
        kwargs["positive_verifier_enabled"] = False
        kwargs["none_review_enabled"] = False
        kwargs["closure_enabled"] = False
        kwargs["closure_max_pairs"] = 0
        super().__init__(*args, **kwargs)
        base = self.compact_prompt_log_path.parent
        self.source_workers = max(1, int(source_workers))
        self.missing_target_retry = bool(missing_target_retry)
        self.conflict_adjudication_enabled = bool(conflict_adjudication_enabled)
        self.conflict_batch_size = max(1, int(conflict_batch_size))
        self.conflict_confidence_margin = max(0.0, float(conflict_confidence_margin))
        self.source_decision_log_path = Path(
            source_decision_log_path or base / "layer02_source_centric_decisions.json"
        )
        self.source_call_audit_path = Path(
            source_call_audit_path or base / "layer02_source_centric_call_audit.json"
        )
        self.conflict_log_path = Path(
            conflict_log_path or base / "layer02_direction_conflicts.json"
        )

    @staticmethod
    def _source_target_index(pairs: list[dict[str, Any]]) -> tuple[list[str], dict[str, set[str]]]:
        targets: dict[str, set[str]] = {}
        for pair in pairs:
            a, b = pair["event_a_key"], pair["event_b_key"]
            targets.setdefault(a, set()).add(b)
            targets.setdefault(b, set()).add(a)
        events = sorted(targets, key=v15._event_sort_key)
        return events, targets

    @staticmethod
    def _event_refs(events: list[str]) -> tuple[dict[str, str], dict[str, str]]:
        key_to_ref = {key: f"E{index:04d}" for index, key in enumerate(events)}
        ref_to_key = {ref: key for key, ref in key_to_ref.items()}
        return key_to_ref, ref_to_key

    def _document_payload(self, state: v15.PipelineState) -> list[dict[str, Any]]:
        profile = state.profile_config or {}
        sentences = profile.get("_input_sentences", []) or []
        tokens = profile.get("_input_tokens", []) or []
        rows = []
        for sid, sentence in enumerate(sentences):
            rows.append({
                "sentence_id": sid,
                "sentence": sentence,
                "tokens": [str(x) for x in (tokens[sid] if sid < len(tokens) else [])],
            })
        return rows

    def _source_prompt(
        self,
        *,
        source_key: str,
        target_keys: list[str],
        key_to_ref: dict[str, str],
        state: v15.PipelineState,
        phase: str,
    ) -> list[dict[str, str]]:
        profile = state.profile_config or {}
        task = profile.get("_input_task_guidance", {}) or {}
        sentences = profile.get("_input_sentences", []) or []
        source_sid = _event_sentence_id(source_key)
        source_sentence = sentences[source_sid] if 0 <= source_sid < len(sentences) else ""
        targets = []
        for key in target_keys:
            sid = _event_sentence_id(key)
            targets.append({
                "target_event_id": key_to_ref[key],
                "target_event_key": key,
                "target_trigger": _event_trigger(key),
                "target_sentence_id": sid,
                "target_sentence": sentences[sid] if 0 <= sid < len(sentences) else "",
            })

        recovery_note = (
            "This is a recovery request for previously missing target IDs. Return every listed target exactly once."
            if phase == "missing_target_recovery"
            else "Return every listed target exactly once."
        )
        system = f"""
You are NeoOLAF Layer 2 for EventStoryLine PLOT_LINK classification.

The SOURCE event is fixed. For every TARGET choose exactly one relation FROM THE
FIXED SOURCE TO THAT TARGET:
- PRECONDITION: the fixed source directly establishes, enables, motivates,
  requires, prepares or sets up the target;
- FALLING_ACTION: the target is directly presented as a consequence,
  continuation, aftermath, follow-up, elaboration or downstream narrative
  development of the fixed source;
- NONE: no direct document-supported PLOT_LINK from the fixed source to the target.

Direction is fixed by this request. Do not return a reverse relation. If only the
reverse direction is supported, return NONE; that reverse direction is evaluated
in the other event's own source request.

EventStoryLine orientation is narrative/annotation-specific and is not always the
same as naive physical causality or temporal order. Read how the story presents
the two events. Headline/body elaboration, legal setup, repeated event description,
and immediate narrative aftermath may be links. Still reject mere chronology,
co-occurrence, shared topic, world-knowledge plausibility, and transitive chains.
A->B and B->C never justify A->C without a direct textual link.

Small source-centric examples:
1. SOURCE = arrest threat. Text: "Officials warned of arrest unless treatment
   began." TARGET = treatment began -> PRECONDITION.
2. SOURCE = arrest. Text: "The arrest was followed by a finding that the order
   had been violated." TARGET = violated -> FALLING_ACTION. Follow the narrative
   annotation direction even when ordinary causality might suggest another view.
3. SOURCE = headline announcement. The body realizes or elaborates that same
   event. TARGET = body realization -> FALLING_ACTION.
4. SOURCE = permit approval. TARGET = rainfall merely mentioned elsewhere -> NONE.
5. SOURCE = A where A enabled B and B caused C. TARGET = C -> NONE unless A and C
   are directly linked in the text.

For positive decisions, cite the source and target sentence IDs. Evidence text is
required, but formatting mistakes must not change the semantic class. {recovery_note}

Return JSON only:
{{"source_event_id":"E0000","decisions":[
  {{"target_event_id":"E0001","relation":"PRECONDITION",
    "evidence_sentence_ids":[0,1],"evidence_text":"...",
    "reason":"...","confidence":0.85}},
  {{"target_event_id":"E0002","relation":"NONE",
    "evidence_sentence_ids":[],"evidence_text":"",
    "reason":"No direct source-to-target link.","confidence":0.8}}
]}}
""".strip()
        user = f"""
Controlled relation definitions:
{v15._json_block(task.get('relation_specs') or [], 7000)}

Source-centric decision guidance:
{v15._json_block((task.get('layer_guidance') or {}).get('layer02', {}), 8000)}

Fixed source:
{v15._json_block({
    'source_event_id': key_to_ref[source_key],
    'source_event_key': source_key,
    'source_trigger': _event_trigger(source_key),
    'source_sentence_id': source_sid,
    'source_sentence': source_sentence,
}, 5000)}

Targets to classify from this fixed source:
{v15._json_block(targets, 30000)}

Authoritative document context:
{v15._json_block(self._document_payload(state), 30000)}

Phase: {phase}. JSON only.
""".strip()
        with self._prompt_lock:
            self._prompt_audit.append({
                "phase": phase,
                "source_event_id": key_to_ref[source_key],
                "source_event_key": source_key,
                "target_count": len(target_keys),
                "target_ids": [key_to_ref[x] for x in target_keys],
                "system_chars": len(system),
                "user_chars": len(user),
                "prompt_kind": "source_centric_directed_checklist",
            })
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _parse_source_response(
        self,
        *,
        parsed: dict[str, Any],
        source_key: str,
        target_keys: list[str],
        key_to_ref: dict[str, str],
        ref_to_key: dict[str, str],
        state: v15.PipelineState,
        phase: str,
    ) -> tuple[dict[str, dict[str, Any]], list[str], list[dict[str, Any]]]:
        rows = parsed.get("decisions") if isinstance(parsed, dict) else []
        rows = rows if isinstance(rows, list) else []
        expected = {key_to_ref[key]: key for key in target_keys}
        profile = state.profile_config or {}
        sentences = profile.get("_input_sentences", []) or []
        source_sid = _event_sentence_id(source_key)
        accepted: dict[str, dict[str, Any]] = {}
        audit: list[dict[str, Any]] = []

        for raw in rows:
            if not isinstance(raw, dict):
                audit.append({"phase": phase, "status": "ignored_non_object", "raw": raw})
                continue
            target_ref = str(raw.get("target_event_id") or raw.get("target_id") or "").strip()
            if target_ref not in expected:
                audit.append({"phase": phase, "status": "ignored_unknown_target", "raw": raw})
                continue
            target_key = expected[target_ref]
            relation = _normalize_source_relation(raw.get("relation", raw.get("decision")))
            if relation not in _ALLOWED_SOURCE_RELATIONS:
                audit.append({
                    "phase": phase, "status": "invalid_relation", "target_event_id": target_ref,
                    "raw_relation": raw.get("relation", raw.get("decision")),
                })
                continue
            target_sid = _event_sentence_id(target_key)
            required = sorted({sid for sid in (source_sid, target_sid) if sid >= 0})
            evidence_ids: list[int] = []
            for value in raw.get("evidence_sentence_ids") or []:
                try:
                    sid = int(value)
                except Exception:
                    continue
                if 0 <= sid < len(sentences) and sid not in evidence_ids:
                    evidence_ids.append(sid)
            evidence_text = str(raw.get("evidence_text") or "").strip()
            evidence_repaired = False
            if relation != "NONE":
                if not set(required).issubset(set(evidence_ids)):
                    evidence_ids = required
                    evidence_repaired = True
                if not evidence_text:
                    evidence_text = " | ".join(
                        sentences[sid] for sid in required if 0 <= sid < len(sentences)
                    )
                    evidence_repaired = True
            else:
                evidence_ids = []
                evidence_text = ""
            candidate = {
                "source_event_id": key_to_ref[source_key],
                "source_event_key": source_key,
                "target_event_id": target_ref,
                "target_event_key": target_key,
                "relation": relation,
                "evidence_sentence_ids": evidence_ids,
                "evidence_text": evidence_text,
                "reason": str(raw.get("reason") or "").strip(),
                "confidence": _clip_confidence(raw.get("confidence")),
                "phase": phase,
                "evidence_repaired": evidence_repaired,
                "status": "accepted",
            }
            previous = accepted.get(target_ref)
            if previous is None or candidate["confidence"] > previous["confidence"]:
                accepted[target_ref] = candidate
            audit.append({**candidate, "status": "parsed"})

        missing = [target_ref for target_ref in expected if target_ref not in accepted]
        return accepted, missing, audit

    def _classify_one_source(
        self,
        *,
        source_key: str,
        target_keys: list[str],
        key_to_ref: dict[str, str],
        ref_to_key: dict[str, str],
        state: v15.PipelineState,
    ) -> tuple[str, dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        call_audit: list[dict[str, Any]] = []
        parse_audit: list[dict[str, Any]] = []
        final: dict[str, dict[str, Any]] = {}

        def invoke(keys: list[str], phase: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
            messages = self._source_prompt(
                source_key=source_key,
                target_keys=keys,
                key_to_ref=key_to_ref,
                state=state,
                phase=phase,
            )
            started = time.time()
            try:
                parsed, meta = self._chat_custom(messages, state)
                accepted, missing, rows = self._parse_source_response(
                    parsed=parsed,
                    source_key=source_key,
                    target_keys=keys,
                    key_to_ref=key_to_ref,
                    ref_to_key=ref_to_key,
                    state=state,
                    phase=phase,
                )
                parse_audit.extend(rows)
                call_audit.append({
                    "source_event_id": key_to_ref[source_key], "source_event_key": source_key,
                    "phase": phase, "requested_targets": len(keys), "resolved_targets": len(accepted),
                    "missing_targets": [key_to_ref[x] for x in keys if key_to_ref[x] in missing],
                    "elapsed_seconds": time.time() - started, **meta,
                })
                return accepted, missing
            except Exception as exc:
                call_audit.append({
                    "source_event_id": key_to_ref[source_key], "source_event_key": source_key,
                    "phase": phase, "requested_targets": len(keys), "resolved_targets": 0,
                    "missing_targets": [key_to_ref[x] for x in keys],
                    "elapsed_seconds": time.time() - started,
                    "status": "error", "error": f"{type(exc).__name__}: {exc}",
                })
                return {}, [key_to_ref[x] for x in keys]

        primary, missing_refs = invoke(target_keys, "primary_source_checklist")
        final.update(primary)
        if missing_refs and self.missing_target_retry:
            missing_keys = [ref_to_key[ref] for ref in missing_refs if ref in ref_to_key]
            recovered, remaining = invoke(missing_keys, "missing_target_recovery")
            final.update(recovered)
            missing_refs = remaining

        # Non-destructive fallback: unresolved target IDs become NONE only after
        # the dedicated recovery request fails. This never deletes valid positives.
        for target_key in target_keys:
            target_ref = key_to_ref[target_key]
            if target_ref in final:
                continue
            final[target_ref] = {
                "source_event_id": key_to_ref[source_key],
                "source_event_key": source_key,
                "target_event_id": target_ref,
                "target_event_key": target_key,
                "relation": "NONE",
                "evidence_sentence_ids": [], "evidence_text": "",
                "reason": "No valid target decision after one focused recovery request.",
                "confidence": 0.0, "phase": "fallback_none", "evidence_repaired": False,
                "status": "fallback_none",
            }
        return source_key, final, call_audit, parse_audit

    def _conflict_prompt(
        self,
        conflicts: list[dict[str, Any]],
        state: v15.PipelineState,
        batch_index: int,
    ) -> list[dict[str, str]]:
        profile = state.profile_config or {}
        sentences = profile.get("_input_sentences", []) or []
        payload = []
        for conflict in conflicts:
            pair = conflict["pair"]
            a_sid = _event_sentence_id(pair["event_a_key"])
            b_sid = _event_sentence_id(pair["event_b_key"])
            context_ids = sorted({sid for sid in (a_sid, b_sid) if 0 <= sid < len(sentences)})
            payload.append({
                "pair_id": pair["pair_id"],
                "event_a": pair["event_a_key"],
                "event_b": pair["event_b_key"],
                "proposal_from_a": conflict["proposal_a"],
                "proposal_from_b": conflict["proposal_b"],
                "context": [{"sentence_id": sid, "sentence": sentences[sid]} for sid in context_ids],
            })
        system = """
You adjudicate only EventStoryLine direction conflicts. Two source-centric calls
both proposed a positive relation for opposite directions of the same unordered
pair. Choose exactly one final decision:
A_PRECONDITION_B, A_FALLING_ACTION_B, B_PRECONDITION_A,
B_FALLING_ACTION_A, or NONE.

Follow the story's direct narrative annotation, not mere chronology or naive world
causality. Reject transitive or unsupported links. Return every pair_id exactly
once. JSON only:
{"decisions":[{"pair_id":"P00001","decision":"NONE","reason":"...","confidence":0.7}]}
""".strip()
        user = f"""
Conflicts:
{v15._json_block(payload, 40000)}
Batch index: {batch_index}. JSON only.
""".strip()
        with self._prompt_lock:
            self._prompt_audit.append({
                "phase": "direction_conflict_adjudication", "batch_index": batch_index,
                "pair_count": len(conflicts), "pair_ids": [x["pair"]["pair_id"] for x in conflicts],
                "system_chars": len(system), "user_chars": len(user),
                "prompt_kind": "compact_conflict_only",
            })
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _adjudicate_conflicts(
        self,
        conflicts: list[dict[str, Any]],
        state: v15.PipelineState,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        resolved: dict[str, dict[str, Any]] = {}
        audit: list[dict[str, Any]] = []
        if not conflicts:
            return resolved, audit
        batches = [conflicts[i:i + self.conflict_batch_size]
                   for i in range(0, len(conflicts), self.conflict_batch_size)]
        for batch_index, batch in enumerate(batches):
            fallback = {}
            for item in batch:
                a, b = item["proposal_a"], item["proposal_b"]
                fallback[item["pair"]["pair_id"]] = a if a["confidence"] >= b["confidence"] else b
            if not self.conflict_adjudication_enabled:
                for item in batch:
                    chosen = fallback[item["pair"]["pair_id"]]
                    resolved[item["pair"]["pair_id"]] = chosen["five_way_row"]
                continue
            try:
                parsed, meta = self._chat_custom(self._conflict_prompt(batch, state, batch_index), state)
                rows = parsed.get("decisions") if isinstance(parsed, dict) else []
                rows = rows if isinstance(rows, list) else []
                by_id: dict[str, dict[str, Any]] = {}
                for raw in rows:
                    if not isinstance(raw, dict):
                        continue
                    pid = str(raw.get("pair_id") or "").strip()
                    decision = str(raw.get("decision") or "").strip().upper()
                    if pid and decision in _FIVE_WAY:
                        by_id[pid] = raw
                for item in batch:
                    pid = item["pair"]["pair_id"]
                    raw = by_id.get(pid)
                    if raw is None:
                        resolved[pid] = fallback[pid]["five_way_row"]
                        audit.append({"pair_id": pid, "status": "fallback_higher_confidence", **meta})
                        continue
                    selected = raw["decision"]
                    # Preserve evidence from the matching directed source proposal;
                    # if NONE, use an empty evidence record.
                    candidates = [item["proposal_a"], item["proposal_b"]]
                    matching = next((x for x in candidates if x["five_way_row"]["decision"] == selected), None)
                    if selected == "NONE":
                        final = {"pair_id": pid, "decision": "NONE", "evidence_sentence_ids": [],
                                 "evidence_text": "", "reason": str(raw.get("reason") or "Conflict adjudicator chose NONE."),
                                 "confidence": _clip_confidence(raw.get("confidence"))}
                    elif matching is not None:
                        final = dict(matching["five_way_row"])
                        final["reason"] = f"Conflict adjudication: {raw.get('reason', final.get('reason', ''))}"
                        final["confidence"] = _clip_confidence(raw.get("confidence"), final.get("confidence", 0.5))
                    else:
                        final = fallback[pid]["five_way_row"]
                        final = dict(final)
                        final["reason"] = "Conflict adjudicator returned a direction/class not supported by either source proposal; kept higher-confidence proposal."
                    resolved[pid] = final
                    audit.append({"pair_id": pid, "status": "adjudicated", "decision": final["decision"], **meta})
            except Exception as exc:
                for item in batch:
                    pid = item["pair"]["pair_id"]
                    resolved[pid] = fallback[pid]["five_way_row"]
                    audit.append({"pair_id": pid, "status": "error_fallback_higher_confidence",
                                  "error": f"{type(exc).__name__}: {exc}"})
        return resolved, audit

    def _run(self, state: v15.PipelineState) -> v15.PipelineState:
        expressions = list(state.linguistic_expressions)
        if self.max_expressions is not None:
            expressions = expressions[: self.max_expressions]
        self._failed_details = []
        self._decisions = []
        self._prompt_audit = []
        self._batch_audit = []

        enriched_events: list[v15.EnrichedExpression] = []
        pairs: list[dict[str, Any]] = []
        for index, expr in enumerate(expressions):
            if expr.label == "event_mention":
                enriched_events.append(self._process_expression_conservative(expr, state))
            elif self._is_relation(expr):
                pairs.append(self._pair_record(expr, index))

        events, targets_by_source = self._source_target_index(pairs)
        key_to_ref, ref_to_key = self._event_refs(events)
        raw_directed: dict[tuple[str, str], dict[str, Any]] = {}
        call_audit: list[dict[str, Any]] = []
        parse_audit: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=min(self.source_workers, max(1, len(events)))) as executor:
            futures = {
                executor.submit(
                    self._classify_one_source,
                    source_key=source_key,
                    target_keys=sorted(targets_by_source[source_key], key=v15._event_sort_key),
                    key_to_ref=key_to_ref,
                    ref_to_key=ref_to_key,
                    state=state,
                ): source_key
                for source_key in events
            }
            for future in as_completed(futures):
                source_key = futures[future]
                try:
                    _, decisions, calls, parsed_rows = future.result()
                except Exception as exc:
                    calls = [{"source_event_id": key_to_ref[source_key], "source_event_key": source_key,
                              "phase": "source_worker", "status": "error",
                              "error": f"{type(exc).__name__}: {exc}"}]
                    parsed_rows = []
                    decisions = {}
                call_audit.extend(calls)
                parse_audit.extend(parsed_rows)
                for row in decisions.values():
                    raw_directed[(row["source_event_key"], row["target_event_key"])] = row

        pair_by_key = {v15._pair_key(p["event_a_key"], p["event_b_key"]): p for p in pairs}
        final_by_pair: dict[str, dict[str, Any]] = {}
        conflicts: list[dict[str, Any]] = []

        for pair in pairs:
            a, b = pair["event_a_key"], pair["event_b_key"]
            row_a = raw_directed.get((a, b), {"relation": "NONE", "confidence": 0.0,
                                                  "evidence_sentence_ids": [], "evidence_text": "",
                                                  "reason": "Missing A-source decision."})
            row_b = raw_directed.get((b, a), {"relation": "NONE", "confidence": 0.0,
                                                  "evidence_sentence_ids": [], "evidence_text": "",
                                                  "reason": "Missing B-source decision."})
            proposals = []
            for source_key, target_key, row in ((a, b, row_a), (b, a, row_b)):
                relation = row.get("relation", "NONE")
                if relation == "NONE":
                    continue
                five_way = _source_to_five_way(relation, source_key, target_key, pair)
                proposals.append({
                    **row,
                    "five_way_row": {
                        "pair_id": pair["pair_id"], "decision": five_way,
                        "evidence_sentence_ids": row.get("evidence_sentence_ids", []),
                        "evidence_text": row.get("evidence_text", ""),
                        "reason": f"Source-centric {key_to_ref[source_key]}: {row.get('reason', '')}",
                        "confidence": row.get("confidence", 0.0),
                    },
                })
            if not proposals:
                final_by_pair[pair["pair_id"]] = {
                    "pair_id": pair["pair_id"], "decision": "NONE",
                    "evidence_sentence_ids": [], "evidence_text": "",
                    "reason": "Both source-centric directions returned NONE.",
                    "confidence": max(row_a.get("confidence", 0.0), row_b.get("confidence", 0.0)),
                }
            elif len(proposals) == 1:
                final_by_pair[pair["pair_id"]] = proposals[0]["five_way_row"]
            else:
                proposals.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
                if proposals[0].get("confidence", 0.0) - proposals[1].get("confidence", 0.0) >= self.conflict_confidence_margin:
                    chosen = dict(proposals[0]["five_way_row"])
                    chosen["reason"] = f"Direction conflict resolved by confidence margin: {chosen.get('reason', '')}"
                    final_by_pair[pair["pair_id"]] = chosen
                else:
                    # Identify A-source/B-source proposals explicitly for the compact adjudicator.
                    prop_a = next(x for x in proposals if x.get("source_event_key") == a)
                    prop_b = next(x for x in proposals if x.get("source_event_key") == b)
                    conflicts.append({"pair": pair, "proposal_a": prop_a, "proposal_b": prop_b})

        conflict_resolved, conflict_audit = self._adjudicate_conflicts(conflicts, state)
        final_by_pair.update(conflict_resolved)

        enriched_relations: list[v15.EnrichedExpression] = []
        for pair in pairs:
            decision = final_by_pair.get(pair["pair_id"], {
                "pair_id": pair["pair_id"], "decision": "NONE",
                "evidence_sentence_ids": [], "evidence_text": "",
                "reason": "No final source-centric decision.", "confidence": 0.0,
                "filtered_invalid_output": True,
            })
            enriched = self._make_enriched_relation(pair, decision, state)
            if enriched is not None:
                enriched_relations.append(enriched)

        # One controlled relation per unordered pair after conflict resolution.
        dedup_relations: dict[tuple[str, str], v15.EnrichedExpression] = {}
        for enriched in enriched_relations:
            triple = v15._parse_relation_instance(enriched.base_expression.text)
            if triple is None:
                continue
            source, _, target = triple
            dedup_relations.setdefault(v15._pair_key(source, target), enriched)

        state.enriched_expressions = [*enriched_events, *dedup_relations.values()]
        self._save_failed_expressions(state)
        write_json(self.source_decision_log_path, sorted(
            raw_directed.values(),
            key=lambda x: (str(x.get("source_event_id", "")), str(x.get("target_event_id", ""))),
        ))
        write_json(self.source_call_audit_path, sorted(
            call_audit, key=lambda x: (str(x.get("source_event_id", "")), str(x.get("phase", "")))
        ))
        write_json(self.conflict_log_path, {
            "conflict_count": len(conflicts), "audit": conflict_audit,
            "resolved_pair_ids": sorted(conflict_resolved),
        })
        write_json(self.decision_log_path, sorted(
            self._decisions, key=lambda row: (str(row.get("pair_id", "")), str(row.get("phase", "")))
        ))
        write_json(self.compact_prompt_log_path, self._prompt_audit)
        # Reuse the familiar batch-audit path for source-call observability.
        write_json(self.compact_prompt_log_path.parent / "layer02_batch_audit.json", call_audit)
        write_json(self.closure_pair_log_path, [])
        write_json(self.compact_prompt_log_path.parent / "layer02_source_parse_audit.json", parse_audit)
        # Explicit empty legacy logs avoid stale files after reruns.
        write_json(self.none_review_log_path, {"selected_pairs": [], "batches": [], "disabled": True})
        write_json(self.verification_log_path, {"input_positive_pairs": [], "batches": [], "disabled": True})

        state.log(
            f"[{self.name}] EventStoryLine v1.6 source-centric relations; events={len(enriched_events)}; "
            f"unordered_pairs={len(pairs)}; source_calls={len(events)}; "
            f"accepted_relations={len(dedup_relations)}; direction_conflicts={len(conflicts)}; "
            "none_review=false; positive_verifier=false; candidate_closure=false"
        )
        return state


def build_pipeline(
    *,
    backends: dict[str, Any],
    rag_adapter: Any,
    profile_config: dict[str, Any],
    relation_catalog_path: str | Path,
    chunk_size: int,
    run_dir: str | Path,
    workers: int = 16,
    verbose: bool = True,
):
    pipeline = _V15_BUILD_PIPELINE(
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
    l2_cfg = v15._layer_cfg(profile_config, "layer02_candidate_enrichment")
    retry_default = int((profile_config.get("orchestration") or {}).get("retry_failed_calls", 1))
    sleep_default = float((profile_config.get("orchestration") or {}).get("retry_sleep_seconds", 1.0))
    pipeline.layers[2] = EventStoryLineSourceCentricCandidateEnrichmentLayer(
        backends["layer02"],
        wikipedia_source=v15.OfflineWikipediaSource(),
        wikidata_source=v15.OfflineWikidataSource(),
        web_search_source=v15.OfflineWebSearchSource(),
        relation_catalog_path=relation_catalog_path,
        decision_log_path=run_dir / "run_logs/layer02_relation_decisions.json",
        compact_prompt_log_path=run_dir / "run_logs/layer02_compact_prompt_audit.json",
        batch_cache_dir=run_dir / "run_logs/layer02_batch_cache",
        closure_pair_log_path=run_dir / "run_logs/layer02_closure_pair_pool.json",
        source_decision_log_path=run_dir / "run_logs/layer02_source_centric_decisions.json",
        source_call_audit_path=run_dir / "run_logs/layer02_source_centric_call_audit.json",
        conflict_log_path=run_dir / "run_logs/layer02_direction_conflicts.json",
        source_workers=int(l2_cfg.get("source_workers", min(workers, 8))),
        missing_target_retry=bool(l2_cfg.get("missing_target_retry", True)),
        conflict_adjudication_enabled=bool(l2_cfg.get("conflict_adjudication_enabled", True)),
        conflict_batch_size=int(l2_cfg.get("conflict_batch_size", 8)),
        conflict_confidence_margin=float(l2_cfg.get("conflict_confidence_margin", 0.15)),
        pair_batch_size=1,
        pair_batch_workers=1,
        context_window_sentences=int(l2_cfg.get("context_window_sentences", 1)),
        use_ontology_evidence=bool(l2_cfg.get("use_ontology_evidence", False)),
        closure_enabled=False,
        closure_max_pairs=0,
        positive_verifier_enabled=False,
        none_review_enabled=False,
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


def run_native_pipeline(**kwargs: Any):
    """Run v1.5's stable execution shell with v1.6 Layer 2 injected."""
    original_build = v15.build_pipeline
    original_make_backend = v15._make_backend

    def tagged_backend(**backend_kwargs: Any):
        tag = str(backend_kwargs.get("layer_tag") or "")
        if tag.startswith("layer02_"):
            backend_kwargs["layer_tag"] = "layer02_source_centric_relations_v1_6"
        elif tag.startswith("layer01_"):
            backend_kwargs["layer_tag"] = "layer01_atomic_event_inventory_v1_6"
        return _V15_MAKE_BACKEND(**backend_kwargs)

    v15.build_pipeline = build_pipeline
    v15._make_backend = tagged_backend
    try:
        final_state = v15.run_native_pipeline(**kwargs)
    finally:
        v15.build_pipeline = original_build
        v15._make_backend = original_make_backend

    run_dir = Path(kwargs["run_dir"]).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        manifest.update({
            "experiment_version": "1.6",
            "layer02_strategy": "source_centric_directed_target_checklist",
            "layer02_source_fixed_direction": True,
            "layer02_source_calls_parallel": True,
            "layer02_missing_target_retry": True,
            "layer02_broad_none_review": False,
            "layer02_positive_verifier": False,
            "layer02_conflict_adjudication_only": True,
            "layer02_evidence_format_repair_non_destructive": True,
            "layer02_candidate_closure": False,
        })
        write_json(manifest_path, manifest)
    fingerprint_path = run_dir / "analysis_input_fingerprint.json"
    if fingerprint_path.is_file():
        fingerprint = read_json(fingerprint_path)
        fingerprint["experiment_version"] = "1.6"
        fingerprint["relation_strategy"] = "source_centric_directed_target_checklist"
        write_json(fingerprint_path, fingerprint)
    return final_state


__all__ = [
    "RELATION_IDS", "analyze_run", "gold_event_index", "indexed_token_table",
    "load_layer_states", "project_event_label", "read_json", "read_jsonl",
    "run_native_pipeline", "seed_ontology_summary", "build_pipeline",
    "EventStoryLineSourceCentricCandidateEnrichmentLayer",
]

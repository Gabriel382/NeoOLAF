"""EventStoryLine native NeoOLAF one-document experiment v1.7.

v1.7 freezes the strong v1.5/v1.6 atomic event inventory and exhaustive
candidate-pair coverage, but recalibrates relation learning around EventStoryLine
PLOT_LINK orientation:

* only text-forward source->target directions are sent to the classifier;
* one compact global plot-role analysis provides SETUP/PIVOT/AFTERMATH context;
* each source is classified in small target batches (default 6) rather than one
  18-target prompt;
* PRECONDITION/FALLING_ACTION are framed as rising-vs-falling storyline roles,
  not naive physical causality;
* direct-causality requirements are relaxed: a documented local storyline
  continuation/elaboration may count even without an explicit causal connective;
* NONE is still available, but is no longer preferred merely because the link is
  narrative rather than physically causal;
* lightweight deterministic label calibration can correct PRECONDITION bias when
  plot-role/evidence signals strongly indicate FALLING_ACTION;
* reverse-direction conflicts disappear by construction;
* cache I/O remains short-path and non-critical through v1.6.

No file under src/neoolaf is modified.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import json
import math
import time

import eventstoryline_native_ablation_v1_6 as v16

v15 = v16.v15
RELATION_IDS = v16.RELATION_IDS
analyze_run = v16.analyze_run
gold_event_index = v16.gold_event_index
indexed_token_table = v16.indexed_token_table
load_layer_states = v16.load_layer_states
project_event_label = v16.project_event_label
read_json = v16.read_json
read_jsonl = v16.read_jsonl
seed_ontology_summary = v16.seed_ontology_summary
write_json = v16.write_json
write_csv_rows = v16.write_csv_rows
state_counts = v16.state_counts
parse_event_key = v16.parse_event_key
resolve_source_centric_cache_dir = v16.resolve_source_centric_cache_dir

_V15_MAKE_BACKEND = v16._V15_MAKE_BACKEND
_V15_BUILD_PIPELINE = v16._V15_BUILD_PIPELINE
_ALLOWED = {"PRECONDITION", "FALLING_ACTION", "NONE"}


def _clip(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
    except Exception:
        x = default
    if math.isnan(x) or math.isinf(x):
        x = default
    return max(0.0, min(1.0, x))


def _position(event_key: str) -> tuple[int, int, int]:
    p = parse_event_key(event_key) or {}
    try:
        return int(p.get("sent_id", 10**9)), int(p.get("start", 10**9)), int(p.get("end", 10**9))
    except Exception:
        return (10**9, 10**9, 10**9)


def _event_trigger(event_key: str) -> str:
    p = parse_event_key(event_key) or {}
    return str(p.get("trigger") or event_key)


def _sent_id(event_key: str) -> int:
    return _position(event_key)[0]


def _normalize_role(value: Any) -> str:
    raw = str(value or "UNCERTAIN").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "RISING": "SETUP", "PRECONDITION": "SETUP", "CONDITION": "SETUP",
        "CLIMAX": "PIVOT", "CENTRAL": "PIVOT", "MAIN": "PIVOT",
        "FALLING": "AFTERMATH", "OUTCOME": "AFTERMATH", "DOWNSTREAM": "AFTERMATH",
        "OTHER": "BACKGROUND", "IRRELEVANT": "BACKGROUND",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in {"SETUP", "PIVOT", "AFTERMATH", "BACKGROUND", "UNCERTAIN"} else "UNCERTAIN"


def _norm_relation(value: Any) -> str | None:
    return v16._normalize_source_relation(value)


class EventStoryLinePlotCalibratedLayer(v16.EventStoryLineSourceCentricCandidateEnrichmentLayer):
    """Forward-only, plot-role calibrated source-centric PLOT_LINK classifier."""

    def __init__(
        self,
        *args: Any,
        target_batch_size: int = 6,
        forward_textual_direction_only: bool = True,
        plot_role_analysis_enabled: bool = True,
        label_calibration_enabled: bool = True,
        relax_narrative_links: bool = True,
        plot_role_log_path: str | Path | None = None,
        direction_filter_log_path: str | Path | None = None,
        calibration_log_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["conflict_adjudication_enabled"] = False
        super().__init__(*args, **kwargs)
        base = self.compact_prompt_log_path.parent
        self.target_batch_size = max(1, int(target_batch_size))
        self.forward_textual_direction_only = bool(forward_textual_direction_only)
        self.plot_role_analysis_enabled = bool(plot_role_analysis_enabled)
        self.label_calibration_enabled = bool(label_calibration_enabled)
        self.relax_narrative_links = bool(relax_narrative_links)
        self.plot_role_log_path = Path(plot_role_log_path or base / "layer02_plot_roles.json")
        self.direction_filter_log_path = Path(direction_filter_log_path or base / "layer02_forward_direction_filter.json")
        self.calibration_log_path = Path(calibration_log_path or base / "layer02_relation_calibration.json")
        self._plot_roles: dict[str, dict[str, Any]] = {}

    def _document_rows(self, state: v15.PipelineState, sentence_ids: set[int] | None = None) -> list[dict[str, Any]]:
        profile = state.profile_config or {}
        sentences = profile.get("_input_sentences", []) or []
        tokens = profile.get("_input_tokens", []) or []
        rows = []
        ids = range(len(sentences)) if sentence_ids is None else sorted(x for x in sentence_ids if 0 <= x < len(sentences))
        for sid in ids:
            rows.append({
                "sentence_id": sid,
                "sentence": sentences[sid],
                "tokens": [str(x) for x in (tokens[sid] if sid < len(tokens) else [])],
            })
        return rows

    def _plot_role_prompt(self, events: list[str], key_to_ref: dict[str, str], state: v15.PipelineState) -> list[dict[str, str]]:
        inventory = [{
            "event_id": key_to_ref[k],
            "event_key": k,
            "trigger": _event_trigger(k),
            "sentence_id": _sent_id(k),
        } for k in events]
        system = """
You analyze EventStoryLine narrative structure before relation classification.
Assign each event one coarse PLOT role based only on the document text:
- SETUP: circumstance/condition/motivation that prepares later story events;
- PIVOT: central or focal event/development in the local storyline;
- AFTERMATH: consequence, continuation, elaboration, follow-up, or later detail;
- BACKGROUND: event mention that is peripheral to the main plot branch;
- UNCERTAIN: insufficient evidence.

Also assign a short branch_id (B1, B2, ...) so events in the same local storyline
can be recognized. Repeated/headline-body realizations may share a branch.
This is NOT relation prediction and must not invent new events.
Return every event exactly once. JSON only:
{"events":[{"event_id":"E0000","role":"PIVOT","branch_id":"B1","reason":"...","confidence":0.8}]}
""".strip()
        user = f"""
Events:
{v15._json_block(inventory, 22000)}

Document:
{v15._json_block(self._document_rows(state), 26000)}

JSON only.
""".strip()
        with self._prompt_lock:
            self._prompt_audit.append({
                "phase": "plot_role_analysis", "prompt_kind": "global_plot_roles",
                "event_count": len(events), "system_chars": len(system), "user_chars": len(user),
            })
        return [{"role":"system","content":system},{"role":"user","content":user}]

    def _analyze_plot_roles(self, events: list[str], key_to_ref: dict[str, str], ref_to_key: dict[str, str], state: v15.PipelineState) -> dict[str, dict[str, Any]]:
        fallback = {k: {"event_id": key_to_ref[k], "event_key": k, "role": "UNCERTAIN", "branch_id": "", "reason": "fallback", "confidence": 0.0} for k in events}
        if not self.plot_role_analysis_enabled or not events:
            write_json(self.plot_role_log_path, list(fallback.values()))
            return fallback
        try:
            parsed, meta = self._chat_custom(self._plot_role_prompt(events, key_to_ref, state), state)
            rows = parsed.get("events") if isinstance(parsed, dict) else []
            rows = rows if isinstance(rows, list) else []
            out = dict(fallback)
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                ref = str(raw.get("event_id") or "").strip()
                key = ref_to_key.get(ref)
                if key is None:
                    continue
                out[key] = {
                    "event_id": ref, "event_key": key,
                    "role": _normalize_role(raw.get("role")),
                    "branch_id": str(raw.get("branch_id") or "").strip(),
                    "reason": str(raw.get("reason") or "").strip(),
                    "confidence": _clip(raw.get("confidence")),
                    "meta_status": meta.get("status", "ok"),
                }
            write_json(self.plot_role_log_path, [out[k] for k in events])
            return out
        except Exception as exc:
            rows = list(fallback.values())
            for x in rows: x["error"] = f"{type(exc).__name__}: {exc}"
            write_json(self.plot_role_log_path, rows)
            return fallback

    def _local_context_ids(self, source_key: str, target_keys: list[str], state: v15.PipelineState) -> set[int]:
        profile = state.profile_config or {}
        sentences = profile.get("_input_sentences", []) or []
        ids: set[int] = set()
        # Headline/title is normally sentence 1 after a URL in normalized EventStoryLine.
        if len(sentences) > 1:
            ids.add(1)
        for key in [source_key, *target_keys]:
            sid = _sent_id(key)
            for x in (sid - 1, sid, sid + 1):
                if 0 <= x < len(sentences): ids.add(x)
        return ids

    def _source_prompt(self, *, source_key: str, target_keys: list[str], key_to_ref: dict[str, str], state: v15.PipelineState, phase: str) -> list[dict[str, str]]:
        profile = state.profile_config or {}
        task = profile.get("_input_task_guidance", {}) or {}
        sentences = profile.get("_input_sentences", []) or []
        source_sid = _sent_id(source_key)
        src_role = self._plot_roles.get(source_key, {})
        targets = []
        for key in target_keys:
            sid = _sent_id(key)
            targets.append({
                "target_event_id": key_to_ref[key], "target_event_key": key,
                "target_trigger": _event_trigger(key), "target_sentence_id": sid,
                "target_sentence": sentences[sid] if 0 <= sid < len(sentences) else "",
                "plot_role": self._plot_roles.get(key, {}),
            })
        recovery_note = "Return every listed target exactly once; this is a missing-ID recovery." if "recovery" in phase else "Return every listed target exactly once."
        relax_note = "When the text clearly places both events on the same storyline branch, a PLOT_LINK may exist without an explicit causal connective; do not overuse NONE." if self.relax_narrative_links else "Require explicit support."
        system = f"""
You are NeoOLAF Layer 2 for EventStoryLine PLOT_LINK classification.

The SOURCE is fixed and ALWAYS occurs earlier in the normalized story text than
all listed TARGETS. Classify ONLY this forward narrative direction. NEVER invert
SOURCE/TARGET merely because the later event is the physical cause, reason, or
precondition of the earlier-mentioned event.

Choose exactly one label for each SOURCE -> TARGET pair:
- PRECONDITION (rising action): SOURCE is setup/circumstance/condition/motivation
  that moves the storyline toward TARGET.
- FALLING_ACTION (falling action): TARGET is a later continuation, consequence,
  aftermath, elaboration, follow-up, explanatory detail, or downstream plot
  development from SOURCE. This includes cases where the later-mentioned TARGET
  semantically explains SOURCE; keep the forward storyline direction.
- NONE: the two events are not connected as a meaningful PLOT_LINK.

IMPORTANT calibration:
1. Narrative direction outranks naive causal inversion.
2. Do not turn a plausible FALLING_ACTION into reverse PRECONDITION.
3. PRECONDITION is for rising/setup movement toward TARGET; FALLING_ACTION is for
   what the story develops after/from SOURCE, including later explanatory detail.
4. {relax_note}
5. Chronology/shared topic alone is still insufficient. Do not invent automatic
   transitive closure, but a non-adjacent relation is allowed when the document
   itself presents the two events as one storyline branch.

Synthetic orientation examples:
- "Officials warned of suspension unless training began." suspension warning ->
  training began = PRECONDITION (setup motivates target).
- "The bridge collapsed, followed by an inspection." collapse -> inspection =
  FALLING_ACTION.
- "She faced expulsion for cheating." If SOURCE=expulsion and TARGET=cheating in
  this normalized mention order, keep SOURCE->TARGET and choose FALLING_ACTION;
  do NOT flip it to cheating PRECONDITION expulsion merely because cheating is
  the real-world reason.
- A headline announces an event and the body later elaborates/details it: the
  forward headline->body link is usually FALLING_ACTION unless the source is
  explicitly a setup required for the target.
- Unrelated rainfall elsewhere in the article = NONE.

Use plot roles as soft evidence, not hard truth:
SETUP->PIVOT tends PRECONDITION; PIVOT/central->AFTERMATH tends FALLING_ACTION.
{recovery_note}

Return JSON only:
{{"source_event_id":"E0000","decisions":[
 {{"target_event_id":"E0001","relation":"FALLING_ACTION","evidence_sentence_ids":[1,2],"evidence_text":"...","reason":"...","confidence":0.84}}
]}}
""".strip()
        user = f"""
Controlled relation definitions:
{v15._json_block(task.get('relation_specs') or [], 6500)}

Fixed source:
{v15._json_block({'source_event_id': key_to_ref[source_key], 'source_event_key': source_key, 'source_trigger': _event_trigger(source_key), 'source_sentence_id': source_sid, 'source_sentence': sentences[source_sid] if 0 <= source_sid < len(sentences) else '', 'plot_role': src_role}, 5500)}

Forward targets (small batch):
{v15._json_block(targets, 18000)}

Local authoritative story context:
{v15._json_block(self._document_rows(state, self._local_context_ids(source_key, target_keys, state)), 22000)}

Phase: {phase}. JSON only.
""".strip()
        with self._prompt_lock:
            self._prompt_audit.append({
                "phase": phase, "source_event_id": key_to_ref[source_key], "source_event_key": source_key,
                "target_count": len(target_keys), "target_ids": [key_to_ref[x] for x in target_keys],
                "system_chars": len(system), "user_chars": len(user),
                "prompt_kind": "v17_forward_plot_calibrated_batch",
            })
        return [{"role":"system","content":system},{"role":"user","content":user}]

    def _classify_one_source(self, *, source_key: str, target_keys: list[str], key_to_ref: dict[str, str], ref_to_key: dict[str, str], state: v15.PipelineState):
        final: dict[str, dict[str, Any]] = {}
        call_audit: list[dict[str, Any]] = []
        parse_audit: list[dict[str, Any]] = []
        ordered = sorted(target_keys, key=v15._event_sort_key)
        batches = [ordered[i:i+self.target_batch_size] for i in range(0, len(ordered), self.target_batch_size)]

        def invoke(keys: list[str], phase: str):
            started = time.time()
            try:
                parsed, meta = self._chat_custom(self._source_prompt(source_key=source_key, target_keys=keys, key_to_ref=key_to_ref, state=state, phase=phase), state)
                accepted, missing, rows = self._parse_source_response(parsed=parsed, source_key=source_key, target_keys=keys, key_to_ref=key_to_ref, ref_to_key=ref_to_key, state=state, phase=phase)
                parse_audit.extend(rows)
                call_audit.append({
                    "source_event_id": key_to_ref[source_key], "source_event_key": source_key,
                    "phase": phase, "requested_targets": len(keys), "resolved_targets": len(accepted),
                    "missing_targets": missing, "elapsed_seconds": time.time()-started, **meta,
                })
                return accepted, missing
            except Exception as exc:
                refs = [key_to_ref[x] for x in keys]
                call_audit.append({
                    "source_event_id": key_to_ref[source_key], "source_event_key": source_key,
                    "phase": phase, "requested_targets": len(keys), "resolved_targets": 0,
                    "missing_targets": refs, "elapsed_seconds": time.time()-started,
                    "status":"error", "error":f"{type(exc).__name__}: {exc}",
                })
                return {}, refs

        missing_all: list[str] = []
        for bi, batch in enumerate(batches):
            accepted, missing = invoke(batch, f"primary_forward_batch_{bi:02d}")
            final.update(accepted)
            missing_all.extend(missing)

        if missing_all and self.missing_target_retry:
            missing_keys = [ref_to_key[r] for r in dict.fromkeys(missing_all) if r in ref_to_key]
            for ri in range(0, len(missing_keys), self.target_batch_size):
                chunk = missing_keys[ri:ri+self.target_batch_size]
                accepted, _ = invoke(chunk, f"missing_target_recovery_{ri//self.target_batch_size:02d}")
                final.update(accepted)

        for key in ordered:
            ref = key_to_ref[key]
            if ref not in final:
                final[ref] = {
                    "source_event_id": key_to_ref[source_key], "source_event_key": source_key,
                    "target_event_id": ref, "target_event_key": key, "relation":"NONE",
                    "evidence_sentence_ids":[], "evidence_text":"",
                    "reason":"No valid decision after focused recovery.", "confidence":0.0,
                    "phase":"fallback_none", "evidence_repaired":False, "status":"fallback_none",
                }
        return source_key, final, call_audit, parse_audit

    def _calibrate(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        result = dict(row)
        before = str(result.get("relation") or "NONE")
        after = before
        src_role = _normalize_role(self._plot_roles.get(result.get("source_event_key", ""), {}).get("role"))
        tgt_role = _normalize_role(self._plot_roles.get(result.get("target_event_key", ""), {}).get("role"))
        evidence = (str(result.get("reason") or "") + " " + str(result.get("evidence_text") or "")).lower()
        setup_cues = ("enable", "require", "allow", "condition", "motivat", "needed", "necessary", "in order", "setup", "prepare")
        fall_cues = ("follow", "aftermath", "consequence", "later", "downstream", "elaborat", "continu", "result", "after", "detail", "explain")
        reason = "no_change"
        if self.label_calibration_enabled and before in {"PRECONDITION","FALLING_ACTION"}:
            has_setup = any(x in evidence for x in setup_cues)
            has_fall = any(x in evidence for x in fall_cues)
            if before == "PRECONDITION":
                # Correct the dominant v1.6 bias only when storyline-role/evidence signals agree.
                if ((src_role in {"PIVOT","AFTERMATH"} and tgt_role in {"AFTERMATH","SETUP"}) or (has_fall and not has_setup)):
                    after = "FALLING_ACTION"
                    reason = "plot_role_or_downstream_evidence_corrected_precondition_bias"
            elif before == "FALLING_ACTION":
                if src_role == "SETUP" and tgt_role == "PIVOT" and has_setup and not has_fall:
                    after = "PRECONDITION"
                    reason = "strong_setup_role_corrected_falling_action"
        result["relation"] = after
        result["pre_calibration_relation"] = before
        result["calibration_reason"] = reason
        return result, {
            "source_event_key": result.get("source_event_key"), "target_event_key": result.get("target_event_key"),
            "source_role": src_role, "target_role": tgt_role, "before": before, "after": after,
            "reason": reason, "confidence": result.get("confidence", 0.0),
        }

    def _run(self, state: v15.PipelineState) -> v15.PipelineState:
        expressions = list(state.linguistic_expressions)
        self._failed_details, self._decisions, self._prompt_audit, self._batch_audit = [], [], [], []
        enriched_events: list[v15.EnrichedExpression] = []
        pairs: list[dict[str, Any]] = []
        for index, expr in enumerate(expressions):
            if expr.label == "event_mention":
                enriched_events.append(self._process_expression_conservative(expr, state))
            elif self._is_relation(expr):
                pairs.append(self._pair_record(expr, index))

        all_event_keys = sorted({k for p in pairs for k in (p["event_a_key"], p["event_b_key"])}, key=v15._event_sort_key)
        key_to_ref, ref_to_key = self._event_refs(all_event_keys)
        self._plot_roles = self._analyze_plot_roles(all_event_keys, key_to_ref, ref_to_key, state)

        targets: dict[str, set[str]] = {}
        direction_audit: list[dict[str, Any]] = []
        for pair in pairs:
            a, b = pair["event_a_key"], pair["event_b_key"]
            if _position(a) <= _position(b): earlier, later = a, b
            else: earlier, later = b, a
            targets.setdefault(earlier, set()).add(later)
            direction_audit.append({
                "pair_id": pair["pair_id"], "earlier_event": earlier, "later_event": later,
                "classified_direction": f"{earlier} -> {later}",
                "reverse_direction_skipped": self.forward_textual_direction_only,
            })
        write_json(self.direction_filter_log_path, direction_audit)

        raw: dict[tuple[str,str], dict[str,Any]] = {}
        call_audit: list[dict[str,Any]] = []
        parse_audit: list[dict[str,Any]] = []
        source_keys = sorted(targets, key=v15._event_sort_key)
        with ThreadPoolExecutor(max_workers=min(self.source_workers, max(1, len(source_keys)))) as ex:
            futs = {
                ex.submit(self._classify_one_source, source_key=s, target_keys=sorted(targets[s], key=v15._event_sort_key), key_to_ref=key_to_ref, ref_to_key=ref_to_key, state=state): s
                for s in source_keys
            }
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    _, decisions, calls, parsed_rows = fut.result()
                except Exception as exc:
                    decisions, parsed_rows = {}, []
                    calls = [{"source_event_id":key_to_ref[s],"source_event_key":s,"phase":"source_worker","status":"error","error":f"{type(exc).__name__}: {exc}"}]
                call_audit.extend(calls); parse_audit.extend(parsed_rows)
                for row in decisions.values():
                    raw[(row["source_event_key"], row["target_event_key"])] = row

        calibration_rows: list[dict[str,Any]] = []
        # Calibrate exactly once, after all source workers have completed.
        for key, row in list(raw.items()):
            calibrated, audit = self._calibrate(row)
            raw[key] = calibrated
            calibration_rows.append(audit)
        write_json(self.calibration_log_path, calibration_rows)

        final_by_pair: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            a, b = pair["event_a_key"], pair["event_b_key"]
            if _position(a) <= _position(b): earlier, later = a, b
            else: earlier, later = b, a
            row = raw.get((earlier, later), {"relation":"NONE","confidence":0.0,"evidence_sentence_ids":[],"evidence_text":"","reason":"Missing forward decision."})
            relation = row.get("relation", "NONE")
            if relation == "NONE":
                final = {"pair_id":pair["pair_id"],"decision":"NONE","evidence_sentence_ids":[],"evidence_text":"","reason":row.get("reason","NONE"),"confidence":row.get("confidence",0.0)}
            else:
                decision = v16._source_to_five_way(relation, earlier, later, pair)
                final = {"pair_id":pair["pair_id"],"decision":decision,
                         "evidence_sentence_ids":row.get("evidence_sentence_ids",[]),
                         "evidence_text":row.get("evidence_text",""),
                         "reason":f"v1.7 forward plot-calibrated: {row.get('reason','')}",
                         "confidence":row.get("confidence",0.0)}
            final_by_pair[pair["pair_id"]] = final

        enriched_relations: list[v15.EnrichedExpression] = []
        for pair in pairs:
            enriched = self._make_enriched_relation(pair, final_by_pair[pair["pair_id"]], state)
            if enriched is not None: enriched_relations.append(enriched)
        dedup: dict[tuple[str,str], v15.EnrichedExpression] = {}
        for e in enriched_relations:
            triple = v15._parse_relation_instance(e.base_expression.text)
            if triple is None: continue
            dedup.setdefault(v15._pair_key(triple[0], triple[2]), e)
        state.enriched_expressions = [*enriched_events, *dedup.values()]
        self._save_failed_expressions(state)
        write_json(self.source_decision_log_path, sorted(raw.values(), key=lambda x:(_position(x.get("source_event_key","")), _position(x.get("target_event_key","")))))
        write_json(self.source_call_audit_path, call_audit)
        write_json(self.conflict_log_path, {"conflict_count":0,"audit":[],"disabled":"forward_textual_direction_only"})
        write_json(self.decision_log_path, sorted(self._decisions, key=lambda r:str(r.get("pair_id",""))))
        write_json(self.compact_prompt_log_path, self._prompt_audit)
        write_json(self.compact_prompt_log_path.parent / "layer02_batch_audit.json", call_audit)
        write_json(self.compact_prompt_log_path.parent / "layer02_source_parse_audit.json", parse_audit)
        write_json(self.closure_pair_log_path, [])
        write_json(self.none_review_log_path, {"disabled":True,"reason":"primary prompt relaxed for narrative PLOT_LINKs"})
        write_json(self.verification_log_path, {"disabled":True,"reason":"v1.7 uses plot-role calibration only"})
        state.log(f"[{self.name}] EventStoryLine v1.7 forward plot-calibrated; events={len(enriched_events)}; unordered_pairs={len(pairs)}; forward_decisions={len(raw)}; accepted_relations={len(dedup)}; target_batch_size={self.target_batch_size}; reverse_directions_skipped=true")
        return state


def build_pipeline(*, backends: dict[str, Any], rag_adapter: Any, profile_config: dict[str, Any], relation_catalog_path: str | Path, chunk_size: int, run_dir: str | Path, workers: int = 16, verbose: bool = True):
    pipeline = _V15_BUILD_PIPELINE(backends=backends, rag_adapter=rag_adapter, profile_config=profile_config, relation_catalog_path=relation_catalog_path, chunk_size=chunk_size, run_dir=run_dir, workers=workers, verbose=verbose)
    run_dir = Path(run_dir)
    l2 = v15._layer_cfg(profile_config, "layer02_candidate_enrichment")
    retry_default = int((profile_config.get("orchestration") or {}).get("retry_failed_calls", 1))
    sleep_default = float((profile_config.get("orchestration") or {}).get("retry_sleep_seconds", 1.0))
    cache_dir = resolve_source_centric_cache_dir() / "v17"
    pipeline.layers[2] = EventStoryLinePlotCalibratedLayer(
        backends["layer02"], wikipedia_source=v15.OfflineWikipediaSource(), wikidata_source=v15.OfflineWikidataSource(), web_search_source=v15.OfflineWebSearchSource(),
        relation_catalog_path=relation_catalog_path,
        decision_log_path=run_dir/"run_logs/layer02_relation_decisions.json",
        compact_prompt_log_path=run_dir/"run_logs/layer02_compact_prompt_audit.json",
        batch_cache_dir=cache_dir,
        closure_pair_log_path=run_dir/"run_logs/layer02_closure_pair_pool.json",
        source_decision_log_path=run_dir/"run_logs/layer02_source_centric_decisions.json",
        source_call_audit_path=run_dir/"run_logs/layer02_source_centric_call_audit.json",
        conflict_log_path=run_dir/"run_logs/layer02_direction_conflicts.json",
        plot_role_log_path=run_dir/"run_logs/layer02_plot_roles.json",
        direction_filter_log_path=run_dir/"run_logs/layer02_forward_direction_filter.json",
        calibration_log_path=run_dir/"run_logs/layer02_relation_calibration.json",
        source_workers=int(l2.get("source_workers", min(workers, 8))),
        target_batch_size=int(l2.get("target_batch_size", 6)),
        missing_target_retry=bool(l2.get("missing_target_retry", True)),
        forward_textual_direction_only=bool(l2.get("forward_textual_direction_only", True)),
        plot_role_analysis_enabled=bool(l2.get("plot_role_analysis_enabled", True)),
        label_calibration_enabled=bool(l2.get("label_calibration_enabled", True)),
        relax_narrative_links=bool(l2.get("relax_narrative_links", True)),
        conflict_adjudication_enabled=False,
        pair_batch_size=1, pair_batch_workers=1,
        context_window_sentences=int(l2.get("context_window_sentences",1)),
        use_ontology_evidence=False, closure_enabled=False, closure_max_pairs=0,
        positive_verifier_enabled=False, none_review_enabled=False,
        max_expressions=None, use_web_search=False, save_intermediate=True, verbose=verbose,
        rag_adapter=rag_adapter, max_concurrency=int(l2.get("max_concurrency",workers)),
        retry_failed_calls=int(l2.get("retry_failed_calls",retry_default)), retry_sleep_seconds=sleep_default,
    )
    return pipeline


def run_native_pipeline(**kwargs: Any):
    original_build = v15.build_pipeline
    original_make = v15._make_backend
    def tagged_backend(**backend_kwargs: Any):
        tag = str(backend_kwargs.get("layer_tag") or "")
        if tag.startswith("layer02_"): backend_kwargs["layer_tag"] = "layer02_forward_plot_relations_v1_7"
        elif tag.startswith("layer01_"): backend_kwargs["layer_tag"] = "layer01_atomic_event_inventory_v1_7"
        return _V15_MAKE_BACKEND(**backend_kwargs)
    v15.build_pipeline = build_pipeline
    v15._make_backend = tagged_backend
    try:
        state = v15.run_native_pipeline(**kwargs)
    finally:
        v15.build_pipeline = original_build; v15._make_backend = original_make
    run_dir = Path(kwargs["run_dir"]).resolve()
    manifest_path = run_dir/"run_manifest.json"
    if manifest_path.is_file():
        m = read_json(manifest_path); m.update({
            "experiment_version":"1.7",
            "layer02_strategy":"forward_textual_plot_role_calibrated_source_batches",
            "layer02_forward_textual_direction_only":True,
            "layer02_target_batch_size":6,
            "layer02_plot_role_analysis":True,
            "layer02_relax_narrative_links":True,
            "layer02_label_calibration":True,
            "layer02_reverse_conflicts":"eliminated_by_construction",
        }); write_json(manifest_path,m)
    fp = run_dir/"analysis_input_fingerprint.json"
    if fp.is_file():
        x=read_json(fp); x["experiment_version"]="1.7"; x["relation_strategy"]="forward_textual_plot_role_calibrated_source_batches"; write_json(fp,x)
    return state


__all__ = ["RELATION_IDS","analyze_run","gold_event_index","indexed_token_table","load_layer_states","project_event_label","read_json","read_jsonl","run_native_pipeline","seed_ontology_summary","build_pipeline","resolve_source_centric_cache_dir","EventStoryLinePlotCalibratedLayer"]

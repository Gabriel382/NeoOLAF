from __future__ import annotations

"""Unified4 v1.3 experiment-side adapters.

Small overlay on ``ragtree_dataset_adapters_v1``.  Nothing under ``src/neoolaf``
is changed.  The patch targets the three observed failure modes from the paid
one-document runs:

* FinCausal: exact proposition-span recall was too low.
* MAVEN-ERE: event-trigger/endpoint recall was too low.
* CausalBank: precision was perfect but the dense normalized graph was severely
  under-generated.

Gold annotations remain unavailable to every pipeline layer.  Gold is used only
by the post-L12 evaluator (and by the explicitly named offline preview helper).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations, permutations
from pathlib import Path
from typing import Any
import json
import os
import re
import shutil
import sys
import time
import traceback

import ragtree_dataset_adapters_v1 as v1

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.enriched_expression import EnrichedExpression
from neoolaf.domain.linguistic_expression import Evidence, LinguisticExpression
from neoolaf.domain.user_guidance import UserGuidance
from neoolaf.ontology.loader import SeedOntologyLoader
from neoolaf.profiles.profile_loader import load_document_profile

from experiments.methods.run_neoolaf import load_user_guidance

# Re-export stable public helpers used by the notebook.
DATASET_DISPLAY = v1.DATASET_DISPLAY
RELATION_IDS = v1.RELATION_IDS
read_json = v1.read_json
read_jsonl = v1.read_jsonl
write_json = v1.write_json
append_jsonl = v1.append_jsonl
state_counts = v1.state_counts
seed_ontology_summary = v1.seed_ontology_summary
triples_from_state = v1.triples_from_state
evaluate_state = v1.evaluate_state
causalbank_event_id = v1.causalbank_event_id
build_document = v1.build_document
choose_chunk_size = v1.choose_chunk_size

_PATCH_VERSION = "unified4-v1.3"


def _role_from_justification(expr: LinguisticExpression) -> str:
    m = re.search(r"(?:role_hint|causal_role_hint)=([A-Z_]+)", str(expr.justification or ""), re.I)
    role = m.group(1).upper() if m else "UNKNOWN"
    return role if role in {"CAUSE", "EFFECT", "UNKNOWN"} else "UNKNOWN"


def _looks_quantified(text: str) -> bool:
    return bool(re.search(
        r"(?:[$€£¥]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:%|percent|million|billion|thousand|people|euros?|dollars?|shekels?|points?|bps)\b)",
        str(text or ""), re.I,
    ))


def _fincausal_deterministic_spans(state: PipelineState) -> list[dict[str, Any]]:
    """High-recall, source-only span proposals.

    The normalized FinCausal files sometimes keep several natural-language
    sentences inside one ``sentences[0]`` item.  Gold fact chunks commonly align
    with sentence boundaries, dash-separated clauses, or discourse-marker
    clauses.  These candidates are generated only from visible source tokens.
    """
    tokens_by_sent = (state.profile_config or {}).get("_input_tokens", []) or []
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    skip_right_markers = {
        "indicating", "indicate", "which", "therefore", "thus", "hence",
        "consequently", "because", "since", "thereby", "so",
    }

    def add(sid: int, start: int, end: int, reason: str) -> None:
        if not (0 <= sid < len(tokens_by_sent) and 0 <= start < end <= len(tokens_by_sent[sid])):
            return
        key = (sid, start, end)
        if key in seen:
            return
        seen.add(key)
        text = " ".join(str(x) for x in tokens_by_sent[sid][start:end]).strip()
        if len(v1._norm(text).split()) < 3:
            return
        out.append({
            "text": text,
            "fact_kind": "QUANTIFIED_FACT" if _looks_quantified(text) else "FACT",
            "causal_role_hint": "UNKNOWN",
            "sent_id": sid,
            "token_start": start,
            "token_end": end,
            "reason": reason,
            "source": "deterministic_source_boundary",
        })

    for sid, raw_tokens in enumerate(tokens_by_sent):
        toks = [str(x) for x in raw_tokens]
        n = len(toks)
        if not n:
            continue
        add(sid, 0, n, "whole visible normalized sentence/chunk")

        # Natural sentence boundaries embedded in one normalized sentence.
        sentence_bounds = [0]
        for i, tok in enumerate(toks):
            # Avoid treating common abbreviations as a sentence break.
            low = tok.lower()
            abbreviation = bool(re.fullmatch(r"(?:[a-z]\.){2,}", low)) or low in {"u.s.", "u.k.", "inc.", "ltd.", "mr.", "mrs.", "dr."}
            if not abbreviation and re.search(r"[.!?][\"')\]]?$", tok):
                sentence_bounds.append(i + 1)
        sentence_bounds.append(n)
        sentence_bounds = sorted(set(x for x in sentence_bounds if 0 <= x <= n))
        for a, b in zip(sentence_bounds, sentence_bounds[1:]):
            add(sid, a, b, "embedded sentence boundary")

        # Dash/semicolon separators often delimit FinCausal fact chunks.
        for i, tok in enumerate(toks):
            if tok in {"-", "--", "—", "–", ";"}:
                add(sid, 0, i, "left side of causal/discourse separator")
                right = i + 1
                if right < n and re.sub(r"[^A-Za-z]", "", toks[right]).lower() in skip_right_markers:
                    right += 1
                add(sid, right, n, "right side of causal/discourse separator")

        # Long comma-bound subordinate clause followed by a fresh capitalized
        # clause (e.g. "while ... year, New York ...").
        for i, tok in enumerate(toks[:-1]):
            if tok.endswith(",") and i >= 8 and toks[i + 1][:1].isupper():
                add(sid, 0, i + 1, "long comma-delimited left fact")
                add(sid, i + 1, n, "long comma-delimited right fact")

    return out


class FinCausalLayer1V13(v1._PromptedLayer1):
    """High-recall FinCausal proposition inventory with role hints.

    One LLM call proposes annotation-compatible spans; deterministic source-only
    span candidates are unioned with it.  Relation creation still happens in
    Layer 2.
    """

    def _run(self, state: PipelineState) -> PipelineState:
        rows = v1._safe_token_rows(state)
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        messages = [
            {"role": "system", "content": """
You are NeoOLAF Layer 1 for FinCausal benchmark endpoint extraction.
Extract ALL source-grounded proposition/fact spans that could be a CAUSE or EFFECT.
Do not output relations. Do not reduce facts to atomic event triggers.

FinCausal annotation behavior to follow:
- endpoints are complete fact/proposition chunks, often whole clauses or sentences;
- an EFFECT is normally a quantified/measurable financial fact;
- a CAUSE may be quantified or qualitative;
- cause/effect semantic direction is independent of textual order;
- pure causal connectives (because, indicating, therefore, resulting in/from, etc.)
  normally stay OUTSIDE the fact span when they only connect two facts;
- preserve source wording. Long/noisy source prefixes are not to be cleaned away
  merely because they look like boilerplate.

Return a HIGH-RECALL inventory. It is better to include an additional plausible
fact span than to omit a benchmark endpoint. For each span also give only a HINT
about its likely role; Layer 2 makes the relation decision.

Synthetic schema examples:
1) "Operating costs rose 12% - because energy prices increased."
   EFFECT = "Operating costs rose 12%"; CAUSE = "energy prices increased."
2) "Revenue fell 8% - indicating customers were leaving."
   EFFECT = "Revenue fell 8%"; CAUSE = "customers were leaving."
3) "Low taxes attracted retirees. Net inflows reached $4 billion."
   CAUSE may be the qualitative first fact; EFFECT may be the quantified second fact.

JSON only:
{"facts":[{"text":"exact source span","fact_kind":"FACT|QUANTIFIED_FACT",
"causal_role_hint":"CAUSE|EFFECT|UNKNOWN","sent_id":0,"token_start":0,
"token_end":4,"reason":"..."}]}
Offsets are zero-based/end-exclusive. If exact offsets are genuinely uncertain,
set them to null but preserve an exact source substring.
""".strip()},
            {"role": "user", "content": (
                f"Task guidance:\n{v1._json_block(task, 9000)}\n\n"
                f"Indexed source:\n{v1._json_block(rows, 42000)}\n\n"
                f"Document:\n{v1._doc_text(state)}\n\nJSON only."
            )},
        ]
        parsed = self._chat(state, messages, "fact_inventory_v13")
        llm_facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        candidates = list(llm_facts if isinstance(llm_facts, list) else [])
        candidates.extend(_fincausal_deterministic_spans(state))

        doc_text = v1._doc_text(state)
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        expressions: list[LinguisticExpression] = []
        audit: list[dict[str, Any]] = []
        # Deduplicate by normalized source text rather than by the generated span
        # key so LLM and deterministic proposals collapse together.
        by_text: dict[str, dict[str, Any]] = {}
        for raw in candidates:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            norm_text = v1._norm(text)
            if len(norm_text.split()) < 3:
                continue
            role = str(raw.get("causal_role_hint") or raw.get("role_hint") or "UNKNOWN").upper()
            if role not in {"CAUSE", "EFFECT", "UNKNOWN"}:
                role = "UNKNOWN"
            kind = str(raw.get("fact_kind") or ("QUANTIFIED_FACT" if _looks_quantified(text) else "FACT")).upper()
            # Prefer a role-bearing LLM row over an UNKNOWN deterministic duplicate.
            prev = by_text.get(norm_text)
            if prev is None or (str(prev.get("causal_role_hint", "UNKNOWN")).upper() == "UNKNOWN" and role != "UNKNOWN"):
                row = dict(raw)
                row["text"] = text
                row["causal_role_hint"] = role
                row["fact_kind"] = kind
                by_text[norm_text] = row

        for raw in by_text.values():
            text = str(raw["text"]).strip()
            sid = raw.get("sent_id")
            start = raw.get("token_start")
            end = raw.get("token_end")
            exact_token = False
            sid_i = start_i = end_i = -1
            try:
                sid_i, start_i, end_i = int(sid), int(start), int(end)
                if 0 <= sid_i < len(tokens) and 0 <= start_i < end_i <= len(tokens[sid_i]):
                    tok_text = " ".join(str(x) for x in tokens[sid_i][start_i:end_i])
                    exact_token = v1._norm(tok_text) == v1._norm(text)
                else:
                    sid_i = start_i = end_i = -1
            except Exception:
                pass
            exact_text = v1._norm(text) in v1._norm(doc_text)
            if not exact_token and not exact_text:
                audit.append({"text": text, "accepted": False, "reason": "not_grounded_in_source"})
                continue
            digest = sha256(f"{sid_i}:{start_i}:{end_i}:{v1._norm(text)}".encode()).hexdigest()[:10]
            key = f"FSPAN:{digest}::{text}"
            role = str(raw.get("causal_role_hint") or "UNKNOWN").upper()
            kind = str(raw.get("fact_kind") or "FACT").upper()
            evidence = v1._token_evidence(state, sid_i, start_i, end_i) if exact_token else v1._whole_document_evidence(state, text)
            expressions.append(LinguisticExpression(
                expr_id=f"expr_f13_{len(expressions):04d}", text=key, label="fact_span",
                justification=(
                    f"kind={kind}; role_hint={role}; quantified={_looks_quantified(text)}; "
                    f"source_text={text}; exact_token={exact_token}; reason={raw.get('reason', '')}"
                ),
                evidence=evidence,
            ))
            audit.append({
                "endpoint": key, "text": text, "kind": kind, "role_hint": role,
                "quantified": _looks_quantified(text), "accepted": True,
                "exact_token": exact_token, "source": raw.get("source", "llm"),
            })

        state.linguistic_expressions = expressions
        v1.write_json(self.audit_path.with_name("layer01_fincausal_fact_inventory_v13.json"), audit)
        state.log(f"[{self.name}] FinCausal v1.3 fact spans={len(expressions)}")
        return state


class FinCausalLayer2V13(v1._TaskLayer2):
    """Recall-oriented relation layer.

    Clear Layer-1 CAUSE/EFFECT role hints are converted deterministically. Every
    remaining unordered pair is still adjudicated by the Layer-2 LLM. This means
    the patch does not depend on role hints being perfect.
    """

    def _run(self, state: PipelineState) -> PipelineState:
        nodes = [x for x in state.linguistic_expressions or [] if x.label == "fact_span"]
        enriched = [v1._node_enriched(x) for x in nodes]
        pairs = list(combinations(nodes, 2))
        pair_rows: list[dict[str, Any]] = []
        deterministic_positive: dict[str, dict[str, Any]] = {}

        for i, (a, b) in enumerate(pairs):
            pid = f"P{i:04d}"
            a_text = a.text.split("::", 1)[-1]
            b_text = b.text.split("::", 1)[-1]
            a_role, b_role = _role_from_justification(a), _role_from_justification(b)
            row = {
                "pair_id": pid,
                "a": a.text, "b": b.text,
                "a_text": a_text, "b_text": b_text,
                "a_role_hint": a_role, "b_role_hint": b_role,
                "a_quantified": _looks_quantified(a_text),
                "b_quantified": _looks_quantified(b_text),
            }
            pair_rows.append(row)
            if a_role == "CAUSE" and b_role == "EFFECT":
                deterministic_positive[pid] = {"pair_id": pid, "decision": "A_CAUSES_B", "reason": "Layer-1 source-grounded CAUSE/EFFECT role hints", "confidence": 0.99, "source": "deterministic_role_hint"}
            elif b_role == "CAUSE" and a_role == "EFFECT":
                deterministic_positive[pid] = {"pair_id": pid, "decision": "B_CAUSES_A", "reason": "Layer-1 source-grounded CAUSE/EFFECT role hints", "confidence": 0.99, "source": "deterministic_role_hint"}

        unresolved = [row for row in pair_rows if row["pair_id"] not in deterministic_positive]
        decisions: dict[str, dict[str, Any]] = dict(deterministic_positive)
        batches = self._batches(unresolved)

        def run_batch(batch_index: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            messages = [
                {"role": "system", "content": """
You are NeoOLAF Layer 2 for FinCausal causal relation classification.
For every supplied unordered proposition pair decide A_CAUSES_B, B_CAUSES_A, or NONE.
Semantic direction is independent of textual order. Endpoints are complete facts,
not event triggers. EFFECT is normally quantified/measurable; CAUSE may be qualitative
or quantified. Preserve recall: if the document directly or implicitly presents one
fact as explaining/enabling the other, output the supported direction. Do not reject a
valid relation merely because the causal connective lies outside the endpoint spans.
A document may contain more than one valid cause/effect pair; judge EACH pair independently.

Synthetic example:
A = "Production costs increased 14%."
B = "Energy prices doubled."
Text = "Production costs increased 14% because energy prices doubled."
Decision = B_CAUSES_A.

Return exactly one row per pair_id, JSON only:
{"decisions":[{"pair_id":"P0000","decision":"A_CAUSES_B|B_CAUSES_A|NONE","reason":"...","confidence":0.9}]}
""".strip()},
                {"role": "user", "content": f"Document:\n{v1._doc_text(state)}\n\nPairs:\n{v1._json_block(batch, 42000)}\n\nJSON only."},
            ]
            parsed = self._chat(state, messages, f"v13_batch_{batch_index:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []

        if batches:
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(batches))) as ex:
                futs = {ex.submit(run_batch, i, b): i for i, b in enumerate(batches)}
                for fut in as_completed(futs):
                    try:
                        rows = fut.result()
                    except Exception as exc:
                        self._record({"phase": "pair_batch", "batch": futs[fut], "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    for row in rows if isinstance(rows, list) else []:
                        if isinstance(row, dict) and row.get("pair_id"):
                            decisions[str(row["pair_id"])] = row

        for pair in pair_rows:
            pid = pair["pair_id"]
            row = decisions.get(pid, {"pair_id": pid, "decision": "NONE", "reason": "missing decision", "confidence": 0.0})
            decision = str(row.get("decision") or "NONE").upper()
            self._record({"phase": "final", **pair, **row})
            if decision == "A_CAUSES_B":
                source, target = pair["a"], pair["b"]
            elif decision == "B_CAUSES_A":
                source, target = pair["b"], pair["a"]
            else:
                continue
            enriched.append(v1._relation_enriched(
                expr_id=f"expr_fr13_{pid}", source=source, relation_id="CAUSE", target=target,
                metadata=self.catalog["CAUSE"], state=state, decision_payload=row,
            ))

        v1.write_json(self.decision_log_path.with_name("layer02_fincausal_pair_pool_v13.json"), pair_rows)
        return self._finish(state, enriched)


def _maven_extract_batch_prompt(batch: list[dict[str, Any]], task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": """
You are NeoOLAF Layer 1A for MAVEN-ERE. Extract an EXHAUSTIVE inventory of event
mentions from the supplied indexed sentences. High event-trigger recall is the priority.

An event mention can be a verb, eventive noun, occurrence, state/change, action,
movement, communication, conflict, legal act, creation/destruction, process, etc.
Use the shortest complete eventive trigger span supported by tokens. Do not omit an
event merely because it is not obviously causal: Layer 2 decides causality later.
Do not output relations or causal direction.

Return exact zero-based/end-exclusive offsets and exact trigger text. JSON only:
{"events":[{"sent_id":0,"start":3,"end":4,"trigger":"flooding","event_type":"Event","reason":"..."}]}
""".strip()},
        {"role": "user", "content": f"Task guidance:\n{v1._json_block(task, 7000)}\n\nIndexed sentences:\n{v1._json_block(batch, 42000)}\n\nJSON only."},
    ]


class MavenLayer1V13(v1._PromptedLayer1):
    """Sentence-batched, high-recall MAVEN mention extraction.

    Most MAVEN gold event clusters are singletons. v1.3 therefore keeps each
    recovered mention as an independent event-cluster endpoint during the native
    pipeline instead of risking destructive over-merging. Post-L12 projection maps
    a mention endpoint to its gold coreference cluster by exact span overlap.
    """

    def _run(self, state: PipelineState) -> PipelineState:
        rows = v1._safe_token_rows(state)
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        cfg = v1._layer_cfg(state.profile_config or {}, "layer01_linguistic_expression_extraction")
        sentence_batch_size = max(1, int(cfg.get("sentence_batch_size", 4)))
        max_concurrency = max(1, int(cfg.get("max_concurrency", 4)))
        batches = [rows[i:i + sentence_batch_size] for i in range(0, len(rows), sentence_batch_size)]
        raw_events: list[dict[str, Any]] = []
        audit_calls: list[dict[str, Any]] = []

        def run_batch(idx: int, batch: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
            parsed = self._chat(state, _maven_extract_batch_prompt(batch, task), f"event_batch_{idx:03d}")
            events = parsed.get("events", []) if isinstance(parsed, dict) else []
            return idx, events if isinstance(events, list) else []

        if batches:
            with ThreadPoolExecutor(max_workers=min(max_concurrency, len(batches))) as ex:
                futs = {ex.submit(run_batch, i, b): i for i, b in enumerate(batches)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        _, events = fut.result()
                        raw_events.extend(x for x in events if isinstance(x, dict))
                        audit_calls.append({"batch": idx, "status": "ok", "events": len(events)})
                    except Exception as exc:
                        audit_calls.append({"batch": idx, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

        # One compact coverage review catches mentions omitted by sentence-local calls.
        existing = []
        for raw in raw_events:
            try:
                existing.append([int(raw.get("sent_id")), int(raw.get("start")), int(raw.get("end"))])
            except Exception:
                pass
        review_messages = [
            {"role": "system", "content": """
You are NeoOLAF Layer 1B MAVEN-ERE coverage review. The first pass already proposed
some event triggers. Return ONLY additional event mentions that were missed.
Be recall-oriented. Include eventive verbs, event nouns, states/changes, actions and
occurrences even if they are not causal. Do not duplicate supplied spans. Exact token
offsets only. No relations. JSON only: {"events":[...]}
""".strip()},
            {"role": "user", "content": (
                f"Existing spans [sent,start,end]:\n{v1._json_block(existing, 12000)}\n\n"
                f"Indexed document:\n{v1._json_block(rows, 50000)}\n\nJSON only."
            )},
        ]
        try:
            parsed = self._chat(state, review_messages, "coverage_review_v13")
            extra = parsed.get("events", []) if isinstance(parsed, dict) else []
            if isinstance(extra, list):
                raw_events.extend(x for x in extra if isinstance(x, dict))
                audit_calls.append({"phase": "coverage_review", "status": "ok", "events": len(extra)})
        except Exception as exc:
            audit_calls.append({"phase": "coverage_review", "status": "error", "error": f"{type(exc).__name__}: {exc}"})

        by_span: dict[tuple[int, int, int], dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        for raw in raw_events:
            try:
                sid, start, end = int(raw.get("sent_id")), int(raw.get("start")), int(raw.get("end"))
            except Exception:
                rejected.append({"raw": raw, "reason": "missing_or_invalid_offset"})
                continue
            if not (0 <= sid < len(tokens) and 0 <= start < end <= len(tokens[sid])):
                rejected.append({"raw": raw, "reason": "out_of_range_offset"})
                continue
            trigger = " ".join(str(x) for x in tokens[sid][start:end]).strip()
            if not v1._norm(trigger):
                continue
            by_span.setdefault((sid, start, end), {
                "sent_id": sid, "start": start, "end": end, "trigger": trigger,
                "event_type": str(raw.get("event_type") or "Event"),
                "reason": str(raw.get("reason") or ""),
            })

        expressions: list[LinguisticExpression] = []
        mention_audit: list[dict[str, Any]] = []
        for sid, start, end in sorted(by_span):
            row = by_span[(sid, start, end)]
            span_key = f"S{sid}[{start}:{end}]::{row['trigger']}"
            digest = sha256(span_key.encode()).hexdigest()[:10]
            key = f"MCL:{digest}::{span_key}"
            expressions.append(LinguisticExpression(
                expr_id=f"expr_m13_{len(expressions):04d}", text=key, label="event_cluster",
                justification=f"singleton_high_recall_event; event_type={row['event_type']}; source-only exact span",
                evidence=v1._token_evidence(state, sid, start, end),
            ))
            mention_audit.append({"cluster_key": key, **row})

        state.linguistic_expressions = expressions
        v1.write_json(self.audit_path.with_name("layer01_maven_calls_v13.json"), audit_calls)
        v1.write_json(self.audit_path.with_name("layer01_maven_mentions_v13.json"), mention_audit)
        v1.write_json(self.audit_path.with_name("layer01_maven_rejected_v13.json"), rejected)
        state.log(f"[{self.name}] MAVEN v1.3 mentions/singleton-clusters={len(expressions)}")
        return state


class MavenLayer2V13(v1._TaskLayer2):
    """Recall-oriented direct five-way causal classification.

    v1.1 first filtered LINK/NONE and then classified subtype. The paid run showed
    that this compounded false negatives. v1.3 classifies existence, direction and
    subtype jointly, with a small review pass for local NONE decisions.
    """

    def _run(self, state: PipelineState) -> PipelineState:
        nodes = [x for x in state.linguistic_expressions or [] if x.label == "event_cluster"]
        enriched = [v1._node_enriched(x) for x in nodes]
        keys = [x.text for x in nodes]
        key_to_id = {k: f"C{i:03d}" for i, k in enumerate(keys)}
        id_to_key = {v: k for k, v in key_to_id.items()}
        cfg = v1._layer_cfg(state.profile_config or {}, "layer02_candidate_enrichment")
        local_distance = max(0, int(cfg.get("local_sentence_distance", 4)))

        pair_set: set[tuple[str, str]] = set()
        for a, b in combinations(keys, 2):
            if v1._cluster_min_distance(a, b) <= local_distance:
                pair_set.add(tuple(sorted((a, b))))

        # Long-range candidate proposal remains label-free and direction-free.
        inventory = [{"cluster_id": key_to_id[k], "positions": v1._maven_cluster_positions(k), "key": k} for k in keys]
        if len(keys) > 1:
            msgs = [
                {"role": "system", "content": """
You are NeoOLAF MAVEN-ERE long-range candidate generation. Propose unordered event
pairs that MIGHT have a CAUSE or PRECONDITION relation. This stage is high-recall:
include plausible causal dependence across distant sentences. Do not classify subtype
or direction. Use only supplied cluster IDs. Return at most 180 pairs. JSON only:
{"pairs":[["C000","C017"], ...]}
""".strip()},
                {"role": "user", "content": f"Clusters:\n{v1._json_block(inventory, 42000)}\n\nDocument:\n{v1._json_block(v1._safe_token_rows(state), 52000)}"},
            ]
            try:
                parsed = self._chat(state, msgs, "long_range_candidates_v13")
                for item in parsed.get("pairs", []) if isinstance(parsed, dict) else []:
                    if isinstance(item, list) and len(item) == 2 and item[0] in id_to_key and item[1] in id_to_key and item[0] != item[1]:
                        pair_set.add(tuple(sorted((id_to_key[item[0]], id_to_key[item[1]]))))
            except Exception as exc:
                self._record({"phase": "long_range_candidates", "status": "error", "error": str(exc)})

        pair_rows = [
            {"pair_id": f"P{i:04d}", "a": a, "b": b, "a_id": key_to_id[a], "b_id": key_to_id[b], "min_sentence_distance": v1._cluster_min_distance(a, b)}
            for i, (a, b) in enumerate(sorted(pair_set))
        ]
        decisions: dict[str, dict[str, Any]] = {}
        batches = self._batches(pair_rows)

        def classify(idx: int, batch: list[dict[str, Any]], review: bool = False) -> list[dict[str, Any]]:
            review_note = "This is a recall-rescue review of prior NONE decisions; promote a pair when causal dependence is reasonably supported." if review else "Judge every pair independently."
            msgs = [
                {"role": "system", "content": f"""
You are NeoOLAF Layer 2 for MAVEN-ERE causal relation extraction.
Allowed decisions exactly: A_CAUSE_B, B_CAUSE_A, A_PRECONDITION_B,
B_PRECONDITION_A, NONE.

CAUSE: given the source event, the target is effectively inevitable.
PRECONDITION: the target would not have happened without the source, but the source
alone need not guarantee the target. PRECONDITION is substantially more common in
MAVEN-ERE; do not force a 50/50 prior.

Infer event-time/semantic direction; NEVER use textual mention order as direction.
A valid relation may be implicit or cross-sentence. Reject mere chronology,
co-occurrence, shared topic, or coreference, but be recall-oriented when the document
supports causal dependence. {review_note}

Return exactly one decision per pair_id, JSON only:
{{"decisions":[{{"pair_id":"P0000","decision":"A_PRECONDITION_B","reason":"...","confidence":0.8}}]}}
""".strip()},
                {"role": "user", "content": f"Document:\n{v1._json_block(v1._safe_token_rows(state), 52000)}\n\nPairs:\n{v1._json_block(batch, 42000)}"},
            ]
            parsed = self._chat(state, msgs, f"{'review' if review else 'class'}_{idx:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []

        if batches:
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(batches))) as ex:
                futs = {ex.submit(classify, i, b, False): i for i, b in enumerate(batches)}
                for fut in as_completed(futs):
                    try:
                        rows = fut.result()
                    except Exception as exc:
                        self._record({"phase": "class", "batch": futs[fut], "status": "error", "error": str(exc)})
                        continue
                    for row in rows if isinstance(rows, list) else []:
                        if isinstance(row, dict) and row.get("pair_id"):
                            decisions[str(row["pair_id"])] = row

        # Review local NONEs only: cheap compared with re-running all pairs and useful
        # against the exact failure mode observed in v1.1.
        by_id = {x["pair_id"]: x for x in pair_rows}
        review_rows = [
            pair for pair in pair_rows
            if pair["min_sentence_distance"] <= 1
            and str(decisions.get(pair["pair_id"], {}).get("decision", "NONE")).upper() == "NONE"
        ]
        review_batches = self._batches(review_rows)
        if review_batches:
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(review_batches))) as ex:
                futs = {ex.submit(classify, i, b, True): i for i, b in enumerate(review_batches)}
                for fut in as_completed(futs):
                    try:
                        rows = fut.result()
                    except Exception as exc:
                        self._record({"phase": "review", "batch": futs[fut], "status": "error", "error": str(exc)})
                        continue
                    for row in rows if isinstance(rows, list) else []:
                        if not isinstance(row, dict) or not row.get("pair_id"):
                            continue
                        decision = str(row.get("decision") or "NONE").upper()
                        if decision != "NONE":
                            row = dict(row); row["source"] = "local_none_recall_review"
                            decisions[str(row["pair_id"])] = row

        for pair in pair_rows:
            pid = pair["pair_id"]
            row = decisions.get(pid, {"pair_id": pid, "decision": "NONE", "reason": "missing decision", "confidence": 0.0})
            decision = str(row.get("decision") or "NONE").upper()
            self._record({"phase": "final", **pair, **row})
            if decision == "A_CAUSE_B": source, rel, target = pair["a"], "CAUSE", pair["b"]
            elif decision == "B_CAUSE_A": source, rel, target = pair["b"], "CAUSE", pair["a"]
            elif decision == "A_PRECONDITION_B": source, rel, target = pair["a"], "PRECONDITION", pair["b"]
            elif decision == "B_PRECONDITION_A": source, rel, target = pair["b"], "PRECONDITION", pair["a"]
            else: continue
            enriched.append(v1._relation_enriched(
                expr_id=f"expr_mr13_{pid}", source=source, relation_id=rel, target=target,
                metadata=self.catalog[rel], state=state, decision_payload=row,
            ))

        v1.write_json(self.decision_log_path.with_name("layer02_maven_candidate_pool_v13.json"), pair_rows)
        return self._finish(state, enriched)


class CausalBankLayer2V13(v1.CausalBankLayer2):
    """Deterministic dense graph generation from independently recovered nodes.

    The paid v1.2 result had P=1.0 and R=0.146 because pair classification was far
    too conservative for the normalized CausalBank graph.  v1.3 removes pair-level
    LLM pruning: all ordered non-self pairs among source-derived lexical nodes receive
    the record's visible causal family.  No gold inventory is consulted.
    """

    def _run(self, state: PipelineState) -> PipelineState:
        nodes = [x for x in state.linguistic_expressions or [] if x.label == "lemma_node"]
        enriched: list[EnrichedExpression] = [v1._node_enriched(x) for x in nodes]
        relation_id, relation_row = self._record_relation(state)
        relation_row = dict(relation_row)
        relation_row["v1_3_dense_mode"] = True
        self._record(relation_row)

        node_labels = [x.text for x in nodes]
        directed_pairs = list(permutations(node_labels, 2))
        for i, (source, target) in enumerate(directed_pairs):
            row = {
                "decision": "A_TO_B",
                "reason": "v1.3 normalized CausalBank dense ordered-pair coverage",
                "confidence": 1.0,
                "source": "deterministic_dense_normalized_contract",
                "record_relation_family": relation_id,
            }
            enriched.append(v1._relation_enriched(
                expr_id=f"expr_cbr13_{i:05d}", source=source, relation_id=relation_id, target=target,
                metadata=self.catalog[relation_id], state=state, decision_payload=row,
            ))

        v1.write_json(self.decision_log_path.with_name("layer02_causalbank_dense_pool_v13.json"), {
            "record_relation_family": relation_id,
            "record_type": (state.profile_config or {}).get("_input_record_type", ""),
            "node_count": len(node_labels),
            "directed_pair_count": len(directed_pairs),
            "pair_llm_calls": 0,
            "gold_used": False,
        })
        self._record({"phase": "dense_summary", "relation": relation_id, "node_count": len(node_labels), "directed_pair_count": len(directed_pairs), "pair_llm_calls": 0})
        return self._finish(state, enriched)


def build_pipeline(
    *, dataset_key: str, backends: dict[str, Any], rag_adapter: Any,
    profile_config: dict[str, Any], relation_catalog_path: str | Path, chunk_size: int,
    run_dir: str | Path, workers: int = 8, verbose: bool = True,
) -> Pipeline:
    # Reuse the already-working v1 construction for native L0/L3-L12, then replace
    # only the experiment-side L1/L2 adapters for v1.3.
    pipeline = v1.build_pipeline(
        dataset_key=dataset_key, backends=backends, rag_adapter=rag_adapter,
        profile_config=profile_config, relation_catalog_path=relation_catalog_path,
        chunk_size=chunk_size, run_dir=run_dir, workers=workers, verbose=verbose,
    )
    run_dir = Path(run_dir)
    l1_cfg = v1._layer_cfg(profile_config, "layer01_linguistic_expression_extraction")
    l2_cfg = v1._layer_cfg(profile_config, "layer02_candidate_enrichment")
    if dataset_key == "fincausal":
        pipeline.layers[1] = FinCausalLayer1V13(
            backends["layer01"], dataset_key=dataset_key,
            audit_path=run_dir / "run_logs/layer01_calls_v13.json",
            temperature=0.0, save_intermediate=True, verbose=verbose,
        )
        pipeline.layers[2] = FinCausalLayer2V13(
            backends["layer02"], dataset_key=dataset_key,
            relation_catalog_path=relation_catalog_path,
            decision_log_path=run_dir / "run_logs/layer02_relation_decisions_v13.json",
            batch_size=int(l2_cfg.get("pair_batch_size", 18)),
            max_concurrency=int(l2_cfg.get("max_concurrency", min(workers, 4))),
            save_intermediate=True, verbose=verbose,
        )
    elif dataset_key == "maven_ere":
        pipeline.layers[1] = MavenLayer1V13(
            backends["layer01"], dataset_key=dataset_key,
            audit_path=run_dir / "run_logs/layer01_calls_v13.json",
            temperature=0.0, save_intermediate=True, verbose=verbose,
        )
        pipeline.layers[2] = MavenLayer2V13(
            backends["layer02"], dataset_key=dataset_key,
            relation_catalog_path=relation_catalog_path,
            decision_log_path=run_dir / "run_logs/layer02_relation_decisions_v13.json",
            batch_size=int(l2_cfg.get("pair_batch_size", 30)),
            max_concurrency=int(l2_cfg.get("max_concurrency", min(workers, 4))),
            save_intermediate=True, verbose=verbose,
        )
    elif dataset_key == "causalbank":
        pipeline.layers[2] = CausalBankLayer2V13(
            backends["layer02"], dataset_key=dataset_key,
            relation_catalog_path=relation_catalog_path,
            decision_log_path=run_dir / "run_logs/layer02_relation_decisions_v13.json",
            batch_size=int(l2_cfg.get("pair_batch_size", 40)),
            max_concurrency=int(l2_cfg.get("max_concurrency", min(workers, 4))),
            save_intermediate=True, verbose=verbose,
        )
    return pipeline


def _prepare_guidance(guidance_path: str | Path, run_dir: Path) -> tuple[Path, UserGuidance]:
    return v1._prepare_guidance(guidance_path, run_dir)


def run_native_pipeline_record(
    *, dataset_key: str, project_root: str | Path, input_jsonl: str | Path, ontology_path: str | Path,
    profile_path: str | Path, guidance_path: str | Path, task_guidance_path: str | Path,
    relation_catalog_path: str | Path, relation_aliases_path: str | Path, run_dir: str | Path,
    model_name: str, api_key: str, host: str = "https://openrouter.ai/api/v1", workers: int = 8,
    max_tokens: int = 8192, request_timeout: int = 180, reasoning_effort: str = "minimal",
    verbose: bool = True, clean_run_dir: bool = True,
) -> PipelineState:
    if dataset_key not in DATASET_DISPLAY:
        raise ValueError(f"Unsupported dataset {dataset_key}")
    project_root = Path(project_root).resolve()
    input_jsonl = Path(input_jsonl).resolve(); ontology_path = Path(ontology_path).resolve()
    profile_path = Path(profile_path).resolve(); guidance_path = Path(guidance_path).resolve()
    task_guidance_path = Path(task_guidance_path).resolve(); relation_catalog_path = Path(relation_catalog_path).resolve()
    run_dir = Path(run_dir).resolve()
    if clean_run_dir and run_dir.exists(): shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "run_logs"; logs_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_jsonl)
    if len(records) != 1:
        raise ValueError(f"Expected exactly one sanitized record, got {len(records)}")
    record = records[0]
    forbidden = {"entities", "relations", "pred_relations", "ontology_links"} & set(record)
    if forbidden:
        raise ValueError(f"Pipeline input contains forbidden gold/precomputed fields: {sorted(forbidden)}")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    profile = load_document_profile(profile_path=profile_path)
    profile_dict = profile.to_state_dict()
    task_guidance = read_json(task_guidance_path)
    profile_dict["_input_task_guidance"] = task_guidance
    profile_dict["_input_sentences"] = record.get("sentences") or []
    profile_dict["_input_tokens"] = record.get("tokens") or []
    profile_dict["_input_record_type"] = record.get("type") or ""
    profile_dict["_input_title"] = record.get("title") or ""
    normalized_guidance_path, guidance = _prepare_guidance(guidance_path, run_dir)

    seed_ontology = SeedOntologyLoader().load(str(ontology_path))
    if not seed_ontology.classes_by_uri and not seed_ontology.properties_by_uri:
        raise RuntimeError(f"Seed ontology loaded no classes/properties: {ontology_path}")

    max_safe = int((profile_dict.get("chunking") or {}).get("max_safe_chunk_chars", 26000))
    chunk_size = choose_chunk_size(record.get("text", ""), max_safe)
    logger = v1.SharedCallLogger(logs_dir)
    backends = {
        "layer01": v1._make_backend(logger=logger, layer_tag=f"{dataset_key}_layer01_v13", model_host=host, api_key=api_key, cfg=v1._layer_cfg(profile_dict, "layer01_linguistic_expression_extraction"), fallback_max_tokens=max_tokens, fallback_timeout=request_timeout, reasoning_effort=reasoning_effort),
        "layer02": v1._make_backend(logger=logger, layer_tag=f"{dataset_key}_layer02_v13", model_host=host, api_key=api_key, cfg=v1._layer_cfg(profile_dict, "layer02_candidate_enrichment"), fallback_max_tokens=4096, fallback_timeout=request_timeout, reasoning_effort=reasoning_effort),
        "layer04": v1._make_backend(logger=logger, layer_tag=f"{dataset_key}_layer04_v13", model_host=host, api_key=api_key, cfg=v1._layer_cfg(profile_dict, "layer04_candidate_relation_extraction"), fallback_max_tokens=384, fallback_timeout=60, reasoning_effort=reasoning_effort),
        "other": v1._make_backend(logger=logger, layer_tag=f"{dataset_key}_other_v13", model_host=host, api_key=api_key, cfg={}, fallback_max_tokens=768, fallback_timeout=90, reasoning_effort=reasoning_effort),
    }
    rag_adapter = v1.v15.v13.v2.OntologyOnlyRAGAdapter(
        seed_ontology, log_path=logs_dir / "ontology_retrieval.jsonl",
        top_k=int((profile_dict.get("rag") or {}).get("top_k", 4)),
        query_expansions=(profile_dict.get("rag") or {}).get("query_expansions", {}) or {},
    )
    pipeline = build_pipeline(
        dataset_key=dataset_key, backends=backends, rag_adapter=rag_adapter,
        profile_config=profile_dict, relation_catalog_path=relation_catalog_path,
        chunk_size=chunk_size, run_dir=run_dir, workers=workers, verbose=verbose,
    )
    state = PipelineState(
        document=build_document(record, input_jsonl), llm_model=model_name,
        user_guidance=guidance, seed_ontology=seed_ontology, artifact_dir=str(run_dir),
        profile_name=profile.name, profile_config=profile_dict,
    )
    runner = Runner(
        pipeline=pipeline, runs_root=str(run_dir.parent), verbose=verbose,
        max_workers=workers, enable_checkpoints=True, save_chunk_checkpoints=False,
    )
    manifest = {
        "dataset": DATASET_DISPLAY[dataset_key], "document_id": record.get("document_id"),
        "title": record.get("title"), "experiment_version": _PATCH_VERSION,
        "model_name": model_name, "input_has_gold": False,
        "forbidden_input_fields": ["entities", "relations", "pred_relations", "ontology_links"],
        "ontology_path": str(ontology_path), "profile_path": str(profile_path),
        "guidance_path": str(guidance_path), "task_guidance_path": str(task_guidance_path),
        "relation_catalog_path": str(relation_catalog_path), "chunk_size": chunk_size,
        "workers": workers, "gold_projection_after_layer12_only": True,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(run_dir / "input_task_guidance.json", task_guidance)
    write_json(run_dir / "effective_user_guidance.json", asdict(guidance))

    started = time.time()
    console_log = logs_dir / "console.log"
    errors_path = logs_dir / "pipeline_errors.jsonl"
    with console_log.open("w", encoding="utf-8") as handle:
        try:
            with redirect_stdout(v1.Tee(sys.stdout, handle)), redirect_stderr(v1.Tee(sys.stderr, handle)):
                final_state = runner.run(state)
        except Exception as exc:
            append_jsonl(errors_path, {
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            raise
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["elapsed_seconds"] = time.time() - started
    manifest["final_state_counts"] = state_counts(final_state)
    manifest["llm_call_count"] = logger.call_index
    write_json(run_dir / "run_manifest.json", manifest)
    return final_state


def offline_causalbank_dense_projection_preview(gold_record: dict[str, Any]) -> dict[str, Any]:
    """Zero-cost posthoc preview of the deterministic v1.3 CausalBank contract.

    Candidate labels are derived ONLY from visible source tokens using the same
    Layer-1 normalizer. Gold is consulted only afterwards to compute metrics.
    """
    record_type = gold_record.get("type") or ""
    relation_id = v1.CausalBankLayer2._relation_from_record_type(record_type)
    if relation_id not in {"BECAUSE", "THEREFORE"}:
        return {"available": False, "reason": f"record type {record_type!r} requires LLM family fallback"}
    labels: list[str] = []
    seen: set[str] = set()
    token_rows = gold_record.get("tokens") or [re.findall(r"\b[\w'-]+\b", str(gold_record.get("text") or ""))]
    for row in token_rows:
        for token in row:
            for stem in v1._causalbank_stem_candidates(str(token)):
                if stem not in seen:
                    seen.add(stem); labels.append(stem)
    gold_entities = gold_record.get("entities") or {}
    label_to_gold = {label: v1.causalbank_event_id(label) for label in labels if v1.causalbank_event_id(label) in gold_entities}
    projected = {
        (label_to_gold[a], relation_id, label_to_gold[b])
        for a, b in permutations(labels, 2)
        if a in label_to_gold and b in label_to_gold and label_to_gold[a] != label_to_gold[b]
    }
    gold_rel = v1._gold_relation_set(gold_record, {relation_id})
    gold_endpoints = {x for s, _, t in gold_rel for x in (s, t)}
    mapped = set(label_to_gold.values())
    ep_tp = len(mapped & gold_endpoints)
    ep_p = ep_tp / len(mapped) if mapped else 0.0
    ep_r = ep_tp / len(gold_endpoints) if gold_endpoints else 0.0
    ep_f = 2 * ep_p * ep_r / (ep_p + ep_r) if ep_p + ep_r else 0.0
    return {
        "available": True, "record_relation_family": relation_id,
        "source_candidate_labels": len(labels), "mapped_gold_endpoints": len(mapped),
        "relation_metrics": v1._metric(projected, gold_rel),
        "endpoint_metrics": {"pred_mapped_unique": len(mapped), "gold_unique": len(gold_endpoints), "tp": ep_tp, "precision": ep_p, "recall": ep_r, "f1": ep_f},
        "gold_used_only_for_posthoc_metrics": True,
    }


def offline_self_test() -> dict[str, Any]:
    base = v1.offline_self_test()
    assert v1.CausalBankLayer2._relation_from_record_type("resulted_from") == "BECAUSE"
    assert v1.CausalBankLayer2._relation_from_record_type("resulted_in") == "THEREFORE"
    return {
        **base,
        "patch_version": _PATCH_VERSION,
        "fincausal": "deterministic boundary union + role-hint pair fallback",
        "maven_ere": "sentence-batched exhaustive mentions + direct five-way relation classification",
        "causalbank": "deterministic all ordered non-self pairs; no pair-level LLM pruning",
    }

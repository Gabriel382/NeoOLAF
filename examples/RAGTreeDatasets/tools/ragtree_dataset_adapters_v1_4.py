from __future__ import annotations

"""Unified4 v1.4 experiment-side adapters.

Small overlay on ``ragtree_dataset_adapters_v1``.  Nothing under ``src/neoolaf``
is changed.  v1.4 preserves the validated FinCausal/CausalBank behavior and revises only
MAVEN-ERE event recall diagnostics and relation precision:

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

_PATCH_VERSION = "unified4-v1.4"


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
mentions from the supplied indexed sentences. Event-trigger recall is the priority.

MAVEN event mentions include eventive verbs AND eventive nouns/nominalizations,
occurrences, states and state changes, movement, communication, conflict, legal acts,
creation/destruction, starts/ends, decisions, participation, processes and outcomes.
Do NOT restrict extraction to events that look causal. Layer 2 handles causality.

Use the SHORTEST complete trigger span supported by the supplied tokens. Preserve
multi-token lexical triggers when the event is genuinely expressed by the phrase
(e.g. "took place"). Do not return whole clauses when one lexical trigger suffices.
Return exact zero-based/end-exclusive offsets and exact source trigger text.
No relations, no causal direction, JSON only:
{"events":[{"sent_id":0,"start":3,"end":4,"trigger":"flooding","event_type":"Event","reason":"..."}]}
""".strip()},
        {"role": "user", "content": f"Task guidance:\n{v1._json_block(task, 9000)}\n\nIndexed sentences:\n{v1._json_block(batch, 42000)}\n\nJSON only."},
    ]


_MAVEN_RESCUE_STOPWORDS = {
    "a","an","the","and","or","but","if","then","than","that","this","these","those",
    "of","to","in","on","at","by","for","from","with","as","into","onto","over","under",
    "near","before","after","during","while","when","where","who","whom","whose","which",
    "is","are","was","were","be","been","being","am","do","does","did","have","has","had",
    "can","could","may","might","must","shall","should","will","would","not","no","yes",
    "it","its","he","she","they","them","their","his","her","we","our","you","your",
    "i","me","my","there","here","very","more","most","less","least","only","also",
    "however","therefore","thus","hence","so","such","some","any","all","each","every",
    "one","two","three","four","five","six","seven","eight","nine","ten",
}
_MAVEN_PARTICLES = {"up","down","out","off","in","on","over","away","back","through","place","apart","together"}


def _maven_candidate_rows_from_tokens(
    tokens_by_sent: list[list[Any]], existing_spans: set[tuple[int, int, int]] | None = None,
    *, cap: int = 360,
) -> list[dict[str, Any]]:
    """Source-only high-recall lexical candidate pool for MAVEN rescue.

    This intentionally over-generates *candidates*, not accepted event mentions.  A
    second LLM pass validates them.  The pool uses only visible tokens and therefore
    does not expose gold event IDs/types/relations to the pipeline.
    """
    existing_spans = set(existing_spans or set())
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    def add(sid: int, start: int, end: int, kind: str) -> None:
        if len(rows) >= cap or (sid, start, end) in existing_spans or (sid, start, end) in seen:
            return
        if not (0 <= sid < len(tokens_by_sent) and 0 <= start < end <= len(tokens_by_sent[sid])):
            return
        trig = " ".join(str(x) for x in tokens_by_sent[sid][start:end]).strip()
        norm = v1._norm(trig)
        if not norm or not re.search(r"[a-zA-Z]", trig):
            return
        seen.add((sid, start, end))
        left = max(0, start - 5); right = min(len(tokens_by_sent[sid]), end + 5)
        context = " ".join(str(x) for x in tokens_by_sent[sid][left:right])
        rows.append({
            "candidate_id": f"E{len(rows):04d}", "sent_id": sid, "start": start, "end": end,
            "trigger": trig, "candidate_kind": kind, "context": context,
        })

    for sid, raw in enumerate(tokens_by_sent):
        toks = [str(x) for x in raw]
        for i, tok in enumerate(toks):
            low = v1._norm(tok)
            if not low or low in _MAVEN_RESCUE_STOPWORDS:
                continue
            if not re.search(r"[a-zA-Z]", tok) or re.fullmatch(r"[\W_]+", tok):
                continue
            # All non-function lexical heads are eligible for LLM event verification.
            add(sid, i, i + 1, "lexical_head")
            # Common multi-token event predicates/particles, including 'took place'.
            if i + 1 < len(toks) and v1._norm(toks[i + 1]) in _MAVEN_PARTICLES:
                add(sid, i, i + 2, "particle_or_multiword_trigger")
    return rows


def _maven_find_exact_trigger_span(tokens_by_sent: list[list[Any]], trigger: str, sid_hint: int | None = None) -> tuple[int, int, int] | None:
    target = v1._norm(trigger)
    if not target:
        return None
    sids = [sid_hint] if isinstance(sid_hint, int) and 0 <= sid_hint < len(tokens_by_sent) else list(range(len(tokens_by_sent)))
    matches: list[tuple[int, int, int]] = []
    target_words = max(1, len(target.split()))
    for sid in sids:
        toks = [str(x) for x in tokens_by_sent[sid]]
        # Permit a little tokenization slack for punctuation-separated triggers.
        max_len = min(8, target_words + 3)
        for start in range(len(toks)):
            for length in range(1, min(max_len, len(toks) - start) + 1):
                end = start + length
                if v1._norm(" ".join(toks[start:end])) == target:
                    matches.append((sid, start, end))
    return matches[0] if len(matches) == 1 else None


def _maven_normalize_raw_event(raw: dict[str, Any], tokens_by_sent: list[list[Any]]) -> tuple[dict[str, Any] | None, str | None]:
    """Repair an LLM event to an exact visible token span without using gold."""
    trigger = str(raw.get("trigger") or raw.get("text") or "").strip()
    try:
        sid = int(raw.get("sent_id")); start = int(raw.get("start")); end = int(raw.get("end"))
    except Exception:
        sid = start = end = -1
    valid = 0 <= sid < len(tokens_by_sent) and 0 <= start < end <= len(tokens_by_sent[sid])
    if valid:
        source_trigger = " ".join(str(x) for x in tokens_by_sent[sid][start:end]).strip()
        if not trigger or v1._norm(trigger) == v1._norm(source_trigger):
            return ({"sent_id": sid, "start": start, "end": end, "trigger": source_trigger,
                     "event_type": str(raw.get("event_type") or "Event"), "reason": str(raw.get("reason") or "")}, None)
        exact = _maven_find_exact_trigger_span(tokens_by_sent, trigger, sid)
        if exact:
            s, a, b = exact
            return ({"sent_id": s, "start": a, "end": b,
                     "trigger": " ".join(str(x) for x in tokens_by_sent[s][a:b]).strip(),
                     "event_type": str(raw.get("event_type") or "Event"),
                     "reason": f"source-only exact-trigger repair; {raw.get('reason','')}"}, None)
        # Offsets themselves remain source-grounded even if the generated trigger string differs.
        if end - start <= 6:
            return ({"sent_id": sid, "start": start, "end": end, "trigger": source_trigger,
                     "event_type": str(raw.get("event_type") or "Event"),
                     "reason": f"source-token offset retained after trigger mismatch; {raw.get('reason','')}"}, "trigger_mismatch_offset_retained")
    if trigger:
        exact = _maven_find_exact_trigger_span(tokens_by_sent, trigger, sid if sid >= 0 else None)
        if exact:
            s, a, b = exact
            return ({"sent_id": s, "start": a, "end": b,
                     "trigger": " ".join(str(x) for x in tokens_by_sent[s][a:b]).strip(),
                     "event_type": str(raw.get("event_type") or "Event"),
                     "reason": f"source-only exact-trigger repair; {raw.get('reason','')}"}, None)
    return None, "unrepairable_offset_or_trigger"


class MavenLayer1V14(v1._PromptedLayer1):
    """High-recall MAVEN mention extraction with a source-only rescue verifier.

    Pass A is sentence-batched generative extraction. Pass B deterministically builds
    an over-complete lexical source candidate pool and asks the model only which
    candidates are event mentions. Pass C is a final whole-document missed-event
    review. No gold IDs/types/relations are available to any pass.
    """

    def _run(self, state: PipelineState) -> PipelineState:
        rows = v1._safe_token_rows(state)
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        cfg = v1._layer_cfg(state.profile_config or {}, "layer01_linguistic_expression_extraction")
        sentence_batch_size = max(1, int(cfg.get("sentence_batch_size", 3)))
        max_concurrency = max(1, int(cfg.get("max_concurrency", 4)))
        rescue_batch_size = max(20, int(cfg.get("rescue_candidate_batch_size", 70)))
        rescue_cap = max(80, int(cfg.get("source_candidate_cap", 360)))
        batches = [rows[i:i + sentence_batch_size] for i in range(0, len(rows), sentence_batch_size)]
        audit_calls: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        normalized: list[dict[str, Any]] = []

        def normalize_many(items: list[dict[str, Any]], source: str) -> None:
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                event, warning = _maven_normalize_raw_event(raw, tokens)
                if event is None:
                    rejected.append({"source": source, "raw": raw, "reason": warning})
                    continue
                event["source"] = source
                if warning:
                    event["repair_warning"] = warning
                normalized.append(event)

        def run_batch(idx: int, batch: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
            parsed = self._chat(state, _maven_extract_batch_prompt(batch, task), f"event_batch_v14_{idx:03d}")
            events = parsed.get("events", []) if isinstance(parsed, dict) else []
            return idx, events if isinstance(events, list) else []

        if batches:
            with ThreadPoolExecutor(max_workers=min(max_concurrency, len(batches))) as ex:
                futs = {ex.submit(run_batch, i, b): i for i, b in enumerate(batches)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        _, events = fut.result()
                        normalize_many([x for x in events if isinstance(x, dict)], "sentence_batch")
                        audit_calls.append({"phase": "sentence_batch", "batch": idx, "status": "ok", "events": len(events)})
                    except Exception as exc:
                        audit_calls.append({"phase": "sentence_batch", "batch": idx, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

        existing_spans = {(x["sent_id"], x["start"], x["end"]) for x in normalized}
        rescue_candidates = _maven_candidate_rows_from_tokens(tokens, existing_spans, cap=rescue_cap)
        candidate_by_id = {x["candidate_id"]: x for x in rescue_candidates}
        rescue_batches = [rescue_candidates[i:i + rescue_batch_size] for i in range(0, len(rescue_candidates), rescue_batch_size)]

        def verify_candidates(idx: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": """
You are NeoOLAF Layer 1B for MAVEN-ERE missed-event rescue. The supplied rows are
SOURCE-TOKEN candidates that were not already extracted. Decide which rows denote a
real event/state/process mention in MAVEN's broad event sense.

Be RECALL-ORIENTED: accept eventive nouns and nominalizations (e.g. election,
creation, attack, distribution, agreement), starts/ends, states/changes, outcomes,
communications, decisions, participation, movement and actions. Reject people,
places, organizations, pure objects, dates/numbers, adjectives with no event/state,
and function/discourse words.

Use ONLY candidate_id values supplied. The candidate span is already exact; do not
invent a new event. Return accepted IDs only, with a short reason. JSON only:
{"accepted":[{"candidate_id":"E0001","event_type":"Event","reason":"eventive noun"}]}
""".strip()},
                {"role": "user", "content": f"Candidates:\n{v1._json_block(batch, 43000)}\n\nJSON only."},
            ]
            parsed = self._chat(state, msgs, f"rescue_verify_v14_{idx:03d}")
            return parsed.get("accepted", []) if isinstance(parsed, dict) else []

        if rescue_batches:
            with ThreadPoolExecutor(max_workers=min(max_concurrency, len(rescue_batches))) as ex:
                futs = {ex.submit(verify_candidates, i, b): i for i, b in enumerate(rescue_batches)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        accepted = fut.result()
                        n = 0
                        for row in accepted if isinstance(accepted, list) else []:
                            if not isinstance(row, dict):
                                continue
                            c = candidate_by_id.get(str(row.get("candidate_id") or ""))
                            if not c:
                                continue
                            normalized.append({
                                "sent_id": c["sent_id"], "start": c["start"], "end": c["end"],
                                "trigger": c["trigger"], "event_type": str(row.get("event_type") or "Event"),
                                "reason": str(row.get("reason") or "source candidate accepted"),
                                "source": "source_candidate_rescue",
                            })
                            n += 1
                        audit_calls.append({"phase": "source_candidate_rescue", "batch": idx, "status": "ok", "accepted": n, "candidate_count": len(rescue_batches[idx])})
                    except Exception as exc:
                        audit_calls.append({"phase": "source_candidate_rescue", "batch": idx, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

        # Final one-shot coverage review after both independent passes.
        existing = sorted({(x["sent_id"], x["start"], x["end"]) for x in normalized})
        review_messages = [
            {"role": "system", "content": """
You are NeoOLAF Layer 1C MAVEN-ERE final coverage review. Return ONLY event mentions
still missing from the supplied existing exact spans. Search especially for eventive
NOUNS/nominalizations, process starts/ends, outcomes, states/changes and multi-token
lexical triggers. Include events even when unrelated to causality. Use the shortest
complete source trigger. Exact token offsets only. No relations. JSON only:
{"events":[{"sent_id":0,"start":1,"end":2,"trigger":"election","event_type":"Event","reason":"missed nominal event"}]}
""".strip()},
            {"role": "user", "content": (
                f"Existing spans [sent,start,end]:\n{v1._json_block(existing, 18000)}\n\n"
                f"Indexed document:\n{v1._json_block(rows, 52000)}\n\nJSON only."
            )},
        ]
        try:
            parsed = self._chat(state, review_messages, "coverage_review_v14")
            extra = parsed.get("events", []) if isinstance(parsed, dict) else []
            normalize_many([x for x in extra if isinstance(x, dict)] if isinstance(extra, list) else [], "final_coverage_review")
            audit_calls.append({"phase": "final_coverage_review", "status": "ok", "events": len(extra) if isinstance(extra, list) else 0})
        except Exception as exc:
            audit_calls.append({"phase": "final_coverage_review", "status": "error", "error": f"{type(exc).__name__}: {exc}"})

        # Exact-span deduplication. We deliberately keep separate mentions instead of
        # aggressive coreference merging; post-L12 maps mention spans to gold clusters.
        by_span: dict[tuple[int, int, int], dict[str, Any]] = {}
        source_priority = {"sentence_batch": 0, "source_candidate_rescue": 1, "final_coverage_review": 2}
        for row in normalized:
            key = (int(row["sent_id"]), int(row["start"]), int(row["end"]))
            if key not in by_span or source_priority.get(row.get("source", ""), 9) < source_priority.get(by_span[key].get("source", ""), 9):
                by_span[key] = row

        expressions: list[LinguisticExpression] = []
        mention_audit: list[dict[str, Any]] = []
        for sid, start, end in sorted(by_span):
            row = by_span[(sid, start, end)]
            trigger = " ".join(str(x) for x in tokens[sid][start:end]).strip()
            span_key = f"S{sid}[{start}:{end}]::{trigger}"
            digest = sha256(span_key.encode()).hexdigest()[:10]
            key = f"MCL:{digest}::{span_key}"
            expressions.append(LinguisticExpression(
                expr_id=f"expr_m14_{len(expressions):04d}", text=key, label="event_cluster",
                justification=f"singleton_high_recall_event_v14; event_type={row.get('event_type','Event')}; source={row.get('source')}; exact source span",
                evidence=v1._token_evidence(state, sid, start, end),
            ))
            mention_audit.append({"cluster_key": key, **row, "trigger": trigger})

        state.linguistic_expressions = expressions
        v1.write_json(self.audit_path.with_name("layer01_maven_calls_v14.json"), audit_calls)
        v1.write_json(self.audit_path.with_name("layer01_maven_mentions_v14.json"), mention_audit)
        v1.write_json(self.audit_path.with_name("layer01_maven_rejected_v14.json"), rejected)
        v1.write_json(self.audit_path.with_name("layer01_maven_source_rescue_pool_v14.json"), rescue_candidates)
        state.log(f"[{self.name}] MAVEN v1.4 mentions/singleton-clusters={len(expressions)}; rescue_pool={len(rescue_candidates)}")
        return state


class MavenLayer2V14(v1._TaskLayer2):
    """Precision-oriented two-gate MAVEN relation extraction.

    Event extraction stays recall-oriented. Relation existence is separated from
    subtype: (1) strict directed causal-dependence gate, (2) independent positive
    verifier, then (3) CAUSE/PRECONDITION subtype classification with direction fixed.
    This removes the v1.3 local-NONE promotion path that inflated false positives.
    """

    def _run(self, state: PipelineState) -> PipelineState:
        nodes = [x for x in state.linguistic_expressions or [] if x.label == "event_cluster"]
        enriched = [v1._node_enriched(x) for x in nodes]
        keys = [x.text for x in nodes]
        key_to_id = {k: f"C{i:03d}" for i, k in enumerate(keys)}
        id_to_key = {v: k for k, v in key_to_id.items()}
        cfg = v1._layer_cfg(state.profile_config or {}, "layer02_candidate_enrichment")
        local_distance = max(0, int(cfg.get("local_sentence_distance", 1)))
        link_min_conf = float(cfg.get("link_min_confidence", 0.68))
        verifier_min_conf = float(cfg.get("verifier_min_confidence", 0.64))
        subtype_min_conf = float(cfg.get("subtype_min_confidence", 0.52))
        max_long = max(0, int(cfg.get("long_range_proposal_max_pairs", 120)))

        pair_set: set[tuple[str, str]] = set()
        for a, b in combinations(keys, 2):
            if v1._cluster_min_distance(a, b) <= local_distance:
                pair_set.add(tuple(sorted((a, b))))

        inventory = [{
            "cluster_id": key_to_id[k], "positions": v1._maven_cluster_positions(k),
            "trigger": k.rsplit("::", 1)[-1],
        } for k in keys]
        if len(keys) > 1 and max_long:
            msgs = [
                {"role": "system", "content": f"""
You are NeoOLAF MAVEN-ERE nonlocal causal-candidate retrieval. From the event inventory,
return unordered pairs that are plausible candidates for CAUSE or PRECONDITION even
when mentions are in distant sentences. This is ONLY candidate retrieval: do not assign
subtype or direction. Prefer pairs with document-supported dependency, purpose/result,
enabling/preventing, initiation/completion, or explicit causal structure; do not return
mere same-topic pairs. Use only supplied cluster IDs. Return at most {max_long} pairs.
JSON only: {{"pairs":[["C000","C017"], ...]}}
""".strip()},
                {"role": "user", "content": f"Clusters:\n{v1._json_block(inventory, 42000)}\n\nDocument:\n{v1._json_block(v1._safe_token_rows(state), 52000)}"},
            ]
            try:
                parsed = self._chat(state, msgs, "long_range_candidates_v14")
                for item in parsed.get("pairs", []) if isinstance(parsed, dict) else []:
                    if isinstance(item, list) and len(item) == 2 and item[0] in id_to_key and item[1] in id_to_key and item[0] != item[1]:
                        pair_set.add(tuple(sorted((id_to_key[item[0]], id_to_key[item[1]]))))
            except Exception as exc:
                self._record({"phase": "long_range_candidates", "status": "error", "error": str(exc)})

        pair_rows = []
        for i, (a, b) in enumerate(sorted(pair_set)):
            pair_rows.append({
                "pair_id": f"P{i:04d}", "a": a, "b": b,
                "a_id": key_to_id[a], "b_id": key_to_id[b],
                "a_trigger": a.rsplit("::", 1)[-1], "b_trigger": b.rsplit("::", 1)[-1],
                "a_positions": v1._maven_cluster_positions(a), "b_positions": v1._maven_cluster_positions(b),
                "min_sentence_distance": v1._cluster_min_distance(a, b),
            })
        v1.write_json(self.decision_log_path.with_name("layer02_maven_candidate_pool_v14.json"), pair_rows)

        # Gate 1: existence + semantic direction only. Deliberately strict.
        gate1: dict[str, dict[str, Any]] = {}
        batches = self._batches(pair_rows)
        def run_gate1(idx: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": """
You are NeoOLAF MAVEN-ERE relation Gate 1. For each event pair decide whether the
DOCUMENT supports a directed causal-dependence relation. Allowed: A_TO_B, B_TO_A, NONE.
Do NOT assign CAUSE vs PRECONDITION yet.

Be precision-oriented. A positive requires a meaningful dependency: A brings about B,
enables/is necessary for B, initiates/completes B, or the document strongly entails
that B depends on A. Reject mere chronology, shared topic, co-occurrence, part-of the
same narrative, lexical association, coreference/same real-world event, and a goal or
intention that is merely stated but not a dependency between the two event mentions.
Direction is semantic/event-time causal direction, NEVER textual mention order.

Return one row per pair_id with confidence and short evidence-based reason. JSON only:
{"decisions":[{"pair_id":"P0000","decision":"A_TO_B|B_TO_A|NONE","confidence":0.82,"reason":"..."}]}
""".strip()},
                {"role": "user", "content": f"Document:\n{v1._json_block(v1._safe_token_rows(state), 52000)}\n\nPairs:\n{v1._json_block(batch, 43000)}"},
            ]
            parsed = self._chat(state, msgs, f"link_gate_v14_{idx:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []

        if batches:
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(batches))) as ex:
                futs = {ex.submit(run_gate1, i, b): i for i, b in enumerate(batches)}
                for fut in as_completed(futs):
                    try: rows = fut.result()
                    except Exception as exc:
                        self._record({"phase": "link_gate", "batch": futs[fut], "status": "error", "error": str(exc)}); continue
                    for row in rows if isinstance(rows, list) else []:
                        if isinstance(row, dict) and row.get("pair_id"):
                            gate1[str(row["pair_id"])] = row

        by_id = {x["pair_id"]: x for x in pair_rows}
        gate1_positive: list[dict[str, Any]] = []
        for pid, pair in by_id.items():
            row = gate1.get(pid, {"pair_id": pid, "decision": "NONE", "confidence": 0.0, "reason": "missing decision"})
            decision = str(row.get("decision") or "NONE").upper()
            try: conf = float(row.get("confidence", 0.0) or 0.0)
            except Exception: conf = 0.0
            self._record({"phase": "link_gate", **pair, **row, "accepted_by_threshold": decision != "NONE" and conf >= link_min_conf})
            if decision not in {"A_TO_B", "B_TO_A"} or conf < link_min_conf:
                continue
            source = pair["a"] if decision == "A_TO_B" else pair["b"]
            target = pair["b"] if decision == "A_TO_B" else pair["a"]
            gate1_positive.append({
                "pair_id": pid, "source": source, "target": target,
                "source_trigger": source.rsplit("::",1)[-1], "target_trigger": target.rsplit("::",1)[-1],
                "gate1_reason": str(row.get("reason") or ""), "gate1_confidence": conf,
            })

        # Gate 2: independently verify positives; can reject or flip direction.
        verified: dict[str, dict[str, Any]] = {}
        pos_batches = self._batches(gate1_positive)
        def run_verify(idx: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": """
You are NeoOLAF MAVEN-ERE positive-link verifier. Every supplied pair passed a first
causal-dependence gate. Independently audit it to reduce false positives.

KEEP only if the document supports a real directed dependence compatible with either:
(1) CAUSE: target is effectively inevitable given source, or
(2) PRECONDITION: target would not have happened without source.
REJECT chronology, topic association, coreference, event/subevent restatement, and
unsupported purpose/goal links. FLIP only when dependence is real but direction is
clearly reversed. Do not use textual mention order.

JSON only: {"decisions":[{"pair_id":"P0000","decision":"KEEP|FLIP|REJECT","confidence":0.8,"reason":"..."}]}
""".strip()},
                {"role": "user", "content": f"Document:\n{v1._json_block(v1._safe_token_rows(state), 52000)}\n\nProvisional positives:\n{v1._json_block(batch, 43000)}"},
            ]
            parsed = self._chat(state, msgs, f"positive_verify_v14_{idx:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []

        if pos_batches:
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(pos_batches))) as ex:
                futs = {ex.submit(run_verify, i, b): i for i, b in enumerate(pos_batches)}
                for fut in as_completed(futs):
                    try: rows = fut.result()
                    except Exception as exc:
                        self._record({"phase": "positive_verify", "batch": futs[fut], "status": "error", "error": str(exc)}); continue
                    for row in rows if isinstance(rows, list) else []:
                        if isinstance(row, dict) and row.get("pair_id"):
                            verified[str(row["pair_id"])] = row

        directed_for_subtype: list[dict[str, Any]] = []
        pos_by_id = {x["pair_id"]: x for x in gate1_positive}
        for pid, item in pos_by_id.items():
            row = verified.get(pid, {"pair_id": pid, "decision": "REJECT", "confidence": 0.0, "reason": "missing verifier decision"})
            decision = str(row.get("decision") or "REJECT").upper()
            try: conf = float(row.get("confidence", 0.0) or 0.0)
            except Exception: conf = 0.0
            self._record({"phase": "positive_verify", **item, **row, "accepted_by_threshold": decision in {"KEEP","FLIP"} and conf >= verifier_min_conf})
            if decision not in {"KEEP", "FLIP"} or conf < verifier_min_conf:
                continue
            source, target = item["source"], item["target"]
            if decision == "FLIP": source, target = target, source
            directed_for_subtype.append({
                "pair_id": pid, "source": source, "target": target,
                "source_trigger": source.rsplit("::",1)[-1], "target_trigger": target.rsplit("::",1)[-1],
                "verifier_reason": str(row.get("reason") or ""), "verifier_confidence": conf,
            })

        subtype: dict[str, dict[str, Any]] = {}
        subtype_batches = self._batches(directed_for_subtype)
        def run_subtype(idx: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": """
You are NeoOLAF MAVEN-ERE relation subtype classification. Direction is already fixed:
SOURCE -> TARGET. Choose exactly CAUSE, PRECONDITION, or REJECT.

CAUSE: given SOURCE, TARGET is effectively inevitable.
PRECONDITION: TARGET would not have happened without SOURCE, although SOURCE alone need
not guarantee TARGET. PRECONDITION is common in MAVEN-ERE; do not force balance.
REJECT if neither definition is actually supported. Never change direction.

JSON only: {"decisions":[{"pair_id":"P0000","decision":"CAUSE|PRECONDITION|REJECT","confidence":0.78,"reason":"..."}]}
""".strip()},
                {"role": "user", "content": f"Document:\n{v1._json_block(v1._safe_token_rows(state), 52000)}\n\nDirected pairs:\n{v1._json_block(batch, 43000)}"},
            ]
            parsed = self._chat(state, msgs, f"subtype_v14_{idx:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []

        if subtype_batches:
            with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(subtype_batches))) as ex:
                futs = {ex.submit(run_subtype, i, b): i for i, b in enumerate(subtype_batches)}
                for fut in as_completed(futs):
                    try: rows = fut.result()
                    except Exception as exc:
                        self._record({"phase": "subtype", "batch": futs[fut], "status": "error", "error": str(exc)}); continue
                    for row in rows if isinstance(rows, list) else []:
                        if isinstance(row, dict) and row.get("pair_id"):
                            subtype[str(row["pair_id"])] = row

        for item in directed_for_subtype:
            pid = item["pair_id"]
            row = subtype.get(pid, {"pair_id": pid, "decision": "REJECT", "confidence": 0.0, "reason": "missing subtype"})
            rel = str(row.get("decision") or "REJECT").upper()
            try: conf = float(row.get("confidence", 0.0) or 0.0)
            except Exception: conf = 0.0
            self._record({"phase": "subtype", **item, **row, "accepted_by_threshold": rel in {"CAUSE","PRECONDITION"} and conf >= subtype_min_conf})
            if rel not in {"CAUSE", "PRECONDITION"} or conf < subtype_min_conf:
                continue
            enriched.append(v1._relation_enriched(
                expr_id=f"expr_mr14_{pid}", source=item["source"], relation_id=rel, target=item["target"],
                metadata=self.catalog[rel], state=state, decision_payload={**row, "pipeline": "v1.4_two_gate"},
            ))

        v1.write_json(self.decision_log_path.with_name("layer02_maven_stage_counts_v14.json"), {
            "event_nodes": len(keys), "candidate_pairs": len(pair_rows),
            "gate1_positive": len(gate1_positive), "verified_positive": len(directed_for_subtype),
            "final_relations": sum(1 for x in enriched if getattr(x.base_expression, 'label', None) == 'relation'),
            "link_min_confidence": link_min_conf, "verifier_min_confidence": verifier_min_conf,
            "subtype_min_confidence": subtype_min_conf, "local_sentence_distance": local_distance,
            "gold_used": False,
        })
        return self._finish(state, enriched)


def offline_maven_rescue_candidate_preview(gold_record: dict[str, Any]) -> dict[str, Any]:
    """Posthoc audit: how much gold endpoint inventory is reachable by source-only rescue candidates.

    Candidate spans are generated before looking at gold. Gold is consulted only for the
    returned diagnostic metrics; this helper is never called from the pipeline.
    """
    tokens = gold_record.get("tokens") or []
    candidates = _maven_candidate_rows_from_tokens(tokens, set(), cap=10000)
    span_set = {(int(x["sent_id"]), int(x["start"]), int(x["end"])) for x in candidates}
    allowed = {"CAUSE", "PRECONDITION"}
    gold_rel = v1._gold_relation_set(gold_record, allowed)
    endpoint_ids = {x for s, _, t in gold_rel for x in (s, t)}
    all_ids = set((gold_record.get("entities") or {}).keys())
    hit_relation: set[str] = set(); hit_all: set[str] = set()
    for eid, entity in (gold_record.get("entities") or {}).items():
        for m in entity.get("mentions", []) or []:
            off = m.get("offset")
            if isinstance(off, list) and len(off) == 2 and m.get("sent_id") is not None:
                span = (int(m["sent_id"]), int(off[0]), int(off[1]))
                if span in span_set:
                    hit_all.add(str(eid))
                    if str(eid) in endpoint_ids: hit_relation.add(str(eid))
    return {
        "source_candidate_count": len(candidates),
        "gold_used_only_after_candidate_generation": True,
        "relation_endpoint_gold_clusters": len(endpoint_ids),
        "relation_endpoint_clusters_reachable": len(hit_relation),
        "relation_endpoint_cluster_recall_ceiling": len(hit_relation)/len(endpoint_ids) if endpoint_ids else 0.0,
        "all_gold_event_clusters": len(all_ids),
        "all_gold_event_clusters_reachable": len(hit_all),
        "all_gold_cluster_recall_ceiling": len(hit_all)/len(all_ids) if all_ids else 0.0,
    }

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
        pipeline.layers[1] = MavenLayer1V14(
            backends["layer01"], dataset_key=dataset_key,
            audit_path=run_dir / "run_logs/layer01_calls_v13.json",
            temperature=0.0, save_intermediate=True, verbose=verbose,
        )
        pipeline.layers[2] = MavenLayer2V14(
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
        "layer01": v1._make_backend(logger=logger, layer_tag=f"{dataset_key}_layer01_v14", model_host=host, api_key=api_key, cfg=v1._layer_cfg(profile_dict, "layer01_linguistic_expression_extraction"), fallback_max_tokens=max_tokens, fallback_timeout=request_timeout, reasoning_effort=reasoning_effort),
        "layer02": v1._make_backend(logger=logger, layer_tag=f"{dataset_key}_layer02_v14", model_host=host, api_key=api_key, cfg=v1._layer_cfg(profile_dict, "layer02_candidate_enrichment"), fallback_max_tokens=4096, fallback_timeout=request_timeout, reasoning_effort=reasoning_effort),
        "layer04": v1._make_backend(logger=logger, layer_tag=f"{dataset_key}_layer04_v14", model_host=host, api_key=api_key, cfg=v1._layer_cfg(profile_dict, "layer04_candidate_relation_extraction"), fallback_max_tokens=384, fallback_timeout=60, reasoning_effort=reasoning_effort),
        "other": v1._make_backend(logger=logger, layer_tag=f"{dataset_key}_other_v14", model_host=host, api_key=api_key, cfg={}, fallback_max_tokens=768, fallback_timeout=90, reasoning_effort=reasoning_effort),
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


def evaluate_state(dataset_key: str, state: PipelineState, gold_record: dict[str, Any]) -> dict[str, Any]:
    """v1.4 adds Layer-1 inventory diagnostics for MAVEN without changing gold isolation."""
    result = v1.evaluate_state(dataset_key, state, gold_record)
    if dataset_key != "maven_ere":
        return result

    labels = [x.text for x in (state.linguistic_expressions or []) if x.label == "event_cluster"]
    mapped_rows = [(label, v1._map_maven_endpoint(label, gold_record)) for label in labels]
    mapped_ids = {gid for _, gid in mapped_rows if gid}
    mapped_mentions = sum(1 for _, gid in mapped_rows if gid)
    gold_rel = v1._gold_relation_set(gold_record, {"CAUSE", "PRECONDITION"})
    relation_endpoint_ids = {x for s, _, t in gold_rel for x in (s, t)}
    all_gold_ids = set((gold_record.get("entities") or {}).keys())
    rel_hits = mapped_ids & relation_endpoint_ids
    all_hits = mapped_ids & all_gold_ids

    def cluster_triggers(eid: str) -> list[str]:
        ent = (gold_record.get("entities") or {}).get(eid) or {}
        return [str(m.get("trigger_word") or "") for m in ent.get("mentions", []) or [] if str(m.get("trigger_word") or "").strip()]

    result["inventory_endpoint_metrics"] = {
        "predicted_event_mentions": len(labels),
        "exact_gold_mapped_mentions": mapped_mentions,
        "mention_exact_mapping_rate": mapped_mentions/len(labels) if labels else 0.0,
        "mapped_unique_gold_clusters": len(mapped_ids),
        "gold_relation_endpoint_clusters": len(relation_endpoint_ids),
        "relation_endpoint_cluster_recall": len(rel_hits)/len(relation_endpoint_ids) if relation_endpoint_ids else 0.0,
        "all_gold_event_clusters": len(all_gold_ids),
        "all_gold_cluster_recall": len(all_hits)/len(all_gold_ids) if all_gold_ids else 0.0,
    }
    result["maven_posthoc_missing_relation_endpoints"] = [
        {"gold_event_id": eid, "gold_triggers": cluster_triggers(eid)}
        for eid in sorted(relation_endpoint_ids - mapped_ids)
    ]
    result["maven_posthoc_unmapped_inventory_mentions"] = [label for label, gid in mapped_rows if gid is None][:100]
    return result


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
        "maven_ere": "dual-pass source rescue event inventory + strict two-gate relation verification",
        "causalbank": "deterministic all ordered non-self pairs; no pair-level LLM pruning",
    }

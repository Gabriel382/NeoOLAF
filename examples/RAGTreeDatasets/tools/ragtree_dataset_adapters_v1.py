from __future__ import annotations

"""Dataset-specific NeoOLAF adapters for the four active RAGTree benchmarks.

Experiment code only: this module intentionally changes no file under src/neoolaf.
Gold entities/relations are never accepted by ``run_native_pipeline_record``.

Adapters:
- FinCausal: proposition/fact spans -> semantic cause/effect direction -> CAUSE.
- MAVEN-ERE: atomic event mentions/coreference clusters -> causal existence -> CAUSE/PRECONDITION.
- CausalBank: lexical stem nodes -> dense BECAUSE/THEREFORE compatibility graph.
- EventStoryLine is delegated to the already-tested v1.7 experiment module.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from hashlib import md5, sha256
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback

import eventstoryline_native_ablation_v1_7 as esl_v17
import eventstoryline_native_ablation_v1_5 as v15
import docred_native_ablation_v4 as v4

from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.documents import Document
from neoolaf.domain.enriched_expression import EnrichedExpression, EnrichmentEvidence
from neoolaf.domain.linguistic_expression import Evidence, LinguisticExpression
from neoolaf.domain.user_guidance import UserGuidance
from neoolaf.layers.layer03_candidate_typing_resolution.component import CandidateTypingResolutionLayer
from neoolaf.ontology.loader import SeedOntologyLoader
from neoolaf.profiles.profile_loader import load_document_profile

from experiments.methods.run_neoolaf import (
    OfflineWebSearchSource,
    OfflineWikipediaSource,
    OfflineWikidataSource,
    load_user_guidance,
)

# Reuse the logging/backend and native L0/L5-L12 construction that already worked
# in the EventStoryLine portable experiments.
SharedCallLogger = v15.SharedCallLogger
TaggedLoggedBackend = v15.TaggedLoggedBackend
OpenAICompatibleBackend = v15.OpenAICompatibleBackend
read_json = v15.read_json
read_jsonl = v15.read_jsonl
write_json = v15.write_json
append_jsonl = v15.append_jsonl
state_counts = v15.state_counts
Tee = v15.Tee
seed_ontology_summary = v15.seed_ontology_summary

DATASET_DISPLAY = {
    "fincausal": "FinCausal",
    "maven_ere": "MAVEN-ERE",
    "causalbank": "CausalBank",
}
RELATION_IDS = {
    "fincausal": ("CAUSE",),
    "maven_ere": ("CAUSE", "PRECONDITION"),
    "causalbank": ("BECAUSE", "THEREFORE"),
}


def _dedup(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _norm(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-]", "", text)
    return text


def _clip(value: Any, default: float = 0.5) -> float:
    try:
        result = float(value)
    except Exception:
        result = default
    if math.isnan(result) or math.isinf(result):
        result = default
    return max(0.0, min(1.0, result))


def _json_block(value: Any, max_chars: int = 30000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= max_chars else text[:max_chars] + "\n... [truncated]"


def _relation_parts(text: Any) -> tuple[str, str, str] | None:
    parts = [x.strip() for x in str(text or "").split("||")]
    return tuple(parts) if len(parts) == 3 and all(parts) else None



def _doc_text(state: PipelineState) -> str:
    return str(getattr(state.document, "cleaned_text", None) or getattr(state.document, "raw_text", "") or "")


def _safe_token_rows(state: PipelineState) -> list[dict[str, Any]]:
    profile = state.profile_config or {}
    sentences = list(profile.get("_input_sentences") or [])
    tokens = list(profile.get("_input_tokens") or [])
    if not sentences and tokens:
        sentences = [" ".join(str(x) for x in row) for row in tokens]
    return [
        {
            "sentence_id": i,
            "sentence": sentences[i] if i < len(sentences) else " ".join(str(x) for x in row),
            "tokens": [str(x) for x in (tokens[i] if i < len(tokens) else [])],
        }
        for i in range(max(len(sentences), len(tokens)))
    ]


def _whole_document_evidence(state: PipelineState, snippet: str | None = None) -> list[Evidence]:
    chunks = list(state.document.chunks or [])
    chunk = chunks[0] if chunks else None
    chunk_id = getattr(chunk, "chunk_id", "chunk_0000")
    text = str(snippet or (getattr(chunk, "text", "") if chunk is not None else ""))
    return [Evidence(
        chunk_id=chunk_id,
        chunk_start_char=-1,
        chunk_end_char=-1,
        doc_start_char=-1,
        doc_end_char=-1,
        snippet=text[:1200],
    )]


def _token_evidence(state: PipelineState, sent_id: int, start: int, end: int) -> list[Evidence]:
    rows = _safe_token_rows(state)
    snippet = rows[sent_id]["sentence"] if 0 <= sent_id < len(rows) else ""
    return _whole_document_evidence(state, snippet)


def _cache_root() -> Path:
    env = os.environ.get("NEOOLAF_RAGTREE_CACHE_DIR")
    if env:
        root = Path(env)
    else:
        root = Path(tempfile.gettempdir()) / "neoolaf_ragtree_unified4_v1_1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cached_chat(
    backend: TaggedLoggedBackend,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    cache_namespace: str,
) -> tuple[Any, dict[str, Any]]:
    payload = json.dumps({"model": model, "messages": messages, "temperature": temperature}, ensure_ascii=False, sort_keys=True)
    digest = sha256(payload.encode("utf-8")).hexdigest()[:28]
    cache_dir = _cache_root() / cache_namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["parsed"], {"cache": "hit", "cache_path": str(path)}
        except Exception:
            pass
    raw = backend.chat(model=model, messages=messages, temperature=temperature)
    parsed = backend.extract_json(raw)
    try:
        path.write_text(json.dumps({"parsed": parsed}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return parsed, {"cache": "miss", "cache_path": str(path)}


def _layer_cfg(profile: dict[str, Any], name: str) -> dict[str, Any]:
    return dict((profile.get("layers") or {}).get(name) or {})


def _relation_catalog(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_json(path).get("relations") or []
    return {str(x["relation_id"]).upper(): x for x in rows}


def _node_enriched(expr: LinguisticExpression) -> EnrichedExpression:
    return EnrichedExpression(
        base_expression=expr,
        aliases=[expr.text],
        synonyms=[],
        lexical_variants=[],
        alias_sources={expr.text: ["source"]},
        synonym_sources={},
        lexical_variant_sources={},
        definition=expr.justification,
        ontology_hints=[f"semantic_role:{expr.label}", "promote_to_ontology:false"],
        enrichment_evidence=[],
    )


def _relation_enriched(
    *,
    expr_id: str,
    source: str,
    relation_id: str,
    target: str,
    metadata: dict[str, Any],
    state: PipelineState,
    decision_payload: dict[str, Any],
    evidence: list[Evidence] | None = None,
) -> EnrichedExpression:
    relation_id = relation_id.upper()
    text = f"{source} || {relation_id} || {target}"
    base = LinguisticExpression(
        expr_id=expr_id,
        text=text,
        label="relation_instance",
        justification=(
            f"controlled_relation={relation_id}; source_label={source}; target_label={target}; "
            f"decision={decision_payload.get('decision')}; reason={decision_payload.get('reason', '')}"
        ),
        evidence=list(evidence or _whole_document_evidence(state)),
    )
    hints = _dedup([
        f"controlled_relation:{relation_id}",
        "promote_to_ontology:true",
        metadata.get("uri"), metadata.get("label"),
        f"source_label:{source}", f"target_label:{target}",
        f"lexical_predicate:{relation_id}",
        f"domain:{', '.join(metadata.get('domain_uris') or [])}" if metadata.get("domain_uris") else None,
        f"range:{', '.join(metadata.get('range_uris') or [])}" if metadata.get("range_uris") else None,
        f"decision_reason:{decision_payload.get('reason', '')}",
    ])
    return EnrichedExpression(
        base_expression=base,
        aliases=_dedup([text, relation_id, metadata.get("label")]),
        synonyms=[], lexical_variants=[],
        alias_sources={x: ["source" if x == text else "task_schema"] for x in _dedup([text, relation_id, metadata.get("label")])},
        synonym_sources={}, lexical_variant_sources={},
        definition=str(metadata.get("comment") or decision_payload.get("reason") or ""),
        ontology_hints=hints,
        enrichment_evidence=[EnrichmentEvidence(
            source="llm",
            content=json.dumps(decision_payload, ensure_ascii=False),
            reference=state.llm_model,
        )],
    )


class GenericRelationCanonicalizingLayer(CandidateTypingResolutionLayer):
    """Native role-based Layer 3 plus exact task-relation label normalization."""

    def __init__(self, *args: Any, relation_catalog_path: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.catalog = _relation_catalog(relation_catalog_path)

    def _run(self, state: PipelineState) -> PipelineState:
        state = super()._run(state)
        audit: list[dict[str, Any]] = []
        for candidate in state.relation_candidates or []:
            relation_id = None
            for hint in candidate.ontology_hints or []:
                text = str(hint)
                if text.lower().startswith("controlled_relation:"):
                    value = text.split(":", 1)[1].strip().upper()
                    if value in self.catalog:
                        relation_id = value
                        break
            original = candidate.canonical_label
            if relation_id:
                meta = self.catalog[relation_id]
                candidate.aliases = _dedup([original, *list(candidate.aliases or [])])
                candidate.canonical_label = relation_id
                candidate.normalized_label = self._normalize_label(relation_id)
                candidate.ontology_hints = _dedup([
                    f"controlled_relation:{relation_id}", "promote_to_ontology:true",
                    meta.get("uri"), meta.get("label"),
                    *list(candidate.ontology_hints or []),
                ])
            audit.append({
                "candidate_id": candidate.candidate_id,
                "original_label": original,
                "canonical_label": candidate.canonical_label,
                "relation_id": relation_id,
                "mention_count": len(candidate.mentions or []),
            })
        if state.artifact_dir:
            path = Path(state.artifact_dir) / self.name / "task_relation_canonicalization.json"
            write_json(path, audit)
        return state


class _PromptedLayer1(BaseLayer):
    name = "layer01_linguistic_expression_extraction"

    def __init__(self, backend: TaggedLoggedBackend, *, dataset_key: str, audit_path: str | Path, **kwargs: Any) -> None:
        super().__init__(save_intermediate=kwargs.get("save_intermediate", True), verbose=kwargs.get("verbose", False))
        self.backend = backend
        self.dataset_key = dataset_key
        self.audit_path = Path(audit_path)
        self.temperature = float(kwargs.get("temperature", 0.0))

    def _chat(self, state: PipelineState, messages: list[dict[str, str]], phase: str) -> Any:
        parsed, meta = _cached_chat(
            self.backend, model=state.llm_model, messages=messages, temperature=self.temperature,
            cache_namespace=f"{self.dataset_key}/layer01/{phase}",
        )
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        if self.audit_path.exists():
            try: rows = json.loads(self.audit_path.read_text(encoding="utf-8"))
            except Exception: rows = []
        rows.append({"phase": phase, **meta, "message_chars": [len(x.get("content", "")) for x in messages]})
        write_json(self.audit_path, rows)
        return parsed


class FinCausalLayer1(_PromptedLayer1):
    """Extract proposition-level causal facts; no gold spans or causal links are exposed."""

    def _run(self, state: PipelineState) -> PipelineState:
        rows = _safe_token_rows(state)
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        messages = [
            {"role": "system", "content": """
You are NeoOLAF Layer 1 for FinCausal linguistic-expression extraction.
Extract proposition/fact spans that could participate in a financial cause-effect statement.
This is NOT event-trigger extraction. Endpoints are whole facts/propositions, often long spans.
A fact may be non-quantified. A quantified fact contains a measurable quantity/value/change.
Do not decide which fact causes which other fact here. Do not output a CAUSE relation.
Preserve source wording and boundaries; causal connectives such as because/due to/resulting from
should normally remain outside the fact span when they only connect two facts.
Return JSON only: {"facts":[{"text":"exact source span","fact_kind":"FACT|QUANTIFIED_FACT","sent_id":0,"token_start":0,"token_end":4,"reason":"..."}]}
Offsets are zero-based, end-exclusive. If a span crosses sentences or exact token offsets are uncertain,
set sent_id/token_start/token_end to null but preserve exact text.
""".strip()},
            {"role": "user", "content": f"Task guidance:\n{_json_block(task, 7000)}\n\nIndexed source:\n{_json_block(rows, 26000)}\n\nDocument:\n{_doc_text(state)}\n\nJSON only."},
        ]
        parsed = self._chat(state, messages, "fact_inventory")
        facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        expressions: list[LinguisticExpression] = []
        audit: list[dict[str, Any]] = []
        seen: set[str] = set()
        doc_text = _doc_text(state)
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        for i, raw in enumerate(facts if isinstance(facts, list) else []):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            sid = raw.get("sent_id")
            start = raw.get("token_start")
            end = raw.get("token_end")
            exact_token = False
            try:
                sid_i, start_i, end_i = int(sid), int(start), int(end)
                if 0 <= sid_i < len(tokens) and 0 <= start_i < end_i <= len(tokens[sid_i]):
                    tok_text = " ".join(str(x) for x in tokens[sid_i][start_i:end_i])
                    exact_token = _norm(tok_text) == _norm(text)
                else:
                    sid_i = start_i = end_i = -1
            except Exception:
                sid_i = start_i = end_i = -1
            exact_text = _norm(text) in _norm(doc_text)
            if not exact_token and not exact_text:
                audit.append({"text": text, "accepted": False, "reason": "not_grounded_in_source"})
                continue
            digest = sha256(f"{sid_i}:{start_i}:{end_i}:{_norm(text)}".encode()).hexdigest()[:10]
            key = f"FSPAN:{digest}::{text}"
            if key in seen:
                continue
            seen.add(key)
            evidence = _token_evidence(state, sid_i, start_i, end_i) if exact_token else _whole_document_evidence(state, text)
            kind = str(raw.get("fact_kind") or "FACT").upper()
            expressions.append(LinguisticExpression(
                expr_id=f"expr_f_{len(expressions):04d}", text=key, label="fact_span",
                justification=f"kind={kind}; source_text={text}; exact_token={exact_token}; reason={raw.get('reason', '')}",
                evidence=evidence,
            ))
            audit.append({"endpoint": key, "text": text, "kind": kind, "accepted": True, "exact_token": exact_token})
        state.linguistic_expressions = expressions
        write_json(self.audit_path.with_name("layer01_fincausal_fact_inventory.json"), audit)
        state.log(f"[{self.name}] FinCausal fact spans={len(expressions)}")
        return state


class MavenLayer1(_PromptedLayer1):
    """Extract atomic event mentions and group coreferent mentions into event endpoints."""

    def _run(self, state: PipelineState) -> PipelineState:
        rows = _safe_token_rows(state)
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        messages = [
            {"role": "system", "content": """
You are NeoOLAF Layer 1 for MAVEN-ERE event extraction and event coreference.
Extract atomic event mentions from the indexed document, then group mentions that refer to the SAME real-world event.
Do not predict CAUSE or PRECONDITION here. Do not use narrative order as causal direction.
Triggers should be the shortest complete eventive trigger span (verb/eventive noun/state) supported by source tokens.
Every returned mention must use exact zero-based sentence/token offsets, end-exclusive.
Use a coref_group such as C000; singleton groups are normal. Group only true event coreference, not merely related events.
Return JSON only:
{"events":[{"mention_id":"M000","sent_id":0,"start":3,"end":4,"trigger":"flooding","event_type":"Catastrophe","coref_group":"C000","reason":"..."}]}
""".strip()},
            {"role": "user", "content": f"Task guidance:\n{_json_block(task, 7000)}\n\nIndexed document:\n{_json_block(rows, 50000)}\n\nJSON only."},
        ]
        parsed = self._chat(state, messages, "event_inventory_coref")
        events = parsed.get("events", []) if isinstance(parsed, dict) else []
        tokens = (state.profile_config or {}).get("_input_tokens", []) or []
        accepted: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        for raw in events if isinstance(events, list) else []:
            if not isinstance(raw, dict):
                continue
            try:
                sid, start, end = int(raw.get("sent_id")), int(raw.get("start")), int(raw.get("end"))
            except Exception:
                continue
            if not (0 <= sid < len(tokens) and 0 <= start < end <= len(tokens[sid])):
                audit.append({"raw": raw, "accepted": False, "reason": "invalid_offset"})
                continue
            source_trigger = " ".join(str(x) for x in tokens[sid][start:end])
            trigger = str(raw.get("trigger") or source_trigger).strip()
            if _norm(trigger) != _norm(source_trigger):
                trigger = source_trigger
            if not _norm(trigger):
                continue
            accepted.append({
                "mention_id": str(raw.get("mention_id") or f"M{len(accepted):03d}"),
                "sent_id": sid, "start": start, "end": end, "trigger": trigger,
                "event_type": str(raw.get("event_type") or "Event"),
                "coref_group": str(raw.get("coref_group") or f"SINGLE_{len(accepted):03d}"),
                "reason": str(raw.get("reason") or ""),
            })
            audit.append({**accepted[-1], "accepted": True})
        # exact-span dedup, then preserve the first assigned coreference group
        by_span: dict[tuple[int, int, int], dict[str, Any]] = {}
        for row in accepted:
            by_span.setdefault((row["sent_id"], row["start"], row["end"]), row)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in by_span.values():
            groups.setdefault(row["coref_group"], []).append(row)
        expressions: list[LinguisticExpression] = []
        cluster_audit: list[dict[str, Any]] = []
        for group_id, members in sorted(groups.items(), key=lambda kv: min((x["sent_id"], x["start"]) for x in kv[1])):
            members = sorted(members, key=lambda x: (x["sent_id"], x["start"], x["end"]))
            span_keys = [f"S{x['sent_id']}[{x['start']}:{x['end']}]::{x['trigger']}" for x in members]
            digest = sha256("|".join(span_keys).encode()).hexdigest()[:10]
            key = f"MCL:{digest}::" + " && ".join(span_keys)
            evidence: list[Evidence] = []
            for x in members:
                evidence.extend(_token_evidence(state, x["sent_id"], x["start"], x["end"]))
            expressions.append(LinguisticExpression(
                expr_id=f"expr_m_{len(expressions):04d}", text=key, label="event_cluster",
                justification=(f"coref_group={group_id}; event_types={_dedup(x['event_type'] for x in members)}; "
                               f"member_count={len(members)}"),
                evidence=evidence,
            ))
            cluster_audit.append({"cluster_key": key, "coref_group": group_id, "members": members})
        state.linguistic_expressions = expressions
        write_json(self.audit_path.with_name("layer01_maven_mentions.json"), audit)
        write_json(self.audit_path.with_name("layer01_maven_clusters.json"), cluster_audit)
        state.log(f"[{self.name}] MAVEN mentions={len(by_span)} clusters={len(expressions)}")
        return state


# CausalBank normalization deliberately emits multiple no-gold lexical normalizations.
# The RAGTree normalized sample mixes lemma/stem-like forms; using a union protects
# endpoint recall without consulting the record's gold entity inventory.
def _causalbank_stem_candidates(token: str) -> list[str]:
    token = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token).lower()
    if not token:
        return []
    values = [token]
    try:
        from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer
        values.extend([
            LancasterStemmer().stem(token),
            PorterStemmer().stem(token),
        ])
        try:
            lemma = WordNetLemmatizer().lemmatize(token)
            values.append(lemma)
        except Exception:
            pass
    except Exception:
        # Tiny dependency-free fallbacks for portability, not a substitute for NLTK.
        for suffix in ("ing", "edly", "edly", "ed", "ies", "es", "s"):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                stem = token[:-len(suffix)] + ("y" if suffix == "ies" else "")
                values.append(stem)
                break
    return _dedup(x for x in values if x and len(x) >= 1)


class CausalBankLayer1(BaseLayer):
    name = "layer01_linguistic_expression_extraction"

    def __init__(self, *, audit_path: str | Path, **kwargs: Any) -> None:
        super().__init__(save_intermediate=kwargs.get("save_intermediate", True), verbose=kwargs.get("verbose", False))
        self.audit_path = Path(audit_path)

    def _run(self, state: PipelineState) -> PipelineState:
        token_rows = (state.profile_config or {}).get("_input_tokens", []) or []
        if not token_rows:
            token_rows = [re.findall(r"\b[\w'-]+\b", _doc_text(state))]
        expressions: list[LinguisticExpression] = []
        seen: set[str] = set()
        audit: list[dict[str, Any]] = []
        for sid, row in enumerate(token_rows):
            for tid, token in enumerate(row):
                token_s = str(token)
                for stem in _causalbank_stem_candidates(token_s):
                    if stem in seen:
                        continue
                    seen.add(stem)
                    expressions.append(LinguisticExpression(
                        expr_id=f"expr_cb_{len(expressions):04d}", text=stem, label="lemma_node",
                        justification=f"source_token={token_s}; sentence_id={sid}; token_id={tid}; no-gold lexical normalization",
                        evidence=_token_evidence(state, sid, tid, tid + 1),
                    ))
                    audit.append({"lemma_candidate": stem, "source_token": token_s, "sent_id": sid, "token_id": tid})
        state.linguistic_expressions = expressions
        write_json(self.audit_path, audit)
        state.log(f"[{self.name}] CausalBank lexical endpoint candidates={len(expressions)}")
        return state


class _TaskLayer2(BaseLayer):
    name = "layer02_candidate_enrichment"

    def __init__(
        self, backend: TaggedLoggedBackend, *, dataset_key: str, relation_catalog_path: str | Path,
        decision_log_path: str | Path, batch_size: int = 24, max_concurrency: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(save_intermediate=kwargs.get("save_intermediate", True), verbose=kwargs.get("verbose", False))
        self.backend = backend
        self.dataset_key = dataset_key
        self.catalog = _relation_catalog(relation_catalog_path)
        self.decision_log_path = Path(decision_log_path)
        self.batch_size = max(1, int(batch_size))
        self.max_concurrency = max(1, int(max_concurrency))
        self.temperature = float(kwargs.get("temperature", 0.0))
        self._decision_lock = threading.Lock()
        self._decisions: list[dict[str, Any]] = []

    def _batches(self, rows: list[Any]) -> list[list[Any]]:
        return [rows[i:i+self.batch_size] for i in range(0, len(rows), self.batch_size)]

    def _chat(self, state: PipelineState, messages: list[dict[str, str]], phase: str) -> Any:
        parsed, _ = _cached_chat(
            self.backend, model=state.llm_model, messages=messages, temperature=self.temperature,
            cache_namespace=f"{self.dataset_key}/layer02/{phase}",
        )
        return parsed

    def _record(self, row: dict[str, Any]) -> None:
        with self._decision_lock:
            self._decisions.append(row)

    def _finish(self, state: PipelineState, enriched: list[EnrichedExpression]) -> PipelineState:
        state.enriched_expressions = enriched
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.decision_log_path, self._decisions)
        state.log(f"[{self.name}] {self.dataset_key}: enriched={len(enriched)} decisions={len(self._decisions)}")
        return state


class FinCausalLayer2(_TaskLayer2):
    def _run(self, state: PipelineState) -> PipelineState:
        nodes = [x for x in state.linguistic_expressions or [] if x.label == "fact_span"]
        enriched = [_node_enriched(x) for x in nodes]
        pairs = list(combinations(nodes, 2))
        pair_rows = []
        for i, (a, b) in enumerate(pairs):
            pair_rows.append({
                "pair_id": f"P{i:04d}", "a": a.text, "b": b.text,
                "a_text": a.text.split("::", 1)[-1], "b_text": b.text.split("::", 1)[-1],
            })
        decisions: dict[str, dict[str, Any]] = {}
        batches = self._batches(pair_rows)
        def run_batch(batch_index: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            messages = [
                {"role": "system", "content": """
You are NeoOLAF Layer 2 for FinCausal relation classification.
For each supplied unordered pair, decide semantic causal direction from the document, never from mention order.
Allowed decisions: A_CAUSES_B, B_CAUSES_A, NONE.
FinCausal endpoints are proposition/fact chunks, not event triggers. An EFFECT is normally a quantified/measurable fact;
a CAUSE may be quantified or non-quantified. Causal connectives can occur between spans and should not reverse semantics.
Return exactly one decision for every pair_id. Use only the supplied endpoint IDs. JSON only:
{"decisions":[{"pair_id":"P0000","decision":"A_CAUSES_B","reason":"...","confidence":0.9}]}
""".strip()},
                {"role": "user", "content": f"Document:\n{_doc_text(state)}\n\nPairs:\n{_json_block(batch, 36000)}\n\nJSON only."},
            ]
            parsed = self._chat(state, messages, f"batch_{batch_index:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, max(1, len(batches)))) as ex:
            futures = {ex.submit(run_batch, i, batch): i for i, batch in enumerate(batches)}
            for fut in as_completed(futures):
                try: rows = fut.result()
                except Exception as exc:
                    rows = []
                    self._record({"batch": futures[fut], "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, dict) and row.get("pair_id"):
                        decisions[str(row["pair_id"])] = row
        by_id = {x["pair_id"]: x for x in pair_rows}
        for pid, pair in by_id.items():
            row = decisions.get(pid, {"pair_id": pid, "decision": "NONE", "reason": "missing decision", "confidence": 0.0})
            decision = str(row.get("decision") or "NONE").upper()
            self._record({**pair, **row})
            if decision == "A_CAUSES_B": source, target = pair["a"], pair["b"]
            elif decision == "B_CAUSES_A": source, target = pair["b"], pair["a"]
            else: continue
            enriched.append(_relation_enriched(
                expr_id=f"expr_fr_{pid}", source=source, relation_id="CAUSE", target=target,
                metadata=self.catalog["CAUSE"], state=state, decision_payload=row,
            ))
        return self._finish(state, enriched)


def _maven_cluster_positions(key: str) -> list[tuple[int, int, int]]:
    return [(int(a), int(b), int(c)) for a, b, c in re.findall(r"S(\d+)\[(\d+):(\d+)\]::", key)]


def _cluster_min_distance(a: str, b: str) -> int:
    pa, pb = _maven_cluster_positions(a), _maven_cluster_positions(b)
    if not pa or not pb: return 10**6
    return min(abs(x[0] - y[0]) for x in pa for y in pb)


class MavenLayer2(_TaskLayer2):
    def _run(self, state: PipelineState) -> PipelineState:
        nodes = [x for x in state.linguistic_expressions or [] if x.label == "event_cluster"]
        enriched = [_node_enriched(x) for x in nodes]
        keys = [x.text for x in nodes]
        key_to_id = {key: f"C{i:03d}" for i, key in enumerate(keys)}
        id_to_key = {v: k for k, v in key_to_id.items()}
        # High-recall deterministic local pool plus one no-label long-range proposal call.
        pair_set: set[tuple[str, str]] = set()
        for a, b in combinations(keys, 2):
            if _cluster_min_distance(a, b) <= 3:
                pair_set.add(tuple(sorted((a, b))))
        inventory = [{"cluster_id": key_to_id[k], "cluster_key": k, "sentence_positions": _maven_cluster_positions(k)} for k in keys]
        if len(keys) > 1:
            messages = [
                {"role": "system", "content": """
You are NeoOLAF Layer 2 candidate generation for MAVEN-ERE.
Propose additional LONG-RANGE unordered event-cluster pairs that deserve causal examination.
Do NOT decide CAUSE/PRECONDITION and do NOT infer direction. Return only supplied cluster IDs.
Favor plausible temporal/causal interaction even across distant sentences; omit purely topical pairs.
JSON only: {"pairs":[["C000","C017"], ...]}
""".strip()},
                {"role": "user", "content": f"Clusters:\n{_json_block(inventory, 36000)}\n\nDocument:\n{_json_block(_safe_token_rows(state), 50000)}\n\nReturn at most 120 additional pairs."},
            ]
            try:
                parsed = self._chat(state, messages, "long_range_candidate_proposal")
                for item in parsed.get("pairs", []) if isinstance(parsed, dict) else []:
                    if isinstance(item, list) and len(item) == 2 and item[0] in id_to_key and item[1] in id_to_key and item[0] != item[1]:
                        pair_set.add(tuple(sorted((id_to_key[item[0]], id_to_key[item[1]]))))
            except Exception as exc:
                self._record({"phase": "long_range_candidate_proposal", "status": "error", "error": str(exc)})
        pair_rows = [
            {"pair_id": f"P{i:04d}", "a": a, "b": b, "a_id": key_to_id[a], "b_id": key_to_id[b], "min_sentence_distance": _cluster_min_distance(a, b)}
            for i, (a, b) in enumerate(sorted(pair_set))
        ]
        # Stage A: relation existence only.
        link_decisions: dict[str, dict[str, Any]] = {}
        stage_a_batches = self._batches(pair_rows)
        def stage_a(idx: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": """
You are NeoOLAF Layer 2A for MAVEN-ERE causal-link existence.
For each unordered event-cluster pair decide LINK or NONE only. Do not choose direction or subtype yet.
A LINK requires document-supported causal dependence, not chronology, co-occurrence, shared topic, coreference, or mere subevent similarity.
Temporal plausibility matters: a cause/precondition must occur before or overlap its consequence in real event time,
but textual mention order is NOT event time. Return every pair_id exactly once. JSON only:
{"decisions":[{"pair_id":"P0000","decision":"LINK|NONE","reason":"...","confidence":0.8}]}
""".strip()},
                {"role": "user", "content": f"Document:\n{_json_block(_safe_token_rows(state), 50000)}\n\nPairs:\n{_json_block(batch, 40000)}"},
            ]
            parsed = self._chat(state, msgs, f"existence_{idx:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, max(1, len(stage_a_batches)))) as ex:
            futs = {ex.submit(stage_a, i, b): i for i, b in enumerate(stage_a_batches)}
            for fut in as_completed(futs):
                try: rows = fut.result()
                except Exception as exc:
                    self._record({"phase": "existence", "batch": futs[fut], "status": "error", "error": str(exc)}); continue
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, dict) and row.get("pair_id"):
                        link_decisions[str(row["pair_id"])] = row
        linked = [pair for pair in pair_rows if str(link_decisions.get(pair["pair_id"], {}).get("decision", "NONE")).upper() == "LINK"]
        # Stage B: direction + CAUSE/PRECONDITION only for accepted LINK pairs.
        class_decisions: dict[str, dict[str, Any]] = {}
        stage_b_batches = self._batches(linked)
        def stage_b(idx: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": """
You are NeoOLAF Layer 2B for MAVEN-ERE causal relation direction and subtype.
Every supplied pair already passed a causal-existence filter.
Allowed decisions exactly:
A_CAUSE_B, B_CAUSE_A, A_PRECONDITION_B, B_PRECONDITION_A.
CAUSE means the target becomes effectively inevitable given the source event.
PRECONDITION means the target would not have happened without the source, but the source alone need not guarantee it.
Infer semantic/event-time direction; NEVER use textual mention order as direction. Return every pair_id exactly once. JSON only.
""".strip()},
                {"role": "user", "content": f"Document:\n{_json_block(_safe_token_rows(state), 50000)}\n\nLinked pairs:\n{_json_block(batch, 40000)}\n\nSchema: {{\"decisions\":[{{\"pair_id\":\"P0000\",\"decision\":\"A_PRECONDITION_B\",\"reason\":\"...\",\"confidence\":0.8}}]}}"},
            ]
            parsed = self._chat(state, msgs, f"class_{idx:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, max(1, len(stage_b_batches)))) as ex:
            futs = {ex.submit(stage_b, i, b): i for i, b in enumerate(stage_b_batches)}
            for fut in as_completed(futs):
                try: rows = fut.result()
                except Exception as exc:
                    self._record({"phase": "class", "batch": futs[fut], "status": "error", "error": str(exc)}); continue
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, dict) and row.get("pair_id"):
                        class_decisions[str(row["pair_id"])] = row
        for pair in pair_rows:
            pid = pair["pair_id"]
            existence = link_decisions.get(pid, {"decision": "NONE", "reason": "missing", "confidence": 0.0})
            class_row = class_decisions.get(pid)
            self._record({"phase": "final", **pair, "existence": existence, "classification": class_row})
            if not class_row: continue
            decision = str(class_row.get("decision") or "").upper()
            if decision == "A_CAUSE_B": source, rel, target = pair["a"], "CAUSE", pair["b"]
            elif decision == "B_CAUSE_A": source, rel, target = pair["b"], "CAUSE", pair["a"]
            elif decision == "A_PRECONDITION_B": source, rel, target = pair["a"], "PRECONDITION", pair["b"]
            elif decision == "B_PRECONDITION_A": source, rel, target = pair["b"], "PRECONDITION", pair["a"]
            else: continue
            enriched.append(_relation_enriched(
                expr_id=f"expr_mr_{pid}", source=source, relation_id=rel, target=target,
                metadata=self.catalog[rel], state=state, decision_payload=class_row,
            ))
        write_json(self.decision_log_path.with_name("layer02_maven_candidate_pool.json"), pair_rows)
        return self._finish(state, enriched)


class CausalBankLayer2(_TaskLayer2):
    """Dense lexical-graph adapter for the two normalized CausalBank relation families.

    The normalized benchmark assigns one relation family to each causal-pattern
    record: BECAUSE for effect/reason-style constructions and THEREFORE for
    cause/consequence-style constructions.  The record ``type`` and raw text are
    pipeline-visible metadata, so they may be used to determine that family;
    gold entities/relations remain unavailable.
    """

    _BECAUSE_TYPE_CUES = (
        "because", "because of", "due to", "owing to", "resulted from", "result from",
        "results from", "caused by", "as a result of", "on account of", "thanks to",
        "attributable to", "attributed to", "stemmed from", "stems from", "derived from",
        "arose from", "arising from", "originated from", "reason for", "consequence of",
    )
    _THEREFORE_TYPE_CUES = (
        "therefore", "thus", "hence", "consequently", "resulted in", "result in",
        "results in", "led to", "lead to", "leads to", "resulting in", "gave rise to",
        "gives rise to", "triggered", "triggers", "produced", "produces", "causes",
        "causing", "which caused", "which led to", "so that",
    )

    @staticmethod
    def _normalise_record_type(value: Any) -> str:
        text = str(value or "").lower().strip().replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", text)

    @classmethod
    def _relation_from_record_type(cls, value: Any) -> str | None:
        text = cls._normalise_record_type(value)
        if not text:
            return None
        # Test reverse/effect->reason markers first because they often contain
        # generic words such as "result" that also occur in forward patterns.
        if any(cue in text for cue in cls._BECAUSE_TYPE_CUES):
            return "BECAUSE"
        if any(cue in text for cue in cls._THEREFORE_TYPE_CUES):
            return "THEREFORE"
        return None

    def _record_relation(self, state: PipelineState) -> tuple[str, dict[str, Any]]:
        profile = state.profile_config or {}
        record_type = str(profile.get("_input_record_type") or "")
        deterministic = self._relation_from_record_type(record_type)
        if deterministic:
            return deterministic, {
                "phase": "record_relation_family",
                "relation": deterministic,
                "source": "record_type_rule",
                "record_type": record_type,
                "reason": "Visible CausalBank pattern tag has an unambiguous causal orientation.",
                "confidence": 1.0,
            }

        messages = [
            {"role": "system", "content": """
You are NeoOLAF Layer 2 for the RAGTree-normalized CausalBank benchmark.
Choose the ONE relation family used by this entire causal-pattern record.
BECAUSE means an effect/result is linguistically explained by a cause/reason (effect BECAUSE reason).
THEREFORE means a cause/reason linguistically leads to an effect/result (cause THEREFORE consequence).
Use only the visible record type and document wording. This is NOT pair classification and no gold annotations are available.
Return JSON only: {"relation":"BECAUSE|THEREFORE","reason":"...","confidence":0.9}
""".strip()},
            {"role": "user", "content": f"Record type: {record_type}\nDocument:\n{_doc_text(state)}\n\nJSON only."},
        ]
        parsed = self._chat(state, messages, "record_relation_family")
        relation = str(parsed.get("relation") if isinstance(parsed, dict) else "").upper()
        if relation not in {"BECAUSE", "THEREFORE"}:
            raise RuntimeError(f"Unable to determine CausalBank relation family from visible input type={record_type!r}")
        row = {
            "phase": "record_relation_family",
            "relation": relation,
            "source": "llm_visible_record_type_and_text",
            "record_type": record_type,
            "reason": parsed.get("reason", "") if isinstance(parsed, dict) else "",
            "confidence": _clip(parsed.get("confidence", 0.5) if isinstance(parsed, dict) else 0.5),
        }
        return relation, row

    def _run(self, state: PipelineState) -> PipelineState:
        nodes = [x for x in state.linguistic_expressions or [] if x.label == "lemma_node"]
        enriched = [_node_enriched(x) for x in nodes]
        relation_id, relation_row = self._record_relation(state)
        self._record(relation_row)

        pairs = list(combinations([x.text for x in nodes], 2))
        pair_rows = [{"pair_id": f"P{i:04d}", "a": a, "b": b} for i, (a, b) in enumerate(pairs)]
        decisions: dict[str, dict[str, Any]] = {}
        batches = self._batches(pair_rows)

        def run_batch(idx: int, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            msgs = [
                {"role": "system", "content": f"""
You are NeoOLAF Layer 2 pair coverage for the RAGTree-normalized CausalBank benchmark.
Endpoints are canonical lexical/stem nodes. This whole record uses the controlled relation label {relation_id}.
The normalized graph is intentionally DENSE and lexical-context based; do NOT reinterpret it as one sparse proposition-level cause->effect edge.
For each unordered lexical pair decide one of: BOTH, A_TO_B, B_TO_A, NONE.
BOTH is valid when both directed lexical associations are supported by participation in the same explicit causal construction/context.
Do not orient solely by token order. Reject only pairs that are not meaningfully part of the causal-context relation represented by this record.
Return every pair_id exactly once. JSON only:
{{"decisions":[{{"pair_id":"P0000","decision":"BOTH","reason":"...","confidence":0.8}}]}}
""".strip()},
                {"role": "user", "content": (
                    f"Record type: {(state.profile_config or {}).get('_input_record_type', '')}\n"
                    f"Controlled relation family: {relation_id}\n"
                    f"Document:\n{_doc_text(state)}\n\nLexical pairs:\n{_json_block(batch, 42000)}\n\nJSON only."
                )},
            ]
            parsed = self._chat(state, msgs, f"dense_{relation_id.lower()}_batch_{idx:03d}")
            return parsed.get("decisions", []) if isinstance(parsed, dict) else []

        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, max(1, len(batches)))) as ex:
            futs = {ex.submit(run_batch, i, b): i for i, b in enumerate(batches)}
            for fut in as_completed(futs):
                try:
                    rows = fut.result()
                except Exception as exc:
                    self._record({"phase": "batch", "relation": relation_id, "batch": futs[fut], "status": "error", "error": str(exc)})
                    continue
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, dict) and row.get("pair_id"):
                        decisions[str(row["pair_id"])] = row

        for pair in pair_rows:
            row = decisions.get(pair["pair_id"], {"decision": "NONE", "reason": "missing", "confidence": 0.0})
            decision = str(row.get("decision") or "NONE").upper()
            self._record({"phase": "pair", "relation": relation_id, **pair, **row})
            directed: list[tuple[str, str]] = []
            if decision == "BOTH":
                directed = [(pair["a"], pair["b"]), (pair["b"], pair["a"])]
            elif decision in {"A_TO_B", f"A_{relation_id}_B"}:
                directed = [(pair["a"], pair["b"])]
            elif decision in {"B_TO_A", f"B_{relation_id}_A"}:
                directed = [(pair["b"], pair["a"])]
            for j, (source, target) in enumerate(directed):
                enriched.append(_relation_enriched(
                    expr_id=f"expr_cbr_{pair['pair_id']}_{j}", source=source, relation_id=relation_id, target=target,
                    metadata=self.catalog[relation_id], state=state,
                    decision_payload={**row, "record_relation_family": relation_id},
                ))

        write_json(self.decision_log_path.with_name("layer02_causalbank_pair_pool.json"), {
            "record_relation_family": relation_id,
            "record_type": (state.profile_config or {}).get("_input_record_type", ""),
            "pairs": pair_rows,
        })
        return self._finish(state, enriched)


class StrictTaskEndpointLayer(v4.StructuredEndpointValidatedRelationLayer):
    """Exact structured endpoint resolution; no dataset-specific type rejection."""
    def _valid_types(self, relation_id: str | None, source: Any, target: Any) -> tuple[bool, str]:
        return True, "task adapter exact endpoint labels; relation schema validated upstream"


def _make_backend(
    *, logger: SharedCallLogger, layer_tag: str, model_host: str, api_key: str,
    cfg: dict[str, Any], fallback_max_tokens: int, fallback_timeout: int, reasoning_effort: str,
) -> TaggedLoggedBackend:
    core = OpenAICompatibleBackend(
        backend_name="openrouter", host=model_host, api_key=api_key,
        timeout=int(cfg.get("request_timeout_seconds", fallback_timeout)),
        max_tokens=int(cfg.get("max_output_tokens", fallback_max_tokens)),
        reasoning_effort=reasoning_effort, exclude_reasoning=True,
    )
    return TaggedLoggedBackend(core, logger, layer_tag=layer_tag, response_hard_cap_chars=cfg.get("response_hard_cap_chars"))


def build_document(record: dict[str, Any], source_path: str | Path) -> Document:
    return v4.build_document(record, source_path)


def choose_chunk_size(text: str, max_safe_chars: int = 26000) -> int:
    return v4.choose_chunk_size(text, max_safe_chars)


def build_pipeline(
    *, dataset_key: str, backends: dict[str, TaggedLoggedBackend], rag_adapter: Any,
    profile_config: dict[str, Any], relation_catalog_path: str | Path, chunk_size: int,
    run_dir: str | Path, workers: int = 8, verbose: bool = True,
) -> Pipeline:
    # v4 builds native L0-L12. We replace only experiment-side Layers 1-4.
    pipeline = v4.build_pipeline(
        backends=backends, rag_adapter=rag_adapter, profile_config=profile_config,
        relation_catalog_path=relation_catalog_path, chunk_size=chunk_size,
        run_dir=run_dir, workers=workers, verbose=verbose,
    )
    run_dir = Path(run_dir)
    l1_cfg = _layer_cfg(profile_config, "layer01_linguistic_expression_extraction")
    l2_cfg = _layer_cfg(profile_config, "layer02_candidate_enrichment")
    l4_cfg = _layer_cfg(profile_config, "layer04_candidate_relation_extraction")
    if dataset_key == "fincausal":
        pipeline.layers[1] = FinCausalLayer1(backends["layer01"], dataset_key=dataset_key, audit_path=run_dir/"run_logs/layer01_calls.json", temperature=0.0, save_intermediate=True, verbose=verbose)
        pipeline.layers[2] = FinCausalLayer2(backends["layer02"], dataset_key=dataset_key, relation_catalog_path=relation_catalog_path, decision_log_path=run_dir/"run_logs/layer02_relation_decisions.json", batch_size=int(l2_cfg.get("pair_batch_size", 12)), max_concurrency=int(l2_cfg.get("max_concurrency", min(workers, 4))), save_intermediate=True, verbose=verbose)
    elif dataset_key == "maven_ere":
        pipeline.layers[1] = MavenLayer1(backends["layer01"], dataset_key=dataset_key, audit_path=run_dir/"run_logs/layer01_calls.json", temperature=0.0, save_intermediate=True, verbose=verbose)
        pipeline.layers[2] = MavenLayer2(backends["layer02"], dataset_key=dataset_key, relation_catalog_path=relation_catalog_path, decision_log_path=run_dir/"run_logs/layer02_relation_decisions.json", batch_size=int(l2_cfg.get("pair_batch_size", 36)), max_concurrency=int(l2_cfg.get("max_concurrency", min(workers, 4))), save_intermediate=True, verbose=verbose)
    elif dataset_key == "causalbank":
        pipeline.layers[1] = CausalBankLayer1(audit_path=run_dir/"run_logs/layer01_causalbank_lexical_inventory.json", save_intermediate=True, verbose=verbose)
        pipeline.layers[2] = CausalBankLayer2(backends["layer02"], dataset_key=dataset_key, relation_catalog_path=relation_catalog_path, decision_log_path=run_dir/"run_logs/layer02_relation_decisions.json", batch_size=int(l2_cfg.get("pair_batch_size", 40)), max_concurrency=int(l2_cfg.get("max_concurrency", min(workers, 4))), save_intermediate=True, verbose=verbose)
    else:
        raise ValueError(dataset_key)
    pipeline.layers[3] = GenericRelationCanonicalizingLayer(
        backends["other"], relation_catalog_path=relation_catalog_path, max_expressions=None,
        temperature=0.0, save_intermediate=True, verbose=verbose, rag_adapter=rag_adapter,
        max_concurrency=1, retry_failed_calls=0, retry_sleep_seconds=0,
    )
    pipeline.layers[4] = StrictTaskEndpointLayer(
        backends["layer04"], endpoint_log_path=run_dir/"run_logs/layer04_endpoint_assignment.json",
        rejection_log_path=run_dir/"run_logs/layer04_constraint_rejections.json", type_constraints={},
        max_relation_mentions=None, temperature=0.0, save_intermediate=True, verbose=verbose,
        rag_adapter=rag_adapter, max_concurrency=int(l4_cfg.get("max_concurrency", min(workers, 4))),
        retry_failed_calls=0, retry_sleep_seconds=0, max_attempts_per_relation=1,
        retry_wait_seconds=0.25, failure_log_path=run_dir/"run_logs/layer04_relation_errors.jsonl",
    )
    return pipeline


def _prepare_guidance(guidance_path: str | Path, run_dir: Path) -> tuple[Path, UserGuidance]:
    # v1.5's normalization adapter handles both native and compact example schemas.
    normalized_path, _ = v15._prepare_user_guidance_for_native_loader(guidance_path, run_dir)
    guidance = load_user_guidance(str(normalized_path)) or UserGuidance()
    return Path(normalized_path), guidance


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
    run_dir.mkdir(parents=True, exist_ok=True); logs_dir = run_dir / "run_logs"; logs_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(input_jsonl)
    if len(records) != 1: raise ValueError(f"Expected exactly one sanitized record, got {len(records)}")
    record = records[0]
    forbidden = {"entities", "relations", "pred_relations", "ontology_links"} & set(record)
    if forbidden: raise ValueError(f"Pipeline input contains forbidden gold/precomputed fields: {sorted(forbidden)}")
    if not api_key: raise ValueError("OPENROUTER_API_KEY is not set")
    profile = load_document_profile(profile_path=profile_path); profile_dict = profile.to_state_dict()
    task_guidance = read_json(task_guidance_path); profile_dict["_input_task_guidance"] = task_guidance
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
    logger = SharedCallLogger(logs_dir)
    backends = {
        "layer01": _make_backend(logger=logger, layer_tag=f"{dataset_key}_layer01", model_host=host, api_key=api_key, cfg=_layer_cfg(profile_dict, "layer01_linguistic_expression_extraction"), fallback_max_tokens=max_tokens, fallback_timeout=request_timeout, reasoning_effort=reasoning_effort),
        "layer02": _make_backend(logger=logger, layer_tag=f"{dataset_key}_layer02", model_host=host, api_key=api_key, cfg=_layer_cfg(profile_dict, "layer02_candidate_enrichment"), fallback_max_tokens=4096, fallback_timeout=request_timeout, reasoning_effort=reasoning_effort),
        "layer04": _make_backend(logger=logger, layer_tag=f"{dataset_key}_layer04_fallback", model_host=host, api_key=api_key, cfg=_layer_cfg(profile_dict, "layer04_candidate_relation_extraction"), fallback_max_tokens=384, fallback_timeout=60, reasoning_effort=reasoning_effort),
        "other": _make_backend(logger=logger, layer_tag=f"{dataset_key}_other", model_host=host, api_key=api_key, cfg={}, fallback_max_tokens=768, fallback_timeout=90, reasoning_effort=reasoning_effort),
    }
    rag_adapter = v15.v13.v2.OntologyOnlyRAGAdapter(
        seed_ontology, log_path=logs_dir/"ontology_retrieval.jsonl",
        top_k=int((profile_dict.get("rag") or {}).get("top_k", 4)),
        query_expansions=(profile_dict.get("rag") or {}).get("query_expansions", {}) or {},
    )
    pipeline = build_pipeline(dataset_key=dataset_key, backends=backends, rag_adapter=rag_adapter, profile_config=profile_dict, relation_catalog_path=relation_catalog_path, chunk_size=chunk_size, run_dir=run_dir, workers=workers, verbose=verbose)
    state = PipelineState(
        document=build_document(record, input_jsonl), llm_model=model_name, user_guidance=guidance,
        seed_ontology=seed_ontology, artifact_dir=str(run_dir), profile_name=profile.name,
        profile_config=profile_dict,
    )
    runner = Runner(pipeline=pipeline, runs_root=str(run_dir.parent), verbose=verbose, max_workers=workers, enable_checkpoints=True, save_chunk_checkpoints=False)
    manifest = {
        "dataset": DATASET_DISPLAY[dataset_key], "document_id": record.get("document_id"), "title": record.get("title"),
        "experiment_version": "unified4-v1.1", "model_name": model_name, "input_has_gold": False,
        "forbidden_input_fields": ["entities", "relations", "pred_relations", "ontology_links"],
        "ontology_path": str(ontology_path), "profile_path": str(profile_path), "guidance_path": str(guidance_path),
        "task_guidance_path": str(task_guidance_path), "relation_catalog_path": str(relation_catalog_path),
        "chunk_size": chunk_size, "workers": workers, "gold_projection_after_layer12_only": True,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir/"run_manifest.json", manifest)
    write_json(run_dir/"input_task_guidance.json", task_guidance)
    write_json(run_dir/"effective_user_guidance.json", asdict(guidance))
    started = time.time(); console_log = logs_dir/"console.log"; errors_path = logs_dir/"pipeline_errors.jsonl"
    with console_log.open("w", encoding="utf-8") as handle:
        try:
            with redirect_stdout(Tee(sys.stdout, handle)), redirect_stderr(Tee(sys.stderr, handle)):
                final_state = runner.run(state)
        except Exception as exc:
            append_jsonl(errors_path, {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            raise
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S"); manifest["elapsed_seconds"] = time.time()-started
    manifest["final_state_counts"] = state_counts(final_state); manifest["llm_call_count"] = logger.call_index
    write_json(run_dir/"run_manifest.json", manifest)
    return final_state


def triples_from_state(state: PipelineState) -> list[tuple[str, str, str]]:
    return v15.v13._triples_from_state(state)


# -------------------------- post-L12 evaluation --------------------------
def _metric(pred: set[tuple[str, str, str]], gold: set[tuple[str, str, str]]) -> dict[str, Any]:
    tp = len(pred & gold); fp = len(pred-gold); fn = len(gold-pred)
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return {"pred": len(pred), "gold": len(gold), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def _gold_relation_set(record: dict[str, Any], allowed: set[str]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for rel, pairs in (record.get("relations") or {}).items():
        rel_u = str(rel).upper()
        if rel_u not in allowed: continue
        for pair in pairs or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                result.add((str(pair[0]), rel_u, str(pair[1])))
    return result


def _gold_entity_maps(record: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    entities = record.get("entities") or {}
    by_id = {str(k): v for k, v in entities.items()}
    text_to_id: dict[str, str] = {}
    for eid, entity in by_id.items():
        for m in entity.get("mentions", []) or []:
            text = str(m.get("trigger_word") or m.get("text") or "").strip()
            if text:
                text_to_id.setdefault(_norm(text), eid)
    return by_id, text_to_id


def _map_fincausal_endpoint(label: str, gold: dict[str, Any]) -> str | None:
    text = label.split("::", 1)[-1] if label.startswith("FSPAN:") else label
    _, text_to_id = _gold_entity_maps(gold)
    return text_to_id.get(_norm(text))


def _map_maven_endpoint(label: str, gold: dict[str, Any]) -> str | None:
    spans = set(_maven_cluster_positions(label))
    if not spans: return None
    matches: list[tuple[int, str]] = []
    for eid, entity in (gold.get("entities") or {}).items():
        entity_spans = set()
        for m in entity.get("mentions", []) or []:
            off = m.get("offset")
            if isinstance(off, list) and len(off) == 2 and m.get("sent_id") is not None:
                entity_spans.add((int(m["sent_id"]), int(off[0]), int(off[1])))
        overlap = len(spans & entity_spans)
        if overlap: matches.append((overlap, str(eid)))
    if not matches: return None
    matches.sort(reverse=True)
    return matches[0][1] if len(matches) == 1 or matches[0][0] > matches[1][0] else None


def _map_causalbank_endpoint(label: str, gold: dict[str, Any]) -> str | None:
    candidate = f"EVENT_{md5(str(label).encode('utf-8')).hexdigest()[:16]}"
    return candidate if candidate in (gold.get("entities") or {}) else None


def evaluate_state(dataset_key: str, state: PipelineState, gold_record: dict[str, Any]) -> dict[str, Any]:
    if dataset_key == "eventstoryline":
        raise ValueError("Use EventStoryLine v1.7 analyze_run for ESL")
    mapper = {
        "fincausal": _map_fincausal_endpoint,
        "maven_ere": _map_maven_endpoint,
        "causalbank": _map_causalbank_endpoint,
    }[dataset_key]
    allowed = set(RELATION_IDS[dataset_key])
    raw_triples = triples_from_state(state)
    projected: set[tuple[str, str, str]] = set()
    endpoint_labels: set[str] = set()
    endpoint_mapped: set[str] = set()
    projection_rows: list[dict[str, Any]] = []
    for source, rel, target in raw_triples:
        rel_u = str(rel).upper().split(" : ", 1)[0]
        if rel_u not in allowed: continue
        source_id = mapper(str(source), gold_record); target_id = mapper(str(target), gold_record)
        endpoint_labels.update([str(source), str(target)])
        if source_id: endpoint_mapped.add(source_id)
        if target_id: endpoint_mapped.add(target_id)
        row = {"source": source, "relation": rel_u, "target": target, "source_gold_id": source_id, "target_gold_id": target_id}
        projection_rows.append(row)
        if source_id and target_id:
            projected.add((source_id, rel_u, target_id))
    gold_rel = _gold_relation_set(gold_record, allowed)
    gold_endpoint_ids = {x for s, _, t in gold_rel for x in (s, t)}
    ep_tp = len(endpoint_mapped & gold_endpoint_ids)
    ep_p = ep_tp/len(endpoint_mapped) if endpoint_mapped else 0.0
    ep_r = ep_tp/len(gold_endpoint_ids) if gold_endpoint_ids else 0.0
    ep_f = 2*ep_p*ep_r/(ep_p+ep_r) if ep_p+ep_r else 0.0
    return {
        "dataset": dataset_key,
        "relation_metrics": _metric(projected, gold_rel),
        "endpoint_metrics": {"pred_mapped_unique": len(endpoint_mapped), "gold_unique": len(gold_endpoint_ids), "tp": ep_tp, "precision": ep_p, "recall": ep_r, "f1": ep_f},
        "raw_native_triple_count": len(raw_triples),
        "projected_triples": sorted(projected),
        "gold_triples": sorted(gold_rel),
        "projection_rows": projection_rows,
    }


def causalbank_event_id(lemma: str) -> str:
    return f"EVENT_{md5(str(lemma).encode('utf-8')).hexdigest()[:16]}"


def offline_self_test() -> dict[str, Any]:
    assert causalbank_event_id("a") == "EVENT_0cc175b9c0f1b6a8"
    assert _relation_parts("A || CAUSE || B") == ("A", "CAUSE", "B")
    assert _maven_cluster_positions("MCL:x::S1[2:3]::hit && S4[5:6]::hit") == [(1,2,3),(4,5,6)]
    assert CausalBankLayer2._relation_from_record_type("resulted_from") == "BECAUSE"
    assert CausalBankLayer2._relation_from_record_type("because_of") == "BECAUSE"
    assert CausalBankLayer2._relation_from_record_type("therefore") == "THEREFORE"
    assert CausalBankLayer2._relation_from_record_type("resulted_in") == "THEREFORE"
    assert set(RELATION_IDS["causalbank"]) == {"BECAUSE", "THEREFORE"}
    return {
        "ok": True,
        "causalbank_hash_a": causalbank_event_id("a"),
        "cache_root": str(_cache_root()),
        "datasets": list(DATASET_DISPLAY),
        "causalbank_relations": list(RELATION_IDS["causalbank"]),
    }

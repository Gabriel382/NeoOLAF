from __future__ import annotations

"""NeoOLAF native DocRED one-document ablation support, v5.

This module is experiment-only: it does not modify ``src/neoolaf``.

Changes relative to v4:

* the single whole-document Layer 1 call performs an explicit country-coverage
  self-check and emits separate country relation instances when supported by
  the document, demonyms, coreference, or very high-confidence geography;
* Layer 2 keeps the ontology RAG call but sends only a compact top-k property
  shortlist, relation-specific rules, and at most two relevant examples;
* transparent profile guardrails canonicalize already-extracted relation
  instances for difficult DocRED distinctions (P127/P749/P361, P159/P276/P131,
  P571/P577, and P17/P27). They never create a new relation instance;
* deterministic entity projection is used only after pipeline execution for
  benchmark evaluation and can map mention variants such as
  ``Athens metropolitan area`` to the DocRED ``Athens`` cluster;
* projection methods, Layer 1 country coverage, raw LLM choices, guardrail
  overrides, compact candidates, and prompt sizes are saved for auditing.

The full native Layer 0--12 pipeline still runs. No direct DocRED extractor,
source-entity anchoring, gold relation hint, closure rule, or post-hoc relation
invention is introduced.
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
import unicodedata

import docred_native_ablation as v2
import docred_native_ablation_v3 as v3
import docred_native_ablation_v4 as v4

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.documents import Document
from neoolaf.domain.enriched_expression import EnrichedExpression, EnrichmentEvidence
from neoolaf.domain.linguistic_expression import LinguisticExpression
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
PriorityOntologyRAGAdapter = v4.PriorityOntologyRAGAdapter
CanonicalizingCandidateTypingResolutionLayer = v4.CanonicalizingCandidateTypingResolutionLayer


def _dedup(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _norm(text: Any) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _relation_id(text: Any) -> str | None:
    match = re.search(r"\bP\d+\b", str(text or ""), re.I)
    return match.group(0).upper() if match else None


def _parse_relation_instance(text: Any) -> tuple[str, str, str] | None:
    parts = [part.strip() for part in str(text or "").split("||")]
    return tuple(parts) if len(parts) == 3 and all(parts) else None


def _json_block(value: Any, max_chars: int = 8000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def _short(text: Any, max_chars: int = 220) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."


class CountryAwareRelationInstanceExtractionLayer(v4.DocREDRelationInstanceExtractionLayer):
    """One native whole-document Layer 1 call with a country coverage checklist."""

    def __init__(self, *args: Any, country_audit_path: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.country_audit_path = Path(country_audit_path)

    def _prompt(self, state: PipelineState, chunk_text: str) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        layer_guidance = (task.get("layer_guidance") or {}).get("layer01", {})
        examples = task.get("relation_instance_examples_v5") or task.get("relation_instance_examples") or []
        country_examples = task.get("country_propagation_examples") or []
        priority_specs = task.get("priority_relation_specs") or []

        compact_specs = [
            {
                "id": item.get("relation_id"),
                "label": item.get("label"),
                "direction": item.get("direction"),
            }
            for item in priority_specs
            if item.get("relation_id") in {
                "P17", "P27", "P127", "P131", "P159", "P361", "P463",
                "P571", "P577", "P749",
            }
        ]

        system = """
You are NeoOLAF Layer 1 for document-level DocRED linguistic-expression extraction.

Return endpoint expressions and relation-instance expressions only.

Endpoint labels:
entity_org, entity_per, entity_loc, entity_time, entity_num, entity_misc

Every relation instance MUST be:
SOURCE || LEXICAL RELATION PHRASE || TARGET
with label relation_instance.

Mandatory relation coverage procedure:
1. Extract every exact named/literal endpoint needed by a relation.
2. Split every independently supported source-target pair. A date and a place
   in the same clause must be two relation instances.
3. Identify country anchors from explicit country names, unambiguous demonyms,
   and phrases such as "the country".
4. For each non-human named organization or location, emit a separate country
   relation instance when the text, a demonym/coreference chain, or very
   high-confidence city-country knowledge supports it. This includes the main
   organization, its explicitly associated corporate group, and named cities.
5. For country relation instances, use one of these lexical predicates exactly:
   "is in country", "demonym implies country", or "the country refers to".
   Normalize a demonym to the country noun only when unambiguous and explain it.
   Do not use country of citizenship for people here.
6. Before returning, self-check that each supported ORG/LOC -> country pair has
   its own relation_instance.

Use exact surface forms whenever possible. A controlled inferred endpoint may
use a canonical country noun. Do not select P-identifiers in Layer 1. Do not
invent obscure facts and do not use gold annotations.

Return JSON only:
{"expressions":[{"text":"...","label":"entity_org","justification":"..."},
{"text":"SOURCE || predicate || TARGET","label":"relation_instance",
"justification":"explicit/inferred; source_type=ORG; target_type=LOC; evidence=..."}],
"coverage_check":{"country_anchor":"...","supported_country_subjects":["..."]}}
""".strip()

        user = f"""
Layer 1 profile:
{_json_block(layer_guidance, 4500)}

Relations to notice at this layer (do not choose their P-ID yet):
{_json_block(compact_specs, 4500)}

Synthetic relation-instance examples:
{_json_block(examples, 6500)}

Synthetic country-propagation examples:
{_json_block(country_examples, 5000)}

Document:
\"\"\"
{chunk_text}
\"\"\"

Extract endpoint expressions and one structured relation instance per supported
pair. Perform the country-coverage self-check. Return JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _run(self, state: PipelineState) -> PipelineState:
        state = super()._run(state)
        endpoints = []
        relations = []
        country_relations = []
        country_predicate_terms = {
            "country", "demonym", "nationality", "national", "sovereign state",
            "is in", "belongs to",
        }
        for expr in state.linguistic_expressions or []:
            parsed = _parse_relation_instance(expr.text)
            if parsed:
                source, predicate, target = parsed
                row = {
                    "expr_id": expr.expr_id,
                    "source": source,
                    "predicate": predicate,
                    "target": target,
                    "justification": expr.justification,
                }
                relations.append(row)
                predicate_n = _norm(predicate)
                if any(term in predicate_n for term in country_predicate_terms):
                    country_relations.append(row)
            else:
                endpoints.append({
                    "expr_id": expr.expr_id,
                    "text": expr.text,
                    "label": expr.label,
                })
        audit = {
            "endpoint_count": len(endpoints),
            "relation_instance_count": len(relations),
            "country_relation_instance_count": len(country_relations),
            "country_relation_instances": country_relations,
            "note": (
                "This is a non-gold runtime audit. It reports what the single native "
                "Layer 1 call emitted and does not create missing relations."
            ),
        }
        self.country_audit_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.country_audit_path, audit)
        state.log(
            f"[{self.name}] country coverage audit; country_relation_instances={len(country_relations)}"
        )
        return state


class CompactGuardrailedCandidateEnrichmentLayer(v4.SelectiveContrastiveCandidateEnrichmentLayer):
    """Relation-only Layer 2 with compact ontology prompts and transparent guardrails."""

    def __init__(self, *args: Any, compact_prompt_log_path: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.compact_prompt_log_path = Path(compact_prompt_log_path)
        self._compact_prompt_rows: list[dict[str, Any]] = []
        self._compact_lock = threading.Lock()

    @staticmethod
    def _endpoint_type_from_state(state: PipelineState, label: str) -> str:
        target = _norm(label)
        mapping = {
            "entity_org": "ORG", "entity_per": "PER", "entity_loc": "LOC",
            "entity_time": "TIME", "entity_num": "NUM", "entity_misc": "MISC",
        }
        for expr in state.linguistic_expressions or []:
            if _parse_relation_instance(expr.text):
                continue
            if _norm(expr.text) == target:
                return mapping.get(str(expr.label).lower(), "UNKNOWN")
        return "UNKNOWN"

    @staticmethod
    def _lexical_candidate_ids(
        predicate: str,
        source: str,
        target: str,
        source_type: str,
        target_type: str,
    ) -> list[str]:
        p = _norm(predicate)
        context = _norm(f"{source} {predicate} {target}")
        ids: list[str] = []

        if any(term in p for term in ["headquarter", "based in", "base in", "based at"]):
            ids += ["P159", "P131", "P276"]
        if any(term in p for term in ["relaunch", "launch", "open", "establish", "found"]):
            if target_type == "TIME":
                ids += ["P571", "P580", "P577"]
            elif source_type == "ORG" and target_type == "LOC":
                ids += ["P159", "P276", "P131"]
        if "part of" in p or "belong" in p or "component of" in p:
            if source_type == "ORG" and target_type == "ORG":
                ids += ["P127", "P749", "P361"]
            else:
                ids += ["P361", "P527"]
        if any(term in context for term in ["subsidiary", "branch", "division", "laboratory", "lab ", "unit of"]):
            ids += ["P749", "P127", "P361"]
        if "member of" in p or "membership" in p:
            ids += ["P463", "P361"]
        if any(term in p for term in ["country", "demonym", "national", "is in"]):
            ids += ["P17", "P27", "P131"]
        if "available on" in p or "broadcast on" in p or "platform" in p:
            ids += ["P400", "P449"]
        if any(term in p for term in ["published", "publication", "released"]):
            ids += ["P577", "P571"]
        if "located in" in p:
            ids += ["P131", "P276", "P159"]
        return _dedup(ids)

    @staticmethod
    def _relevant_rows(
        rows: Iterable[dict[str, Any]],
        *,
        candidate_ids: set[str],
        predicate: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        predicate_tokens = set(_norm(predicate).split())
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in rows or []:
            row_ids = {
                value.upper()
                for value in re.findall(r"\bP\d+\b", json.dumps(row, ensure_ascii=False), re.I)
            }
            text_tokens = set(_norm(json.dumps(row, ensure_ascii=False)).split())
            score = 4 * len(row_ids & candidate_ids) + len(predicate_tokens & text_tokens)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def _compact_catalog_rows(
        self,
        candidate_ids: list[str],
        type_constraints: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for relation_id in candidate_ids:
            item = self.catalog.get(relation_id)
            if not item:
                continue
            constraint = type_constraints.get(relation_id, {})
            rows.append({
                "id": relation_id,
                "label": item.get("label"),
                "definition": _short(item.get("comment"), 210),
                "source_types": constraint.get("source") or [],
                "target_types": constraint.get("target") or [],
            })
        return rows

    def _candidate_ids(
        self,
        *,
        state: PipelineState,
        source: str,
        predicate: str,
        target: str,
        source_type: str,
        target_type: str,
        grounding_text: str,
    ) -> list[str]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        allowed = set(task.get("allowed_relation_ids") or self.catalog)
        profile_sets = task.get("predicate_candidate_sets") or []
        profile_ids: list[str] = []
        predicate_n = _norm(predicate)
        for row in profile_sets:
            triggers = [_norm(x) for x in row.get("triggers", [])]
            if any(trigger and trigger in predicate_n for trigger in triggers):
                profile_ids.extend(row.get("candidate_relation_ids") or [])
        lexical = self._lexical_candidate_ids(predicate, source, target, source_type, target_type)
        retrieved = self._candidate_ids_from_grounding(grounding_text)
        cap = int((state.profile_config or {}).get("layer02_compact_candidate_cap", 5))
        ordered = _dedup([*profile_ids, *lexical, *retrieved])
        result = [rid for rid in ordered if rid in allowed and rid in self.catalog][:cap]
        if not result:
            result = [rid for rid in task.get("priority_relation_ids", []) if rid in self.catalog][:cap]
        return result

    @staticmethod
    def _guardrail_relation_id(
        *,
        selected_id: str | None,
        predicate: str,
        source: str,
        target: str,
        source_type: str,
        target_type: str,
        candidate_ids: Iterable[str],
    ) -> tuple[str | None, str | None]:
        p = _norm(predicate)
        context = _norm(f"{source} {predicate} {target}")
        candidates = set(candidate_ids)

        def choose(relation_id: str, reason: str) -> tuple[str | None, str | None]:
            if relation_id in candidates:
                return relation_id, reason
            return selected_id, None

        if source_type == "ORG" and target_type == "ORG":
            if any(term in context for term in ["subsidiary", "branch", "division", "laboratory", "lab ", "unit of"]):
                return choose("P749", "Explicit branch/subsidiary/unit wording requires child-to-parent P749.")
            if "part of" in p and any(term in context for term in ["group", "company", "corporate", "media"]):
                return choose("P127", "Corporate/media-group part-of wording is canonicalized to P127 owned by; P749 is reserved for explicit branch/subsidiary units.")

        if source_type == "ORG" and target_type == "LOC":
            if any(term in p for term in ["based in", "headquarter", "base in", "based at"]):
                return choose("P159", "Organization base/headquarters city requires P159.")
            if any(term in p for term in ["relaunch", "launch", "opened", "established in"]):
                return choose("P159", "DocRED organization launch/relaunch location convention uses P159 rather than generic P276/P131.")

        if source_type in {"ORG", "LOC", "MISC"} and target_type == "LOC":
            if any(term in p for term in ["country", "demonym", "national"]):
                return choose("P17", "Non-human entity/place to sovereign country requires P17, not P27.")

        if target_type == "TIME" and source_type in {"ORG", "LOC", "MISC"}:
            if any(term in p for term in ["relaunch", "launch", "establish", "found", "inception"]):
                return choose("P571", "Organization/entity establishment or relaunch date requires P571.")

        return selected_id, None

    def _contrastive_prompt_compact(
        self,
        *,
        expr: LinguisticExpression,
        state: PipelineState,
        source: str,
        predicate: str,
        target: str,
        source_type: str,
        target_type: str,
        candidate_ids: list[str],
    ) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        constraints = (state.profile_config or {}).get("docred_type_constraints", {}) or {}
        candidate_set = set(candidate_ids)
        rules = self._relevant_rows(
            task.get("contrastive_rules") or [],
            candidate_ids=candidate_set,
            predicate=predicate,
            limit=3,
        )
        examples = self._relevant_rows(
            task.get("relation_examples") or [],
            candidate_ids=candidate_set,
            predicate=predicate,
            limit=2,
        )
        candidates = self._compact_catalog_rows(candidate_ids, constraints)

        system = """
You are NeoOLAF Layer 2. Classify one already-extracted relation instance into
exactly one supplied DocRED ontology property, or found=false.

Use endpoint types, direction, lexical meaning, specificity, and the compact
ontology definitions. Prefer the specific benchmark property over a generic
property. Do not create endpoints or relation instances. Do not use gold data.

Return JSON only:
{"found":true,"selected_relation_id":"P159","decision":"brief contrastive reason"}
or {"found":false,"decision":"brief reason"}.
""".strip()
        user = f"""
INSTANCE: {source} || {predicate} || {target}
TYPES: {source_type} -> {target_type}
EVIDENCE: {_short(expr.justification, 500)}
CANDIDATES: {_json_block(candidates, 3500)}
RELEVANT_RULES: {_json_block(rules, 1800)}
RELEVANT_EXAMPLES: {_json_block(examples, 2200)}
Choose one candidate ID or found=false. JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _process_relation_contrastive(self, expr: LinguisticExpression, state: PipelineState) -> EnrichedExpression:
        parsed_instance = _parse_relation_instance(expr.text)
        if parsed_instance is None:
            raise ValueError(f"Malformed relation instance: {expr.text}")
        source, predicate, target = parsed_instance
        source_type = self._type_from_justification(expr.justification, "source_type")
        target_type = self._type_from_justification(expr.justification, "target_type")
        if source_type == "UNKNOWN":
            source_type = self._endpoint_type_from_state(state, source)
        if target_type == "UNKNOWN":
            target_type = self._endpoint_type_from_state(state, target)

        grounding_text = ""
        if self.rag_adapter is not None:
            grounding = self.rag_adapter.ground(GroundingRequest(
                layer_name=self.name,
                query=f"{predicate} {source_type} {target_type}",
                payload={
                    "relation_instance": expr.text,
                    "source": source,
                    "predicate": predicate,
                    "target": target,
                    "source_type": source_type,
                    "target_type": target_type,
                    "justification": expr.justification,
                },
                preferred_sources=["ontology"],
                top_k=6,
            ))
            grounding_text = build_grounding_context(grounding)

        candidate_ids = self._candidate_ids(
            state=state,
            source=source,
            predicate=predicate,
            target=target,
            source_type=source_type,
            target_type=target_type,
            grounding_text=grounding_text,
        )
        messages = self._contrastive_prompt_compact(
            expr=expr,
            state=state,
            source=source,
            predicate=predicate,
            target=target,
            source_type=source_type,
            target_type=target_type,
            candidate_ids=candidate_ids,
        )
        raw = self.ollama_backend.chat(
            model=state.llm_model,
            messages=messages,
            temperature=0.0,
        )
        parsed = self.ollama_backend.extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Layer 2 compact response is not a JSON object")

        found = bool(parsed.get("found", False))
        raw_selected_id = _relation_id(parsed.get("selected_relation_id"))
        if found and raw_selected_id not in candidate_ids:
            raise ValueError(
                f"Layer 2 selected {raw_selected_id}, outside compact candidates {candidate_ids}"
            )
        selected_id, override_reason = self._guardrail_relation_id(
            selected_id=raw_selected_id if found else None,
            predicate=predicate,
            source=source,
            target=target,
            source_type=source_type,
            target_type=target_type,
            candidate_ids=candidate_ids,
        )
        if selected_id is not None:
            found = True
        if found and selected_id not in self.catalog:
            raise ValueError(f"Layer 2 selected invalid relation ID: {selected_id}")

        if found:
            item = self.catalog[selected_id]
            canonical = f"{selected_id} : {item['label']}"
            decision_text = str(parsed.get("decision") or "").strip()
            if override_reason:
                decision_text = f"{decision_text} PROFILE_GUARDRAIL: {override_reason}".strip()
            hints = _dedup([
                f"controlled_relation:{canonical}",
                "promote_to_ontology:true",
                item.get("uri"),
                item.get("label"),
                f"source_label:{source}",
                f"target_label:{target}",
                f"lexical_predicate:{predicate}",
                f"source_type:{source_type}",
                f"target_type:{target_type}",
                f"contrastive_decision:{decision_text}",
            ])
            definition = str(item.get("comment") or item.get("label") or "").strip()
        else:
            canonical = None
            decision_text = str(parsed.get("decision") or "No supported ontology relation selected.")
            hints = _dedup([
                "promote_to_ontology:false",
                f"source_label:{source}",
                f"target_label:{target}",
                f"lexical_predicate:{predicate}",
                f"source_type:{source_type}",
                f"target_type:{target_type}",
                f"contrastive_decision:{decision_text}",
            ])
            definition = decision_text

        decision_row = {
            "expr_id": expr.expr_id,
            "relation_instance": expr.text,
            "source": source,
            "predicate": predicate,
            "target": target,
            "source_type": source_type,
            "target_type": target_type,
            "candidate_relation_ids": candidate_ids,
            "found": found,
            "raw_selected_relation_id": raw_selected_id,
            "selected_relation_id": selected_id,
            "canonical_relation": canonical,
            "guardrail_override": bool(override_reason and selected_id != raw_selected_id),
            "guardrail_reason": override_reason,
            "decision": decision_text,
            "ontology_hints": hints,
            "system_chars": len(messages[0]["content"]),
            "user_chars": len(messages[1]["content"]),
        }
        self._record_decision(decision_row)
        with self._compact_lock:
            self._compact_prompt_rows.append({
                key: decision_row[key]
                for key in [
                    "expr_id", "relation_instance", "candidate_relation_ids",
                    "raw_selected_relation_id", "selected_relation_id",
                    "guardrail_override", "system_chars", "user_chars",
                ]
            })

        aliases = _dedup([expr.text, predicate])
        return EnrichedExpression(
            base_expression=expr,
            aliases=aliases,
            synonyms=[],
            lexical_variants=[],
            alias_sources={value: ["source"] for value in aliases},
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
        self._compact_prompt_rows = []
        state = super()._run(state)
        self.compact_prompt_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            self.compact_prompt_log_path,
            sorted(self._compact_prompt_rows, key=lambda row: row.get("expr_id", "")),
        )
        return state


# ---------------------------------------------------------------------------
# Post-run mention-aware benchmark projection (evaluation only)
# ---------------------------------------------------------------------------

_ADMIN_SUFFIXES = [
    ("metropolitan", "area"), ("metro", "area"), ("urban", "area"),
    ("city",), ("county",), ("province",), ("region",), ("district",),
    ("municipality",),
]
_STOPWORDS = {"the", "of"}


def _tokens(text: Any) -> tuple[str, ...]:
    return tuple(token for token in _norm(text).split() if token not in _STOPWORDS)


def _strip_admin_suffix(tokens: tuple[str, ...]) -> tuple[str, ...]:
    result = tokens
    changed = True
    while changed and result:
        changed = False
        for suffix in _ADMIN_SUFFIXES:
            if len(result) > len(suffix) and result[-len(suffix):] == suffix:
                result = result[:-len(suffix)]
                changed = True
                break
    return result


def _candidate_values(candidate: Any) -> list[str]:
    values = [getattr(candidate, "canonical_label", "")]
    values.extend(getattr(candidate, "aliases", []) or [])
    values.extend(getattr(candidate, "synonyms", []) or [])
    values.extend(getattr(candidate, "lexical_variants", []) or [])
    values.extend(getattr(mention, "text", "") for mention in getattr(candidate, "mentions", []) or [])
    return _dedup(values)


def _gold_alias_rows(gold: dict[str, Any]) -> dict[str, list[str]]:
    return {
        entity_id: _dedup(
            mention.get("trigger_word")
            for mention in payload.get("mentions", [])
            if mention.get("trigger_word")
        )
        for entity_id, payload in gold.get("entities", {}).items()
    }


def project_values_to_gold(values: Iterable[str], gold: dict[str, Any]) -> dict[str, Any]:
    candidate_values = _dedup(values)
    gold_aliases = _gold_alias_rows(gold)
    matches: list[dict[str, Any]] = []

    for candidate_value in candidate_values:
        c_norm = _norm(candidate_value)
        c_tokens = _tokens(candidate_value)
        c_stripped = _strip_admin_suffix(c_tokens)
        if not c_norm:
            continue
        for entity_id, aliases in gold_aliases.items():
            for alias in aliases:
                g_norm = _norm(alias)
                g_tokens = _tokens(alias)
                g_stripped = _strip_admin_suffix(g_tokens)
                method = None
                score = 0.0
                if c_norm == g_norm:
                    method, score = "exact", 1.0
                elif c_stripped and c_stripped == g_stripped:
                    method, score = "administrative_suffix_normalization", 0.96
                elif len(g_tokens) >= 1 and len(c_tokens) > len(g_tokens):
                    # Require a token-boundary prefix/suffix match, not arbitrary substring.
                    if c_tokens[: len(g_tokens)] == g_tokens or c_tokens[-len(g_tokens):] == g_tokens:
                        method, score = "token_boundary_containment", 0.88
                elif len(c_tokens) >= 2 and len(g_tokens) > len(c_tokens):
                    if g_tokens[: len(c_tokens)] == c_tokens or g_tokens[-len(c_tokens):] == c_tokens:
                        method, score = "reverse_token_boundary_containment", 0.84
                if method:
                    matches.append({
                        "entity_id": entity_id,
                        "candidate_value": candidate_value,
                        "gold_alias": alias,
                        "method": method,
                        "score": score,
                    })

    if not matches:
        return {
            "entity_id": None,
            "method": "unmapped",
            "score": 0.0,
            "candidate_value": candidate_values[0] if candidate_values else None,
            "gold_alias": None,
            "ambiguous": False,
        }
    best_score = max(row["score"] for row in matches)
    best = [row for row in matches if row["score"] == best_score]
    best_ids = {row["entity_id"] for row in best}
    if len(best_ids) != 1:
        return {
            "entity_id": None,
            "method": "ambiguous",
            "score": best_score,
            "candidate_value": best[0]["candidate_value"],
            "gold_alias": None,
            "ambiguous": True,
            "candidate_entity_ids": sorted(best_ids),
        }
    selected = next(row for row in best if row["entity_id"] in best_ids)
    return {**selected, "ambiguous": False}


def project_candidate_to_gold(candidate: Any, gold: dict[str, Any]) -> dict[str, Any]:
    if candidate is None:
        return {"entity_id": None, "method": "missing_candidate", "score": 0.0, "ambiguous": False}
    return project_values_to_gold(_candidate_values(candidate), gold)


def native_predictions_v5(
    state: PipelineState,
    gold: dict[str, Any],
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> list[dict[str, Any]]:
    _, label_to_id = v2.relation_lookup(catalog_path, aliases_path)
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in [
            *(state.entity_candidates or []), *(state.event_candidates or []),
            *(state.attribute_candidates or []), *(state.relation_candidates or []),
        ]
    }
    relation_by_id = {candidate.candidate_id: candidate for candidate in state.relation_candidates or []}
    rows: list[dict[str, Any]] = []
    for triple in state.candidate_triples or []:
        subject = candidate_by_id.get(triple.subject_id)
        obj = candidate_by_id.get(triple.object_id)
        relation_candidate = relation_by_id.get(triple.predicate_id)
        relation_id = v2.map_predicate(
            triple.predicate_label,
            getattr(relation_candidate, "ontology_hints", []) if relation_candidate else [],
            label_to_id,
        )
        head_projection = project_candidate_to_gold(subject, gold)
        tail_projection = project_candidate_to_gold(obj, gold)
        rows.append({
            "triple_id": triple.triple_id,
            "subject_label": triple.subject_label,
            "predicate_label": triple.predicate_label,
            "object_label": triple.object_label,
            "head_id": head_projection.get("entity_id"),
            "relation_id": relation_id,
            "tail_id": tail_projection.get("entity_id"),
            "fully_mapped": bool(head_projection.get("entity_id") and relation_id and tail_projection.get("entity_id")),
            "head_projection_method": head_projection.get("method"),
            "head_projection_score": head_projection.get("score"),
            "head_matched_value": head_projection.get("candidate_value"),
            "tail_projection_method": tail_projection.get("method"),
            "tail_projection_score": tail_projection.get("score"),
            "tail_matched_value": tail_projection.get("candidate_value"),
            "confidence": triple.confidence,
            "justification": triple.justification,
        })
    return rows


def assertion_predictions_v5(
    state: PipelineState,
    gold: dict[str, Any],
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> list[dict[str, Any]]:
    _, label_to_id = v2.relation_lookup(catalog_path, aliases_path)
    candidates = {
        c.candidate_id: c
        for c in [
            *(state.entity_candidates or []), *(state.event_candidates or []),
            *(state.attribute_candidates or []),
        ]
    }
    relations = {c.candidate_id: c for c in state.relation_candidates or []}
    rows: list[dict[str, Any]] = []
    for assertion in state.candidate_relation_assertions or []:
        src = candidates.get(assertion.source_candidate_id)
        dst = candidates.get(assertion.target_candidate_id)
        rel = relations.get(assertion.relation_candidate_id)
        relation_id = v2.map_predicate(
            assertion.relation_label,
            getattr(rel, "ontology_hints", []) if rel else [],
            label_to_id,
        )
        head = project_candidate_to_gold(src, gold)
        tail = project_candidate_to_gold(dst, gold)
        rows.append({
            "assertion_id": assertion.assertion_id,
            "head_id": head.get("entity_id"),
            "relation_id": relation_id,
            "tail_id": tail.get("entity_id"),
            "fully_mapped": bool(head.get("entity_id") and relation_id and tail.get("entity_id")),
            "source_label": assertion.source_candidate_label,
            "predicate_label": assertion.relation_label,
            "target_label": assertion.target_candidate_label,
            "head_projection_method": head.get("method"),
            "tail_projection_method": tail.get("method"),
        })
    return rows


def write_entity_projection_audit(
    run_dir: str | Path,
    state: PipelineState,
    gold: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in [
        *(state.entity_candidates or []), *(state.event_candidates or []),
        *(state.attribute_candidates or []),
    ]:
        projection = project_candidate_to_gold(candidate, gold)
        rows.append({
            "candidate_id": candidate.candidate_id,
            "canonical_label": candidate.canonical_label,
            "candidate_values": " | ".join(_candidate_values(candidate)),
            **projection,
        })
    analysis_dir = Path(run_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_json(analysis_dir / "entity_projection_audit_v5.json", rows)
    if rows:
        with (analysis_dir / "entity_projection_audit_v5.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def write_cumulative_evaluation_v5(
    run_dir: str | Path,
    gold: dict[str, Any],
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name, state in load_layer_states(run_dir):
        if index < 4:
            predictions = []
        elif index == 4:
            predictions = assertion_predictions_v5(state, gold, catalog_path, aliases_path)
        else:
            predictions = native_predictions_v5(state, gold, catalog_path, aliases_path)
        evaluation = v2.strict_evaluate(predictions, gold)
        rows.append({
            "layer_index": index,
            "layer_name": name,
            "mapped_predictions": sum(1 for row in predictions if row.get("fully_mapped")),
            **{key: evaluation[key] for key in [
                "predicted", "gold", "true_positive", "false_positive",
                "false_negative", "precision", "recall", "f1",
            ]},
        })
    analysis_dir = Path(run_dir) / "analysis"
    write_json(analysis_dir / "cumulative_strict_evaluation_v5.json", rows)
    if rows:
        with (analysis_dir / "cumulative_strict_evaluation_v5.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def write_relation_trace_v5(
    *,
    run_dir: str | Path,
    gold_jsonl: str | Path,
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    gold = read_jsonl(gold_jsonl)[0]
    states = {index: state for index, _, state in load_layer_states(run_dir)}
    l1, l2, l3, l4, l5 = (states.get(i) for i in [1, 2, 3, 4, 5])
    decisions_path = run_dir / "run_logs/layer02_contrastive_decisions.json"
    decisions = read_json(decisions_path) if decisions_path.is_file() else []

    l1_endpoint_ids: set[str] = set()
    l1_relation_keys: set[tuple[str, str]] = set()
    for expr in (l1.linguistic_expressions if l1 else []):
        parsed = _parse_relation_instance(expr.text)
        if parsed:
            source, _, target = parsed
            src = project_values_to_gold([source], gold).get("entity_id")
            dst = project_values_to_gold([target], gold).get("entity_id")
            if src and dst:
                l1_relation_keys.add((src, dst))
        else:
            projected = project_values_to_gold([expr.text], gold).get("entity_id")
            if projected:
                l1_endpoint_ids.add(projected)

    l2_keys: set[tuple[str, str, str]] = set()
    for row in decisions:
        rid = row.get("selected_relation_id")
        src = project_values_to_gold([row.get("source")], gold).get("entity_id")
        dst = project_values_to_gold([row.get("target")], gold).get("entity_id")
        if src and rid and dst:
            l2_keys.add((src, rid, dst))

    l3_keys: set[tuple[str, str, str]] = set()
    for candidate in (l3.relation_candidates if l3 else []):
        rid = _relation_id(candidate.canonical_label) or _relation_id(" ".join(candidate.ontology_hints or []))
        for value in [*(candidate.aliases or []), *(m.text for m in candidate.mentions or [])]:
            parsed = _parse_relation_instance(value)
            if not parsed:
                continue
            src = project_values_to_gold([parsed[0]], gold).get("entity_id")
            dst = project_values_to_gold([parsed[2]], gold).get("entity_id")
            if src and rid and dst:
                l3_keys.add((src, rid, dst))

    l4_rows = assertion_predictions_v5(l4, gold, catalog_path, aliases_path) if l4 else []
    l4_keys = {(r["head_id"], r["relation_id"], r["tail_id"]) for r in l4_rows if r.get("fully_mapped")}
    l5_rows = native_predictions_v5(l5, gold, catalog_path, aliases_path) if l5 else []
    l5_keys = {(r["head_id"], r["relation_id"], r["tail_id"]) for r in l5_rows if r.get("fully_mapped")}

    def entity_label(entity_id: str) -> str:
        mentions = gold["entities"][entity_id].get("mentions") or []
        return mentions[0].get("trigger_word") if mentions else entity_id

    rows: list[dict[str, Any]] = []
    for head, relation_id, tail in sorted(v2.gold_triples(gold)):
        source_available = head in l1_endpoint_ids
        target_available = tail in l1_endpoint_ids
        relation_instance = (head, tail) in l1_relation_keys
        linked_l2 = (head, relation_id, tail) in l2_keys
        linked_l3 = (head, relation_id, tail) in l3_keys
        assertion_l4 = (head, relation_id, tail) in l4_keys
        triple_l5 = (head, relation_id, tail) in l5_keys
        if not source_available or not target_available:
            failure = "layer01_endpoint_missing_after_projection"
        elif not relation_instance:
            failure = "layer01_relation_instance_missing"
        elif not linked_l2:
            failure = "layer02_wrong_or_missing_controlled_relation"
        elif not linked_l3:
            failure = "layer03_candidate_typing_or_resolution"
        elif not assertion_l4:
            failure = "layer04_endpoint_direction_or_type_validation"
        elif not triple_l5:
            failure = "layer05_triple_materialization_or_projection"
        else:
            failure = "survived_to_layer05"
        rows.append({
            "head_id": head,
            "source": entity_label(head),
            "relation_id": relation_id,
            "tail_id": tail,
            "target": entity_label(tail),
            "source_available_layer01": source_available,
            "target_available_layer01": target_available,
            "relation_instance_layer01": relation_instance,
            "canonical_relation_layer02": linked_l2,
            "relation_candidate_layer03": linked_l3,
            "assertion_layer04": assertion_l4,
            "triple_layer05": triple_l5,
            "first_failure": failure,
        })
    analysis_dir = run_dir / "analysis"
    write_json(analysis_dir / "gold_relation_trace_v5.json", rows)
    if rows:
        with (analysis_dir / "gold_relation_trace_v5.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return rows


def write_native_views_v5(
    run_dir: str | Path,
    state: PipelineState,
    gold: dict[str, Any],
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    analysis_dir = Path(run_dir) / "analysis"
    mapped = native_predictions_v5(state, gold, catalog_path, aliases_path)
    relations = {c.candidate_id: c for c in state.relation_candidates or []}
    lexical: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    for triple, mapped_row in zip(state.candidate_triples or [], mapped):
        rel = relations.get(triple.predicate_id)
        row = {
            "triple_id": triple.triple_id,
            "subject_label": triple.subject_label,
            "predicate_label": triple.predicate_label,
            "object_label": triple.object_label,
            "relation_candidate_id": triple.predicate_id,
            "relation_aliases": list(getattr(rel, "aliases", []) or []) if rel else [],
            "ontology_hints": list(getattr(rel, "ontology_hints", []) or []) if rel else [],
            "confidence": triple.confidence,
            "justification": triple.justification,
        }
        lexical.append(row)
        if mapped_row.get("relation_id"):
            canonical.append({**row, **mapped_row})
    gold_set = v2.gold_triples(gold)
    not_in_gold = [
        {**row, "manual_review_required": True}
        for row in mapped
        if row.get("fully_mapped") and (row["head_id"], row["relation_id"], row["tail_id"]) not in gold_set
    ]
    files = {
        "native_lexical_triples_v5": lexical,
        "ontology_canonical_triples_v5": canonical,
        "strict_docred_predictions_v5": mapped,
        "predictions_not_in_gold_manual_review_v5": not_in_gold,
    }
    for name, rows in files.items():
        write_json(analysis_dir / f"{name}.json", rows)
        if rows:
            with (analysis_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    return files


# ---------------------------------------------------------------------------
# Pipeline construction and execution
# ---------------------------------------------------------------------------


def _layer_cfg(profile: dict[str, Any], layer_name: str) -> dict[str, Any]:
    return dict((profile.get("layers") or {}).get(layer_name) or {})


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


def choose_chunk_size(text: str, max_safe_chars: int = 24000) -> int:
    return v4.choose_chunk_size(text, max_safe_chars)


def build_document(record: dict[str, Any], source_path: str | Path) -> Document:
    return v4.build_document(record, source_path)


def build_pipeline(
    *,
    backends: dict[str, TaggedLoggedBackend],
    rag_adapter: PriorityOntologyRAGAdapter,
    profile_config: dict[str, Any],
    relation_catalog_path: str | Path,
    chunk_size: int,
    run_dir: str | Path,
    workers: int = 16,
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
    retry_default = int((profile_config.get("orchestration") or {}).get("retry_failed_calls", 1))
    sleep_default = float((profile_config.get("orchestration") or {}).get("retry_sleep_seconds", 1.0))
    l2_cfg = _layer_cfg(profile_config, "layer02_candidate_enrichment")
    run_dir = Path(run_dir)

    pipeline.layers[1] = CountryAwareRelationInstanceExtractionLayer(
        backends["layer01"],
        decision_log_path=run_dir / "run_logs/layer01_relation_instances.json",
        country_audit_path=run_dir / "run_logs/layer01_country_coverage_audit.json",
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
    pipeline.layers[2] = CompactGuardrailedCandidateEnrichmentLayer(
        backends["layer02"],
        wikipedia_source=OfflineWikipediaSource(),
        wikidata_source=OfflineWikidataSource(),
        web_search_source=OfflineWebSearchSource(),
        relation_catalog_path=relation_catalog_path,
        decision_log_path=run_dir / "run_logs/layer02_contrastive_decisions.json",
        compact_prompt_log_path=run_dir / "run_logs/layer02_compact_prompt_audit.json",
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
    relation_catalog_path: str | Path,
    relation_aliases_path: str | Path,
    run_dir: str | Path,
    model_name: str,
    api_key: str,
    host: str = "https://openrouter.ai/api/v1",
    workers: int = 16,
    max_tokens: int = 4096,
    request_timeout: int = 120,
    reasoning_effort: str = "minimal",
    verbose: bool = True,
    clean_run_dir: bool = True,
) -> PipelineState:
    project_root = Path(project_root).resolve()
    input_jsonl = Path(input_jsonl).resolve()
    ontology_path = Path(ontology_path).resolve()
    profile_path = Path(profile_path).resolve()
    guidance_path = Path(guidance_path).resolve()
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
        raise ValueError(f"This notebook expects exactly one input document, found {len(records)}")
    record = records[0]
    if "entities" in record or "relations" in record:
        raise ValueError("Pipeline input must not contain gold entities or relations.")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    profile = load_document_profile(profile_path=profile_path)
    profile_dict = profile.to_state_dict()
    profile_dict["_input_task_guidance"] = record.get("task_guidance") or {}
    guidance = load_user_guidance(str(guidance_path)) or UserGuidance()
    guidance = v3.merge_input_task_guidance(guidance, record)
    write_json(run_dir / "input_task_guidance.json", record.get("task_guidance") or {})
    write_json(run_dir / "effective_user_guidance.json", asdict(guidance))

    seed_ontology = SeedOntologyLoader().load(str(ontology_path))
    if len(seed_ontology.properties_by_uri) < 90:
        raise RuntimeError(f"Only {len(seed_ontology.properties_by_uri)} ontology properties were loaded.")

    chunk_size = choose_chunk_size(record["text"], int(profile.get("chunking.max_safe_chunk_chars", 24000)))
    logger = SharedCallLogger(logs_dir)
    backends = {
        "layer01": _make_backend(
            logger=logger, layer_tag="layer01_country_aware", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer01_linguistic_expression_extraction"),
            fallback_max_tokens=max_tokens, fallback_timeout=request_timeout,
            reasoning_effort=reasoning_effort,
        ),
        "layer02": _make_backend(
            logger=logger, layer_tag="layer02_compact_topk", model_host=host, api_key=api_key,
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
    aliases = read_json(relation_aliases_path)
    rag_adapter = PriorityOntologyRAGAdapter(
        seed_ontology,
        log_path=logs_dir / "ontology_retrieval.jsonl",
        top_k=int(profile.get("rag.top_k", 6)),
        query_expansions=profile.get("rag.query_expansions", {}) or {},
        relation_aliases=aliases,
        priority_property_ids=profile.get("rag.priority_property_ids", []) or [],
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
        "document_id": record["document_id"],
        "title": record.get("title"),
        "model_name": model_name,
        "profile_name": profile.name,
        "profile_path": str(profile_path),
        "guidance_path": str(guidance_path),
        "ontology_path": str(ontology_path),
        "ontology_classes": len(seed_ontology.classes_by_uri),
        "ontology_properties": len(seed_ontology.properties_by_uri),
        "input_has_gold": False,
        "chunk_size": chunk_size,
        "whole_document_single_chunk_expected": len(record["text"]) <= chunk_size,
        "workers": workers,
        "layer02_compact_candidate_cap": profile_dict.get("layer02_compact_candidate_cap", 5),
        "layer01_country_coverage_check": True,
        "layer02_profile_guardrails": True,
        "mention_aware_projection_after_execution_only": True,
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
                final_state = runner.run(state, from_layer=0, to_layer=12, run_dir=run_dir)
        except Exception as exc:
            append_jsonl(errors_path, {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            raise

    manifest.update({
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 3),
        "final_counts": state_counts(final_state),
    })
    write_json(run_dir / "run_manifest.json", manifest)
    return final_state


def analyze_run(
    *,
    run_dir: str | Path,
    gold_jsonl: str | Path,
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> dict[str, Any]:
    # Retain all earlier analyses for comparison, then replace the principal
    # benchmark-facing metrics with the audited v5 projection.
    summary = v4.analyze_run(
        run_dir=run_dir,
        gold_jsonl=gold_jsonl,
        catalog_path=catalog_path,
        aliases_path=aliases_path,
    )
    run_dir = Path(run_dir)
    gold = read_jsonl(gold_jsonl)[0]
    states = {index: state for index, _, state in load_layer_states(run_dir)}
    final_state = states[max(states)]

    predictions = native_predictions_v5(final_state, gold, catalog_path, aliases_path)
    evaluation = v2.strict_evaluate(predictions, gold)
    cumulative = write_cumulative_evaluation_v5(run_dir, gold, catalog_path, aliases_path)
    trace = write_relation_trace_v5(
        run_dir=run_dir,
        gold_jsonl=gold_jsonl,
        catalog_path=catalog_path,
        aliases_path=aliases_path,
    )
    projection_audit = write_entity_projection_audit(run_dir, final_state, gold)
    native_views = write_native_views_v5(run_dir, final_state, gold, catalog_path, aliases_path)
    failure_counts: dict[str, int] = {}
    for row in trace:
        failure_counts[row["first_failure"]] = failure_counts.get(row["first_failure"], 0) + 1

    summary["strict_evaluation_before_v5_projection"] = summary.get("strict_evaluation")
    summary["strict_evaluation"] = evaluation
    summary["strict_evaluation_v5"] = evaluation
    summary["strict_docred_predictions_v5"] = predictions
    summary["cumulative_strict_evaluation_v5"] = cumulative
    summary["gold_relation_trace_v5"] = trace
    summary["failure_counts_v5"] = failure_counts
    summary["entity_projection_audit_v5"] = projection_audit
    summary.update(native_views)
    write_json(run_dir / "analysis/analysis_summary_v5.json", summary)
    return summary

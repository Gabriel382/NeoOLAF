from __future__ import annotations

"""NeoOLAF native DocRED one-document ablation support, v4.

This experiment module changes no file under ``src/neoolaf``. It keeps the full
Layer 0--12 NeoOLAF pipeline while improving only profile/guidance-driven
orchestration for DocRED:

* one whole-document Layer 1 call emits named endpoints and one structured
  relation instance per source/predicate/target pair;
* Layer 2 deterministically preserves entity/date expressions and sends only
  relation instances to a contrastive ontology-property classifier;
* Layer 3 uses NeoOLAF's existing role-based typing and promotes only
  document-supported relation candidates (no empty vocabulary candidates);
* Layer 4 resolves exact structured endpoints deterministically when possible,
  otherwise falls back to NeoOLAF's existing parallel endpoint-selection task;
* ontology/domain-range and coarse DocRED type constraints reject impossible
  assertions such as publication-date -> location;
* every decision, rejection, prompt, response, retrieval, layer state, log and
  error remains inspectable.

No direct DocRED relation extractor, source-entity anchoring, gold leakage,
closure rule, or post-hoc relation invention is used.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import json
import re
import shutil
import sys
import threading
import time
import traceback

import docred_native_ablation as v2
import docred_native_ablation_v3 as v3

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner
from neoolaf.domain.documents import Document
from neoolaf.domain.enriched_expression import EnrichedExpression, EnrichmentEvidence
from neoolaf.domain.linguistic_expression import Evidence, LinguisticExpression
from neoolaf.domain.relation_assertion import CandidateRelationAssertion
from neoolaf.domain.user_guidance import UserGuidance
from neoolaf.grounding.rag.formatting import build_grounding_context
from neoolaf.grounding.rag.types import GroundingRequest
from neoolaf.ontology.loader import SeedOntologyLoader
from neoolaf.profiles.profile_loader import load_document_profile

from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
from neoolaf.layers.layer01_linguistic_expression_extraction.component import LinguisticExpressionExtractionLayer
from neoolaf.layers.layer02_candidate_enrichment.component import CandidateEnrichmentLayer
from neoolaf.layers.layer05_candidate_triple_generation.component import CandidateTripleGenerationLayer
from neoolaf.layers.layer06_concept_relation_induction.component import ConceptRelationInductionLayer
from neoolaf.layers.layer07_hierarchisation.component import HierarchisationLayer
from neoolaf.layers.layer08_axiom_schemata_extraction.component import AxiomSchemataExtractionLayer
from neoolaf.layers.layer09_general_axiom_extraction.component import GeneralAxiomExtractionLayer
from neoolaf.layers.layer10_validation_reasoning.component import ValidationReasoningLayer
from neoolaf.layers.layer11_inference_completion.component import InferenceCompletionLayer
from neoolaf.layers.layer12_serialization.component import SerializationLayer

from experiments.methods.run_neoolaf import (
    OfflineWebSearchSource,
    OfflineWikipediaSource,
    OfflineWikidataSource,
    OpenAICompatibleBackend,
    load_user_guidance,
)

# Re-export notebook helpers.
read_json = v3.read_json
read_jsonl = v3.read_jsonl
write_json = v3.write_json
append_jsonl = v3.append_jsonl
load_layer_states = v3.load_layer_states
state_counts = v3.state_counts
safe_name = v3.safe_name
Tee = v3.Tee
LAYER_NAMES = v3.LAYER_NAMES
SharedCallLogger = v3.SharedCallLogger
TaggedLoggedBackend = v3.TaggedLoggedBackend
PriorityOntologyRAGAdapter = v3.PriorityOntologyRAGAdapter
CanonicalizingCandidateTypingResolutionLayer = v3.CanonicalizingCandidateTypingResolutionLayer


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


def _norm(text: str | None) -> str:
    value = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    value = re.sub(r"[^\w\s\-]", "", value)
    return value


def _relation_id(text: str | None) -> str | None:
    match = re.search(r"\bP\d+\b", str(text or ""), re.IGNORECASE)
    return match.group(0).upper() if match else None


def _parse_relation_instance(text: str | None) -> tuple[str, str, str] | None:
    parts = [part.strip() for part in str(text or "").split("||")]
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def _json_block(value: Any, max_chars: int = 14000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= max_chars else text[:max_chars] + "\n... [truncated]"


class DocREDRelationInstanceExtractionLayer(LinguisticExpressionExtractionLayer):
    """Layer 1 with a DocRED-specific output contract expressed via guidance.

    This is still one native Layer 1 call over the whole document. It does not
    predict canonical ontology properties; it only extracts endpoint nodes and
    source/predicate/target relation instances for Layer 2 to classify.
    """

    def __init__(self, *args: Any, decision_log_path: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decision_log_path = Path(decision_log_path)

    def _prompt(self, state: PipelineState, chunk_text: str) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        layer_guidance = (task.get("layer_guidance") or {}).get("layer01", {})
        examples = task.get("relation_instance_examples") or task.get("relation_examples") or []
        priority_specs = task.get("priority_relation_specs") or task.get("relation_specs") or []
        system = """
You are NeoOLAF Layer 1 for document-level DocRED linguistic-expression extraction.

Extract two kinds of expressions only:
1. named or literal relation endpoints;
2. relation instances grounded in this document.

Endpoint labels must be one of:
- entity_org
- entity_per
- entity_loc
- entity_time
- entity_num
- entity_misc

Every relation instance MUST use this exact text form:
SOURCE || LEXICAL RELATION PHRASE || TARGET
and the label relation_instance.

Rules:
- create one relation instance for each independently supported source-target pair;
- never collapse a date relation and a location relation into one expression;
- preserve exact endpoint surface forms whenever possible;
- relation instances may be explicit or controlled high-confidence inferences;
- mark inferred relations clearly in justification;
- use demonyms, aliases, coreference, “the country”, and common city-country
  containment only when confidence is high and evidence is explained;
- do not choose P-identifiers here; Layer 2 performs ontology classification;
- do not output topical phrases that are neither endpoints nor relations;
- do not expose or use gold annotations.

Return JSON only:
{
  "expressions": [
    {"text": "Northstar TV", "label": "entity_org", "justification": "Named organization endpoint."},
    {"text": "Northstar TV || is based in || Harbor City", "label": "relation_instance", "justification": "Explicit ORG->LOC statement; source_type=ORG; target_type=LOC."}
  ]
}
""".strip()
        user = f"""
Layer-specific guidance:
{_json_block(layer_guidance, 5000)}

Priority relation definitions and direction rules (use them only to notice relevant relation instances; do not select the final P-ID here):
{_json_block(priority_specs, 9000)}

Synthetic examples:
{_json_block(examples, 8000)}

Document:
\"\"\"
{chunk_text}
\"\"\"

Extract endpoint expressions and separate structured relation instances. Return JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _run(self, state: PipelineState) -> PipelineState:
        chunks = list(state.document.chunks)
        if self.max_chunks is not None:
            chunks = chunks[: self.max_chunks]
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
            for item in parsed.get("expressions", []) or []:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                label = str(item.get("label") or "").strip().lower()
                justification = str(item.get("justification") or "").strip()
                if not text:
                    continue
                if label == "relation_instance":
                    parsed_instance = _parse_relation_instance(text)
                    if parsed_instance is None:
                        decisions.append({
                            "status": "rejected",
                            "reason": "relation_instance_not_source_pipe_predicate_pipe_target",
                            "text": text,
                            "label": label,
                        })
                        continue
                elif label not in {
                    "entity_org", "entity_per", "entity_loc", "entity_time",
                    "entity_num", "entity_misc",
                }:
                    decisions.append({
                        "status": "rejected",
                        "reason": "unsupported_layer01_label",
                        "text": text,
                        "label": label,
                    })
                    continue

                match_span = self._find_expression_span(text, chunk.text)
                if match_span is not None:
                    chunk_start, chunk_end = match_span
                    doc_start = chunk.start_char + chunk_start
                    doc_end = chunk.start_char + chunk_end
                    snippet = self._build_snippet(chunk.text, chunk_start, chunk_end)
                else:
                    chunk_start = chunk_end = doc_start = doc_end = -1
                    snippet = chunk.text[:1000]

                expressions.append(LinguisticExpression(
                    expr_id=f"expr_{expr_counter:05d}",
                    text=text,
                    label=label,
                    justification=justification,
                    evidence=[Evidence(
                        chunk_id=chunk.chunk_id,
                        chunk_start_char=chunk_start,
                        chunk_end_char=chunk_end,
                        doc_start_char=doc_start,
                        doc_end_char=doc_end,
                        snippet=snippet,
                    )],
                ))
                decisions.append({
                    "status": "accepted",
                    "expr_id": f"expr_{expr_counter:05d}",
                    "text": text,
                    "label": label,
                    "relation_instance": _parse_relation_instance(text),
                    "justification": justification,
                })
                expr_counter += 1

        dedup: dict[tuple[str, str], LinguisticExpression] = {}
        for expr in expressions:
            dedup.setdefault((_norm(expr.text), expr.label), expr)
        state.linguistic_expressions = list(dedup.values())
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.decision_log_path, decisions)
        state.log(
            f"[{self.name}] DocRED structured extraction; expressions={len(state.linguistic_expressions)}; "
            f"relation_instances={sum(1 for x in state.linguistic_expressions if x.label == 'relation_instance')}"
        )
        return state


class SelectiveContrastiveCandidateEnrichmentLayer(CandidateEnrichmentLayer):
    """Layer 2: deterministic nodes, parallel contrastive relation linking."""

    def __init__(
        self,
        *args: Any,
        relation_catalog_path: str | Path,
        decision_log_path: str | Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        catalog = read_json(relation_catalog_path)["relations"]
        self.catalog = {item["relation_id"].upper(): item for item in catalog}
        self.decision_log_path = Path(decision_log_path)
        self._decision_lock = threading.Lock()
        self._decisions: list[dict[str, Any]] = []

    def _is_relation(self, expr: LinguisticExpression) -> bool:
        return expr.label == "relation_instance" or _parse_relation_instance(expr.text) is not None

    def _compact_catalog(self, relation_ids: Iterable[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for relation_id in _dedup(relation_ids):
            item = self.catalog.get(relation_id.upper())
            if not item:
                continue
            rows.append({
                "relation_id": item["relation_id"],
                "label": item["label"],
                "definition": item.get("comment") or "",
                "domain_uris": item.get("domain_uris") or [],
                "range_uris": item.get("range_uris") or [],
            })
        return rows

    def _candidate_ids_from_grounding(self, grounding_text: str) -> list[str]:
        return _dedup(match.upper() for match in re.findall(r"\bP\d+\b", grounding_text or "", re.I))

    def _contrastive_prompt(
        self,
        *,
        expr: LinguisticExpression,
        state: PipelineState,
        grounding_text: str,
    ) -> list[dict[str, str]]:
        task = (state.profile_config or {}).get("_input_task_guidance", {}) or {}
        parsed_instance = _parse_relation_instance(expr.text)
        if parsed_instance is None:
            raise ValueError(f"Not a structured relation instance: {expr.text}")
        source, predicate, target = parsed_instance
        layer_guidance = (task.get("layer_guidance") or {}).get("layer02", {})
        contrastive_rules = task.get("contrastive_rules") or []
        examples = task.get("relation_examples") or []
        allowed = task.get("allowed_relation_ids") or list(self.catalog)
        retrieved_ids = self._candidate_ids_from_grounding(grounding_text)
        priority_ids = task.get("priority_relation_ids") or []
        candidate_ids = _dedup([*retrieved_ids, *priority_ids])
        candidate_ids = [x for x in candidate_ids if x in set(allowed) and x in self.catalog][:12]
        if not candidate_ids:
            candidate_ids = [x for x in priority_ids if x in self.catalog][:12]

        system = """
You are NeoOLAF Layer 2 for ontology-constrained candidate enrichment.

You receive exactly one relation instance already extracted by Layer 1. Your
only semantic task is to select the single best DocRED/Wikidata property from
the supplied ontology candidates, or return found=false when none is supported.

Use source type, target type, direction, predicate specificity, ontology
definition, domain and range. Prefer the most specific supported property.
Contrast close alternatives explicitly. Never create a new source, target or
relation instance. Never use gold annotations.

Return JSON only:
{
  "found": true,
  "selected_relation_id": "P159",
  "selected_relation_label": "headquarters location",
  "aliases": ["is based in"],
  "synonyms": [],
  "lexical_variants": [],
  "definition": "...",
  "decision": "P159 is preferred over P131 because ..."
}

or {"found": false, "decision": "..."}.
""".strip()
        user = f"""
Relation instance:
- source: {source}
- lexical predicate: {predicate}
- target: {target}
- Layer 1 justification: {expr.justification}

Layer-specific rules:
{_json_block(layer_guidance, 5000)}

Contrastive rules:
{_json_block(contrastive_rules, 7000)}

Synthetic examples:
{_json_block(examples, 7000)}

Retrieved ontology evidence:
{grounding_text}

Allowed candidate properties for this decision:
{_json_block(self._compact_catalog(candidate_ids), 12000)}

Choose exactly one relation only when supported. Return JSON only.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _record_decision(self, row: dict[str, Any]) -> None:
        with self._decision_lock:
            self._decisions.append(row)

    def _process_relation_contrastive(self, expr: LinguisticExpression, state: PipelineState) -> EnrichedExpression:
        parsed_instance = _parse_relation_instance(expr.text)
        if parsed_instance is None:
            raise ValueError(f"Malformed relation instance: {expr.text}")
        source, predicate, target = parsed_instance
        grounding_text = ""
        if self.rag_adapter is not None:
            grounding = self.rag_adapter.ground(GroundingRequest(
                layer_name=self.name,
                query=f"{predicate} {source} {target}",
                payload={
                    "relation_instance": expr.text,
                    "source": source,
                    "predicate": predicate,
                    "target": target,
                    "justification": expr.justification,
                },
                preferred_sources=["ontology"],
                top_k=10,
            ))
            grounding_text = build_grounding_context(grounding)

        messages = self._contrastive_prompt(expr=expr, state=state, grounding_text=grounding_text)
        raw = self.ollama_backend.chat(
            model=state.llm_model,
            messages=messages,
            temperature=0.0,
        )
        parsed = self.ollama_backend.extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Layer 2 contrastive response is not a JSON object")
        found = bool(parsed.get("found", False))
        selected_id = _relation_id(parsed.get("selected_relation_id"))
        if found and selected_id not in self.catalog:
            raise ValueError(f"Layer 2 selected invalid relation ID: {selected_id}")

        if found:
            item = self.catalog[selected_id]
            canonical = f"{selected_id} : {item['label']}"
            hints = _dedup([
                f"controlled_relation:{canonical}",
                "promote_to_ontology:true",
                item.get("uri"),
                item.get("label"),
                f"source_label:{source}",
                f"target_label:{target}",
                f"lexical_predicate:{predicate}",
                f"source_type:{self._type_from_justification(expr.justification, 'source_type')}",
                f"target_type:{self._type_from_justification(expr.justification, 'target_type')}",
                f"domain:{', '.join(item.get('domain_uris') or [])}" if item.get("domain_uris") else None,
                f"range:{', '.join(item.get('range_uris') or [])}" if item.get("range_uris") else None,
                f"contrastive_decision:{parsed.get('decision', '')}",
            ])
            definition = str(parsed.get("definition") or item.get("comment") or "").strip()
        else:
            canonical = None
            hints = _dedup([
                "promote_to_ontology:false",
                f"source_label:{source}",
                f"target_label:{target}",
                f"lexical_predicate:{predicate}",
                f"contrastive_decision:{parsed.get('decision', '')}",
            ])
            definition = str(parsed.get("decision") or "No supported ontology relation selected.")

        self._record_decision({
            "expr_id": expr.expr_id,
            "relation_instance": expr.text,
            "source": source,
            "predicate": predicate,
            "target": target,
            "found": found,
            "selected_relation_id": selected_id,
            "canonical_relation": canonical,
            "decision": parsed.get("decision"),
            "ontology_hints": hints,
        })
        return EnrichedExpression(
            base_expression=expr,
            aliases=_dedup([expr.text, predicate, *(parsed.get("aliases") or [])]),
            synonyms=_dedup(parsed.get("synonyms") or []),
            lexical_variants=_dedup(parsed.get("lexical_variants") or []),
            alias_sources={value: ["source" if value in {expr.text, predicate} else "llm"] for value in _dedup([expr.text, predicate, *(parsed.get("aliases") or [])])},
            synonym_sources={value: ["llm"] for value in _dedup(parsed.get("synonyms") or [])},
            lexical_variant_sources={value: ["llm"] for value in _dedup(parsed.get("lexical_variants") or [])},
            definition=definition,
            ontology_hints=hints,
            enrichment_evidence=[EnrichmentEvidence(
                source="llm",
                content=json.dumps(parsed, ensure_ascii=False),
                reference=state.llm_model,
            )],
        )

    @staticmethod
    def _type_from_justification(justification: str, key: str) -> str:
        match = re.search(rf"\b{re.escape(key)}\s*=\s*([A-Za-z]+)", justification or "", re.I)
        return match.group(1).upper() if match else "UNKNOWN"

    def _process_relation_with_retries(self, index: int, expr: LinguisticExpression, state: PipelineState) -> EnrichedExpression | None:
        last_exc: Exception | None = None
        for attempt in range(self.retry_failed_calls + 1):
            try:
                return self._process_relation_contrastive(expr, state)
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry_failed_calls and self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        if last_exc is not None:
            self._record_failure(expr, index, last_exc, attempt=self.retry_failed_calls)
        return None

    def _run(self, state: PipelineState) -> PipelineState:
        expressions = list(state.linguistic_expressions)
        if self.max_expressions is not None:
            expressions = expressions[: self.max_expressions]
        self._failed_details = []
        self._decisions = []

        relation_jobs: list[tuple[int, LinguisticExpression]] = []
        enriched_by_index: dict[int, EnrichedExpression] = {}
        for index, expr in enumerate(expressions):
            if self._is_relation(expr):
                relation_jobs.append((index, expr))
            else:
                enriched_by_index[index] = self._process_expression_conservative(expr, state)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {
                executor.submit(self._process_relation_with_retries, index, expr, state): index
                for index, expr in relation_jobs
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self._record_failure(expressions[index], index, exc, attempt="unhandled")
                    continue
                if result is not None:
                    enriched_by_index[index] = result

        state.enriched_expressions = [enriched_by_index[i] for i in sorted(enriched_by_index)]
        self._save_failed_expressions(state)
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.decision_log_path, sorted(self._decisions, key=lambda x: x.get("expr_id", "")))
        state.log(
            f"[{self.name}] selective enrichment; total={len(expressions)}; "
            f"deterministic_nodes={len(expressions)-len(relation_jobs)}; "
            f"relation_llm_calls={len(relation_jobs)}; enriched={len(state.enriched_expressions)}; "
            f"failed={len(self._failed_details)}"
        )
        return state


class StructuredEndpointValidatedRelationLayer(v3.ParallelCandidateRelationExtractionLayer):
    """Layer 4 with exact endpoint resolution and coarse type/range validation."""

    def __init__(
        self,
        *args: Any,
        endpoint_log_path: str | Path,
        rejection_log_path: str | Path,
        type_constraints: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.endpoint_log_path = Path(endpoint_log_path)
        self.rejection_log_path = Path(rejection_log_path)
        self.type_constraints = {str(k).upper(): v for k, v in (type_constraints or {}).items()}
        self._decision_lock = threading.Lock()
        self._endpoint_decisions: list[dict[str, Any]] = []
        self._rejections: list[dict[str, Any]] = []

    @staticmethod
    def _hint_value(hints: Iterable[str], prefix: str) -> str | None:
        prefix_l = prefix.lower()
        for hint in hints or []:
            text = str(hint)
            if text.lower().startswith(prefix_l):
                return text.split(":", 1)[1].strip()
        return None

    @staticmethod
    def _candidate_roles(candidate: Any) -> set[str]:
        roles: set[str] = set()
        for hint in candidate.ontology_hints or []:
            text = str(hint)
            if text.lower().startswith("semantic_role:"):
                role = text.split(":", 1)[1].strip().lower()
                if "org" in role or role in {"organization", "organisation"}:
                    roles.add("ORG")
                elif "per" in role or role in {"person", "human"}:
                    roles.add("PER")
                elif "loc" in role or role in {"location", "city", "country"}:
                    roles.add("LOC")
                elif "time" in role or role == "date":
                    roles.add("TIME")
                elif "num" in role or role == "number":
                    roles.add("NUM")
                else:
                    roles.add("MISC")
            upper = text.upper()
            for coarse in ["ORG", "PER", "LOC", "TIME", "NUM", "MISC"]:
                if re.search(rf"\b{coarse}\b", upper):
                    roles.add(coarse)
        return roles or {"UNKNOWN"}

    def _candidate_matches(self, state: PipelineState, expected: str) -> list[Any]:
        expected_n = _norm(expected)
        result: list[Any] = []
        for candidate in [
            *(state.entity_candidates or []),
            *(state.event_candidates or []),
            *(state.attribute_candidates or []),
        ]:
            labels = [candidate.canonical_label, *(candidate.aliases or []), *(m.text for m in candidate.mentions or [])]
            if any(_norm(label) == expected_n for label in labels):
                result.append(candidate)
        return result

    def _valid_types(self, relation_id: str | None, source: Any, target: Any) -> tuple[bool, str]:
        if not relation_id or relation_id not in self.type_constraints:
            return True, "no_coarse_constraint"
        cfg = self.type_constraints[relation_id]
        allowed_source = set(cfg.get("source") or [])
        allowed_target = set(cfg.get("target") or [])
        source_roles = self._candidate_roles(source)
        target_roles = self._candidate_roles(target)
        source_ok = not allowed_source or bool(source_roles & allowed_source)
        target_ok = not allowed_target or bool(target_roles & allowed_target)
        reason = (
            f"relation={relation_id}; source_roles={sorted(source_roles)} allowed={sorted(allowed_source)}; "
            f"target_roles={sorted(target_roles)} allowed={sorted(allowed_target)}"
        )
        return source_ok and target_ok, reason

    def _record_endpoint(self, row: dict[str, Any]) -> None:
        with self._decision_lock:
            self._endpoint_decisions.append(row)

    def _record_rejection(self, row: dict[str, Any]) -> None:
        with self._decision_lock:
            self._rejections.append(row)

    def _process_relation_mention(
        self,
        *,
        state: PipelineState,
        relation_mention: dict[str, Any],
        chunk_to_local_candidates: dict[str, list[dict[str, Any]]],
    ) -> CandidateRelationAssertion | None:
        relation_candidate = relation_mention["relation_candidate"]
        hints = list(relation_candidate.ontology_hints or [])
        source_label = self._hint_value(hints, "source_label:")
        target_label = self._hint_value(hints, "target_label:")
        relation_id = _relation_id(relation_candidate.canonical_label) or _relation_id(" ".join(hints))

        if source_label and target_label:
            source_matches = self._candidate_matches(state, source_label)
            target_matches = self._candidate_matches(state, target_label)
            if len(source_matches) == 1 and len(target_matches) == 1:
                source = source_matches[0]
                target = target_matches[0]
                valid, reason = self._valid_types(relation_id, source, target)
                if not valid:
                    self._record_rejection({
                        "mode": "structured_exact",
                        "relation_candidate_id": relation_candidate.candidate_id,
                        "relation_id": relation_id,
                        "source": source.canonical_label,
                        "target": target.canonical_label,
                        "reason": reason,
                    })
                    return None
                self._record_endpoint({
                    "mode": "structured_exact",
                    "relation_candidate_id": relation_candidate.candidate_id,
                    "relation_id": relation_id,
                    "source": source.canonical_label,
                    "target": target.canonical_label,
                    "llm_call": False,
                    "type_validation": reason,
                })
                return CandidateRelationAssertion(
                    assertion_id="pending",
                    relation_candidate_id=relation_candidate.candidate_id,
                    relation_label=relation_candidate.canonical_label,
                    source_candidate_id=source.candidate_id,
                    source_candidate_label=source.canonical_label,
                    source_candidate_type=source.candidate_type,
                    target_candidate_id=target.candidate_id,
                    target_candidate_label=target.canonical_label,
                    target_candidate_type=target.candidate_type,
                    chunk_id=relation_mention["chunk_id"],
                    justification=(
                        "Exact Layer 1 source/target labels resolved to Layer 3 candidates; "
                        "predicate was canonically linked by Layer 2."
                    ),
                    confidence=1.0,
                    evidence=relation_mention["evidence"],
                )

        assertion = super()._process_relation_mention(
            state=state,
            relation_mention=relation_mention,
            chunk_to_local_candidates=chunk_to_local_candidates,
        )
        if assertion is None:
            self._record_endpoint({
                "mode": "native_llm_fallback",
                "relation_candidate_id": relation_candidate.candidate_id,
                "relation_id": relation_id,
                "source_hint": source_label,
                "target_hint": target_label,
                "llm_call": True,
                "found": False,
            })
            return None
        source = self._find_candidate_by_id(state, assertion.source_candidate_id)
        target = self._find_candidate_by_id(state, assertion.target_candidate_id)
        valid, reason = self._valid_types(relation_id, source, target)
        if not valid:
            self._record_rejection({
                "mode": "native_llm_fallback",
                "relation_candidate_id": relation_candidate.candidate_id,
                "relation_id": relation_id,
                "source": assertion.source_candidate_label,
                "target": assertion.target_candidate_label,
                "reason": reason,
            })
            return None
        self._record_endpoint({
            "mode": "native_llm_fallback",
            "relation_candidate_id": relation_candidate.candidate_id,
            "relation_id": relation_id,
            "source": assertion.source_candidate_label,
            "target": assertion.target_candidate_label,
            "llm_call": True,
            "found": True,
            "type_validation": reason,
        })
        return assertion

    def _run(self, state: PipelineState) -> PipelineState:
        self._endpoint_decisions = []
        self._rejections = []
        state = super()._run(state)
        self.endpoint_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.endpoint_log_path, self._endpoint_decisions)
        write_json(self.rejection_log_path, self._rejections)
        state.log(
            f"[{self.name}] endpoint modes: exact={sum(x.get('mode') == 'structured_exact' for x in self._endpoint_decisions)}; "
            f"llm={sum(x.get('mode') == 'native_llm_fallback' for x in self._endpoint_decisions)}; "
            f"rejected_by_type={len(self._rejections)}"
        )
        return state


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
    return v2.choose_chunk_size(text, max_safe_chars)


def build_document(record: dict[str, Any], source_path: str | Path) -> Document:
    return v2.build_document(record, source_path)


def build_pipeline(
    *,
    backends: dict[str, TaggedLoggedBackend],
    rag_adapter: PriorityOntologyRAGAdapter,
    profile_config: dict[str, Any],
    relation_catalog_path: str | Path,
    chunk_size: int,
    run_dir: str | Path,
    workers: int = 12,
    verbose: bool = True,
) -> Pipeline:
    workers = max(1, int(workers))
    l2_cfg = _layer_cfg(profile_config, "layer02_candidate_enrichment")
    l4_cfg = _layer_cfg(profile_config, "layer04_candidate_relation_extraction")
    retry_default = int((profile_config.get("orchestration") or {}).get("retry_failed_calls", 1))
    sleep_default = float((profile_config.get("orchestration") or {}).get("retry_sleep_seconds", 1.0))
    l2_workers = int(l2_cfg.get("max_concurrency", workers))
    l4_workers = int(l4_cfg.get("max_concurrency", min(workers, 8)))
    run_dir = Path(run_dir)

    layers = [
        PreprocessingLayer(
            chunk_size=chunk_size,
            overlap=0,
            enable_chunking=True,
            translate=False,
            save_intermediate=True,
            verbose=verbose,
            profile_config=profile_config,
        ),
        DocREDRelationInstanceExtractionLayer(
            backends["layer01"],
            decision_log_path=run_dir / "run_logs/layer01_relation_instances.json",
            max_chunks=1,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_backend=rag_adapter,
            max_concurrency=1,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
            rag_enabled=False,
        ),
        SelectiveContrastiveCandidateEnrichmentLayer(
            backends["layer02"],
            wikipedia_source=OfflineWikipediaSource(),
            wikidata_source=OfflineWikidataSource(),
            web_search_source=OfflineWebSearchSource(),
            relation_catalog_path=relation_catalog_path,
            decision_log_path=run_dir / "run_logs/layer02_contrastive_decisions.json",
            max_expressions=None,
            use_web_search=False,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=l2_workers,
            retry_failed_calls=int(l2_cfg.get("retry_failed_calls", retry_default)),
            retry_sleep_seconds=sleep_default,
        ),
        CanonicalizingCandidateTypingResolutionLayer(
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
        ),
        StructuredEndpointValidatedRelationLayer(
            backends["layer04"],
            endpoint_log_path=run_dir / "run_logs/layer04_endpoint_assignment.json",
            rejection_log_path=run_dir / "run_logs/layer04_constraint_rejections.json",
            type_constraints=(profile_config.get("docred_type_constraints") or {}),
            max_relation_mentions=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=l4_workers,
            retry_failed_calls=int(l4_cfg.get("retry_failed_calls", retry_default)),
            retry_sleep_seconds=sleep_default,
            max_attempts_per_relation=int(l4_cfg.get("max_attempts_per_relation", 1)),
            retry_wait_seconds=float(l4_cfg.get("retry_wait_seconds", 0.5)),
            failure_log_path=run_dir / "run_logs/layer04_relation_errors.jsonl",
        ),
        CandidateTripleGenerationLayer(
            max_assertions=None,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
            save_intermediate=True,
            verbose=verbose,
        ),
        ConceptRelationInductionLayer(
            backends["other"],
            max_concept_inputs=None,
            max_relation_inputs=None,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
        ),
        HierarchisationLayer(
            backends["other"],
            max_concept_pairs=None,
            max_relation_pairs=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            rag_adapter=rag_adapter,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        AxiomSchemataExtractionLayer(
            backends["other"],
            max_relation_schema_inputs=None,
            max_subclass_inputs=None,
            temperature=0.0,
            rag_adapter=rag_adapter,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        GeneralAxiomExtractionLayer(
            backends["other"],
            max_schema_inputs=None,
            max_description_inputs=None,
            temperature=0.0,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        ValidationReasoningLayer(
            max_triples=None,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        InferenceCompletionLayer(
            max_inferred_triples=None,
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
            retry_failed_calls=retry_default,
            retry_sleep_seconds=sleep_default,
        ),
        SerializationLayer(
            output_subdir="exports",
            save_intermediate=True,
            verbose=verbose,
            max_concurrency=workers,
        ),
    ]
    return Pipeline(layers=layers, verbose=verbose, continue_from_last=False)


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
    workers: int = 12,
    max_tokens: int = 4096,
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
            logger=logger, layer_tag="layer01", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer01_linguistic_expression_extraction"),
            fallback_max_tokens=max_tokens, fallback_timeout=request_timeout,
            reasoning_effort=reasoning_effort,
        ),
        "layer02": _make_backend(
            logger=logger, layer_tag="layer02_relation_only", model_host=host, api_key=api_key,
            cfg=_layer_cfg(profile_dict, "layer02_candidate_enrichment"),
            fallback_max_tokens=512, fallback_timeout=60,
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
        top_k=int(profile.get("rag.top_k", 10)),
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
        "selective_layer02": True,
        "structured_layer04_exact_endpoint_resolution": True,
        "empty_schema_candidates_injected": False,
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


def write_relation_trace_v4(*, run_dir: str | Path, gold_jsonl: str | Path, catalog_path: str | Path) -> list[dict[str, Any]]:
    """Corrected layer-first relation trace for one DocRED document."""
    run_dir = Path(run_dir)
    gold = read_jsonl(gold_jsonl)[0]
    states = {index: state for index, _, state in load_layer_states(run_dir)}
    l1 = states.get(1)
    l2 = states.get(2)
    l3 = states.get(3)
    l4 = states.get(4)
    l5 = states.get(5)

    relation_decisions = read_json(run_dir / "run_logs/layer02_contrastive_decisions.json") if (run_dir / "run_logs/layer02_contrastive_decisions.json").is_file() else []
    decision_by_id = {row.get("selected_relation_id"): [] for row in relation_decisions if row.get("selected_relation_id")}
    for row in relation_decisions:
        if row.get("selected_relation_id"):
            decision_by_id.setdefault(row["selected_relation_id"], []).append(row)

    def entity_label(entity_id: str) -> str:
        entity = gold["entities"][entity_id]
        mentions = entity.get("mentions") or []
        return mentions[0]["trigger_word"] if mentions else entity_id

    rows: list[dict[str, Any]] = []
    for relation_label, pairs in gold.get("relations", {}).items():
        relation_id = _relation_id(relation_label)
        for source_id, target_id in pairs:
            source_label = entity_label(source_id)
            target_label = entity_label(target_id)
            l1_texts = [_norm(x.text) for x in (l1.linguistic_expressions if l1 else [])]
            source_l1 = _norm(source_label) in l1_texts
            target_l1 = _norm(target_label) in l1_texts
            l1_instances = [
                x for x in (l1.linguistic_expressions if l1 else [])
                if x.label == "relation_instance" and _parse_relation_instance(x.text)
            ]
            exact_l1_instance = any(
                _norm(_parse_relation_instance(x.text)[0]) == _norm(source_label)
                and _norm(_parse_relation_instance(x.text)[2]) == _norm(target_label)
                for x in l1_instances
            )
            linked_l2 = any(
                row.get("selected_relation_id") == relation_id
                and _norm(row.get("source")) == _norm(source_label)
                and _norm(row.get("target")) == _norm(target_label)
                for row in relation_decisions
            )
            linked_l3 = any(
                _relation_id(c.canonical_label) == relation_id
                and any(
                    (lambda parsed: parsed and _norm(parsed[0]) == _norm(source_label) and _norm(parsed[2]) == _norm(target_label))(
                        _parse_relation_instance(alias)
                    )
                    for alias in [*list(c.aliases or []), *(m.text for m in c.mentions or [])]
                )
                for c in (l3.relation_candidates if l3 else [])
            )
            assertion_l4 = any(
                _relation_id(a.relation_label) == relation_id
                and _norm(a.source_candidate_label) == _norm(source_label)
                and _norm(a.target_candidate_label) == _norm(target_label)
                for a in (l4.candidate_relation_assertions if l4 else [])
            )
            triple_l5 = any(
                _relation_id(t.predicate_label) == relation_id
                and _norm(t.subject_label) == _norm(source_label)
                and _norm(t.object_label) == _norm(target_label)
                for t in (l5.candidate_triples if l5 else [])
            )
            if not source_l1 or not target_l1:
                failure = "layer01_endpoint_missing"
            elif not exact_l1_instance:
                failure = "layer01_relation_instance_missing"
            elif not linked_l2:
                failure = "layer02_wrong_or_missing_controlled_relation"
            elif not linked_l3:
                failure = "layer03_candidate_typing_or_resolution"
            elif not assertion_l4:
                failure = "layer04_endpoint_direction_or_type_validation"
            elif not triple_l5:
                failure = "layer05_triple_materialization"
            else:
                failure = "survived_to_layer05"
            rows.append({
                "gold_relation_id": relation_id,
                "gold_relation_label": relation_label,
                "source": source_label,
                "target": target_label,
                "source_available_layer01": source_l1,
                "target_available_layer01": target_l1,
                "relation_instance_layer01": exact_l1_instance,
                "canonical_relation_layer02": linked_l2,
                "relation_candidate_layer03": linked_l3,
                "assertion_layer04": assertion_l4,
                "triple_layer05": triple_l5,
                "first_failure": failure,
            })
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    import csv
    path = analysis_dir / "gold_relation_trace_v4.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def analyze_run(
    *,
    run_dir: str | Path,
    gold_jsonl: str | Path,
    catalog_path: str | Path,
    aliases_path: str | Path,
) -> dict[str, Any]:
    summary = v3.analyze_run(
        run_dir=run_dir,
        gold_jsonl=gold_jsonl,
        catalog_path=catalog_path,
        aliases_path=aliases_path,
    )
    trace = write_relation_trace_v4(
        run_dir=run_dir,
        gold_jsonl=gold_jsonl,
        catalog_path=catalog_path,
    )
    counts: dict[str, int] = {}
    for row in trace:
        counts[row["first_failure"]] = counts.get(row["first_failure"], 0) + 1
    summary["gold_relation_trace_v4"] = trace
    summary["failure_counts_v4"] = counts
    write_json(Path(run_dir) / "analysis/analysis_summary_v4.json", summary)
    return summary

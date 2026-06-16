from __future__ import annotations

# Standard library imports
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports
from tqdm.auto import tqdm

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.state_serialization import dump_json, load_json, to_jsonable
from neoolaf.domain.enriched_expression import EnrichedExpression, EnrichmentEvidence
from neoolaf.domain.linguistic_expression import LinguisticExpression
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend


class CandidateEnrichmentLayer(BaseLayer):
    """
    Layer 2: candidate enrichment.

    The generic strategy keeps the original LLM/RAG enrichment graph. Profiles can
    also select a conservative, ontology-aware strategy for already structured
    Layer 1 outputs. In that mode, Layer 2 does not broaden expressions with
    open-domain knowledge. It normalizes lightly, keeps provenance, and attaches
    profile/ontology links that downstream layers can use for typing, triple
    generation, and ontology population.
    """

    name = "layer02_candidate_enrichment"

    def __init__(
        self,
        ollama_backend: OllamaBackend,
        wikipedia_source: Any | None = None,
        wikidata_source: Any | None = None,
        wordnet_source: Any | None = None,
        web_search_source: Any | None = None,
        max_expressions: int | None = None,
        use_web_search: bool = True,
        save_intermediate: bool = True,
        verbose: bool = False,
        rag_adapter=None,
        max_concurrency: int = 1,
        retry_failed_calls: int = 0,
        retry_sleep_seconds: float = 2.0,
        failed_expressions_file: str | None = None,
    ) -> None:
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.ollama_backend = ollama_backend
        self.wikipedia_source = wikipedia_source
        self.wikidata_source = wikidata_source
        self.wordnet_source = wordnet_source
        self.web_search_source = web_search_source
        self.max_expressions = max_expressions
        self.use_web_search = use_web_search
        self.rag_adapter = rag_adapter
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.retry_failed_calls = max(0, int(retry_failed_calls or 0))
        self.retry_sleep_seconds = float(retry_sleep_seconds or 0.0)
        self.failed_expressions_file = failed_expressions_file
        self._failed_details: list[dict[str, Any]] = []

    def _run(self, state: PipelineState) -> PipelineState:
        """Enrich Layer 1 expressions and keep provenance for lexical items."""
        expressions = list(state.linguistic_expressions)
        expressions = self._filter_failed_expressions(expressions)
        if self.max_expressions is not None:
            expressions = expressions[: self.max_expressions]

        self._failed_details = []
        enriched: list[EnrichedExpression] = []

        strategy = self._strategy(state)
        if self.verbose:
            print(f"[NeoOLAF][Layer 2] strategy={strategy}")

        if self._is_conservative_strategy(strategy):
            # Conservative/profile-driven strategies are deterministic and do not
            # call the LLM. Avoid tqdm here because notebooks can flush tqdm
            # stderr after the process has already printed the final run summary,
            # which makes it look like a second Layer 2 run is starting.
            if self.verbose:
                print(
                    f"[NeoOLAF][Layer 2] deterministic normalization for "
                    f"{len(expressions)} expressions; no LLM calls."
                )
            for index, expr in enumerate(expressions):
                result = self._process_expression_with_retries(index, expr, state, strategy)
                if result is not None:
                    enriched.append(result)
        elif self.max_concurrency <= 1:
            iterator = expressions
            if self.verbose:
                iterator = tqdm(expressions, desc="Layer 2 - expressions", leave=False)
            for index, expr in enumerate(iterator):
                result = self._process_expression_with_retries(index, expr, state, strategy)
                if result is not None:
                    enriched.append(result)
        else:
            enriched_by_index: dict[int, EnrichedExpression] = {}
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                futures = {
                    executor.submit(self._process_expression_with_retries, index, expr, state, strategy): index
                    for index, expr in enumerate(expressions)
                }
                completed = as_completed(futures)
                if self.verbose:
                    completed = tqdm(
                        completed,
                        total=len(futures),
                        desc=f"Layer 2 - expressions x{self.max_concurrency}",
                        leave=False,
                    )
                for future in completed:
                    index = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # Defensive: _process should not raise.
                        self._record_failure(expressions[index], index, exc, attempt="unhandled")
                        continue
                    if result is not None:
                        enriched_by_index[index] = result
            enriched = [enriched_by_index[i] for i in sorted(enriched_by_index)]

        state.enriched_expressions = enriched
        self._save_failed_expressions(state)
        state.log(
            f"[layer02_candidate_enrichment] strategy={strategy}; "
            f"enriched {len(state.enriched_expressions)} expressions; failed={len(self._failed_details)}"
        )
        return state

    def _strategy(self, state: PipelineState) -> str:
        """Return the configured Layer 2 strategy."""
        layer_cfg = (state.profile_config or {}).get("layers", {}).get(self.name, {})
        return str(layer_cfg.get("strategy", "generic_llm_enrichment"))

    def _is_conservative_strategy(self, strategy: str) -> bool:
        return strategy in {
            "conservative_normalization",
            "ontology_aware_conservative_normalization",
            "role_based_normalization",
            "xquality_conservative_normalization",
        }

    def _filter_failed_expressions(self, expressions: list[LinguisticExpression]) -> list[LinguisticExpression]:
        """Restrict the run to expression IDs listed in a previous failed_expressions.json."""
        if not self.failed_expressions_file:
            return expressions
        path = Path(self.failed_expressions_file)
        if not path.exists():
            raise FileNotFoundError(f"Layer 2 failed expressions file not found: {path}")
        data = load_json(path)
        ids: set[str] = set()
        for item in data.get("failed_expressions", data if isinstance(data, list) else []):
            if isinstance(item, str):
                ids.add(item)
            elif isinstance(item, dict):
                value = item.get("expr_id") or item.get("id")
                if value:
                    ids.add(str(value))
        return [expr for expr in expressions if expr.expr_id in ids]

    def _process_expression_with_retries(
        self,
        index: int,
        expr: LinguisticExpression,
        state: PipelineState,
        strategy: str,
    ) -> EnrichedExpression | None:
        """Process one expression, retrying transient provider/JSON failures."""
        last_exc: Exception | None = None
        for attempt in range(self.retry_failed_calls + 1):
            try:
                return self._process_expression(expr, state, strategy)
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry_failed_calls:
                    if self.retry_sleep_seconds > 0:
                        time.sleep(self.retry_sleep_seconds)
                    continue
        if last_exc is not None:
            self._record_failure(expr, index, last_exc, attempt=self.retry_failed_calls)
        return None

    def _process_expression(
        self,
        expr: LinguisticExpression,
        state: PipelineState,
        strategy: str | None = None,
    ) -> EnrichedExpression:
        """Run the selected enrichment strategy for one expression."""
        strategy = strategy or self._strategy(state)
        if self._is_conservative_strategy(strategy):
            return self._process_expression_conservative(expr, state)
        return self._process_expression_generic(expr, state)

    # ------------------------------------------------------------------
    # Conservative profile strategy
    # ------------------------------------------------------------------
    def _process_expression_conservative(
        self,
        expr: LinguisticExpression,
        state: PipelineState,
    ) -> EnrichedExpression:
        """Profile-driven enrichment without open-domain semantic drift."""
        profile = state.profile_config or {}
        role = self._normalize_role(expr.label)
        canonical_text = self._canonicalize_expression(expr.text, role, profile)

        ontology_link = self._role_ontology_link(role, profile)
        aliases = self._dedup([expr.text, canonical_text, *self._profile_aliases_for_role(role, profile)])
        synonyms = self._dedup(self._profile_synonyms_for_role(role, profile))
        lexical_variants = self._dedup([self._normalize_label(expr.text), self._normalize_label(canonical_text)])

        alias_sources = {alias: ["source" if alias == expr.text else "profile"] for alias in aliases}
        synonym_sources = {syn: ["profile"] for syn in synonyms}
        lexical_variant_sources = {variant: ["normalizer"] for variant in lexical_variants}

        ontology_hints = self._dedup(
            [
                f"semantic_role:{role}",
                f"candidate_family:{ontology_link.get('candidate_family', 'entity')}",
                f"ontology_status:{ontology_link.get('ontology_status', 'candidate_instance')}",
                f"promote_to_ontology:{str(ontology_link.get('promote_to_ontology', False)).lower()}",
                ontology_link.get("class_uri"),
                ontology_link.get("class_label"),
            ]
        )

        definition = ontology_link.get("definition") or (
            f"Profile-constrained {role} node extracted from the source document and linked to the seed ontology."
        )
        evidence_content = {
            "strategy": "ontology_aware_conservative_normalization",
            "semantic_role": role,
            "canonical_text": canonical_text,
            "ontology_link": ontology_link,
        }
        return EnrichedExpression(
            base_expression=expr,
            aliases=aliases,
            synonyms=synonyms,
            lexical_variants=lexical_variants,
            alias_sources=alias_sources,
            synonym_sources=synonym_sources,
            lexical_variant_sources=lexical_variant_sources,
            definition=definition,
            ontology_hints=ontology_hints,
            enrichment_evidence=[
                EnrichmentEvidence(
                    source="profile",
                    content=json.dumps(evidence_content, ensure_ascii=False),
                    reference=profile.get("profile_name"),
                )
            ],
        )

    def _normalize_role(self, label: str | None) -> str:
        role = (label or "unknown").strip().lower()
        role = re.sub(r"[^a-z0-9_\-]+", "_", role)
        role = role.strip("_") or "unknown"
        role_aliases = {
            "alarm_label": "alarm",
            "alarm_text": "alarm",
            "message_label": "message",
            "message_text": "message",
            "responsible_actor": "responsible",
            "responsible_items": "responsible",
            "reference_items": "reference",
            "intervention_items": "intervention",
            "effect_items": "effect",
            "cause_items": "cause",
        }
        return role_aliases.get(role, role)

    def _profile_aliases_for_role(self, role: str, profile: dict[str, Any]) -> list[str]:
        aliases = profile.get("table_extraction", {}).get("field_aliases", {})
        if not isinstance(aliases, dict):
            return []
        return [str(x) for x in aliases.get(role, []) if x]

    def _profile_synonyms_for_role(self, role: str, profile: dict[str, Any]) -> list[str]:
        synonyms = profile.get("ontology_linking", {}).get("node_roles", {}).get(role, {}).get("synonyms", [])
        return [str(x) for x in synonyms if x]

    def _role_ontology_link(self, role: str, profile: dict[str, Any]) -> dict[str, Any]:
        node_roles = profile.get("ontology_linking", {}).get("node_roles", {})
        if isinstance(node_roles, dict) and isinstance(node_roles.get(role), dict):
            return dict(node_roles[role])
        return {
            "class_label": role.title(),
            "class_uri": f"http://neoolaf.org/ontology/{role.title().replace('_', '')}",
            "candidate_family": "entity",
            "ontology_status": "candidate_instance",
            "promote_to_ontology": False,
        }

    def _canonicalize_expression(self, text: str, role: str, profile: dict[str, Any]) -> str:
        value = str(text or "").strip()
        if not value:
            return value
        mappings = profile.get("canonical_node_label_mappings", {})
        role_mapping = {}
        if isinstance(mappings, dict):
            role_mapping = mappings.get(role, {}) or mappings.get("global", {}) or {}
        if isinstance(role_mapping, dict):
            lowered = value.lower()
            for source, target in role_mapping.items():
                if lowered == str(source).lower():
                    return str(target)
        return value

    # ------------------------------------------------------------------
    # Original generic LLM/RAG strategy
    # ------------------------------------------------------------------
    def _process_expression_generic(self, expr: LinguisticExpression, state: PipelineState) -> EnrichedExpression:
        """Run the enrichment graph and convert its raw output into an EnrichedExpression."""
        from neoolaf.layers.layer02_candidate_enrichment.graph import Layer02EnrichmentGraphFactory
        from neoolaf.resources.knowledge_sources.wordnet_source import WordNetSource
        from neoolaf.resources.knowledge_sources.wikipedia_source import WikipediaSource
        from neoolaf.resources.knowledge_sources.wikidata_source import WikidataSource
        from neoolaf.resources.knowledge_sources.web_search_source import WebSearchSource

        wikipedia_source = self.wikipedia_source or WikipediaSource()
        wikidata_source = self.wikidata_source or WikidataSource()
        wordnet_source = self.wordnet_source or WordNetSource()
        web_search_source = self.web_search_source or WebSearchSource()

        graph = Layer02EnrichmentGraphFactory(
            ollama_backend=self.ollama_backend,
            model_name=state.llm_model,
            wikipedia_source=wikipedia_source,
            wikidata_source=wikidata_source,
            wordnet_source=wordnet_source,
            web_search_source=web_search_source,
            user_guidance=state.user_guidance,
            use_web_search=self.use_web_search,
            seed_ontology=state.seed_ontology,
            rag_adapter=self.rag_adapter,
        ).build()

        result = graph.invoke({"expression": expr})
        enrichment_result = result.get("enrichment_result", {}) or {}
        gathered_evidence = result.get("gathered_evidence", {}) or {}

        evidence_objects = self._convert_evidence(
            gathered_evidence=gathered_evidence,
            enrichment_result=enrichment_result,
            llm_model=state.llm_model,
        )

        alias_sources: dict[str, list[str]] = {}
        synonym_sources: dict[str, list[str]] = {}
        lexical_variant_sources: dict[str, list[str]] = {}

        self._add_items_with_source(alias_sources, gathered_evidence.get("wordnet", {}).get("aliases", []), "wordnet")
        self._add_items_with_source(synonym_sources, gathered_evidence.get("wordnet", {}).get("synonyms", []), "wordnet")
        self._add_items_with_source(lexical_variant_sources, gathered_evidence.get("wordnet", {}).get("lexical_variants", []), "wordnet")
        self._add_items_with_source(alias_sources, gathered_evidence.get("wikipedia", {}).get("aliases", []), "wikipedia")
        self._add_items_with_source(alias_sources, gathered_evidence.get("wikidata", {}).get("aliases", []), "wikidata")
        self._add_items_with_source(alias_sources, gathered_evidence.get("wikidata", {}).get("labels", []), "wikidata")

        self._add_items_with_source(alias_sources, enrichment_result.get("aliases", []), "llm")
        self._add_items_with_source(synonym_sources, enrichment_result.get("synonyms", []), "llm")
        self._add_items_with_source(lexical_variant_sources, enrichment_result.get("lexical_variants", []), "llm")

        return EnrichedExpression(
            base_expression=expr,
            aliases=list(alias_sources.keys()),
            synonyms=list(synonym_sources.keys()),
            lexical_variants=list(lexical_variant_sources.keys()),
            alias_sources=alias_sources,
            synonym_sources=synonym_sources,
            lexical_variant_sources=lexical_variant_sources,
            definition=enrichment_result.get("definition"),
            ontology_hints=self._dedup(enrichment_result.get("ontology_hints", [])),
            enrichment_evidence=evidence_objects,
        )

    def _record_failure(self, expr: LinguisticExpression, index: int, exc: Exception, attempt: int | str) -> None:
        """Record one failed expression without crashing the full layer."""
        detail = {
            "index": index,
            "expr_id": expr.expr_id,
            "text": expr.text,
            "label": expr.label,
            "attempt": attempt,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        self._failed_details.append(detail)

    def _save_failed_expressions(self, state: PipelineState) -> None:
        """Persist failed expression IDs and diagnostics for restartable runs."""
        if state.artifact_dir is None:
            return
        layer_dir = Path(state.artifact_dir) / self.name
        layer_dir.mkdir(parents=True, exist_ok=True)
        failed_payload = {
            "layer": self.name,
            "failed_count": len(self._failed_details),
            "failed_expressions": self._failed_details,
        }
        dump_json(layer_dir / "failed_expressions.json", failed_payload)
        details_dir = layer_dir / "failed_expression_details"
        details_dir.mkdir(parents=True, exist_ok=True)
        for item in self._failed_details:
            safe_id = str(item.get("expr_id", item.get("index", "unknown"))).replace("/", "_")
            dump_json(details_dir / f"{safe_id}.json", item)

    def _add_items_with_source(self, mapping: Dict[str, List[str]], items: List[str], source: str) -> None:
        """Add lexical items to a provenance mapping."""
        for item in items or []:
            if isinstance(item, dict):
                item = str(item)
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized:
                continue
            mapping.setdefault(normalized, [])
            if source not in mapping[normalized]:
                mapping[normalized].append(source)

    def _dedup(self, items: List[Any]) -> List[str]:
        """Deduplicate strings while preserving order."""
        cleaned = []
        for item in items or []:
            if item is None:
                continue
            if isinstance(item, dict):
                item = str(item)
            if not isinstance(item, str):
                item = str(item)
            item = item.strip()
            if item:
                cleaned.append(item)
        return list(dict.fromkeys(cleaned))

    def _normalize_label(self, text: str) -> str:
        text = str(text or "").lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\-]", "", text)
        return text

    def _convert_evidence(self, gathered_evidence: dict, enrichment_result: dict, llm_model: str) -> List[EnrichmentEvidence]:
        """Convert gathered evidence and the final LLM synthesis into typed evidence objects."""
        evidences: list[EnrichmentEvidence] = []

        wordnet_data = gathered_evidence.get("wordnet", {}) or {}
        for definition in wordnet_data.get("definitions", []) or []:
            evidences.append(EnrichmentEvidence(source="wordnet", content=str(definition), reference=None))

        wikipedia_data = gathered_evidence.get("wikipedia", {}) or {}
        if wikipedia_data.get("found"):
            evidences.append(
                EnrichmentEvidence(
                    source="wikipedia",
                    content=wikipedia_data.get("summary", ""),
                    reference=wikipedia_data.get("url"),
                )
            )

        wikidata_data = gathered_evidence.get("wikidata", {}) or {}
        for item in wikidata_data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            text = f"{', '.join(item.get('labels', []))} -- {item.get('description', '')}".strip()
            evidences.append(EnrichmentEvidence(source="wikidata", content=text, reference=item.get("url")))

        web_data = gathered_evidence.get("web", {}) or {}
        for item in web_data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            text = f"{item.get('title', '')} -- {item.get('body', '')}".strip()
            evidences.append(EnrichmentEvidence(source="web", content=text, reference=item.get("href")))

        evidences.append(
            EnrichmentEvidence(
                source="llm",
                content=json.dumps(to_jsonable(enrichment_result), ensure_ascii=False),
                reference=llm_model,
            )
        )
        return evidences

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """Serialize enriched expressions with lexical provenance."""
        return {
            "layer": self.name,
            "num_enriched_expressions": len(state.enriched_expressions),
            "failed_count": len(getattr(self, "_failed_details", [])),
            "enriched_expressions": [
                {
                    "base_expression": {
                        "expr_id": item.base_expression.expr_id,
                        "text": item.base_expression.text,
                        "label": item.base_expression.label,
                    },
                    "aliases": item.aliases,
                    "synonyms": item.synonyms,
                    "lexical_variants": item.lexical_variants,
                    "alias_sources": item.alias_sources,
                    "synonym_sources": item.synonym_sources,
                    "lexical_variant_sources": item.lexical_variant_sources,
                    "definition": item.definition,
                    "ontology_hints": item.ontology_hints,
                    "evidence": [
                        {"source": ev.source, "content": ev.content, "reference": ev.reference}
                        for ev in item.enrichment_evidence
                    ],
                }
                for item in state.enriched_expressions
            ],
        }

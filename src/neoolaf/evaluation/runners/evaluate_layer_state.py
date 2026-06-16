from __future__ import annotations

"""Layer-wise evaluation helpers for NeoOLAF ablation studies.

These utilities evaluate a saved `<run_dir>/<layer_name>/state.json` directly,
without requiring layer 12 serialization.  They are intentionally permissive so
that early layers can still be scored against XQuality Machine 32 gold labels.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.evaluation.datasets.xquality import XQualityGold, load_xquality_gold
from neoolaf.evaluation.metrics.extraction import evaluate_extraction
from neoolaf.evaluation.profiles.registry import get_profile
from neoolaf.evaluation.schema.artifact import EvalDocument, EvalEntity, EvalRelation, EvaluationArtifact


LAYER_NAMES = [
    "layer00_preprocessing",
    "layer01_linguistic_expression_extraction",
    "layer02_candidate_enrichment",
    "layer03_candidate_typing_resolution",
    "layer04_candidate_relation_extraction",
    "layer05_candidate_triple_generation",
    "layer06_concept_relation_induction",
    "layer07_hierarchisation",
    "layer08_axiom_schemata_extraction",
    "layer09_general_axiom_extraction",
    "layer10_validation_reasoning",
    "layer11_inference_completion",
    "layer12_serialization",
]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def convert_xquality_excel_to_json(excel_path: str | Path, json_path: str | Path) -> Path:
    """Convert a flat XQuality triplet Excel file to the JSON format used by evaluators."""
    excel_path = Path(excel_path)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(excel_path)
    rows = df.fillna("").to_dict(orient="records")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def load_xquality_gold_any(path: str | Path) -> XQualityGold:
    """Load XQuality gold truth from JSON, XLSX, or XLS."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        temp_json = path.with_suffix(".json")
        convert_xquality_excel_to_json(path, temp_json)
        return load_xquality_gold(temp_json)
    return load_xquality_gold(path)


def gold_to_artifact(gold: XQualityGold, *, profile: str) -> EvaluationArtifact:
    artifact = EvaluationArtifact(
        method="gold",
        dataset="xquality",
        profile=profile,
        run_id="gold",
    )
    doc_id = "xquality"
    artifact.documents.append(EvalDocument(document_id=doc_id, metadata={"dataset": "xquality"}))
    artifact.entities_by_doc[doc_id] = gold.entities
    artifact.relations_by_doc[doc_id] = gold.relations
    return artifact


def _evidence_to_text(evidence: Any) -> str:
    if not evidence:
        return ""
    if isinstance(evidence, list):
        parts = []
        for item in evidence:
            snippet = getattr(item, "snippet", None)
            if snippet:
                parts.append(str(snippet))
            elif isinstance(item, dict) and item.get("snippet"):
                parts.append(str(item["snippet"]))
        return " | ".join(parts)
    snippet = getattr(evidence, "snippet", None)
    return str(snippet or "")


def artifact_from_layer_state(
    state: PipelineState,
    *,
    layer_name: str | None = None,
    profile: str = "xquality_loose",
    run_id: str = "neoolaf_layer",
) -> EvaluationArtifact:
    """Build a canonical EvaluationArtifact from any saved NeoOLAF state."""
    doc_id = state.document.doc_id or "xquality"
    artifact = EvaluationArtifact(
        method="neoolaf",
        dataset="xquality",
        profile=profile,
        run_id=run_id,
        metadata={"layer_name": layer_name or "unknown", "source_state": "PipelineState"},
    )
    artifact.documents.append(
        EvalDocument(
            document_id=doc_id,
            text=state.document.cleaned_text or state.document.raw_text,
            source_path=state.document.source_path,
        )
    )

    entities: dict[str, EvalEntity] = {}
    relations: list[EvalRelation] = []

    def add_entity(label: str | None, *, id: str | None = None, type_: str | None = None, evidence: str | None = None, raw: Any = None) -> None:
        label = str(label or "").strip()
        if not label:
            return
        key = label.lower()
        if key not in entities:
            entities[key] = EvalEntity(
                label=label,
                id=id,
                type=type_,
                evidence=evidence,
                provenance_present=bool(evidence),
                raw=_jsonable(raw) if raw is not None else {},
            )

    # Layer 1: linguistic expressions.
    for expression in getattr(state, "linguistic_expressions", []) or []:
        add_entity(
            getattr(expression, "label", None) or getattr(expression, "text", None),
            id=getattr(expression, "expr_id", None),
            type_="linguistic_expression",
            evidence=_evidence_to_text(getattr(expression, "evidence", None)),
            raw=expression,
        )

    # Layer 2: enriched expressions and aliases.
    for enriched in getattr(state, "enriched_expressions", []) or []:
        base = getattr(enriched, "base_expression", None)
        add_entity(
            getattr(base, "label", None) or getattr(base, "text", None),
            id=getattr(base, "expr_id", None),
            type_="enriched_expression",
            evidence=_evidence_to_text(getattr(base, "evidence", None)),
            raw=enriched,
        )
        for alias in list(getattr(enriched, "aliases", []) or []) + list(getattr(enriched, "synonyms", []) or []):
            add_entity(alias, type_="alias_or_synonym", raw={"source": "layer02"})

    # Layer 3+: typed candidates.
    for field_name, type_name in [
        ("entity_candidates", "entity"),
        ("event_candidates", "event"),
        ("attribute_candidates", "attribute"),
    ]:
        for candidate in getattr(state, field_name, []) or []:
            evidence = ""
            mentions = getattr(candidate, "mentions", []) or []
            if mentions:
                evidence = _evidence_to_text(getattr(mentions[0], "evidence", None))
            add_entity(
                getattr(candidate, "canonical_label", None),
                id=getattr(candidate, "candidate_id", None),
                type_=type_name,
                evidence=evidence,
                raw=candidate,
            )

    # Relation candidates as entity-like labels for early-layer coverage.
    for candidate in getattr(state, "relation_candidates", []) or []:
        add_entity(
            getattr(candidate, "canonical_label", None),
            id=getattr(candidate, "candidate_id", None),
            type_="relation_candidate",
            raw=candidate,
        )

    # Layer 4: candidate relation assertions.
    for assertion in getattr(state, "candidate_relation_assertions", []) or []:
        head = getattr(assertion, "source_candidate_label", "")
        relation = getattr(assertion, "relation_label", "")
        tail = getattr(assertion, "target_candidate_label", "")
        if head and relation and tail:
            relations.append(
                EvalRelation(
                    head=head,
                    relation=relation,
                    tail=tail,
                    evidence=getattr(assertion, "justification", "") or _evidence_to_text(getattr(assertion, "evidence", None)),
                    confidence=getattr(assertion, "confidence", None),
                    provenance_present=bool(getattr(assertion, "chunk_id", None) or getattr(assertion, "evidence", None)),
                    provenance={"chunk_id": getattr(assertion, "chunk_id", "")},
                    raw=_jsonable(assertion),
                )
            )
            add_entity(head, type_="relation_source")
            add_entity(tail, type_="relation_target")

    # Layer 5+: candidate triples.
    for triple in getattr(state, "candidate_triples", []) or []:
        head = getattr(triple, "subject_label", "")
        relation = getattr(triple, "predicate_label", "")
        tail = getattr(triple, "object_label", "")
        if head and relation and tail:
            relations.append(
                EvalRelation(
                    head=head,
                    relation=relation,
                    tail=tail,
                    evidence=getattr(triple, "justification", "") or _evidence_to_text(getattr(triple, "provenance", None)),
                    confidence=getattr(triple, "confidence", None),
                    provenance_present=bool(getattr(triple, "chunk_id", None) or getattr(triple, "provenance", None)),
                    provenance={"chunk_id": getattr(triple, "chunk_id", "")},
                    raw=_jsonable(triple),
                )
            )
            add_entity(head, id=getattr(triple, "subject_id", None), type_=getattr(triple, "subject_type", None))
            add_entity(tail, id=getattr(triple, "object_id", None), type_=getattr(triple, "object_type", None))

    # Layer 6 ontology candidates can be compared as semantic labels.
    for concept in getattr(state, "concept_candidates", []) or []:
        add_entity(
            getattr(concept, "label", None),
            id=getattr(concept, "concept_id", None),
            type_="concept_candidate",
            evidence=getattr(concept, "justification", None),
            raw=concept,
        )
    for relation in getattr(state, "ontology_relation_candidates", []) or []:
        add_entity(
            getattr(relation, "label", None),
            id=getattr(relation, "relation_id", None),
            type_="ontology_relation_candidate",
            evidence=getattr(relation, "justification", None),
            raw=relation,
        )

    artifact.entities_by_doc[doc_id] = sorted(entities.values(), key=lambda e: e.label.lower())
    artifact.relations_by_doc[doc_id] = relations
    return artifact


def evaluate_layer_state(
    *,
    state_path: str | Path,
    gold_path: str | Path,
    profile_name: str = "xquality_loose",
    layer_name: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one saved layer state against XQuality gold truth."""
    state_path = Path(state_path)
    state = PipelineState.load_json(str(state_path))
    gold = load_xquality_gold_any(gold_path)
    profile = get_profile(profile_name)

    pred_artifact = artifact_from_layer_state(
        state,
        layer_name=layer_name or state_path.parent.name,
        profile=profile_name,
        run_id=state_path.parent.name,
    )
    gold_artifact = gold_to_artifact(gold, profile=profile_name)
    extraction = evaluate_extraction(pred_artifact, gold_artifact, profile)

    result: dict[str, Any] = {
        "state_path": str(state_path),
        "layer_name": layer_name or state_path.parent.name,
        "profile": profile_name,
        "extraction": extraction,
        "counts": extraction.get("counts", {}),
        "prompt_stats": _load_prompt_stats_for_state(state_path),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def _load_prompt_stats_for_state(state_path: Path) -> dict[str, Any]:
    path = state_path.parent / "prompt_stats.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": f"Could not parse {path}"}


def find_layer_state_files(run_dir: str | Path) -> list[Path]:
    """Return layer state files in layer order."""
    run_dir = Path(run_dir)
    out: list[Path] = []
    for layer_name in LAYER_NAMES:
        path = run_dir / layer_name / "state.json"
        if path.exists():
            out.append(path)
    return out


def evaluate_run_layers(
    *,
    run_dir: str | Path,
    gold_path: str | Path,
    profile_name: str = "xquality_loose",
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all layer state files found in a run directory."""
    results = []
    for state_path in find_layer_state_files(run_dir):
        layer_name = state_path.parent.name
        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir) / f"{layer_name}_eval.json"
        results.append(
            evaluate_layer_state(
                state_path=state_path,
                gold_path=gold_path,
                profile_name=profile_name,
                layer_name=layer_name,
                output_path=output_path,
            )
        )
    return results

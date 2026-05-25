"""High-level evaluation runner for method-specific outputs."""

from __future__ import annotations

from pathlib import Path

from neoolaf.evaluation.adapters.neoolaf_json_adapter import artifact_from_neoolaf_exports
from neoolaf.evaluation.adapters.singlepass_adapter import artifact_from_singlepass
from neoolaf.evaluation.adapters.taxodriven_ttl_adapter import artifact_from_taxodriven_ttl
from neoolaf.evaluation.adapters.ttl_adapter import ontology_from_ttl
from neoolaf.evaluation.datasets.xquality import dedup_relations, load_xquality_gold
from neoolaf.evaluation.metrics.extraction import evaluate_extraction
from neoolaf.evaluation.metrics.ontology import evaluate_ontology
from neoolaf.evaluation.metrics.validation import evaluate_validation
from neoolaf.evaluation.profiles.registry import get_profile
from neoolaf.evaluation.reports.writers import write_run_outputs
from neoolaf.evaluation.schema.artifact import EvaluationArtifact


def _xquality_gold_artifact(gold_path: str | Path, profile_name: str) -> tuple[EvaluationArtifact, object]:
    """Build a gold artifact and keep the XQualityGold helper."""
    gold = load_xquality_gold(gold_path)
    artifact = EvaluationArtifact(method="gold", dataset="xquality", profile=profile_name, run_id="gold")
    artifact.documents = gold.documents
    artifact.entities_by_doc = {"xquality": gold.entities}
    artifact.relations_by_doc = {"xquality": gold.relations}
    return artifact, gold


def _canonicalize_xquality_pred_artifact(pred: EvaluationArtifact, xq_gold: object, profile) -> EvaluationArtifact:
    """Apply XQuality-specific canonicalization to predicted artifacts."""
    if pred.dataset != "xquality":
        return pred

    new_entities_by_doc = {}
    for doc_id, entities in pred.entities_by_doc.items():
        out = []
        for entity in entities:
            label = entity.label
            if profile.use_alias_maps:
                label = xq_gold.canonicalize_alarm_label(label)
            entity.label = label
            out.append(entity)
        new_entities_by_doc[doc_id] = out
    pred.entities_by_doc = new_entities_by_doc

    new_relations_by_doc = {}
    for doc_id, relations in pred.relations_by_doc.items():
        out = []
        for relation in relations:
            if profile.use_alarm_number_anchoring:
                relation.head = xq_gold.canonicalize_alarm_label(relation.head)
                relation.tail = xq_gold.canonicalize_alarm_label(relation.tail)
            if pred.method == "neoolaf" and profile.gt_guided_canonicalization:
                out.extend(xq_gold.canonicalize_neoolaf_relation(relation, profile))
            else:
                out.append(relation)
        new_relations_by_doc[doc_id] = dedup_relations(out)
    pred.relations_by_doc = new_relations_by_doc
    return pred


def build_method_artifact(
    *,
    dataset: str,
    method: str,
    profile_name: str,
    input_path: str | Path | None = None,
    kg_ttl: str | Path | None = None,
    ontology_ttl: str | Path | None = None,
    run_id: str | None = None,
    modality: str | None = None,
) -> EvaluationArtifact:
    """Build a method artifact from the corresponding adapter."""
    method_norm = method.strip().lower()
    run_id = run_id or method_norm

    if method_norm == "singlepass":
        if input_path is None:
            raise ValueError("--input is required for method=singlepass")
        return artifact_from_singlepass(input_path, dataset=dataset, profile=profile_name, run_id=run_id)

    if method_norm in {"taxodrivenkg", "taxodriven"}:
        if kg_ttl is None:
            raise ValueError("--kg-ttl is required for method=taxodrivenkg")
        return artifact_from_taxodriven_ttl(kg_ttl, ontology_ttl, dataset=dataset, profile=profile_name, run_id=run_id)

    if method_norm == "neoolaf":
        if input_path is None:
            raise ValueError("--input is required for method=neoolaf")
        return artifact_from_neoolaf_exports(input_path, dataset=dataset, profile=profile_name, run_id=run_id, modality=modality)

    raise ValueError(f"Unsupported method: {method}")


def evaluate_run(
    *,
    dataset: str,
    method: str,
    profile_name: str,
    gold_path: str | Path,
    output_dir: str | Path,
    input_path: str | Path | None = None,
    kg_ttl: str | Path | None = None,
    ontology_ttl: str | Path | None = None,
    ontology_path: str | Path | None = None,
    run_id: str | None = None,
    modality: str | None = None,
) -> dict:
    """Evaluate one method output against one dataset gold file."""
    profile = get_profile(profile_name)
    dataset_norm = dataset.strip().lower()

    if dataset_norm != "xquality":
        raise ValueError("evaluate currently supports dataset=xquality. Use evaluate-jsonl for generic JSONL datasets.")

    gold_artifact, xq_gold = _xquality_gold_artifact(gold_path, profile.name)
    pred_artifact = build_method_artifact(
        dataset=dataset_norm,
        method=method,
        profile_name=profile.name,
        input_path=input_path,
        kg_ttl=kg_ttl,
        ontology_ttl=ontology_ttl,
        run_id=run_id,
        modality=modality,
    )
    pred_artifact = _canonicalize_xquality_pred_artifact(pred_artifact, xq_gold, profile)

    # XQuality gold is a single-document artifact. If a method artifact has a
    # different document id, align it to xquality for global comparison.
    if len(pred_artifact.relations_by_doc) == 1 and "xquality" not in pred_artifact.relations_by_doc:
        only_doc = next(iter(pred_artifact.relations_by_doc))
        pred_artifact.relations_by_doc = {"xquality": pred_artifact.relations_by_doc[only_doc]}
        pred_artifact.entities_by_doc = {"xquality": pred_artifact.entities_by_doc.get(only_doc, [])}
        pred_artifact.documents = gold_artifact.documents

    extraction = evaluate_extraction(pred_artifact, gold_artifact, profile)
    validation = evaluate_validation(pred_artifact, gold_artifact, profile)
    seed_ontology = ontology_from_ttl(ontology_path) if ontology_path else None
    ontology = evaluate_ontology(pred_artifact.global_ontology, seed_ontology=seed_ontology)

    summary = {
        "method": pred_artifact.method,
        "dataset": dataset_norm,
        "profile": profile.name,
        "run_id": pred_artifact.run_id,
        "config": profile.to_dict(),
        "extraction": extraction,
        "validation": validation,
        "ontology": ontology,
        "errors": [],
    }
    write_run_outputs(output_dir, pred_artifact, summary)
    return summary

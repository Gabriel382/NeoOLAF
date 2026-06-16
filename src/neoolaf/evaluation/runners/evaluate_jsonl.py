"""Evaluation runner for generic relation extraction JSONL datasets."""

from __future__ import annotations

from pathlib import Path

from neoolaf.evaluation.adapters.jsonl_relation_adapter import artifact_from_prediction_jsonl
from neoolaf.evaluation.adapters.ttl_adapter import ontology_from_ttl
from neoolaf.evaluation.datasets.generic_jsonl import load_generic_gold_jsonl
from neoolaf.evaluation.metrics.extraction import evaluate_extraction
from neoolaf.evaluation.metrics.ontology import evaluate_ontology
from neoolaf.evaluation.metrics.validation import evaluate_validation
from neoolaf.evaluation.profiles.registry import get_profile
from neoolaf.evaluation.reports.writers import write_run_outputs
from neoolaf.evaluation.schema.artifact import EvaluationArtifact


def evaluate_jsonl(
    *,
    gold_jsonl_path: str | Path,
    prediction_jsonl_path: str | Path,
    profile_name: str,
    output_dir: str | Path,
    ontology_path: str | Path | None = None,
    dataset: str = "generic",
    method: str = "jsonl",
    type_filter: str | list[str] = "all",
    run_id: str | None = None,
) -> dict:
    """Evaluate predictions on a generic JSONL dataset."""
    profile = get_profile(profile_name)
    documents, entities_by_doc, relations_by_doc = load_generic_gold_jsonl(gold_jsonl_path, type_filter=type_filter)
    gold_artifact = EvaluationArtifact(
        method="gold",
        dataset=dataset,
        profile=profile.name,
        run_id="gold",
        documents=documents,
        entities_by_doc=entities_by_doc,
        relations_by_doc=relations_by_doc,
    )
    pred_artifact = artifact_from_prediction_jsonl(
        prediction_jsonl_path,
        dataset=dataset,
        profile=profile.name,
        method=method,
        run_id=run_id or Path(prediction_jsonl_path).stem,
    )

    extraction = evaluate_extraction(pred_artifact, gold_artifact, profile)
    validation = evaluate_validation(pred_artifact, gold_artifact, profile)
    ontology = evaluate_ontology(ontology_from_ttl(ontology_path) if ontology_path else None)

    total_docs = len(documents)
    pred_doc_ids = {doc.document_id for doc in pred_artifact.documents}
    gold_doc_ids = {doc.document_id for doc in documents}
    missing_predictions = len(gold_doc_ids - pred_doc_ids)
    parsed_failures = sum(1 for doc in pred_artifact.documents if doc.metadata.get("parsed_ok") is False)

    summary = {
        "method": pred_artifact.method,
        "dataset": dataset,
        "profile": profile.name,
        "run_id": pred_artifact.run_id,
        "config": profile.to_dict(),
        "extraction": extraction,
        "validation": validation,
        "ontology": ontology,
        "total_docs": total_docs,
        "missing_predictions": missing_predictions,
        "parsed_failures": parsed_failures,
        "errors": [],
    }
    write_run_outputs(output_dir, pred_artifact, summary)
    return summary

from __future__ import annotations

"""No-gold evaluation runner for NeoOLAF.

This runner evaluates a completed NeoOLAF PipelineState without requiring a gold
truth.  It combines automatic checks that already exist in the evaluation
package with a source-grounded table/role support check.
"""

import csv
import json
from pathlib import Path
from typing import Any, Optional

from neoolaf.core.pipeline_state import PipelineState
from neoolaf.evaluation.no_gold.bleu_score import bleu_to_dict, compute_bleu_scores
from neoolaf.evaluation.no_gold.faithfulness import faithfulness_to_dict, compute_faithfulness
from neoolaf.evaluation.no_gold.ontology_alignment import (
    alignment_to_dict,
    compute_ontology_alignment,
    load_reference_from_json,
    load_reference_from_rdf,
)
from neoolaf.evaluation.no_gold.source_grounding import (
    compute_source_grounding,
    source_grounding_to_dict,
)
from neoolaf.evaluation.no_gold.validation_outcomes import (
    compute_validation_outcomes,
    outcomes_to_dict,
)
from neoolaf.evaluation.no_gold.llm_judge import (
    compute_llm_judge,
    compute_llm_judge_panel,
    llm_judge_to_dict,
    llm_judge_panel_to_dict,
)


def _safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_reference(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_reference_from_json(str(path))
    return load_reference_from_rdf(str(path))


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _generate_markdown_report(summary: dict[str, Any]) -> str:
    validation = summary.get("validation", {})
    source_grounding = summary.get("source_grounding", {})
    faithfulness = summary.get("faithfulness", {})
    bleu = summary.get("bleu", {})
    alignment = summary.get("ontology_alignment")

    lines: list[str] = []
    lines.append("# NeoOLAF No-Gold Evaluation Report")
    lines.append("")
    lines.append(f"State: `{summary.get('state_path')}`")
    lines.append(f"Document: `{summary.get('document')}`")
    lines.append("")

    lines.append("## Structural and validation outcomes")
    lines.append(f"- Valid: `{validation.get('is_valid')}`")
    issues = validation.get("issues", {})
    lines.append(f"- Issues: {issues.get('total', 0)} total, {issues.get('errors', 0)} errors, {issues.get('warnings', 0)} warnings")
    graph = validation.get("graph_health", {})
    lines.append(f"- Candidate triples: {graph.get('candidate_triples', 0)}")
    lines.append(f"- Inferred triples: {graph.get('inferred_triples', 0)}")
    lines.append(f"- Average triple confidence: {_pct(graph.get('avg_triple_confidence'))}")
    ont = validation.get("ontology_health", {})
    lines.append(f"- Orphan concept ratio: {_pct(ont.get('orphan_concept_ratio'))}")
    lines.append(f"- Domain/range coverage: {_pct(ont.get('domain_range_coverage'))}")
    lines.append("")

    lines.append("## Source-grounded automatic checks")
    lines.append(f"- Provenance coverage: {_pct(source_grounding.get('provenance_coverage'))}")
    lines.append(f"- Relation marker support: {_pct(source_grounding.get('relation_marker_support_rate'))}")
    lines.append(f"- Endpoint support: {_pct(source_grounding.get('endpoint_support_rate'))}")
    lines.append(f"- Table record support: {_pct(source_grounding.get('table_record_support_rate'))}")
    lines.append(f"- Source grounding rate: {_pct(source_grounding.get('source_grounding_rate'))}")
    lines.append(f"- Average support score: {_pct(source_grounding.get('average_support_score'))}")
    lines.append("")

    llm_judge = summary.get("llm_judge")
    if llm_judge is not None:
        lines.append("## Optional LLM-as-a-judge checks")
        lines.append(f"- Judge model: `{llm_judge.get('model')}`")
        lines.append(f"- Judged triples: {llm_judge.get('judged_count', 0)}")
        lines.append(f"- Valid / weak / invalid: {llm_judge.get('valid_count', 0)} / {llm_judge.get('weak_count', 0)} / {llm_judge.get('invalid_count', 0)}")
        lines.append(f"- Average judge score: {_pct(llm_judge.get('average_score'))}")
        lines.append(f"- Supported rate: {_pct(llm_judge.get('supported_rate'))}")
        lines.append(f"- Relation-supported rate: {_pct(llm_judge.get('relation_supported_rate'))}")
        lines.append(f"- Direction-correct rate: {_pct(llm_judge.get('direction_correct_rate'))}")
        lines.append("")

    llm_judge_panel = summary.get("llm_judge_panel")
    if llm_judge_panel is not None:
        lines.append("## Optional multi-judge LLM panel")
        lines.append(f"- Judge model: `{llm_judge_panel.get('model')}`")
        lines.append(f"- Judged triples: {llm_judge_panel.get('judged_count', 0)}")
        lines.append(f"- Valid / weak / invalid / inconclusive: {llm_judge_panel.get('valid_count', 0)} / {llm_judge_panel.get('weak_count', 0)} / {llm_judge_panel.get('invalid_count', 0)} / {llm_judge_panel.get('inconclusive_count', 0)}")
        lines.append(f"- Average final score: {_pct(llm_judge_panel.get('average_score'))}")
        lines.append(f"- Supported rate: {_pct(llm_judge_panel.get('supported_rate'))}")
        lines.append(f"- Relation-supported rate: {_pct(llm_judge_panel.get('relation_supported_rate'))}")
        lines.append(f"- Direction-correct rate: {_pct(llm_judge_panel.get('direction_correct_rate'))}")
        lines.append(f"- Agreement high/medium/low: {llm_judge_panel.get('high_agreement_count', 0)} / {llm_judge_panel.get('medium_agreement_count', 0)} / {llm_judge_panel.get('low_agreement_count', 0)}")
        lines.append("")

    lines.append("## Legacy faithfulness checks")
    lines.append(f"- Provenance coverage: {_pct(faithfulness.get('provenance_coverage'))}")
    lines.append(f"- Textual grounding rate: {_pct(faithfulness.get('textual_grounding_rate'))}")
    lines.append(f"- Contradiction rate: {_pct(faithfulness.get('contradiction_rate'))}")
    lines.append("")

    lines.append("## BLEU-style textual overlap")
    lines.append(f"- Pairs evaluated: {bleu.get('scores_count', 0)}")
    lines.append(f"- Average BLEU: {_pct(bleu.get('avg_bleu'))}")
    lines.append(f"- Median BLEU: {_pct(bleu.get('median_bleu'))}")
    lines.append("")

    if alignment is not None:
        lines.append("## Reference ontology alignment")
        concepts = alignment.get("concepts", {})
        relations = alignment.get("relations", {})
        hierarchy = alignment.get("hierarchy", {})
        lines.append(f"- Concept alignment: {_pct(concepts.get('alignment_rate'))} ({concepts.get('aligned', 0)}/{concepts.get('total', 0)})")
        lines.append(f"- Relation alignment: {_pct(relations.get('alignment_rate'))} ({relations.get('aligned', 0)}/{relations.get('total', 0)})")
        lines.append(f"- Hierarchy alignment: {_pct(hierarchy.get('alignment_rate'))} ({hierarchy.get('aligned', 0)}/{hierarchy.get('total', 0)})")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("- This report is fully automatic and does not require a manually annotated gold truth.")
    lines.append("- Source grounding is based on provenance snippets, table markers, ontology roles, and relation-specific field markers.")
    lines.append("- BLEU is included only as a weak lexical-overlap signal, not as the main KG-quality metric.")
    return "\n".join(lines) + "\n"


def evaluate_no_gold_state(
    *,
    state_path: str | Path,
    output_dir: str | Path,
    reference_ontology_path: str | Path | None = None,
    alignment_threshold: float = 0.75,
    include_bleu: bool = True,
    llm_judge_model: str | None = None,
    llm_judge_max_items: int = 50,
    llm_judge_only_weak: bool = True,
    llm_judge_temperature: float = 0.0,
    llm_judge_max_tokens: int = 1200,
    llm_judge_max_workers: int = 4,
    llm_judge_panel: bool = False,
    llm_judge_count_subjudge_parse_errors: bool = False,
) -> dict[str, Any]:
    """Run automatic no-gold evaluation on a saved PipelineState."""
    state_path = Path(state_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = PipelineState.load_json(str(state_path))

    validation = compute_validation_outcomes(state)
    source_grounding = compute_source_grounding(state)
    faithfulness = compute_faithfulness(state)

    bleu_result: dict[str, Any]
    if include_bleu:
        try:
            bleu_result = bleu_to_dict(compute_bleu_scores(state))
        except Exception as exc:  # NLTK data may be missing in some environments.
            bleu_result = {"error": str(exc), "scores_count": 0}
    else:
        bleu_result = {"skipped": True, "reason": "disabled_by_user"}

    alignment_result: dict[str, Any] | None = None
    if reference_ontology_path is not None:
        reference = _load_reference(reference_ontology_path)
        alignment_result = alignment_to_dict(
            compute_ontology_alignment(
                state,
                reference,
                threshold=alignment_threshold,
            )
        )

    source_grounding_dict = source_grounding_to_dict(source_grounding)
    faithfulness_dict = faithfulness_to_dict(faithfulness)

    llm_judge_result: dict[str, Any] | None = None
    llm_judge_panel_result: dict[str, Any] | None = None
    if llm_judge_model and llm_judge_panel:
        panel_report = compute_llm_judge_panel(
            state,
            model=llm_judge_model,
            source_grounding=source_grounding,
            max_items=llm_judge_max_items,
            only_weak=llm_judge_only_weak,
            temperature=llm_judge_temperature,
            max_tokens=llm_judge_max_tokens,
            max_workers=llm_judge_max_workers,
            count_subjudge_parse_errors=llm_judge_count_subjudge_parse_errors,
        )
        llm_judge_panel_result = llm_judge_panel_to_dict(panel_report)
    elif llm_judge_model:
        llm_judge_report = compute_llm_judge(
            state,
            model=llm_judge_model,
            source_grounding=source_grounding,
            max_items=llm_judge_max_items,
            only_weak=llm_judge_only_weak,
            temperature=llm_judge_temperature,
            max_tokens=llm_judge_max_tokens,
            max_workers=llm_judge_max_workers,
        )
        llm_judge_result = llm_judge_to_dict(llm_judge_report)

    summary: dict[str, Any] = {
        "mode": "no_gold",
        "state_path": str(state_path),
        "document": getattr(state.document, "doc_id", None),
        "source_path": getattr(state.document, "source_path", None),
        "llm_model": getattr(state, "llm_model", None),
        "validation": outcomes_to_dict(validation),
        "source_grounding": {k: v for k, v in source_grounding_dict.items() if k != "per_triple"},
        "faithfulness": faithfulness_dict,
        "bleu": bleu_result,
    }
    if llm_judge_result is not None:
        summary["llm_judge"] = {k: v for k, v in llm_judge_result.items() if k != "items"}
    if llm_judge_panel_result is not None:
        summary["llm_judge_panel"] = {k: v for k, v in llm_judge_panel_result.items() if k != "items"}
    if alignment_result is not None:
        summary["ontology_alignment"] = alignment_result

    _safe_write_json(output_dir / "no_gold.summary.json", summary)
    _safe_write_json(output_dir / "no_gold.source_grounding.full.json", source_grounding_dict)

    # Compact CSV for quick notebook inspection.
    _write_csv(
        output_dir / "no_gold.source_grounding.per_triple.csv",
        source_grounding_dict.get("per_triple", []),
    )

    if llm_judge_result is not None:
        _safe_write_json(output_dir / "no_gold.llm_judge.full.json", llm_judge_result)
        _write_csv(output_dir / "no_gold.llm_judge.per_triple.csv", llm_judge_result.get("items", []))

    if llm_judge_panel_result is not None:
        _safe_write_json(output_dir / "no_gold.llm_judge_panel.full.json", llm_judge_panel_result)
        _write_csv(output_dir / "no_gold.llm_judge_panel.per_triple.csv", llm_judge_panel_result.get("items", []))

    if alignment_result is not None:
        concept_rows = alignment_result.get("concepts", {}).get("pairs", [])
        relation_rows = alignment_result.get("relations", {}).get("pairs", [])
        _write_csv(output_dir / "no_gold.ontology_alignment.concepts.csv", concept_rows)
        _write_csv(output_dir / "no_gold.ontology_alignment.relations.csv", relation_rows)

    (output_dir / "no_gold.report.md").write_text(
        _generate_markdown_report(summary),
        encoding="utf-8",
    )

    return summary

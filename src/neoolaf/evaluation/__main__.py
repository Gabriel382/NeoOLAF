"""Module entry point for NeoOLAF evaluation.

Examples
--------
Evaluate NeoOLAF on XQuality with the configurable domain evaluator:

    python scripts/evaluation/run_eval.py evaluate \
      --dataset xquality \
      --method neoolaf \
      --profile xquality_relaxed_fair \
      --input runs/run_20260408_091832/data/exports \
      --gold data/XQuality/Examples/XQuality_all_triplets_flat_en.json \
      --output outputs/evaluation/xquality/neoolaf/xquality_relaxed_fair

Evaluate another method with the legacy/general evaluator:

    python scripts/evaluation/run_eval.py evaluate \
      --dataset xquality \
      --method singlepass \
      --profile xquality_loose \
      --input outputs_singlepass_xquality \
      --gold data/XQuality/Examples/XQuality_all_triplets_flat_en.json \
      --ontology-path data/ontology/ContextOntology-COInd4.owl \
      --output outputs/evaluation/xquality/singlepass/xquality_loose
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neoolaf.evaluation.runners.compare_runs import compare_runs
from neoolaf.evaluation.runners.evaluate_jsonl import evaluate_jsonl
from neoolaf.evaluation.runners.evaluate_run import evaluate_run
from neoolaf.evaluation.runners.evaluate_no_gold import evaluate_no_gold_state


def parse_type_filter(raw: str) -> str | list[str]:
    """Parse type-filter CLI option."""
    raw = raw.strip()

    if raw.lower() == "all":
        return "all"

    if "," in raw:
        return [item.strip() for item in raw.split(",") if item.strip()]

    return raw


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by method-specific evaluation commands."""
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. xquality")
    parser.add_argument(
        "--method",
        required=True,
        help="Method name, e.g. singlepass, taxodrivenkg, neoolaf",
    )
    parser.add_argument("--profile", required=True, help="Evaluation profile name")
    parser.add_argument("--gold", required=True, help="Gold file path")
    parser.add_argument("--output", required=True, help="Output directory")

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input directory or file. For NeoOLAF XQuality, this should usually be "
            "the exports folder containing kg_local.json, kg_inferred.json, "
            "ontology_local.ttl, and ontology_inferred.ttl."
        ),
    )

    parser.add_argument("--kg-ttl", default=None, help="KG TTL path for TaxoDrivenKG")
    parser.add_argument("--ontology-ttl", default=None, help="Generated ontology TTL path")
    parser.add_argument("--ontology-path", default=None, help="Reference/seed ontology path")
    parser.add_argument("--run-id", default=None, help="Optional run identifier")
    parser.add_argument("--modality", default=None, help="Optional NeoOLAF modality/ablation name")

    # Explicit NeoOLAF JSON/TTL export paths.
    # These are used by the configurable domain evaluator.
    parser.add_argument("--kg-local-json", default=None, help="NeoOLAF local KG JSON path")
    parser.add_argument("--kg-inferred-json", default=None, help="NeoOLAF inferred KG JSON path")
    parser.add_argument("--ontology-local-ttl", default=None, help="NeoOLAF local ontology TTL path")
    parser.add_argument("--ontology-inferred-ttl", default=None, help="NeoOLAF inferred ontology TTL path")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="NeoOLAF evaluation runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one method output",
    )
    add_common_eval_args(evaluate_parser)

    jsonl_parser = subparsers.add_parser(
        "evaluate-jsonl",
        help="Evaluate a generic relation extraction JSONL prediction file",
    )
    jsonl_parser.add_argument("--profile", required=True)
    jsonl_parser.add_argument("--gold-jsonl-path", required=True)
    jsonl_parser.add_argument("--prediction-jsonl-path", required=True)
    jsonl_parser.add_argument("--ontology-path", default=None)
    jsonl_parser.add_argument("--type-filter", default="all")
    jsonl_parser.add_argument("--dataset", default="generic")
    jsonl_parser.add_argument("--method", default="jsonl")
    jsonl_parser.add_argument("--run-id", default=None)
    jsonl_parser.add_argument("--output", required=True)

    no_gold_parser = subparsers.add_parser(
        "no-gold",
        help="Run automatic no-gold evaluation from a saved NeoOLAF state.json",
    )
    no_gold_parser.add_argument("--state", required=True, help="Path to a saved NeoOLAF state.json")
    no_gold_parser.add_argument("--output", required=True, help="Output directory")
    no_gold_parser.add_argument(
        "--reference-ontology",
        default=None,
        help="Optional reference/seed ontology path for automatic ontology alignment",
    )
    no_gold_parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=0.75,
        help="Fuzzy matching threshold for ontology alignment",
    )
    no_gold_parser.add_argument(
        "--no-bleu",
        action="store_true",
        help="Disable BLEU-style lexical overlap evaluation",
    )
    no_gold_parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable optional LLM-as-a-judge evaluation. This may call a paid/provider API.",
    )
    no_gold_parser.add_argument(
        "--llm-judge-panel",
        action="store_true",
        help="Use a multi-judge panel instead of a single judge: blue support, red critic, profile judge, and arbiter.",
    )
    no_gold_parser.add_argument(
        "--judge-model",
        default=None,
        help="LiteLLM model name for LLM-as-a-judge, e.g. openrouter/openai/gpt-oss-20b",
    )
    no_gold_parser.add_argument(
        "--judge-max-items",
        type=int,
        default=50,
        help="Maximum number of triples to judge with the LLM",
    )
    no_gold_parser.add_argument(
        "--judge-all",
        action="store_true",
        help="Judge all sampled triples instead of focusing on weak/unsupported triples",
    )
    no_gold_parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature used by the LLM judge",
    )
    no_gold_parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1200,
        help="Maximum output tokens for each LLM judge call",
    )
    no_gold_parser.add_argument(
        "--judge-max-workers",
        type=int,
        default=4,
        help="Maximum parallel LLM judge calls. Lower this if your provider rate-limits requests.",
    )
    no_gold_parser.add_argument(
        "--count-subjudge-parse-errors",
        action="store_true",
        help=(
            "For --llm-judge-panel, count recoverable subjudge parse failures in parse_error_count. "
            "Disabled by default because the panel can recover from individual subjudge failures."
        ),
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple evaluation runs",
    )
    compare_parser.add_argument("--runs-dir", required=True)
    compare_parser.add_argument("--output", required=True)

    batch_parser = subparsers.add_parser(
        "batch-evaluate",
        help="Evaluate all subdirectories in an ablation runs directory",
    )
    batch_parser.add_argument("--dataset", required=True)
    batch_parser.add_argument("--method", default="neoolaf")
    batch_parser.add_argument("--profile", required=True)
    batch_parser.add_argument("--runs-dir", required=True)
    batch_parser.add_argument("--gold", required=True)
    batch_parser.add_argument("--ontology-path", default=None)
    batch_parser.add_argument("--output", required=True)

    return parser


def resolve_neoolaf_export_paths(args: argparse.Namespace) -> dict[str, str | None]:
    """Resolve NeoOLAF export paths from either --input or explicit path args."""
    input_dir = Path(args.input) if args.input else None

    kg_local_json = args.kg_local_json
    kg_inferred_json = args.kg_inferred_json
    ontology_local_ttl = args.ontology_local_ttl
    ontology_inferred_ttl = args.ontology_inferred_ttl

    if input_dir is not None:
        kg_local_json = kg_local_json or str(input_dir / "kg_local.json")
        kg_inferred_json = kg_inferred_json or str(input_dir / "kg_inferred.json")
        ontology_local_ttl = ontology_local_ttl or str(input_dir / "ontology_local.ttl")
        ontology_inferred_ttl = ontology_inferred_ttl or str(input_dir / "ontology_inferred.ttl")

    return {
        "kg_local_json": kg_local_json,
        "kg_inferred_json": kg_inferred_json,
        "ontology_local_ttl": ontology_local_ttl,
        "ontology_inferred_ttl": ontology_inferred_ttl,
    }


def evaluate_xquality_neoolaf_with_domain_runner(args: argparse.Namespace) -> None:
    """Evaluate NeoOLAF on XQuality using the configurable domain KG evaluator."""
    from neoolaf.evaluation.domains.xquality import (
        build_neoolaf_json_method_config,
        build_xquality_domain_config,
    )
    from neoolaf.evaluation.runners.evaluate_domain_kg import (
        EvaluationInput,
        evaluate_domain_kg,
    )

    paths = resolve_neoolaf_export_paths(args)

    if not paths["kg_local_json"] and not paths["kg_inferred_json"]:
        raise ValueError(
            "For --dataset xquality --method neoolaf, provide either --input pointing "
            "to the NeoOLAF exports folder, or provide explicit --kg-local-json and/or "
            "--kg-inferred-json paths."
        )

    evaluation_input = EvaluationInput(
        local_json_path=paths["kg_local_json"],
        inferred_json_path=paths["kg_inferred_json"],
        ontology_local_path=paths["ontology_local_ttl"],
        ontology_inferred_path=paths["ontology_inferred_ttl"],
        gold_path=args.gold,
        output_dir=args.output,
    )

    evaluate_domain_kg(
        input_data=evaluation_input,
        method=build_neoolaf_json_method_config(),
        domain=build_xquality_domain_config(),
        profile=args.profile,
    )


def main() -> None:
    """Execute the selected evaluation command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        # IMPORTANT:
        # XQuality + NeoOLAF uses the new configurable domain evaluator.
        # This must happen before the legacy/general evaluate_run call,
        # because profiles such as xquality_relaxed_fair are not part of
        # the old profile registry.
        if args.dataset == "xquality" and args.method == "neoolaf":
            evaluate_xquality_neoolaf_with_domain_runner(args)
            return

        evaluate_run(
            dataset=args.dataset,
            method=args.method,
            profile_name=args.profile,
            input_path=args.input,
            kg_ttl=args.kg_ttl,
            ontology_ttl=args.ontology_ttl,
            gold_path=args.gold,
            ontology_path=args.ontology_path,
            output_dir=args.output,
            run_id=args.run_id,
            modality=args.modality,
        )
        return

    if args.command == "evaluate-jsonl":
        evaluate_jsonl(
            gold_jsonl_path=args.gold_jsonl_path,
            prediction_jsonl_path=args.prediction_jsonl_path,
            profile_name=args.profile,
            ontology_path=args.ontology_path,
            dataset=args.dataset,
            method=args.method,
            type_filter=parse_type_filter(args.type_filter),
            output_dir=args.output,
            run_id=args.run_id,
        )
        return

    if args.command == "no-gold":
        if (args.llm_judge or args.llm_judge_panel) and not args.judge_model:
            parser.error("--llm-judge or --llm-judge-panel requires --judge-model")

        evaluate_no_gold_state(
            state_path=args.state,
            output_dir=args.output,
            reference_ontology_path=args.reference_ontology,
            alignment_threshold=args.alignment_threshold,
            include_bleu=not args.no_bleu,
            llm_judge_model=args.judge_model if (args.llm_judge or args.llm_judge_panel) else None,
            llm_judge_max_items=args.judge_max_items,
            llm_judge_only_weak=not args.judge_all,
            llm_judge_temperature=args.judge_temperature,
            llm_judge_max_tokens=args.judge_max_tokens,
            llm_judge_max_workers=args.judge_max_workers,
            llm_judge_panel=args.llm_judge_panel,
            llm_judge_count_subjudge_parse_errors=args.count_subjudge_parse_errors,
        )
        return

    if args.command == "compare":
        compare_runs(args.runs_dir, args.output)
        return

    if args.command == "batch-evaluate":
        runs_dir = Path(args.runs_dir)
        output_root = Path(args.output)

        for run_dir in sorted(item for item in runs_dir.iterdir() if item.is_dir()):
            # For XQuality + NeoOLAF ablations, use the new configurable evaluator.
            if args.dataset == "xquality" and args.method == "neoolaf":
                from neoolaf.evaluation.domains.xquality import (
                    build_neoolaf_json_method_config,
                    build_xquality_domain_config,
                )
                from neoolaf.evaluation.runners.evaluate_domain_kg import (
                    EvaluationInput,
                    evaluate_domain_kg,
                )

                evaluation_input = EvaluationInput(
                    local_json_path=run_dir / "kg_local.json",
                    inferred_json_path=run_dir / "kg_inferred.json",
                    ontology_local_path=run_dir / "ontology_local.ttl",
                    ontology_inferred_path=run_dir / "ontology_inferred.ttl",
                    gold_path=args.gold,
                    output_dir=output_root / run_dir.name,
                )

                evaluate_domain_kg(
                    input_data=evaluation_input,
                    method=build_neoolaf_json_method_config(),
                    domain=build_xquality_domain_config(),
                    profile=args.profile,
                )

            else:
                evaluate_run(
                    dataset=args.dataset,
                    method=args.method,
                    profile_name=args.profile,
                    input_path=run_dir,
                    gold_path=args.gold,
                    ontology_path=args.ontology_path,
                    output_dir=output_root / run_dir.name,
                    run_id=run_dir.name,
                    modality=run_dir.name,
                )

        compare_runs(output_root, output_root / "comparison")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
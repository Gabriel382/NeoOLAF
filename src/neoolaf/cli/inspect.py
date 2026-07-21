from __future__ import annotations

import argparse
import json

from neoolaf.integration import build_integrity_report, inspect_run
from neoolaf.version import get_version_info


def add_inspect_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "inspect-run",
        help="Inspect an existing NeoOLAF run without modifying it",
    )
    parser.add_argument("--run-dir", required=True, help="Existing NeoOLAF run directory")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--integrity",
        action="store_true",
        help="Include SHA-256 integrity metadata for original exports",
    )


def _payload(args: argparse.Namespace) -> dict:
    snapshot = inspect_run(args.run_dir)
    payload = {
        "neoolaf": get_version_info().to_dict(),
        "run": snapshot.to_dict(),
    }
    if args.integrity:
        payload["integrity"] = build_integrity_report(args.run_dir)
    return payload


def inspect_command(args: argparse.Namespace) -> None:
    payload = _payload(args)
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    run = payload["run"]
    version = payload["neoolaf"]
    print(f"NeoOLAF run: {run['run_dir']}")
    print(f"Inspector pipeline identity: {version['scientific_pipeline']} ({version['layer_contract']})")
    print(f"Status: {run['status']}")

    config = run.get("run_config", {})
    profile = config.get("resolved_profile_name") or config.get("profile")
    model = config.get("model")
    if profile:
        print(f"Profile: {profile}")
    if model:
        print(f"Model: {model}")

    print("\nLayers:")
    symbols = {
        "completed": "[OK]",
        "skipped": "[SKIP]",
        "started": "[RUNNING]",
        "not_started": "[--]",
    }
    for layer in run["layers"]:
        symbol = symbols.get(layer["status"], "[?]")
        elapsed = (
            f" ({layer['elapsed_seconds']:.3f}s)"
            if layer.get("elapsed_seconds") is not None
            else ""
        )
        print(f"  {symbol:9} L{layer['index']:02d} {layer['name']}{elapsed}")

    print("\nExports:")
    exports = {item["name"]: item for item in run["exports"]}
    for name in (
        "ontology_local.ttl",
        "ontology_inferred.ttl",
        "kg_local.ttl",
        "kg_inferred.ttl",
        "kg_local.json",
        "kg_inferred.json",
    ):
        item = exports.get(name)
        if item:
            print(f"  [OK] {name} ({item['size_bytes']} bytes, sha256={item['sha256']})")
        else:
            print(f"  [--] {name}")

    if run.get("final_state_path"):
        print(f"\nFinal JSON state: {run['final_state_path']}")
    elif run.get("checkpoint_path"):
        print(f"\nTrusted-local checkpoint available: {run['checkpoint_path']}")

    warnings = run.get("warnings", [])
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

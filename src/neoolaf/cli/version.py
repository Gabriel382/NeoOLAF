from __future__ import annotations

import argparse
import json

from neoolaf.version import get_version_info


def add_version_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "version",
        help="Show package and scientific-pipeline identity metadata",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")


def version_command(args: argparse.Namespace) -> None:
    info = get_version_info()
    if args.format == "json":
        print(json.dumps(info.to_dict(), indent=2, ensure_ascii=False))
        return

    print(f"NeoOLAF {info.package_version}")
    print(f"Scientific pipeline: {info.scientific_pipeline}")
    print(f"Layer contract: {info.layer_contract}")
    print(f"Scientific release tag: {info.scientific_release_tag}")
    print(f"Run inspection contract: {info.run_inspection_contract_version}")
    print(f"Python: {info.python_version}")

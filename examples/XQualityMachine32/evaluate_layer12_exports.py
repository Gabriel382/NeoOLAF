from __future__ import annotations

"""Convenience command builder for evaluating NeoOLAF Layer 12 exports on XQuality.

Run from the NeoOLAF repository root after Layer 12 has produced the exports folder.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()

GOLD_JSON = ROOT / "data" / "XQuality" / "Examples" / "XQuality_all_triplets_flat_en.json"
SEED_ONTOLOGY = ROOT / "data" / "ontology" / "ContextOntology-COInd4.owl"
EXPORT_DIR = ROOT / "examples" / "XQualityMachine32" / "runs" / "xquality_machine32" / "layer12_from_l11" / "exports"
EVAL_DIR = ROOT / "examples" / "XQualityMachine32" / "runs" / "xquality_machine32" / "eval_layer12"

cmd = [
    sys.executable,
    "-m",
    "neoolaf.evaluation",
    "evaluate",
    "--dataset",
    "xquality",
    "--method",
    "neoolaf",
    "--profile",
    "xquality_relaxed_recall",
    "--input",
    str(EXPORT_DIR),
    "--gold",
    str(GOLD_JSON),
    "--ontology-path",
    str(SEED_ONTOLOGY),
    "--output",
    str(EVAL_DIR),
]

print("Running:")
print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
subprocess.run(cmd, check=True)

from __future__ import annotations

"""Read-only integration surface for external applications."""

from neoolaf.integration.integrity import (
    build_integrity_report,
    hash_directory,
    hash_exports,
    hash_file,
    verify_exports_unchanged,
)
from neoolaf.integration.run_reader import inspect_run, load_final_state
from neoolaf.integration.run_snapshot import ExportSnapshot, LayerSnapshot, RunSnapshot

__all__ = [
    "ExportSnapshot",
    "LayerSnapshot",
    "RunSnapshot",
    "build_integrity_report",
    "hash_directory",
    "hash_exports",
    "hash_file",
    "inspect_run",
    "load_final_state",
    "verify_exports_unchanged",
]

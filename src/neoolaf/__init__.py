from __future__ import annotations

"""Public NeoOLAF package metadata and read-only integration surface."""

from neoolaf.integration import (
    build_integrity_report,
    hash_directory,
    hash_exports,
    hash_file,
    inspect_run,
    load_final_state,
    verify_exports_unchanged,
)
from neoolaf.version import (
    RUN_INSPECTION_CONTRACT_VERSION,
    SCIENTIFIC_LAYER_RANGE,
    SCIENTIFIC_PIPELINE_ID,
    SCIENTIFIC_RELEASE_TAG,
    __version__,
    get_version_info,
)

__all__ = [
    "RUN_INSPECTION_CONTRACT_VERSION",
    "SCIENTIFIC_LAYER_RANGE",
    "SCIENTIFIC_PIPELINE_ID",
    "SCIENTIFIC_RELEASE_TAG",
    "__version__",
    "build_integrity_report",
    "get_version_info",
    "hash_directory",
    "hash_exports",
    "hash_file",
    "inspect_run",
    "load_final_state",
    "verify_exports_unchanged",
]

from __future__ import annotations

"""Version and scientific-pipeline identity metadata for NeoOLAF.

This module is deliberately independent from the pipeline implementation.  It
allows external applications, including NeoOLAF Studio, to verify which
scientific contract they are inspecting without importing or instantiating any
pipeline layer.
"""

from dataclasses import asdict, dataclass
import platform


__version__ = "0.1.0"
SCIENTIFIC_PIPELINE_ID = "neoolaf-kes-2026"
SCIENTIFIC_LAYER_RANGE = "L0-L12"
SCIENTIFIC_RELEASE_TAG = "v0.1.0-kes2026"
RUN_INSPECTION_CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class VersionInfo:
    package_version: str
    scientific_pipeline: str
    layer_contract: str
    scientific_release_tag: str
    run_inspection_contract_version: str
    python_version: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def get_version_info() -> VersionInfo:
    """Return machine-readable package and scientific-contract metadata."""

    return VersionInfo(
        package_version=__version__,
        scientific_pipeline=SCIENTIFIC_PIPELINE_ID,
        layer_contract=SCIENTIFIC_LAYER_RANGE,
        scientific_release_tag=SCIENTIFIC_RELEASE_TAG,
        run_inspection_contract_version=RUN_INSPECTION_CONTRACT_VERSION,
        python_version=platform.python_version(),
    )

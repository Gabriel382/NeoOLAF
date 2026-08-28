from __future__ import annotations

"""Read-only hashing helpers used to prove run and export immutability."""

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from neoolaf.integration.run_contract import EXPORT_NAMES, EXPORT_SEARCH_ROOTS


_CHUNK_SIZE = 1024 * 1024


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""

    file_path = Path(path)
    digest = sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path: str | Path) -> str:
    """Hash a directory deterministically, including relative paths and bytes."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    digest = sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = bytes.fromhex(hash_file(file_path))
        digest.update(file_digest)
    return digest.hexdigest()


def _find_export(run_dir: Path, name: str) -> Path | None:
    for relative_root in EXPORT_SEARCH_ROOTS:
        candidate = run_dir / relative_root / name
        if candidate.is_file():
            return candidate

    # Compatibility fallback for copied/legacy runs.  Sort for deterministic
    # behavior and prefer the shallowest path.
    matches = sorted(
        (path for path in run_dir.rglob(name) if path.is_file()),
        key=lambda path: (len(path.relative_to(run_dir).parts), path.as_posix()),
    )
    return matches[0] if matches else None


def hash_exports(run_dir: str | Path) -> dict[str, str]:
    """Return hashes for all canonical exports found in a run directory."""

    root = Path(run_dir).resolve()
    result: dict[str, str] = {}
    for name in EXPORT_NAMES:
        path = _find_export(root, name)
        if path is not None:
            result[name] = hash_file(path)
    return result


def verify_exports_unchanged(
    before: Mapping[str, str], after: Mapping[str, str]
) -> bool:
    """Return ``True`` only when two export-hash mappings are identical."""

    return dict(before) == dict(after)


def build_integrity_report(run_dir: str | Path) -> dict[str, Any]:
    """Build a reproducibility report from existing files only."""

    root = Path(run_dir).resolve()
    run_config_path = root / "run_config.json"
    input_sha256: str | None = None
    input_path: str | None = None

    if run_config_path.is_file():
        import json

        try:
            config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        raw_input = config.get("input_pdf") if isinstance(config, dict) else None
        if isinstance(raw_input, str) and raw_input:
            input_path = raw_input
            candidate = Path(raw_input).expanduser()
            if not candidate.is_absolute():
                candidate = (root / candidate).resolve()
            if candidate.is_file():
                input_sha256 = hash_file(candidate)

    return {
        "run_directory": str(root),
        "run_config_sha256": hash_file(run_config_path) if run_config_path.is_file() else None,
        "input_path": input_path,
        "input_sha256": input_sha256,
        "exports": hash_exports(root),
    }

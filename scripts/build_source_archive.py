from __future__ import annotations

"""Build a clean NeoOLAF source ZIP and reject probable embedded secrets."""

import argparse
from pathlib import Path
import re
import sys
from zipfile import ZIP_DEFLATED, ZipFile


EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "runs",
}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo", ".pkl", ".gz"}
EXCLUDED_EXACT_FILES = {".env"}

# Patterns require an actual non-placeholder value, so .env.example remains
# distributable.  The fragments are joined to avoid embedding an example secret
# token in this source file itself.
SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:OPENROUTER|OPENAI)_API_KEY\s*=\s*['\"]?[^\s'\"]{8,}"),
    re.compile(r"\b" + "sk" + "-or-v1-" + r"[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b" + "sk" + r"-[A-Za-z0-9_-]{20,}"),
)
TEXT_SCAN_LIMIT = 5 * 1024 * 1024


def _excluded(relative: Path, output: Path, root: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
        return True
    if relative.name in EXCLUDED_EXACT_FILES:
        return True
    if relative.name.startswith(".env.") and relative.name != ".env.example":
        return True
    if relative.suffix in EXCLUDED_FILE_SUFFIXES or relative.name.endswith(".pkl.gz"):
        return True
    try:
        return (root / relative).resolve() == output.resolve()
    except OSError:
        return False


def _scan_for_secrets(path: Path) -> list[str]:
    if path.stat().st_size > TEXT_SCAN_LIMIT:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)]


def build_archive(root: Path, output: Path) -> tuple[int, int]:
    root = root.resolve()
    output = output.resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError(f"Not a NeoOLAF source root (pyproject.toml missing): {root}")

    files: list[tuple[Path, Path]] = []
    secret_findings: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if _excluded(relative, output, root):
            continue
        matches = _scan_for_secrets(path)
        if matches:
            secret_findings.append(f"{relative}: probable secret pattern")
        files.append((path, relative))

    if secret_findings:
        joined = "\n".join(f"  - {finding}" for finding in secret_findings)
        raise RuntimeError(f"Refusing to build archive; probable secrets detected:\n{joined}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in files:
            archive.write(path, arcname=f"NeoOLAF/{relative.as_posix()}")

    return len(files), output.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--output", default=Path("dist/NeoOLAF-source.zip"), type=Path)
    args = parser.parse_args()

    output = args.output
    if not output.is_absolute():
        output = args.root / output
    try:
        count, size = build_archive(args.root, output)
    except Exception as exc:
        print(f"[NeoOLAF] Source archive failed: {exc}", file=sys.stderr)
        return 2

    print(f"[NeoOLAF] Source archive created: {output}")
    print(f"[NeoOLAF] Files: {count}; bytes: {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

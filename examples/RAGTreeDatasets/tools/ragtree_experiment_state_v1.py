from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import os
import tempfile

DATASET_KEYS = ("eventstoryline", "fincausal", "maven_ere", "causalbank")
DEFAULT_MANIFEST = {
    "schema_version": 1,
    "eventstoryline": {
        "status": "READY_5", "locked": False, "best_version": "v1.7",
        "one_doc_completed": True, "smoke5_already_run": False,
        "smoke5_record_keys": [], "smoke5_completed_record_keys": [],
    },
    "fincausal": {
        "status": "NOT_STARTED", "locked": False, "best_version": "unified-v1",
        "one_doc_completed": False, "smoke5_already_run": False,
        "smoke5_record_keys": [], "smoke5_completed_record_keys": [],
    },
    "maven_ere": {
        "status": "NOT_STARTED", "locked": False, "best_version": "unified-v1",
        "one_doc_completed": False, "smoke5_already_run": False,
        "smoke5_record_keys": [], "smoke5_completed_record_keys": [],
    },
    "causalbank": {
        "status": "NOT_STARTED", "locked": False, "best_version": "unified-v1",
        "one_doc_completed": False, "smoke5_already_run": False,
        "smoke5_record_keys": [], "smoke5_completed_record_keys": [],
    },
}

GOLD_KEYS = {
    "entities", "relations", "pred_relations", "labels", "gold", "gold_relations",
    "ontology_links", "relation_mentions", "gold_entities", "gold_events",
}


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: JSONL record is not an object")
            row = dict(row)
            row["__line_index__"] = line_no - 1
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_gold(record: dict[str, Any]) -> dict[str, Any]:
    """Return the pipeline-visible record. Gold and RAGTree ontology links are removed."""
    clean = {k: deepcopy(v) for k, v in record.items() if k not in GOLD_KEYS and not k.startswith("__")}
    return clean


def record_key(dataset_key: str, record: dict[str, Any], line_index: int | None = None) -> str:
    """Stable key that does not assume document_id is unique (important for FinCausal)."""
    idx = record.get("__line_index__") if line_index is None else line_index
    payload = {
        "dataset": dataset_key,
        "line_index": idx,
        "document_id": record.get("document_id"),
        "title": record.get("title"),
        "type": record.get("type"),
        "text": record.get("text"),
    }
    digest = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{dataset_key}:{idx}:{digest}"




def target_relation_count(dataset_key: str, record: dict[str, Any]) -> int:
    """Count scored target relations in a gold-bearing record.

    Used only by the experiment controller before a paid *development sanity* run
    and by post-L12 integrity checks. The returned count is never passed to NeoOLAF.
    """
    relations = record.get("relations") or {}
    if not isinstance(relations, dict):
        return 0
    target_labels = {
        "eventstoryline": {"PRECONDITION", "FALLING_ACTION"},
        "fincausal": {"CAUSE"},
        "maven_ere": {"CAUSE", "PRECONDITION"},
        "causalbank": {"BECAUSE", "THEREFORE"},
    }.get(dataset_key, set(relations))
    total = 0
    for label, pairs in relations.items():
        if label not in target_labels or not isinstance(pairs, list):
            continue
        total += sum(1 for pair in pairs if isinstance(pair, (list, tuple)) and len(pair) == 2)
    return total


def gold_contract_summary(dataset_key: str, record: dict[str, Any]) -> dict[str, Any]:
    entities = record.get("entities") or {}
    return {
        "dataset": dataset_key,
        "record_key": record_key(dataset_key, record),
        "document_id": record.get("document_id"),
        "title": record.get("title"),
        "line_index": record.get("__line_index__"),
        "gold_entity_count": len(entities) if isinstance(entities, dict) else 0,
        "gold_target_relation_count": target_relation_count(dataset_key, record),
    }


def first_evaluable_dev_record(dataset_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministically choose a development sanity record.

    For FinCausal, an empty-relation row cannot test the extractor, so choose the
    first source-order record with at least one scored CAUSE relation. This is a
    fixed validity criterion, not score-based/cherry-picked selection. Other
    datasets retain the historical first-row behavior.
    """
    if not rows:
        raise ValueError(f"No records available for {dataset_key}")
    if dataset_key != "fincausal":
        return rows[0]
    for row in rows:
        if target_relation_count(dataset_key, row) > 0:
            return row
    raise RuntimeError("FinCausal JSONL contains no evaluable CAUSE-bearing record.")


def load_manifest(live_path: str | Path, template_path: str | Path | None = None) -> dict[str, Any]:
    live_path = Path(live_path)
    if live_path.exists():
        data = read_json(live_path)
    elif template_path and Path(template_path).exists():
        data = deepcopy(read_json(template_path))
        atomic_write_json(live_path, data)
    else:
        data = deepcopy(DEFAULT_MANIFEST)
        atomic_write_json(live_path, data)
    for key in DATASET_KEYS:
        defaults = DEFAULT_MANIFEST[key]
        data.setdefault(key, {})
        for field, value in defaults.items():
            data[key].setdefault(field, deepcopy(value))
    return data


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest)


def assert_run_allowed(
    manifest: dict[str, Any], dataset_key: str, mode: str, *, force: bool = False
) -> None:
    if dataset_key not in DATASET_KEYS:
        raise KeyError(dataset_key)
    entry = manifest[dataset_key]
    if entry.get("locked") and not force:
        raise RuntimeError(f"{dataset_key} is LOCKED. Set FORCE_RUN[{dataset_key!r}]=True only intentionally.")
    if mode == "smoke5" and entry.get("smoke5_already_run") and not force:
        raise RuntimeError(
            f"{dataset_key}: the single paid smoke-5 is already marked complete. "
            "Refusing to run it again without FORCE_RUN."
        )
    if mode == "full" and not entry.get("locked") and not force:
        raise RuntimeError(f"{dataset_key}: full benchmark requires a frozen/LOCKED configuration.")


def select_records_for_mode(
    dataset_key: str,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    mode: str,
    *,
    preferred_document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"No records available for {dataset_key}")
    keyed = [(record_key(dataset_key, row), row) for row in rows]
    by_key = {k: row for k, row in keyed}

    if mode == "full":
        return rows

    if mode == "one_doc":
        entry = manifest[dataset_key]
        # A one-doc development sample is frozen independently from the smoke-5.
        # This avoids an old empty FinCausal first row silently becoming the dev
        # sample while preserving deterministic source-order selection.
        frozen_key = entry.get("one_doc_record_key")
        if frozen_key in by_key:
            frozen = by_key[frozen_key]
            if dataset_key != "fincausal" or target_relation_count(dataset_key, frozen) > 0:
                return [frozen]
            # Old/invalid frozen FinCausal key: repair without an API call.
            entry.pop("one_doc_record_key", None)
            entry.pop("one_doc_selection", None)

        if preferred_document_ids:
            preferred = set(preferred_document_ids)
            for key, row in keyed:
                if str(row.get("document_id")) in preferred:
                    if dataset_key == "fincausal" and target_relation_count(dataset_key, row) <= 0:
                        continue
                    entry["one_doc_record_key"] = key
                    entry["one_doc_selection"] = {
                        "criterion": "preferred_document_id",
                        **gold_contract_summary(dataset_key, row),
                    }
                    return [row]

        row = first_evaluable_dev_record(dataset_key, rows)
        key = record_key(dataset_key, row)
        entry["one_doc_record_key"] = key
        entry["one_doc_selection"] = {
            "criterion": (
                "first source-order record with >=1 scored CAUSE relation"
                if dataset_key == "fincausal" else "historical first source-order record"
            ),
            **gold_contract_summary(dataset_key, row),
        }
        return [row]

    if mode != "smoke5":
        raise ValueError(f"Unsupported RUN_MODE={mode!r}")

    entry = manifest[dataset_key]
    locked_keys = list(entry.get("smoke5_record_keys") or [])
    if locked_keys:
        missing = [k for k in locked_keys if k not in by_key]
        if missing:
            raise RuntimeError(
                f"{dataset_key}: saved smoke5 record keys are not present in the current JSONL: {missing}"
            )
        return [by_key[k] for k in locked_keys]

    selected: list[dict[str, Any]] = []
    if preferred_document_ids:
        wanted = list(preferred_document_ids)
        for doc_id in wanted:
            match = next((row for _, row in keyed if str(row.get("document_id")) == doc_id), None)
            if match is not None:
                selected.append(match)
    seen = {record_key(dataset_key, row) for row in selected}
    for key, row in keyed:
        if len(selected) >= 5:
            break
        if key not in seen:
            selected.append(row)
            seen.add(key)
    if len(selected) < 5:
        raise RuntimeError(f"{dataset_key}: smoke5 requires 5 records, found {len(selected)}")
    entry["smoke5_record_keys"] = [record_key(dataset_key, row) for row in selected[:5]]
    entry["status"] = "SMOKE5_SELECTED"
    return selected[:5]


def mark_record_complete(
    manifest: dict[str, Any], dataset_key: str, mode: str, key: str
) -> None:
    entry = manifest[dataset_key]
    if mode == "one_doc":
        entry["one_doc_completed"] = True
        entry["status"] = "READY_5"
    elif mode == "smoke5":
        done = list(entry.get("smoke5_completed_record_keys") or [])
        if key not in done:
            done.append(key)
        entry["smoke5_completed_record_keys"] = done
        selected = list(entry.get("smoke5_record_keys") or [])
        if selected and set(selected).issubset(set(done)):
            entry["smoke5_already_run"] = True
            entry["status"] = "SMOKE5"
        else:
            entry["status"] = "SMOKE5_IN_PROGRESS"


def pending_smoke_records(dataset_key: str, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    done = set(manifest[dataset_key].get("smoke5_completed_record_keys") or [])
    return [row for row in rows if record_key(dataset_key, row) not in done]


def find_existing(paths: Iterable[str | Path]) -> Path | None:
    for item in paths:
        if not item:
            continue
        path = Path(item).expanduser()
        if path.exists():
            return path.resolve()
    return None


def discover_ragtree_root(project_root: str | Path | None = None) -> Path:
    env = os.environ.get("RAGTREE_ROOT")
    candidates: list[str | Path] = []
    if env:
        candidates.append(env)
    candidates.extend([
        r"C:\Users\galencarmedeiro\RAGTree",
        r"C:\Users\galencarmedeiro\Documents\git\postdoc\RAGTree",
    ])
    if project_root:
        root = Path(project_root).resolve()
        candidates.extend([root.parent / "RAGTree", root.parent / "ragtree"])
    found = find_existing(candidates)
    if found is None:
        raise FileNotFoundError(
            "Could not locate RAGTree. Set RAGTREE_ROOT, e.g. "
            r"$env:RAGTREE_ROOT='C:\Users\galencarmedeiro\RAGTree'"
        )
    return found


def discover_preprocessed_dir(ragtree_root: str | Path) -> Path:
    root = Path(ragtree_root)
    env = os.environ.get("RAGTREE_PREPROCESSED_DIR")
    candidates = [env] if env else []
    candidates.extend([root / "data" / "preprocessed", root / "preprocessed"])
    found = find_existing(candidates)
    if found is None:
        raise FileNotFoundError(f"Could not locate RAGTree preprocessed dir under {root}")
    return found


def discover_ontology_dir(ragtree_root: str | Path) -> Path:
    root = Path(ragtree_root)
    env = os.environ.get("RAGTREE_ONTOLOGY_DIR")
    candidates = [env] if env else []
    candidates.extend([root / "data" / "ontology"])
    found = find_existing(candidates)
    if found is None:
        raise FileNotFoundError(f"Could not locate RAGTree ontology dir under {root}")
    return found


def locate_ontology_files(ontology_root: str | Path) -> dict[str, Path]:
    root = Path(ontology_root)
    mapping = {
        "eventstoryline": root / "OWLTime" / "time.ttl",
        "fincausal": root / "FIBO-CorePlus" / "fibo-core-plus.ttl",
        "maven_ere": root / "EventKG" / "EventKGSchema.ttl",
        "causalbank": root / "WordNet-Full" / "wordnet.ttl",
    }
    missing = {k: str(v) for k, v in mapping.items() if not v.exists()}
    if missing:
        raise FileNotFoundError("Missing ontology files: " + json.dumps(missing, indent=2))
    return {k: v.resolve() for k, v in mapping.items()}


def locate_dataset_files(preprocessed_dir: str | Path) -> dict[str, Path]:
    root = Path(preprocessed_dir)
    names = {
        "eventstoryline": "eventstoryline.jsonl",
        "fincausal": "fincausal.jsonl",
        "maven_ere": "maven_ere.jsonl",
        "causalbank": "causalbank.jsonl",
    }
    mapping = {k: root / name for k, name in names.items()}
    missing = {k: str(v) for k, v in mapping.items() if not v.exists()}
    if missing:
        raise FileNotFoundError("Missing normalized dataset files: " + json.dumps(missing, indent=2))
    return {k: v.resolve() for k, v in mapping.items()}

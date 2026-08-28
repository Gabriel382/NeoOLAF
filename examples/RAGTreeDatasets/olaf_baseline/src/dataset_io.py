from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_ragtree_preprocessed(start: Path) -> Path:
    candidates = [
        start,
        *start.parents,
        start.parent / "RAGTree",
        start.parent.parent / "RAGTree",
        Path(r"C:\Users\galencarmedeiro\RAGTree"),
    ]

    checked = []

    for p in candidates:
        p = p.resolve()

        # Current local layout.
        direct = p / "preprocessed"
        checked.append(direct)
        if direct.is_dir():
            return direct

        # Older/alternate layout.
        nested = p / "data" / "preprocessed"
        checked.append(nested)
        if nested.is_dir():
            return nested

    raise FileNotFoundError(
        "Could not find RAGTree preprocessed directory. Checked:\n"
        + "\n".join(str(x) for x in checked)
    )


def locate_dataset(preprocessed_dir: Path, dataset_key: str) -> Path:
    preferred = {
        "docred": ["docred_causal.jsonl", "docred.jsonl"],
        "fincausal": ["fincausal_causal.jsonl", "fincausal.jsonl"],
        "eventstoryline": ["eventstoryline_causal.jsonl", "eventstoryline.jsonl"],
    }
    for name in preferred[dataset_key]:
        p = preprocessed_dir / name
        if p.is_file():
            return p

    tokens = {
        "docred": ("docred",),
        "fincausal": ("fincausal",),
        "eventstoryline": ("eventstoryline", "event_story_line"),
    }[dataset_key]

    candidates = []
    for p in preprocessed_dir.glob("*.jsonl"):
        low = p.name.lower()
        if any(t in low for t in tokens):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No JSONL found for {dataset_key} under {preprocessed_dir}"
        )

    # Prefer the shortest / simplest matching filename.
    return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]


def positive_gold_relation_count(row: dict) -> int:
    total = 0
    for rel, pairs in (row.get("relations") or {}).items():
        if str(rel).lower() in {"null", "none", ""}:
            continue
        total += len(pairs or [])
    return total


def choose_positive_row(rows: Iterable[dict]) -> dict:
    for row in rows:
        if positive_gold_relation_count(row) > 0:
            return row
    raise RuntimeError("No positive-gold document found.")


def strip_gold(row: dict) -> dict:
    clean = dict(row)
    for key in ("entities", "relations", "pred_relations", "ontology_links"):
        clean.pop(key, None)
    return clean

#!/usr/bin/env python3
"""Evaluate raw-text DocRED entity + relation predictions.

Entity evaluation is by exact/alias-normalized cluster matching against DocRED
gold entity clusters. Relation evaluation first maps predicted entity labels to
gold entity IDs, then computes strict (head_id, relation_id, tail_id) P/R/F1.
Gold is used only here, not in extraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def norm(x: object) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\"'`]+|[\"'`]+$", "", s)
    return s


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def doc_id(record: Dict[str, Any]) -> str:
    return str(record.get("document_id") or record.get("id") or record.get("doc_id") or record.get("title"))


def rel_id(label: object) -> str:
    s = str(label or "").strip()
    if " : " in s:
        return s.split(" : ", 1)[0].strip()
    m = re.match(r"^(P\d+)", s)
    if m:
        return m.group(1)
    return s


def filter_gold(records: Iterable[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    if split == "all":
        return list(records)
    return [r for r in records if r.get("type") == split or r.get("split") == split]


def gold_entities(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    value = record.get("entities")
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(value, dict):
        for eid, ent in value.items():
            aliases = []
            if isinstance(ent, dict):
                mentions = ent.get("mentions") or []
                if isinstance(mentions, list):
                    for m in mentions:
                        if isinstance(m, dict):
                            label = m.get("trigger_word") or m.get("name") or m.get("text")
                            if label:
                                aliases.append(str(label).strip())
                typ = ent.get("type") or "entity"
            else:
                typ = "entity"
            label = aliases[0] if aliases else str(eid)
            out[str(eid)] = {"id": str(eid), "label": label, "aliases": sorted(set(aliases or [label])), "type": str(typ)}
    elif isinstance(value, list):
        for i, ent in enumerate(value):
            if isinstance(ent, dict):
                eid = str(ent.get("id") or ent.get("entity_id") or f"entity_{i}")
                label = str(ent.get("label") or ent.get("name") or ent.get("text") or eid)
                aliases = [label] + [str(a) for a in (ent.get("aliases") or []) if str(a).strip()] if isinstance(ent.get("aliases"), list) else [label]
                out[eid] = {"id": eid, "label": label, "aliases": sorted(set(aliases)), "type": str(ent.get("type") or "entity")}
    return out


def gold_alias_index(entities: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for eid, ent in entities.items():
        for alias in [ent.get("label"), *(ent.get("aliases") or [])]:
            key = norm(alias)
            if key and key not in idx:
                idx[key] = eid
    return idx


def gold_triples(record: Dict[str, Any]) -> Set[Tuple[str, str, str]]:
    triples: Set[Tuple[str, str, str]] = set()
    relations = record.get("relations") or {}
    if isinstance(relations, dict):
        for rlabel, pairs in relations.items():
            rid = rel_id(rlabel)
            if not isinstance(pairs, list):
                continue
            for pair in pairs:
                if isinstance(pair, list) and len(pair) >= 2:
                    triples.add((str(pair[0]), rid, str(pair[1])))
                elif isinstance(pair, dict):
                    h = pair.get("head") or pair.get("head_id") or pair.get("subject")
                    t = pair.get("tail") or pair.get("tail_id") or pair.get("object")
                    if h and t:
                        triples.add((str(h), rid, str(t)))
    return triples


def predicted_entities(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e for e in record.get("prediction", {}).get("entities", []) if isinstance(e, dict)]


def predicted_relations(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in record.get("prediction", {}).get("relations", []) if isinstance(r, dict)]


def pred_entity_aliases(ent: Dict[str, Any]) -> List[str]:
    aliases = []
    for x in [ent.get("label"), ent.get("name"), ent.get("text"), *(ent.get("aliases") or [])]:
        if x and str(x).strip():
            aliases.append(str(x).strip())
    return sorted(set(aliases))


def map_pred_entities_to_gold(pred_entities: List[Dict[str, Any]], gold_idx: Dict[str, str]) -> Tuple[Dict[str, str], Set[str], List[Dict[str, Any]]]:
    """Return pred local key -> gold id, matched gold IDs, and FP entities."""
    mapping: Dict[str, str] = {}
    matched_gold: Set[str] = set()
    false_positive_entities: List[Dict[str, Any]] = []
    for ent in pred_entities:
        local = str(ent.get("entity_id") or ent.get("id") or ent.get("label") or "")
        candidate_gold = None
        for alias in pred_entity_aliases(ent):
            key = norm(alias)
            if key in gold_idx:
                candidate_gold = gold_idx[key]
                break
        if candidate_gold:
            mapping[local] = candidate_gold
            # Also map label/aliases to gold for relation endpoint mapping.
            for alias in pred_entity_aliases(ent):
                mapping[norm(alias)] = candidate_gold
            matched_gold.add(candidate_gold)
        else:
            false_positive_entities.append(ent)
    return mapping, matched_gold, false_positive_entities


def map_relation_endpoint(value: object, pred_entity_map: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    return pred_entity_map.get(s) or pred_entity_map.get(norm(s))


def pred_triples_mapped(record: Dict[str, Any], pred_entity_map: Dict[str, str]) -> Tuple[Set[Tuple[str, str, str]], List[Dict[str, Any]]]:
    triples: Set[Tuple[str, str, str]] = set()
    unmapped: List[Dict[str, Any]] = []
    for rel in predicted_relations(record):
        h_raw = rel.get("head_entity_id") or rel.get("head_id") or rel.get("head") or rel.get("subject")
        t_raw = rel.get("tail_entity_id") or rel.get("tail_id") or rel.get("tail") or rel.get("object")
        h = map_relation_endpoint(h_raw, pred_entity_map)
        t = map_relation_endpoint(t_raw, pred_entity_map)
        r = rel.get("relation_id") or rel_id(rel.get("relation"))
        if h and t and r:
            triples.add((h, str(r), t))
        else:
            unmapped.append(rel)
    return triples, unmapped


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def evaluate(gold_path: str | Path, pred_path: str | Path, split: str = "dev") -> Dict[str, Any]:
    gold_records = filter_gold(load_jsonl(gold_path), split)
    pred_records = load_jsonl(pred_path)
    pred_by_id = {doc_id(r): r for r in pred_records}

    global_gold_entities = 0
    global_pred_entities = 0
    global_entity_tp = 0
    global_entity_fp = 0
    global_entity_fn = 0

    all_gold_rel: Set[Tuple[str, str, str, str]] = set()
    all_pred_rel: Set[Tuple[str, str, str, str]] = set()
    unmapped_pred_relations = 0
    per_doc: List[Dict[str, Any]] = []

    endpoint_gold_ids: Set[Tuple[str, str]] = set()
    endpoint_pred_gold_ids: Set[Tuple[str, str]] = set()

    for gold in gold_records:
        gid = doc_id(gold)
        pred = pred_by_id.get(gid, {"document_id": gid, "prediction": {"entities": [], "relations": []}})
        gents = gold_entities(gold)
        gidx = gold_alias_index(gents)
        pents = predicted_entities(pred)
        pmap, matched, fp_ents = map_pred_entities_to_gold(pents, gidx)

        ent_tp = len(matched)
        ent_fp = len(fp_ents)
        ent_fn = len(set(gents) - matched)
        global_gold_entities += len(gents)
        global_pred_entities += len(pents)
        global_entity_tp += ent_tp
        global_entity_fp += ent_fp
        global_entity_fn += ent_fn

        gtrip = gold_triples(gold)
        ptrip, unmapped = pred_triples_mapped(pred, pmap)
        unmapped_pred_relations += len(unmapped)

        for h, r, t in gtrip:
            all_gold_rel.add((gid, h, r, t))
            endpoint_gold_ids.add((gid, h))
            endpoint_gold_ids.add((gid, t))
        for h, r, t in ptrip:
            all_pred_rel.add((gid, h, r, t))
            endpoint_pred_gold_ids.add((gid, h))
            endpoint_pred_gold_ids.add((gid, t))

        tp = len(ptrip & gtrip)
        fp = len(ptrip - gtrip) + len(unmapped)
        fn = len(gtrip - ptrip)
        row = {
            "document_id": gid,
            "title": pred.get("title") or gold.get("title"),
            "parsed_ok": bool(pred.get("parsed_ok")),
            "gold_entities": len(gents),
            "pred_entities": len(pents),
            "entity_TP": ent_tp,
            "entity_FP": ent_fp,
            "entity_FN": ent_fn,
            **{f"entity_{k}": v for k, v in prf(ent_tp, ent_fp, ent_fn).items()},
            "gold_relations": len(gtrip),
            "pred_relations_mapped": len(ptrip),
            "pred_relations_unmapped": len(unmapped),
            "relation_TP": tp,
            "relation_FP": fp,
            "relation_FN": fn,
            **{f"relation_{k}": v for k, v in prf(tp, fp, fn).items()},
        }
        per_doc.append(row)

    rel_tp = len(all_pred_rel & all_gold_rel)
    rel_fp = len(all_pred_rel - all_gold_rel) + unmapped_pred_relations
    rel_fn = len(all_gold_rel - all_pred_rel)

    endpoint_tp = len(endpoint_pred_gold_ids & endpoint_gold_ids)
    endpoint_fp = len(endpoint_pred_gold_ids - endpoint_gold_ids)
    endpoint_fn = len(endpoint_gold_ids - endpoint_pred_gold_ids)

    return {
        "split": split,
        "docs_gold": len(gold_records),
        "docs_pred": len(pred_records),
        "entity_inventory": {
            "gold": global_gold_entities,
            "pred": global_pred_entities,
            "TP": global_entity_tp,
            "FP": global_entity_fp,
            "FN": global_entity_fn,
            **prf(global_entity_tp, global_entity_fp, global_entity_fn),
        },
        "entity_endpoint": {
            "gold": len(endpoint_gold_ids),
            "pred": len(endpoint_pred_gold_ids),
            "TP": endpoint_tp,
            "FP": endpoint_fp,
            "FN": endpoint_fn,
            **prf(endpoint_tp, endpoint_fp, endpoint_fn),
        },
        "relation_strict": {
            "gold": len(all_gold_rel),
            "pred": len(all_pred_rel) + unmapped_pred_relations,
            "TP": rel_tp,
            "FP": rel_fp,
            "FN": rel_fn,
            "unmapped_pred_relations": unmapped_pred_relations,
            **prf(rel_tp, rel_fp, rel_fn),
        },
        "per_document": per_doc,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-jsonl-path", required=True)
    ap.add_argument("--pred-jsonl-path", required=True)
    ap.add_argument("--type-filter", default="dev")
    ap.add_argument("--output-json-path", required=True)
    ap.add_argument("--output-csv-path", default=None)
    args = ap.parse_args(argv)
    result = evaluate(args.gold_jsonl_path, args.pred_jsonl_path, args.type_filter)
    out = Path(args.output_json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_csv_path:
        csv_path = Path(args.output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = result["per_document"]
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("RAW TEXT DOCRED ENTITY + RELATION EVALUATION")
    print(f"docs_gold={result['docs_gold']} docs_pred={result['docs_pred']}")
    for section in ["entity_inventory", "entity_endpoint", "relation_strict"]:
        m = result[section]
        print(f"{section}: gold={m['gold']} pred={m['pred']} TP={m['TP']} FP={m['FP']} FN={m['FN']} precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ontology/KG alignment metrics for OLAF/NeoOLAF-style outputs.

This module is intentionally dependency-light. It uses rdflib for RDF parsing and
Python's difflib for optional fuzzy matching. It can evaluate whether a generated
KG uses entities and predicates that are represented in a generated/seed ontology.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import json
import math
import re
from difflib import SequenceMatcher

import pandas as pd
from rdflib import Graph, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    try:
        n = float(n or 0.0)
        d = float(d or 0.0)
    except Exception:
        return default
    if d == 0:
        return default
    return n / d


def harmonic(values: Iterable[float], default: float = 0.0) -> float:
    vals = []
    for v in values:
        try:
            x = float(v or 0.0)
        except Exception:
            return default
        if not math.isfinite(x) or x <= 0:
            return default
        vals.append(x)
    if not vals:
        return default
    return len(vals) / sum(1.0 / v for v in vals)


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^\w\s<>.,:+#]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def local_name(uri: Any) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rstrip("/").rsplit("/", 1)[-1]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def is_missing_node(value: Any) -> bool:
    text = clean_text(value)
    return text == "" or text in {"?", "??", "???", "-", "_"}


@dataclass
class MatchResult:
    input_label: str
    matched: bool
    match_kind: str | None = None
    score: float = 0.0
    uri: str | None = None
    canonical_label: str | None = None
    term_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OntologyIndex:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.terms: dict[str, dict[str, Any]] = {}
        self.by_norm: dict[str, list[str]] = {}
        self.property_uris: set[str] = set()
        self.class_uris: set[str] = set()
        self.individual_uris: set[str] = set()
        self.subclass_edges: set[tuple[str, str]] = set()
        self._build()

    @classmethod
    def from_path(cls, path: str | Path) -> "OntologyIndex":
        path = Path(path)
        g = Graph()
        fmt = "turtle" if path.suffix.lower() in {".ttl", ".turtle"} else None
        g.parse(str(path), format=fmt)
        return cls(g)

    def _add_label_index(self, uri: str, label: str) -> None:
        norm = normalize_label(label)
        if not norm:
            return
        self.by_norm.setdefault(norm, [])
        if uri not in self.by_norm[norm]:
            self.by_norm[norm].append(uri)

    def _labels_for(self, s: URIRef) -> list[str]:
        labels = [local_name(s)]
        for p in (RDFS.label, SKOS.altLabel, SKOS.prefLabel):
            for o in self.graph.objects(s, p):
                labels.append(str(o))
        out = []
        seen = set()
        for label in labels:
            text = clean_text(label)
            if text and text not in seen:
                out.append(text)
                seen.add(text)
        return out

    def _term_type(self, s: URIRef) -> str:
        types = set(self.graph.objects(s, RDF.type))
        if OWL.Class in types or RDFS.Class in types:
            return "class"
        if OWL.ObjectProperty in types:
            return "object_property"
        if OWL.DatatypeProperty in types:
            return "datatype_property"
        if OWL.AnnotationProperty in types or RDF.Property in types:
            return "property"
        return "individual_or_resource"

    def _build(self) -> None:
        property_types = {OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty, RDF.Property}
        class_types = {OWL.Class, RDFS.Class}

        candidates = set()
        for s in self.graph.subjects():
            if isinstance(s, URIRef):
                candidates.add(s)
        for s, _, o in self.graph.triples((None, RDF.type, None)):
            if isinstance(s, URIRef):
                candidates.add(s)

        for s in candidates:
            uri = str(s)
            term_type = self._term_type(s)
            labels = self._labels_for(s)
            domains = [str(o) for o in self.graph.objects(s, RDFS.domain)]
            ranges = [str(o) for o in self.graph.objects(s, RDFS.range)]
            self.terms[uri] = {
                "uri": uri,
                "local_name": local_name(s),
                "labels": labels,
                "canonical_label": labels[0] if labels else local_name(s),
                "term_type": term_type,
                "domains": domains,
                "ranges": ranges,
            }
            for label in labels:
                self._add_label_index(uri, label)

            rdf_types = set(self.graph.objects(s, RDF.type))
            if rdf_types & class_types:
                self.class_uris.add(uri)
            if rdf_types & property_types:
                self.property_uris.add(uri)
            if rdf_types and not (rdf_types & class_types) and not (rdf_types & property_types):
                self.individual_uris.add(uri)

        for s, _, o in self.graph.triples((None, RDFS.subClassOf, None)):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                self.subclass_edges.add((str(s), str(o)))

    def match(self, label: Any, allowed_types: set[str] | None = None, fuzzy_threshold: float = 0.92) -> MatchResult:
        raw = clean_text(label)
        norm = normalize_label(raw)
        if not norm:
            return MatchResult(input_label=raw, matched=False, match_kind="empty")

        def allowed(uri: str) -> bool:
            if allowed_types is None:
                return True
            return self.terms.get(uri, {}).get("term_type") in allowed_types

        exact_uris = [uri for uri in self.by_norm.get(norm, []) if allowed(uri)]
        if exact_uris:
            uri = exact_uris[0]
            t = self.terms[uri]
            return MatchResult(raw, True, "exact", 1.0, uri, t["canonical_label"], t["term_type"])

        # fuzzy match against all normalized ontology labels
        best_uri = None
        best_norm = None
        best_score = 0.0
        for candidate_norm, uris in self.by_norm.items():
            candidate_uris = [u for u in uris if allowed(u)]
            if not candidate_uris:
                continue
            score = SequenceMatcher(None, norm, candidate_norm).ratio()
            if score > best_score:
                best_score = score
                best_uri = candidate_uris[0]
                best_norm = candidate_norm
        if best_uri is not None and best_score >= fuzzy_threshold:
            t = self.terms[best_uri]
            return MatchResult(raw, True, "fuzzy", float(best_score), best_uri, t["canonical_label"], t["term_type"])
        return MatchResult(raw, False, "none", float(best_score), best_uri, best_norm, None)

    def is_subclass_or_same(self, child_uri: str | None, parent_uri: str | None) -> bool:
        if not child_uri or not parent_uri:
            return False
        if child_uri == parent_uri:
            return True
        # small graph, simple DFS
        seen = set()
        stack = [child_uri]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for c, p in self.subclass_edges:
                if c == cur:
                    if p == parent_uri:
                        return True
                    stack.append(p)
        return False

    def summary(self) -> dict[str, Any]:
        classes = [self.terms[u] for u in self.class_uris if u in self.terms]
        props = [self.terms[u] for u in self.property_uris if u in self.terms]
        return {
            "ontology_term_count": len(self.terms),
            "ontology_class_count": len(classes),
            "ontology_property_count": len(props),
            "ontology_object_property_count": sum(1 for p in props if p.get("term_type") == "object_property"),
            "ontology_datatype_property_count": sum(1 for p in props if p.get("term_type") == "datatype_property"),
            "ontology_subclass_axiom_count": len(self.subclass_edges),
            "ontology_properties_with_domain_count": sum(1 for p in props if p.get("domains")),
            "ontology_properties_with_range_count": sum(1 for p in props if p.get("ranges")),
            "ontology_properties_with_domain_range_count": sum(1 for p in props if p.get("domains") and p.get("ranges")),
            "ontology_property_domain_range_rate": safe_div(sum(1 for p in props if p.get("domains") and p.get("ranges")), len(props)),
            "ontology_classes_with_altlabel_count": sum(1 for c in classes if len(c.get("labels", [])) > 1),
            "ontology_classes_with_altlabel_rate": safe_div(sum(1 for c in classes if len(c.get("labels", [])) > 1), len(classes)),
        }


def load_olaf_excel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df["_sheet"] = sheet
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find any of {candidates}; available columns: {list(df.columns)}")


def olaf_excel_to_triples(df: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    source_col = pick_column(df, ["Source Node", "source", "subject", "head"])
    target_col = pick_column(df, ["Target Node", "target", "object", "tail"])
    relation_col = pick_column(df, ["Relation", "predicate", "relation"])
    triples = []
    valid_rows = []
    skipped_rows = []
    for i, row in df.iterrows():
        s = clean_text(row.get(source_col))
        p = clean_text(row.get(relation_col))
        o = clean_text(row.get(target_col))
        reason = None
        if is_missing_node(s) or is_missing_node(o):
            reason = "missing_source_or_target"
        elif not p:
            reason = "missing_relation"
        if reason:
            skipped = row.to_dict()
            skipped["skip_reason"] = reason
            skipped_rows.append(skipped)
            continue
        t = {"subject": s, "predicate": p, "object": o, "source_row": int(i)}
        triples.append(t)
        valid = row.to_dict()
        valid.update(t)
        valid_rows.append(valid)
    return triples, pd.DataFrame(valid_rows), pd.DataFrame(skipped_rows)


def write_neoolaf_kg(triples: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"triples": []}
    for i, t in enumerate(triples):
        data["triples"].append({
            "id": f"olaf_{i:05d}",
            "subject": t.get("subject"),
            "predicate": t.get("predicate"),
            "object": t.get("object"),
            "confidence": t.get("confidence", 1.0),
            "provenance": {"source": "OLAF Excel", "source_row": t.get("source_row")},
        })
    for name in ["kg_inferred.json", "kg.json", "kg_local.json"]:
        (out_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir / "kg_inferred.json"


def evaluate_kg_ontology_alignment(
    triples: list[dict[str, Any]],
    ontology_path: str | Path,
    *,
    node_fuzzy_threshold: float = 0.92,
    predicate_fuzzy_threshold: float = 0.90,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, OntologyIndex]:
    onto = OntologyIndex.from_path(ontology_path)
    node_types = {"class", "individual_or_resource"}
    prop_types = {"object_property", "datatype_property", "property"}

    details = []
    unique_nodes = set()
    unique_preds = set()
    for i, t in enumerate(triples):
        s = clean_text(t.get("subject"))
        p = clean_text(t.get("predicate"))
        o = clean_text(t.get("object"))
        unique_nodes.update([s, o])
        unique_preds.add(p)
        sm = onto.match(s, allowed_types=node_types, fuzzy_threshold=node_fuzzy_threshold)
        pm = onto.match(p, allowed_types=prop_types, fuzzy_threshold=predicate_fuzzy_threshold)
        om = onto.match(o, allowed_types=node_types, fuzzy_threshold=node_fuzzy_threshold)

        prop = onto.terms.get(pm.uri or "", {})
        domains = prop.get("domains", []) or []
        ranges = prop.get("ranges", []) or []
        has_domain = bool(domains)
        has_range = bool(ranges)
        has_domain_range = has_domain and has_range
        domain_consistent = None
        range_consistent = None
        if pm.matched and sm.matched and has_domain:
            domain_consistent = any(onto.is_subclass_or_same(sm.uri, d) for d in domains)
        if pm.matched and om.matched and has_range:
            range_consistent = any(onto.is_subclass_or_same(om.uri, r) for r in ranges)

        details.append({
            "triple_index": i,
            "subject": s,
            "predicate": p,
            "object": o,
            "subject_aligned": sm.matched,
            "subject_match_kind": sm.match_kind,
            "subject_match_score": sm.score,
            "subject_uri": sm.uri,
            "subject_canonical_label": sm.canonical_label,
            "predicate_aligned": pm.matched,
            "predicate_match_kind": pm.match_kind,
            "predicate_match_score": pm.score,
            "predicate_uri": pm.uri,
            "predicate_canonical_label": pm.canonical_label,
            "object_aligned": om.matched,
            "object_match_kind": om.match_kind,
            "object_match_score": om.score,
            "object_uri": om.uri,
            "object_canonical_label": om.canonical_label,
            "fully_aligned": sm.matched and pm.matched and om.matched,
            "predicate_has_domain": has_domain,
            "predicate_has_range": has_range,
            "predicate_has_domain_range": has_domain_range,
            "domain_consistent_when_applicable": domain_consistent,
            "range_consistent_when_applicable": range_consistent,
            "domain_uris": " | ".join(domains),
            "range_uris": " | ".join(ranges),
        })

    detail_df = pd.DataFrame(details)
    # unique alignment tables
    unique_node_rows = []
    for n in sorted(x for x in unique_nodes if x):
        m = onto.match(n, allowed_types=node_types, fuzzy_threshold=node_fuzzy_threshold)
        unique_node_rows.append({"label": n, "aligned": m.matched, "match_kind": m.match_kind, "score": m.score, "uri": m.uri, "canonical_label": m.canonical_label})
    unique_pred_rows = []
    for p in sorted(x for x in unique_preds if x):
        m = onto.match(p, allowed_types=prop_types, fuzzy_threshold=predicate_fuzzy_threshold)
        prop = onto.terms.get(m.uri or "", {})
        unique_pred_rows.append({
            "predicate": p, "aligned": m.matched, "match_kind": m.match_kind, "score": m.score, "uri": m.uri,
            "canonical_label": m.canonical_label, "has_domain": bool(prop.get("domains")), "has_range": bool(prop.get("ranges")),
            "has_domain_range": bool(prop.get("domains")) and bool(prop.get("ranges")),
        })
    unique_node_df = pd.DataFrame(unique_node_rows)
    unique_pred_df = pd.DataFrame(unique_pred_rows)

    n_triples = len(detail_df)
    summary = onto.summary()
    summary.update({
        "kg_triple_count": n_triples,
        "kg_unique_node_count": len(unique_nodes),
        "kg_unique_predicate_count": len(unique_preds),
        "kg_subject_alignment_rate": safe_div(detail_df["subject_aligned"].sum() if n_triples else 0, n_triples),
        "kg_object_alignment_rate": safe_div(detail_df["object_aligned"].sum() if n_triples else 0, n_triples),
        "kg_node_mention_alignment_rate": safe_div((detail_df["subject_aligned"].sum() + detail_df["object_aligned"].sum()) if n_triples else 0, 2 * n_triples),
        "kg_predicate_alignment_rate": safe_div(detail_df["predicate_aligned"].sum() if n_triples else 0, n_triples),
        "kg_full_triple_alignment_rate": safe_div(detail_df["fully_aligned"].sum() if n_triples else 0, n_triples),
        "kg_unique_node_alignment_rate": safe_div(unique_node_df["aligned"].sum() if len(unique_node_df) else 0, len(unique_node_df)),
        "kg_unique_predicate_alignment_rate": safe_div(unique_pred_df["aligned"].sum() if len(unique_pred_df) else 0, len(unique_pred_df)),
        "kg_predicate_domain_axiom_rate": safe_div(detail_df["predicate_has_domain"].sum() if n_triples else 0, n_triples),
        "kg_predicate_range_axiom_rate": safe_div(detail_df["predicate_has_range"].sum() if n_triples else 0, n_triples),
        "kg_predicate_domain_range_axiom_rate": safe_div(detail_df["predicate_has_domain_range"].sum() if n_triples else 0, n_triples),
        "kg_unique_predicate_domain_range_axiom_rate": safe_div(unique_pred_df["has_domain_range"].sum() if len(unique_pred_df) else 0, len(unique_pred_df)),
    })
    applicable_domain = detail_df[detail_df["domain_consistent_when_applicable"].notna()] if n_triples else pd.DataFrame()
    applicable_range = detail_df[detail_df["range_consistent_when_applicable"].notna()] if n_triples else pd.DataFrame()
    summary["kg_domain_consistency_rate_when_applicable"] = safe_div(applicable_domain["domain_consistent_when_applicable"].sum() if len(applicable_domain) else 0, len(applicable_domain))
    summary["kg_range_consistency_rate_when_applicable"] = safe_div(applicable_range["range_consistent_when_applicable"].sum() if len(applicable_range) else 0, len(applicable_range))
    summary["kg_alignment_hmean"] = harmonic([
        summary["kg_unique_node_alignment_rate"],
        summary["kg_unique_predicate_alignment_rate"],
        summary["kg_full_triple_alignment_rate"],
    ])
    summary["kg_schema_alignment_hmean"] = harmonic([
        summary["kg_unique_predicate_alignment_rate"],
        summary["kg_unique_predicate_domain_range_axiom_rate"],
    ])
    summary["kg_ontology_alignment_score"] = harmonic([
        summary["kg_alignment_hmean"],
        summary["kg_schema_alignment_hmean"],
    ])

    inventory_df = pd.DataFrame(onto.terms.values())
    return summary, detail_df, inventory_df, onto


def evaluate_ontology_seed_alignment(
    generated_ontology_path: str | Path,
    seed_ontology_path: str | Path,
    *,
    class_threshold: float = 0.92,
    property_threshold: float = 0.90,
) -> tuple[dict[str, Any], pd.DataFrame]:
    gen = OntologyIndex.from_path(generated_ontology_path)
    seed = OntologyIndex.from_path(seed_ontology_path)
    rows = []
    for uri, t in gen.terms.items():
        tt = t.get("term_type")
        if tt == "class":
            allowed = {"class"}
            threshold = class_threshold
        elif tt in {"object_property", "datatype_property", "property"}:
            allowed = {"object_property", "datatype_property", "property"}
            threshold = property_threshold
        else:
            continue
        # Try every label and keep the best match.
        best = MatchResult(t.get("canonical_label", ""), False, "none", 0.0)
        for label in t.get("labels", []) or [t.get("local_name", "")]:
            m = seed.match(label, allowed_types=allowed, fuzzy_threshold=threshold)
            if m.matched and (not best.matched or m.score > best.score):
                best = m
            elif not best.matched and m.score > best.score:
                best = m
        rows.append({
            "generated_uri": uri,
            "generated_label": t.get("canonical_label"),
            "generated_term_type": tt,
            "aligned_to_seed": best.matched,
            "seed_match_kind": best.match_kind,
            "seed_match_score": best.score,
            "seed_uri": best.uri,
            "seed_canonical_label": best.canonical_label,
        })
    df = pd.DataFrame(rows)
    summary = {
        "generated_alignable_term_count": len(df),
        "generated_seed_aligned_term_count": int(df["aligned_to_seed"].sum()) if len(df) else 0,
        "generated_seed_alignment_rate": safe_div(int(df["aligned_to_seed"].sum()) if len(df) else 0, len(df)),
    }
    for typ in ["class", "object_property", "datatype_property", "property"]:
        sub = df[df["generated_term_type"] == typ] if len(df) else pd.DataFrame()
        summary[f"seed_alignment_{typ}_count"] = len(sub)
        summary[f"seed_alignment_{typ}_aligned_count"] = int(sub["aligned_to_seed"].sum()) if len(sub) else 0
        summary[f"seed_alignment_{typ}_rate"] = safe_div(int(sub["aligned_to_seed"].sum()) if len(sub) else 0, len(sub))
    return summary, df

"""Semantic validation/alignment metrics for KG/ontology outputs (v2 strict CV).

Metrics intended for XQuality-style no-gold validation slides:

STR: support traceability rate. Fraction of valid triples with explicit textual support/source-span metadata.
CR: conflict/unresolved ratio. Fraction of raw rows that are invalid/unresolved.
PC: provenance coverage. Fraction of valid triples with provenance metadata.
OC: ontology compliance. Fraction of valid triples whose subject, predicate and object align to ontology vocabulary.
CV: constraint violation rate. In v2 this is configurable. By default, CV is the strict-schema
    violation rate: a valid triple is considered violating if it is not ontology-compliant, if its
    predicate has no explicit domain/range constraints, or if it violates explicit domain/range constraints.
DVS: domain validation score. Binary pass/fail based on thresholds over STR/CR/PC/OC/CV.

This v2 keeps the older observed-alignment CV as CV_observed_alignment, because a generated ontology can
lexically contain every KG term and therefore produce CV=0 without providing real schema validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import json
import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

try:
    import rdflib
    from rdflib import RDF, RDFS, OWL
except Exception:  # pragma: no cover
    rdflib = None
    RDF = RDFS = OWL = None


EMPTY_MARKERS = {"", "?", "nan", "none", "null", "na", "n/a", "-", "_"}


@dataclass
class SemanticValidationConfig:
    node_fuzzy_threshold: float = 0.92
    predicate_fuzzy_threshold: float = 0.90
    allow_lexical_source_support_for_str: bool = False
    min_label_chars_for_source_match: int = 3
    # "strict_schema" is recommended for slide CV.
    # "observed_alignment" reproduces the old behavior: CV = non-alignment or explicit D/R violation only.
    cv_mode: str = "strict_schema"
    require_domain_and_range_for_strict_cv: bool = True
    count_invalid_rows_in_cv: bool = False
    dvs_min_str: float = 0.95
    dvs_max_cr: float = 0.05
    dvs_min_pc: float = 0.95
    dvs_min_oc: float = 0.95
    dvs_max_cv: float = 0.05


def normalize_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = str(value).strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_label(value))


def is_empty_value(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    return str(value).strip().lower() in EMPTY_MARKERS


def uri_local_name(uri: Any) -> str:
    s = str(uri).rstrip("/#")
    if "#" in s:
        s = s.rsplit("#", 1)[-1]
    elif "/" in s:
        s = s.rsplit("/", 1)[-1]
    return re.sub(r"%[0-9A-Fa-f]{2}", " ", s)


def _literal_str(value: Any) -> str:
    return str(value).strip()


def _labels_from_ttl_regex(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    labels = set()
    property_labels = set()
    class_labels = set()
    for m in re.finditer(r"(?:rdfs:label|skos:altLabel)\s+\"([^\"]+)\"", text):
        labels.add(m.group(1))
    for m in re.finditer(r"<([^>]+)>\s+a\s+([^;\.]+)", text):
        local = uri_local_name(m.group(1))
        labels.add(local)
        rdf_type = m.group(2)
        if "ObjectProperty" in rdf_type or "DatatypeProperty" in rdf_type or "Property" in rdf_type:
            property_labels.add(local)
        if "Class" in rdf_type:
            class_labels.add(local)
    # Regex fallback intentionally cannot recover domain/range robustly.
    return {
        "all_labels": labels,
        "node_labels": labels | class_labels,
        "predicate_labels": property_labels,
        "property_domain_range": {},
        "properties_with_domain": set(),
        "properties_with_range": set(),
        "property_count": len(property_labels),
        "class_count": len(class_labels),
    }


def load_ontology_index(ontology_path: str | Path) -> Dict[str, Any]:
    path = Path(ontology_path)
    if not path.exists():
        raise FileNotFoundError(f"Ontology not found: {path}")
    if rdflib is None:
        return _labels_from_ttl_regex(path)

    graph = rdflib.Graph()
    try:
        graph.parse(str(path))
    except Exception:
        parsed = False
        for fmt in ["xml", "turtle", "nt", "n3"]:
            try:
                graph = rdflib.Graph()
                graph.parse(str(path), format=fmt)
                parsed = True
                break
            except Exception:
                pass
        if not parsed:
            return _labels_from_ttl_regex(path)

    all_labels: set[str] = set()
    node_labels: set[str] = set()
    predicate_labels: set[str] = set()
    class_uris: set[Any] = set()
    property_uris: set[Any] = set()
    properties_with_domain: set[str] = set()
    properties_with_range: set[str] = set()
    property_domain_range: Dict[str, Dict[str, set[str]]] = {}

    for s in graph.subjects(RDF.type, OWL.Class):
        class_uris.add(s)
    for ptype in [OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property]:
        for s in graph.subjects(RDF.type, ptype):
            property_uris.add(s)

    def labels_for_resource(res: Any) -> set[str]:
        out = {uri_local_name(res)}
        for pred in [RDFS.label, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#altLabel")]:
            for lit in graph.objects(res, pred):
                out.add(_literal_str(lit))
        return {x for x in out if normalize_compact(x)}

    for res in set(graph.subjects()) | set(graph.predicates()) | set(graph.objects()):
        if isinstance(res, rdflib.URIRef):
            all_labels.add(uri_local_name(res))
            for lab in graph.objects(res, RDFS.label):
                all_labels.add(_literal_str(lab))
            for lab in graph.objects(res, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#altLabel")):
                all_labels.add(_literal_str(lab))

    for c in class_uris:
        labs = labels_for_resource(c)
        node_labels.update(labs)
        all_labels.update(labs)

    for p in property_uris:
        labs = labels_for_resource(p)
        predicate_labels.update(labs)
        all_labels.update(labs)
        domains = {lab for d in graph.objects(p, RDFS.domain) for lab in labels_for_resource(d)}
        ranges = {lab for r in graph.objects(p, RDFS.range) for lab in labels_for_resource(r)}
        for lab in labs:
            key = normalize_compact(lab)
            if not key:
                continue
            if domains:
                properties_with_domain.add(key)
            if ranges:
                properties_with_range.add(key)
            property_domain_range[key] = {"domain": domains, "range": ranges, "labels": labs}

    if not predicate_labels:
        for p in property_uris:
            predicate_labels.add(uri_local_name(p))

    return {
        "all_labels": all_labels,
        "node_labels": node_labels or all_labels,
        "predicate_labels": predicate_labels,
        "property_domain_range": property_domain_range,
        "properties_with_domain": properties_with_domain,
        "properties_with_range": properties_with_range,
        "property_count": len(property_uris),
        "class_count": len(class_uris),
    }


def _build_norm_index(labels: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for lab in labels:
        n = normalize_compact(lab)
        if n and n not in out:
            out[n] = str(lab)
    return out


def align_label(value: Any, labels: Iterable[str], threshold: float = 0.90) -> Dict[str, Any]:
    compact = normalize_compact(value)
    if not compact:
        return {"aligned": False, "kind": "empty", "score": 0.0, "match": None}
    index = _build_norm_index(labels)
    if compact in index:
        return {"aligned": True, "kind": "exact", "score": 1.0, "match": index[compact]}
    best_score = 0.0
    best_label = None
    for n, lab in index.items():
        score = SequenceMatcher(None, compact, n).ratio()
        if score > best_score:
            best_score = score
            best_label = lab
    return {
        "aligned": best_score >= threshold,
        "kind": "fuzzy" if best_score >= threshold else "none",
        "score": float(best_score),
        "match": best_label,
    }


def find_triple_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = {str(c).lower().strip(): c for c in df.columns}
    source_candidates = ["source node", "source", "subject", "head", "from", "source_label", "subject_label"]
    target_candidates = ["target node", "target", "object", "tail", "to", "target_label", "object_label"]
    relation_candidates = ["relation", "predicate", "property", "edge", "relation_label", "predicate_label"]

    def pick(candidates: Sequence[str], fallback_index: int) -> str:
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return df.columns[fallback_index]

    return pick(source_candidates, 0), pick(relation_candidates, 2 if len(df.columns) >= 3 else 1), pick(target_candidates, 1 if len(df.columns) >= 2 else 0)


def load_triple_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Triple file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                rows = obj.get("triples") or obj.get("relations") or obj.get("edges") or []
            else:
                rows = obj
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported triple file type: {path.suffix}")


def provenance_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    patterns = [
        "provenance", "evidence", "support", "chunk", "page", "section", "span", "offset",
        "record_id", "record_no", "source_text", "source_file", "source_path", "document_id", "doc_id",
    ]
    excluded = {"source node", "source", "source_label", "subject", "subject_label"}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in excluded:
            continue
        if any(p in lc for p in patterns):
            cols.append(c)
    return cols


def support_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    patterns = ["evidence", "support", "source_text", "chunk", "page", "section", "span", "offset", "record_id", "record_no"]
    excluded = {"source node", "source", "source_label", "subject", "subject_label"}
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc in excluded:
            continue
        if any(p in lc for p in patterns):
            cols.append(c)
    return cols


def row_has_any_value(row: pd.Series, cols: Sequence[str]) -> bool:
    return any(not is_empty_value(row.get(c)) for c in cols)


def source_text_support(row: pd.Series, source_col: str, relation_col: str, target_col: str, source_text_norm: str, min_chars: int = 3) -> bool:
    if not source_text_norm:
        return False
    labels = [row.get(source_col), row.get(target_col)]
    hits = 0
    useful = 0
    for lab in labels:
        norm = normalize_label(lab)
        if len(norm.replace(" ", "")) >= min_chars:
            useful += 1
            if norm in source_text_norm:
                hits += 1
    return useful > 0 and hits == useful


def compute_semantic_validation_metrics(
    triples_df: pd.DataFrame,
    ontology_path: str | Path,
    source_text_path: str | Path | None = None,
    config: SemanticValidationConfig | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    config = config or SemanticValidationConfig()
    if config.cv_mode not in {"strict_schema", "observed_alignment"}:
        raise ValueError("config.cv_mode must be 'strict_schema' or 'observed_alignment'")

    source_col, relation_col, target_col = find_triple_columns(triples_df)
    ontology = load_ontology_index(ontology_path)

    raw_count = len(triples_df)
    valid_mask = []
    for _, row in triples_df.iterrows():
        valid_mask.append(
            not is_empty_value(row.get(source_col))
            and not is_empty_value(row.get(relation_col))
            and not is_empty_value(row.get(target_col))
        )
    triples_df = triples_df.copy()
    triples_df["_valid_triple"] = valid_mask
    valid_df = triples_df[triples_df["_valid_triple"]].copy()
    valid_count = len(valid_df)
    invalid_count = raw_count - valid_count

    prov_cols = provenance_columns(triples_df)
    supp_cols = support_columns(triples_df)

    source_text_norm = ""
    if source_text_path is not None and Path(source_text_path).exists():
        source_text_norm = normalize_label(Path(source_text_path).read_text(encoding="utf-8", errors="ignore"))

    detail_rows = []
    support_count = 0
    provenance_count = 0
    full_ontology_aligned_count = 0
    observed_cv_count = 0
    strict_schema_cv_count = 0
    explicit_domain_range_violation_count = 0
    domain_applicable_count = 0
    range_applicable_count = 0
    complete_domain_range_applicable_count = 0
    missing_domain_range_constraint_count = 0

    node_labels = ontology["node_labels"] or ontology["all_labels"]
    predicate_labels = ontology["predicate_labels"]  # no fallback to all_labels: predicates must be properties

    for idx, row in triples_df.iterrows():
        valid = bool(row["_valid_triple"])
        rec = {
            "row_index": idx,
            "subject": row.get(source_col),
            "predicate": row.get(relation_col),
            "object": row.get(target_col),
            "valid_triple": valid,
        }
        if not valid:
            rec.update({
                "has_support_trace": False,
                "has_provenance": False,
                "subject_aligned": False,
                "predicate_aligned": False,
                "object_aligned": False,
                "ontology_compliant": False,
                "formal_domain_range_violation": False,
                "predicate_domain_axiom_applicable": False,
                "predicate_range_axiom_applicable": False,
                "complete_domain_range_axioms_applicable": False,
                "missing_domain_range_constraint": False,
                "observed_alignment_constraint_violation": True,
                "strict_schema_constraint_violation": True,
                "constraint_violation": True,
            })
            detail_rows.append(rec)
            continue

        has_prov = row_has_any_value(row, prov_cols)
        has_supp = row_has_any_value(row, supp_cols)
        if config.allow_lexical_source_support_for_str and not has_supp:
            has_supp = source_text_support(row, source_col, relation_col, target_col, source_text_norm, config.min_label_chars_for_source_match)

        s_align = align_label(row.get(source_col), node_labels, config.node_fuzzy_threshold)
        p_align = align_label(row.get(relation_col), predicate_labels, config.predicate_fuzzy_threshold)
        o_align = align_label(row.get(target_col), node_labels, config.node_fuzzy_threshold)
        full_aligned = bool(s_align["aligned"] and p_align["aligned"] and o_align["aligned"])

        formal_violation = False
        predicate_norm_candidates = [
            normalize_compact(row.get(relation_col)),
            normalize_compact(p_align.get("match")),
        ]
        pdr = None
        for cand in predicate_norm_candidates:
            if cand and cand in ontology.get("property_domain_range", {}):
                pdr = ontology["property_domain_range"][cand]
                break

        domain_covered = False
        range_covered = False
        if pdr:
            domains = pdr.get("domain") or set()
            ranges = pdr.get("range") or set()
            if domains:
                domain_covered = True
                if not align_label(row.get(source_col), domains, config.node_fuzzy_threshold)["aligned"]:
                    formal_violation = True
            if ranges:
                range_covered = True
                if not align_label(row.get(target_col), ranges, config.node_fuzzy_threshold)["aligned"]:
                    formal_violation = True

        complete_dr = bool(domain_covered and range_covered)
        if config.require_domain_and_range_for_strict_cv:
            missing_dr = not complete_dr
        else:
            missing_dr = not (domain_covered or range_covered)

        observed_violation = (not full_aligned) or formal_violation
        strict_violation = (not full_aligned) or formal_violation or missing_dr
        selected_violation = strict_violation if config.cv_mode == "strict_schema" else observed_violation

        support_count += int(has_supp)
        provenance_count += int(has_prov)
        full_ontology_aligned_count += int(full_aligned)
        observed_cv_count += int(observed_violation)
        strict_schema_cv_count += int(strict_violation)
        explicit_domain_range_violation_count += int(formal_violation)
        domain_applicable_count += int(domain_covered)
        range_applicable_count += int(range_covered)
        complete_domain_range_applicable_count += int(complete_dr)
        missing_domain_range_constraint_count += int(missing_dr)

        rec.update({
            "has_support_trace": bool(has_supp),
            "has_provenance": bool(has_prov),
            "subject_aligned": bool(s_align["aligned"]),
            "subject_match_kind": s_align["kind"],
            "subject_match_score": s_align["score"],
            "subject_match": s_align["match"],
            "predicate_aligned": bool(p_align["aligned"]),
            "predicate_match_kind": p_align["kind"],
            "predicate_match_score": p_align["score"],
            "predicate_match": p_align["match"],
            "object_aligned": bool(o_align["aligned"]),
            "object_match_kind": o_align["kind"],
            "object_match_score": o_align["score"],
            "object_match": o_align["match"],
            "ontology_compliant": full_aligned,
            "formal_domain_range_violation": bool(formal_violation),
            "predicate_domain_axiom_applicable": bool(domain_covered),
            "predicate_range_axiom_applicable": bool(range_covered),
            "complete_domain_range_axioms_applicable": bool(complete_dr),
            "missing_domain_range_constraint": bool(missing_dr),
            "observed_alignment_constraint_violation": bool(observed_violation),
            "strict_schema_constraint_violation": bool(strict_violation),
            "constraint_violation": bool(selected_violation),
        })
        detail_rows.append(rec)

    str_rate = support_count / valid_count if valid_count else 0.0
    pc_rate = provenance_count / valid_count if valid_count else 0.0
    oc_rate = full_ontology_aligned_count / valid_count if valid_count else 0.0
    observed_cv_rate = observed_cv_count / valid_count if valid_count else 1.0
    strict_schema_cv_rate = strict_schema_cv_count / valid_count if valid_count else 1.0
    if config.count_invalid_rows_in_cv and raw_count:
        # Optional global CV that also counts unresolved rows. CR already reports this, so default is False.
        selected_valid_count = strict_schema_cv_count if config.cv_mode == "strict_schema" else observed_cv_count
        cv_rate = (selected_valid_count + invalid_count) / raw_count
    else:
        cv_rate = strict_schema_cv_rate if config.cv_mode == "strict_schema" else observed_cv_rate
    cr_rate = invalid_count / raw_count if raw_count else 0.0

    domain_axiom_coverage = domain_applicable_count / valid_count if valid_count else 0.0
    range_axiom_coverage = range_applicable_count / valid_count if valid_count else 0.0
    schema_constraint_coverage = complete_domain_range_applicable_count / valid_count if valid_count else 0.0

    dvs = float(
        str_rate >= config.dvs_min_str
        and cr_rate <= config.dvs_max_cr
        and pc_rate >= config.dvs_min_pc
        and oc_rate >= config.dvs_min_oc
        and cv_rate <= config.dvs_max_cv
    )

    summary = {
        "raw_triple_row_count": raw_count,
        "valid_triple_count": valid_count,
        "invalid_or_unresolved_triple_count": invalid_count,
        "support_trace_count": support_count,
        "provenance_count": provenance_count,
        "ontology_compliant_triple_count": full_ontology_aligned_count,
        "constraint_violation_count": strict_schema_cv_count if config.cv_mode == "strict_schema" else observed_cv_count,
        "observed_alignment_constraint_violation_count": observed_cv_count,
        "strict_schema_constraint_violation_count": strict_schema_cv_count,
        "explicit_domain_range_violation_count": explicit_domain_range_violation_count,
        "missing_domain_range_constraint_count": missing_domain_range_constraint_count,
        "domain_axiom_applicable_count": domain_applicable_count,
        "range_axiom_applicable_count": range_applicable_count,
        "complete_domain_range_applicable_count": complete_domain_range_applicable_count,
        "STR": str_rate,
        "CR": cr_rate,
        "PC": pc_rate,
        "OC": oc_rate,
        "CV": cv_rate,
        "CV_observed_alignment": observed_cv_rate,
        "CV_strict_schema": strict_schema_cv_rate,
        "domain_axiom_coverage": domain_axiom_coverage,
        "range_axiom_coverage": range_axiom_coverage,
        "schema_constraint_coverage": schema_constraint_coverage,
        "DVS": dvs,
        "ontology_class_count": ontology.get("class_count", 0),
        "ontology_property_count": ontology.get("property_count", 0),
        "ontology_properties_with_domain_count": len(ontology.get("properties_with_domain", set())),
        "ontology_properties_with_range_count": len(ontology.get("properties_with_range", set())),
        "provenance_columns_detected": ";".join(map(str, prov_cols)),
        "support_columns_detected": ";".join(map(str, supp_cols)),
        "source_column": source_col,
        "relation_column": relation_col,
        "target_column": target_col,
        "node_fuzzy_threshold": config.node_fuzzy_threshold,
        "predicate_fuzzy_threshold": config.predicate_fuzzy_threshold,
        "allow_lexical_source_support_for_str": config.allow_lexical_source_support_for_str,
        "cv_mode": config.cv_mode,
        "require_domain_and_range_for_strict_cv": config.require_domain_and_range_for_strict_cv,
        "count_invalid_rows_in_cv": config.count_invalid_rows_in_cv,
    }
    return summary, pd.DataFrame(detail_rows)

"""Cumulative prefix triple consolidation utilities for NeoOLAF ablation.

The goal is to evaluate a more faithful stop-after-layer diagnostic than
independent prefix finalization:

    stop at layer k -> use all triples generated from prefixes <= k
    -> merge semantically equivalent triples
    -> keep a compact/budgeted KG
    -> evaluate the resulting KG.

This does not use the gold annotations for consolidation. Gold is only used
later by the normal evaluation runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_ALLOWED_RELATIONS = {"CAUSES", "TRIGGERS", "REQUIRES", "HANDLED_BY", "REFERENCES"}

RELATION_ALIASES = {
    "CAUSE": "CAUSES",
    "CAUSES": "CAUSES",
    "CAUSED_BY": "CAUSES",
    "TRIGGER": "TRIGGERS",
    "TRIGGERS": "TRIGGERS",
    "ACTIVATES": "TRIGGERS",
    "REQUIRE": "REQUIRES",
    "REQUIRES": "REQUIRES",
    "NEEDS": "REQUIRES",
    "HANDLED_BY": "HANDLED_BY",
    "HANDLES": "HANDLED_BY",
    "RESOLVED_BY": "HANDLED_BY",
    "REFERENCE": "REFERENCES",
    "REFERENCES": "REFERENCES",
    "MENTIONS": "REFERENCES",
}

GENERIC_ENTITY_TERMS = {
    "message", "alarm", "alarme", "operator", "system", "machine", "component",
    "condition", "action", "issue", "problem", "error", "warning", "fault",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_text(text: Any) -> str:
    """Normalize labels for matching, without destroying technical identifiers."""
    text = str(text or "").strip()
    text = strip_accents(text).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Any) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_text(text)))


def token_jaccard(a: Any, b: Any) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def string_similarity(a: Any, b: Any) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return max(token_jaccard(na, nb), SequenceMatcher(None, na, nb).ratio())


def normalize_relation(rel: Any) -> str:
    rel = str(rel or "").strip()
    rel = strip_accents(rel).upper()
    rel = rel.replace(" ", "_").replace("-", "_")
    rel = re.sub(r"[^A-Z0-9_]+", "", rel)
    rel = re.sub(r"_+", "_", rel).strip("_")
    return RELATION_ALIASES.get(rel, rel)


def compact_label(labels: Iterable[str]) -> str:
    """Pick a stable representative label for a cluster."""
    clean = [str(x).strip() for x in labels if str(x or "").strip()]
    if not clean:
        return ""
    counts = Counter(clean)
    # Prefer frequent, reasonably short labels. Avoid extremely verbose generations.
    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], len(kv[0]), kv[0].lower()),
    )
    return ranked[0][0]


def label_verbosity_penalty(label: str) -> float:
    words = normalize_text(label).split()
    if len(words) <= 8:
        return 0.0
    return min(1.0, (len(words) - 8) / 20.0)


def generic_entity_penalty(label: str) -> float:
    toks = tokenize(label)
    if not toks:
        return 0.5
    if len(toks) <= 2 and toks & GENERIC_ENTITY_TERMS:
        return 0.35
    if toks <= GENERIC_ENTITY_TERMS:
        return 0.5
    return 0.0


@dataclass
class TripleRecord:
    source_stop_index: int
    source_layer_name: str
    raw_index: int
    head: str
    predicate: str
    tail: str
    confidence: float | None = None
    evidence: str = ""
    chunk_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_key(self) -> tuple[str, str, str]:
        return (normalize_text(self.head), self.predicate, normalize_text(self.tail))

    @property
    def text(self) -> str:
        return f"{self.head} {self.predicate} {self.tail}"


@dataclass
class TripleCluster:
    cluster_id: str
    predicate: str
    records: list[TripleRecord] = field(default_factory=list)

    def add(self, rec: TripleRecord) -> None:
        self.records.append(rec)

    @property
    def source_layers(self) -> set[int]:
        return {r.source_stop_index for r in self.records}

    @property
    def support_count(self) -> int:
        return len(self.records)

    @property
    def first_seen(self) -> int:
        return min(self.source_layers) if self.records else -1

    @property
    def last_seen(self) -> int:
        return max(self.source_layers) if self.records else -1

    @property
    def head(self) -> str:
        return compact_label(r.head for r in self.records)

    @property
    def tail(self) -> str:
        return compact_label(r.tail for r in self.records)

    @property
    def evidence(self) -> str:
        ev = [r.evidence for r in self.records if r.evidence]
        if not ev:
            return ""
        return compact_label(ev)

    @property
    def chunk_id(self) -> str:
        chunks = [r.chunk_id for r in self.records if r.chunk_id]
        if not chunks:
            return ""
        return compact_label(chunks)

    @property
    def avg_confidence(self) -> float | None:
        vals = [float(r.confidence) for r in self.records if isinstance(r.confidence, (int, float))]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @property
    def text(self) -> str:
        return f"{self.head} {self.predicate} {self.tail}"


def triple_similarity_to_cluster(rec: TripleRecord, cluster: TripleCluster) -> float:
    if rec.predicate != cluster.predicate:
        return 0.0
    h = string_similarity(rec.head, cluster.head)
    t = string_similarity(rec.tail, cluster.tail)
    # Direction matters, so head and tail are compared separately.
    return 0.52 * h + 0.42 * t + 0.06


def cluster_score(cluster: TripleCluster, stop_index: int) -> float:
    """Gold-free score used to rank clusters for a compact KG.

    The score rewards cross-layer stability and repeated support, while mildly
    penalizing generic/verbose labels. It intentionally does not use the gold.
    """
    if not cluster.records:
        return 0.0

    layers_seen = cluster.source_layers
    layer_count = max(1, stop_index + 1)
    support_layers_ratio = len(layers_seen) / layer_count
    recency = 1.0 if stop_index in layers_seen else 0.5 + 0.5 * (cluster.last_seen / max(1, stop_index))
    repetition = math.log1p(cluster.support_count)
    confidence = cluster.avg_confidence if cluster.avg_confidence is not None else 0.5

    penalty = (
        0.35 * generic_entity_penalty(cluster.head)
        + 0.35 * generic_entity_penalty(cluster.tail)
        + 0.20 * label_verbosity_penalty(cluster.head)
        + 0.20 * label_verbosity_penalty(cluster.tail)
    )

    return (
        2.25 * support_layers_ratio
        + 1.10 * repetition
        + 0.85 * recency
        + 0.40 * confidence
        - penalty
    )


def cluster_to_export_triple(cluster: TripleCluster, stop_index: int, score: float) -> dict[str, Any]:
    confidence = cluster.avg_confidence
    out = {
        "subject": cluster.head,
        "predicate": cluster.predicate,
        "object": cluster.tail,
        "head": cluster.head,
        "relation": cluster.predicate,
        "tail": cluster.tail,
        "subject_label": cluster.head,
        "predicate_label": cluster.predicate,
        "object_label": cluster.tail,
        "justification": cluster.evidence,
        "chunk_id": cluster.chunk_id,
        "confidence": float(confidence) if confidence is not None else float(min(1.0, score / 6.0)),
        "consolidation": {
            "cluster_id": cluster.cluster_id,
            "stop_index": stop_index,
            "score": score,
            "support_count": cluster.support_count,
            "support_layers": sorted(cluster.source_layers),
            "first_seen": cluster.first_seen,
            "last_seen": cluster.last_seen,
            "source_layer_names": sorted({r.source_layer_name for r in cluster.records}),
        },
    }
    return out


def build_triple_record(
    raw_triple: dict[str, Any],
    *,
    source_stop_index: int,
    source_layer_name: str,
    raw_index: int,
) -> TripleRecord | None:
    """Extract a flat TripleRecord from common prefix/NeoOLAF triple layouts."""
    def nested(obj: Any, *keys: str) -> Any:
        cur = obj
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur

    def label(value: Any) -> str:
        if isinstance(value, dict):
            for k in ("label", "text", "name", "value", "id"):
                if value.get(k):
                    return str(value[k]).strip()
            return ""
        return str(value or "").strip()

    head = (
        raw_triple.get("subject_label")
        or raw_triple.get("subject")
        or raw_triple.get("head")
        or raw_triple.get("s")
        or nested(raw_triple, "subject", "label")
        or nested(raw_triple, "head", "label")
    )
    pred = (
        raw_triple.get("predicate_label")
        or raw_triple.get("predicate")
        or raw_triple.get("relation")
        or raw_triple.get("rel")
        or raw_triple.get("p")
        or nested(raw_triple, "predicate", "label")
        or nested(raw_triple, "relation", "label")
    )
    tail = (
        raw_triple.get("object_label")
        or raw_triple.get("object")
        or raw_triple.get("tail")
        or raw_triple.get("o")
        or nested(raw_triple, "object", "label")
        or nested(raw_triple, "tail", "label")
    )

    head_label = label(head)
    tail_label = label(tail)
    pred_label = normalize_relation(label(pred))

    if not head_label or not pred_label or not tail_label:
        return None

    conf = raw_triple.get("confidence")
    if not isinstance(conf, (int, float)):
        conf = None

    evidence = (
        raw_triple.get("justification")
        or raw_triple.get("evidence")
        or raw_triple.get("support_text")
        or raw_triple.get("source_text")
        or raw_triple.get("context")
        or ""
    )
    chunk_id = raw_triple.get("chunk_id") or raw_triple.get("chunkid") or raw_triple.get("source_id") or ""

    return TripleRecord(
        source_stop_index=int(source_stop_index),
        source_layer_name=str(source_layer_name),
        raw_index=int(raw_index),
        head=head_label,
        predicate=pred_label,
        tail=tail_label,
        confidence=conf,
        evidence=str(evidence or ""),
        chunk_id=str(chunk_id or ""),
        raw=raw_triple,
    )


class IncrementalConsolidator:
    """Greedy cumulative merger for prefix triples."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.86,
        allowed_relations: set[str] | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.allowed_relations = allowed_relations or set(DEFAULT_ALLOWED_RELATIONS)
        self.clusters_by_predicate: dict[str, list[TripleCluster]] = defaultdict(list)
        self.exact_index: dict[tuple[str, str, str], TripleCluster] = {}
        self.next_cluster_id = 0

    @property
    def clusters(self) -> list[TripleCluster]:
        out: list[TripleCluster] = []
        for clusters in self.clusters_by_predicate.values():
            out.extend(clusters)
        return out

    def add_record(self, rec: TripleRecord) -> None:
        if self.allowed_relations and rec.predicate not in self.allowed_relations:
            return

        key = rec.normalized_key
        exact = self.exact_index.get(key)
        if exact is not None:
            exact.add(rec)
            return

        best_cluster = None
        best_sim = 0.0
        for cluster in self.clusters_by_predicate.get(rec.predicate, []):
            sim = triple_similarity_to_cluster(rec, cluster)
            if sim > best_sim:
                best_cluster = cluster
                best_sim = sim

        if best_cluster is not None and best_sim >= self.similarity_threshold:
            best_cluster.add(rec)
            self.exact_index[key] = best_cluster
            return

        cluster = TripleCluster(cluster_id=f"c{self.next_cluster_id:06d}", predicate=rec.predicate)
        self.next_cluster_id += 1
        cluster.add(rec)
        self.clusters_by_predicate[rec.predicate].append(cluster)
        self.exact_index[key] = cluster

    def add_records(
        self,
        records: Iterable[TripleRecord],
        *,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        records_iter = records
        if show_progress and tqdm is not None:
            try:
                total = len(records)  # type: ignore[arg-type]
            except Exception:
                total = None
            records_iter = tqdm(records, total=total, desc=progress_desc or "Merging triples", leave=False)
        for rec in records_iter:
            self.add_record(rec)

    def ranked_clusters(self, stop_index: int) -> list[tuple[TripleCluster, float]]:
        scored = [(c, cluster_score(c, stop_index)) for c in self.clusters]
        scored.sort(key=lambda item: (-item[1], -item[0].support_count, item[0].cluster_id))
        return scored

    def selected_export_triples(
        self,
        *,
        stop_index: int,
        budget: int | None = None,
        min_score: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ranked = self.ranked_clusters(stop_index)
        if min_score is not None:
            ranked = [(c, s) for c, s in ranked if s >= min_score]
        if budget is not None and budget > 0:
            ranked = ranked[:budget]

        triples = [cluster_to_export_triple(c, stop_index, s) for c, s in ranked]
        support_counts = [c.support_count for c, _ in ranked]
        support_layers = [len(c.source_layers) for c, _ in ranked]
        scores = [s for _, s in ranked]
        metadata = {
            "stop_index": stop_index,
            "candidate_cluster_count": len(self.clusters),
            "selected_cluster_count": len(triples),
            "budget": budget,
            "min_score": min_score,
            "avg_cluster_score": sum(scores) / len(scores) if scores else 0.0,
            "avg_support_count": sum(support_counts) / len(support_counts) if support_counts else 0.0,
            "avg_support_layers": sum(support_layers) / len(support_layers) if support_layers else 0.0,
            "max_support_layers": max(support_layers) if support_layers else 0,
        }
        return triples, metadata

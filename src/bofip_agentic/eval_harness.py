from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class QueryGold:
    query_id: str
    query: str
    category: str
    gold_doc_refs: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class EvalResult:
    query_id: str
    query: str
    category: str
    doc_hit: bool
    passage_hit: bool
    doc_rank: int | None
    passage_rank: int | None
    retrieved_doc_refs: list[str]
    retrieved_chunk_ids: list[str]
    gold_doc_refs: list[str] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class EvalMetrics:
    queries_count: int
    categories_count: dict[str, int]
    doc_hit_at: dict[int, float]
    passage_hit_at: dict[int, float]
    mrr_doc: float
    mrr_passage: float
    ndcg_doc_at: dict[int, float]
    ndcg_passage_at: dict[int, float]
    per_query: list[EvalResult] = field(default_factory=list)


LayerTarget = Literal["doc", "chunk"]


@dataclass(frozen=True)
class RetrievalLayer:
    name: str
    target: LayerTarget


@dataclass
class LayerEvalResult:
    layer: str
    target: LayerTarget
    hit: bool
    rank: int | None
    retrieved_ids: list[str]
    gold_ids: list[str] = field(default_factory=list)


@dataclass
class LayeredQueryResult:
    query_id: str
    query: str
    category: str
    layers: dict[str, LayerEvalResult]
    first_miss_layer: str | None = None


@dataclass
class LayeredEvalMetrics:
    queries_count: int
    categories_count: dict[str, int]
    layer_hit_at: dict[str, dict[int, float]]
    layer_mrr: dict[str, float]
    layer_ndcg_at: dict[str, dict[int, float]]
    transition_misses: dict[str, int]
    first_miss_counts: dict[str, int]
    per_query: list[LayeredQueryResult] = field(default_factory=list)


def _binary_relevance(items: list[str], golds: set[str]) -> list[int]:
    return [1 if item in golds else 0 for item in items]


def _ndcg(relevance: list[int], k: int) -> float:
    ideal = sorted(relevance, reverse=True)[:k]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance[:k]))
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _first_hit_rank(items: list[str], golds: set[str]) -> int | None:
    for idx, item in enumerate(items):
        if item in golds:
            return idx + 1
    return None


def evaluate(
    queries: list[QueryGold],
    *,
    retrieve_docs,
    retrieve_chunks,
    k_values: list[int] | None = None,
) -> EvalMetrics:
    if k_values is None:
        k_values = [1, 3, 5, 8]

    max_k = max(k_values)
    per_query: list[EvalResult] = []
    categories_count: dict[str, int] = {}

    for qg in queries:
        cat = qg.category or "unknown"
        categories_count[cat] = categories_count.get(cat, 0) + 1

        gold_doc_set = set(qg.gold_doc_refs)
        gold_chunk_set = set(qg.gold_chunk_ids)

        retrieved_docs = retrieve_docs(qg.query) if retrieve_docs else []
        retrieved_doc_refs = list(retrieved_docs)[:max_k]

        retrieved_chunks = retrieve_chunks(qg.query) if retrieve_chunks else []
        retrieved_chunk_ids = list(retrieved_chunks)[:max_k]

        doc_rank = _first_hit_rank(retrieved_doc_refs, gold_doc_set)
        passage_rank = _first_hit_rank(retrieved_chunk_ids, gold_chunk_set)

        per_query.append(
            EvalResult(
                query_id=qg.query_id,
                query=qg.query,
                category=cat,
                doc_hit=doc_rank is not None,
                passage_hit=passage_rank is not None,
                doc_rank=doc_rank,
                passage_rank=passage_rank,
                retrieved_doc_refs=retrieved_doc_refs,
                retrieved_chunk_ids=retrieved_chunk_ids,
                gold_doc_refs=qg.gold_doc_refs,
                gold_chunk_ids=qg.gold_chunk_ids,
            )
        )

    total = len(queries)

    doc_hit_at: dict[int, float] = {}
    passage_hit_at: dict[int, float] = {}
    ndcg_doc_at: dict[int, float] = {}
    ndcg_passage_at: dict[int, float] = {}
    mrr_doc_sum = 0.0
    mrr_passage_sum = 0.0

    for k in k_values:
        doc_hits = sum(
            1 for r in per_query
            if any(ref in set(r.gold_doc_refs) for ref in r.retrieved_doc_refs[:k])
        )
        passage_hits = sum(
            1 for r in per_query
            if any(cid in set(r.gold_chunk_ids) for cid in r.retrieved_chunk_ids[:k])
        )
        doc_hit_at[k] = doc_hits / total if total else 0.0
        passage_hit_at[k] = passage_hits / total if total else 0.0

        ndcg_doc_values = 0.0
        ndcg_passage_values = 0.0
        for r in per_query:
            doc_rel = _binary_relevance(r.retrieved_doc_refs[:k], set(r.gold_doc_refs))
            ndcg_doc_values += _ndcg(doc_rel, k)
            chunk_rel = _binary_relevance(r.retrieved_chunk_ids[:k], set(r.gold_chunk_ids))
            ndcg_passage_values += _ndcg(chunk_rel, k)
        ndcg_doc_at[k] = ndcg_doc_values / total if total else 0.0
        ndcg_passage_at[k] = ndcg_passage_values / total if total else 0.0

    for r in per_query:
        doc_rank = r.doc_rank
        passage_rank = r.passage_rank
        if doc_rank and doc_rank >= 1:
            mrr_doc_sum += 1.0 / doc_rank
        if passage_rank and passage_rank >= 1:
            mrr_passage_sum += 1.0 / passage_rank

    mrr_doc = mrr_doc_sum / total if total else 0.0
    mrr_passage = mrr_passage_sum / total if total else 0.0

    return EvalMetrics(
        queries_count=total,
        categories_count=categories_count,
        doc_hit_at=doc_hit_at,
        passage_hit_at=passage_hit_at,
        mrr_doc=mrr_doc,
        mrr_passage=mrr_passage,
        ndcg_doc_at=ndcg_doc_at,
        ndcg_passage_at=ndcg_passage_at,
        per_query=per_query,
    )


def _gold_ids_for_layer(query_gold: QueryGold, target: LayerTarget) -> list[str]:
    if target == "doc":
        return query_gold.gold_doc_refs
    return query_gold.gold_chunk_ids


def evaluate_layers(
    queries: list[QueryGold],
    *,
    retrieve_layers,
    layers: list[RetrievalLayer],
    k_values: list[int] | None = None,
) -> LayeredEvalMetrics:
    """Evaluate retrieval at named pipeline layers.

    This is deliberately generic: callers decide which layer emits document refs
    and which emits chunk ids. The harness only compares emitted ids to gold ids.
    """
    if k_values is None:
        k_values = [1, 3, 5, 8]

    max_k = max(k_values)
    per_query: list[LayeredQueryResult] = []
    categories_count: dict[str, int] = {}
    layer_names = [layer.name for layer in layers]

    for qg in queries:
        cat = qg.category or "unknown"
        categories_count[cat] = categories_count.get(cat, 0) + 1
        retrieved_by_layer = retrieve_layers(qg.query) or {}
        layer_results: dict[str, LayerEvalResult] = {}
        first_miss_layer: str | None = None

        for layer in layers:
            gold_ids = list(_gold_ids_for_layer(qg, layer.target))
            retrieved_ids = list(retrieved_by_layer.get(layer.name, []))[:max_k]
            rank = _first_hit_rank(retrieved_ids, set(gold_ids))
            hit = rank is not None
            layer_results[layer.name] = LayerEvalResult(
                layer=layer.name,
                target=layer.target,
                hit=hit,
                rank=rank,
                retrieved_ids=retrieved_ids,
                gold_ids=gold_ids,
            )
            if gold_ids and not hit and first_miss_layer is None:
                first_miss_layer = layer.name

        per_query.append(
            LayeredQueryResult(
                query_id=qg.query_id,
                query=qg.query,
                category=cat,
                layers=layer_results,
                first_miss_layer=first_miss_layer,
            )
        )

    total = len(queries)
    layer_hit_at: dict[str, dict[int, float]] = {name: {} for name in layer_names}
    layer_ndcg_at: dict[str, dict[int, float]] = {name: {} for name in layer_names}
    layer_mrr: dict[str, float] = {name: 0.0 for name in layer_names}
    transition_misses: dict[str, int] = {}
    first_miss_counts: dict[str, int] = {}

    for layer in layers:
        name = layer.name
        for k in k_values:
            hits = 0
            ndcg_sum = 0.0
            for result in per_query:
                layer_result = result.layers[name]
                gold_set = set(layer_result.gold_ids)
                if any(item in gold_set for item in layer_result.retrieved_ids[:k]):
                    hits += 1
                relevance = _binary_relevance(layer_result.retrieved_ids[:k], gold_set)
                ndcg_sum += _ndcg(relevance, k)
            layer_hit_at[name][k] = hits / total if total else 0.0
            layer_ndcg_at[name][k] = ndcg_sum / total if total else 0.0

        mrr_sum = 0.0
        for result in per_query:
            rank = result.layers[name].rank
            if rank and rank >= 1:
                mrr_sum += 1.0 / rank
        layer_mrr[name] = mrr_sum / total if total else 0.0

    for result in per_query:
        if result.first_miss_layer:
            first_miss_counts[result.first_miss_layer] = first_miss_counts.get(result.first_miss_layer, 0) + 1
        for left, right in zip(layer_names, layer_names[1:]):
            left_hit = result.layers[left].hit
            right_result = result.layers[right]
            if left_hit and right_result.gold_ids and not right_result.hit:
                key = f"{left}->{right}"
                transition_misses[key] = transition_misses.get(key, 0) + 1

    return LayeredEvalMetrics(
        queries_count=total,
        categories_count=categories_count,
        layer_hit_at=layer_hit_at,
        layer_mrr=layer_mrr,
        layer_ndcg_at=layer_ndcg_at,
        transition_misses=transition_misses,
        first_miss_counts=first_miss_counts,
        per_query=per_query,
    )

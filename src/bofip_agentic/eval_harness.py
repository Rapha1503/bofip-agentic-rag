from __future__ import annotations

import math
from dataclasses import dataclass, field


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

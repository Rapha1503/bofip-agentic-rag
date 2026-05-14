"""Ablation testing — measure each pipeline component's impact."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_cleanroom.rag_runtime import RagRuntime
from bofip_cleanroom.eval_harness import QueryGold, evaluate
from bofip_cleanroom.jsonio import read_jsonl
from bofip_cleanroom.settings import INTERIM_DIR

QUERIES_PATH = INTERIM_DIR / "eval_queries_v1.jsonl"
GOLD_PATH = INTERIM_DIR / "passage_gold_v3.jsonl"
LIMIT = 5

configs = [
    ("BM25 only", dict(use_dense=False, use_chunk_dense=False, use_anchor_filter=False, use_reranker=False)),
    ("Dense only", dict(use_lexical=False, use_reranker=False)),
    ("Full (no reranker)", dict(use_reranker=False)),
    ("Full (+ reranker)", dict(use_reranker=True)),
]

rt = RagRuntime.from_local_corpus(corpus="commentary", device="cpu")

# Load queries and gold
query_rows = read_jsonl(QUERIES_PATH)
gold_rows = read_jsonl(GOLD_PATH)
gold_map = {g["query_id"]: g.get("gold_chunk_ids", []) for g in gold_rows}
queries = []
for row in query_rows[:LIMIT]:
    qid = row["query_id"]
    queries.append(QueryGold(
        query_id=qid, query=row["query"], category=row.get("category", ""),
        gold_doc_refs=row.get("gold_doc_refs", []),
        gold_chunk_ids=gold_map.get(qid, []),
    ))

print(f"{'Config':<30} {'doc@1':>8} {'doc@5':>8} {'pass@1':>8} {'pass@5':>8} {'MRR_doc':>8} {'MRR_pass':>8}")
print("-" * 80)

for name, kwargs in configs:
    results = {}  # cache per query
    def get_result(q):
        if q not in results:
            results[q] = rt.retrieve(q, top_docs=5, **kwargs)
        return results[q]

    def retrieve_docs(q):
        return [h.boi_reference for h in get_result(q).stage1_hits]

    def retrieve_chunks(q):
        return [c.chunk_id for c in get_result(q).stage2_chunks]

    metrics = evaluate(queries, retrieve_docs=retrieve_docs, retrieve_chunks=retrieve_chunks)
    print(f"{name:<30} {metrics.doc_hit_at.get(1,0):>8.3f} {metrics.doc_hit_at.get(5,0):>8.3f} "
          f"{metrics.passage_hit_at.get(1,0):>8.3f} {metrics.passage_hit_at.get(5,0):>8.3f} "
          f"{metrics.mrr_doc:>8.3f} {metrics.mrr_passage:>8.3f}")

"""Ablation testing: measure each retrieval component's impact."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.eval_harness import QueryGold, evaluate
from bofip_cleanroom.jsonio import read_jsonl
from bofip_cleanroom.rag_runtime import RagRuntime
from bofip_cleanroom.settings import INTERIM_DIR


CONFIGS = [
    ("BM25 only", dict(use_dense=False, use_chunk_dense=False, use_anchor_filter=False, use_reranker=False)),
    ("Dense only", dict(use_lexical=False, use_reranker=False)),
    ("Full (no reranker)", dict(use_reranker=False)),
    ("Full (+ reranker)", dict(use_reranker=True)),
]


def _load_queries(queries_path: Path, gold_path: Path, *, limit: int) -> list[QueryGold]:
    query_rows = read_jsonl(queries_path)
    gold_rows = read_jsonl(gold_path)
    gold_map = {g["query_id"]: g.get("gold_chunk_ids", []) for g in gold_rows}
    selected_rows = query_rows[:limit] if limit else query_rows
    return [
        QueryGold(
            query_id=row["query_id"],
            query=row["query"],
            category=row.get("category", ""),
            gold_doc_refs=row.get("gold_doc_refs", []),
            gold_chunk_ids=gold_map.get(row["query_id"], []),
        )
        for row in selected_rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval ablations over the BOFiP eval set.")
    parser.add_argument("--queries", type=Path, default=INTERIM_DIR / "eval_queries_v1.jsonl")
    parser.add_argument("--gold", type=Path, default=INTERIM_DIR / "passage_gold_v3.jsonl")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    queries = _load_queries(args.queries, args.gold, limit=args.limit)
    if not queries:
        print("No queries loaded.")
        return 1

    runtime = RagRuntime.from_local_corpus(corpus="commentary", device=args.device)

    print(
        f"{'Config':<30} {'doc@1':>8} {'doc@5':>8} {'pass@1':>8} "
        f"{'pass@5':>8} {'MRR_doc':>8} {'MRR_pass':>8}"
    )
    print("-" * 86)

    for name, kwargs in CONFIGS:
        results = {}

        def get_result(query: str):
            if query not in results:
                results[query] = runtime.retrieve(query, top_docs=5, **kwargs)
            return results[query]

        def retrieve_docs(query: str) -> list[str]:
            return [hit.boi_reference for hit in get_result(query).stage1_hits]

        def retrieve_chunks(query: str) -> list[str]:
            return [chunk.chunk_id for chunk in get_result(query).stage2_chunks]

        metrics = evaluate(queries, retrieve_docs=retrieve_docs, retrieve_chunks=retrieve_chunks)
        print(
            f"{name:<30} {metrics.doc_hit_at.get(1, 0):>8.3f} {metrics.doc_hit_at.get(5, 0):>8.3f} "
            f"{metrics.passage_hit_at.get(1, 0):>8.3f} {metrics.passage_hit_at.get(5, 0):>8.3f} "
            f"{metrics.mrr_doc:>8.3f} {metrics.mrr_passage:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.eval_harness import EvalMetrics, QueryGold, evaluate
from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.preview_runtime import (
    DEFAULT_PREVIEW_CORPUS,
    Phase8bPreviewRuntime,
)
from bofip_cleanroom.rag_runtime import RagRuntime
from bofip_cleanroom.settings import INTERIM_DIR, REPORTS_DIR, ensure_data_dirs


def _load_query_golds(path: str) -> list[QueryGold]:
    rows = read_jsonl(Path(path))
    return [
        QueryGold(
            query_id=row["query_id"],
            query=row["query"],
            category=row.get("category", ""),
            gold_doc_refs=row.get("gold_doc_refs", []),
            gold_chunk_ids=row.get("gold_chunk_ids", []),
            note=row.get("note", ""),
        )
        for row in rows
    ]


def _metrics_to_dict(metrics: EvalMetrics) -> dict:
    return {
        "queries_count": metrics.queries_count,
        "categories_count": metrics.categories_count,
        "doc_hit_at": metrics.doc_hit_at,
        "passage_hit_at": metrics.passage_hit_at,
        "mrr_doc": metrics.mrr_doc,
        "mrr_passage": metrics.mrr_passage,
        "ndcg_doc_at": metrics.ndcg_doc_at,
        "ndcg_passage_at": metrics.ndcg_passage_at,
        "per_query": [asdict(r) for r in metrics.per_query],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Standardized retrieval evaluation.")
    parser.add_argument(
        "--queries",
        type=str,
        default=str(INTERIM_DIR / "eval_queries_v1.jsonl"),
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=str(INTERIM_DIR / "passage_gold_v3.jsonl"),
    )
    parser.add_argument("--corpus", type=str, default=DEFAULT_PREVIEW_CORPUS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument("--chunks-per-doc", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--case-ids", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--runtime",
        type=str,
        default="rag",
        choices=["phase8b", "rag"],
        help="Which retrieval runtime: phase8b (no reranker) or rag (with cross-encoder reranker).",
    )
    args = parser.parse_args()

    ensure_data_dirs()
    query_golds = _load_query_golds(args.queries)
    gold_rows = _load_query_golds(args.gold)
    gold_map = {g.query_id: g.gold_chunk_ids for g in gold_rows}
    for qg in query_golds:
        if qg.query_id in gold_map:
            qg.gold_chunk_ids = gold_map[qg.query_id]
    if args.case_ids:
        selected = {v.strip() for v in args.case_ids.split(",") if v.strip()}
        query_golds = [q for q in query_golds if q.query_id in selected]
    if args.limit > 0:
        query_golds = query_golds[:args.limit]
    if not query_golds:
        print("No queries loaded.")
        return 1

    if args.runtime == "rag":
        runtime = RagRuntime.from_local_corpus(
            corpus=args.corpus,
            device=args.device,
        )
    elif args.runtime == "phase8b":
        runtime = Phase8bPreviewRuntime.from_local_corpus(
            corpus=args.corpus,
            device=args.device,
        )
    else:
        print(f"Unknown runtime: {args.runtime}")
        return 1

    _cache: dict[str, tuple[list[str], list[str]]] = {}

    def _retrieve_pair(query: str) -> tuple[list[str], list[str]]:
        if query in _cache:
            return _cache[query]
        result = runtime.retrieve(
            query,
            top_docs=args.top_docs,
            chunks_per_doc=args.chunks_per_doc,
            max_chunks=args.max_chunks,
        )
        docs = [hit.boi_reference for hit in result.stage1_hits]
        chunks_out = [chunk.chunk_id for chunk in result.stage2_chunks]
        _cache[query] = (docs, chunks_out)
        return docs, chunks_out

    def _retrieve_docs(query: str) -> list[str]:
        return _retrieve_pair(query)[0]

    def _retrieve_chunks(query: str) -> list[str]:
        return _retrieve_pair(query)[1]

    total = len(query_golds)
    for idx, qg in enumerate(query_golds, 1):
        _retrieve_pair(qg.query)
        print(f"  [{idx}/{total}] {qg.query_id}", flush=True)

    metrics = evaluate(
        query_golds,
        retrieve_docs=_retrieve_docs,
        retrieve_chunks=_retrieve_chunks,
    )

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"eval_metrics_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "runtime": args.runtime,
        "corpus": args.corpus,
        "top_docs": args.top_docs,
        "chunks_per_doc": args.chunks_per_doc,
        "max_chunks": args.max_chunks,
        "metrics": _metrics_to_dict(metrics),
    }
    write_json(report_path, payload)
    print(f"Evaluation report written to: {report_path}")
    print(f"Queries: {metrics.queries_count}")
    print(f"Doc hit@1: {metrics.doc_hit_at.get(1, 0):.4f}")
    print(f"Doc hit@3: {metrics.doc_hit_at.get(3, 0):.4f}")
    print(f"Doc hit@5: {metrics.doc_hit_at.get(5, 0):.4f}")
    print(f"Passage hit@1: {metrics.passage_hit_at.get(1, 0):.4f}")
    print(f"Passage hit@3: {metrics.passage_hit_at.get(3, 0):.4f}")
    print(f"Passage hit@5: {metrics.passage_hit_at.get(5, 0):.4f}")
    print(f"MRR doc: {metrics.mrr_doc:.4f}")
    print(f"MRR passage: {metrics.mrr_passage:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

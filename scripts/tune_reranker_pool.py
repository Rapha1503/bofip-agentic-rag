"""
Tune reranker candidate pool size with retrieval-only metrics and latency.

This script helps choose a practical reranker pool size (e.g. 25 vs 50)
based on measurable trade-offs:
- Recall@K / Precision@K / HitRate
- Average latency per query

Usage:
    python scripts/tune_reranker_pool.py
    python scripts/tune_reranker_pool.py --pools 15 20 25 30 --k 5 10 20
    python scripts/tune_reranker_pool.py --dataset scripts/test_questions_expanded.json --pools 20 30 --k 5 10 20
    python scripts/tune_reranker_pool.py --verbose
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25 import get_bm25_index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import BOFIPVectorStore

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_POOLS = [10, 15, 20, 25, 30, 40, 50]
DEFAULT_KS = [5, 10, 20]
DEFAULT_DATASET = Path(__file__).parent / "test_questions.json"


def _normalize_reference(reference: str) -> str:
    ref = (reference or "").replace("\xa0", " ").replace("’", "'").strip()
    ref = re.sub(r"\s+", " ", ref)
    ref = ref.rstrip(".,;:")
    return ref.upper()


def _matches_expected(reference: str, expected_refs: List[str]) -> bool:
    ref = _normalize_reference(reference)
    if not ref:
        return False
    for expected in expected_refs:
        exp = _normalize_reference(expected)
        if exp and ref.startswith(exp):
            return True
    return False


def _evaluate_query(
    retriever: HybridRetriever,
    query_data: Dict[str, Any],
    ks: List[int],
) -> Dict[str, Any]:
    query = query_data["query"]
    expected = query_data.get("expected_boi", [])
    max_k = max(ks)

    t0 = time.perf_counter()
    results = retriever.search_simple(query, n_results=max_k)
    latency_s = time.perf_counter() - t0

    retrieved_refs = [r.get("metadata", {}).get("boi_reference", "") for r in results]
    per_k: Dict[str, Dict[str, float]] = {}

    for k in ks:
        topk_refs = retrieved_refs[:k]
        hit_count = sum(1 for ref in topk_refs if _matches_expected(ref, expected))
        unique_expected_hits = set()
        for exp in expected:
            if any((ref or "").startswith(exp) for ref in topk_refs):
                unique_expected_hits.add(exp)

        recall = (len(unique_expected_hits) / len(expected)) if expected else 0.0
        precision = (hit_count / k) if k > 0 else 0.0
        hit = 1.0 if hit_count > 0 else 0.0

        per_k[str(k)] = {
            "recall": recall,
            "precision": precision,
            "hit": hit,
        }

    return {
        "id": query_data.get("id", ""),
        "query": query,
        "expected_boi": expected,
        "retrieved_refs": retrieved_refs,
        "latency_s": latency_s,
        "metrics": per_k,
    }


def _aggregate(details: List[Dict[str, Any]], ks: List[int]) -> Dict[str, Any]:
    n = max(1, len(details))
    summary: Dict[str, Any] = {
        "avg_latency_s": sum(d["latency_s"] for d in details) / n,
        "p95_latency_s": sorted(d["latency_s"] for d in details)[min(n - 1, int(0.95 * n))],
        "metrics": {},
    }

    for k in ks:
        key = str(k)
        recall = sum(d["metrics"][key]["recall"] for d in details) / n
        precision = sum(d["metrics"][key]["precision"] for d in details) / n
        hit_rate = sum(d["metrics"][key]["hit"] for d in details) / n
        summary["metrics"][key] = {
            "avg_recall": recall,
            "avg_precision": precision,
            "hit_rate": hit_rate,
        }

    return summary


def _score_candidate(summary: Dict[str, Any], primary_k: int = 5) -> tuple:
    """
    Ranking tuple for best candidate:
    1) maximize Recall@primary_k
    2) maximize Precision@primary_k
    3) minimize avg latency
    """
    m = summary["metrics"].get(str(primary_k), {})
    recall = m.get("avg_recall", 0.0)
    precision = m.get("avg_precision", 0.0)
    latency = summary.get("avg_latency_s", 1e9)
    return (recall, precision, -latency)


def _run_one_setting(
    questions: List[Dict[str, Any]],
    retriever: HybridRetriever,
    ks: List[int],
    rerank_pool_size: int,
) -> Dict[str, Any]:
    retriever.rerank_pool_size = rerank_pool_size
    logger.info(f"Evaluating pool size {rerank_pool_size} on {len(questions)} questions")

    details = []
    for i, q in enumerate(questions, start=1):
        logger.info(f"  [{i}/{len(questions)}] {q.get('id', '?')} - {q.get('query', '')[:70]}")
        details.append(_evaluate_query(retriever, q, ks))

    summary = _aggregate(details, ks)
    return {
        "reranker_enabled": True,
        "rerank_pool_size": rerank_pool_size,
        "summary": summary,
        "details": details,
    }


def _run_no_reranker(
    questions: List[Dict[str, Any]],
    vector_store: BOFIPVectorStore,
    bm25_index,
    ks: List[int],
) -> Dict[str, Any]:
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=None,
        use_reranker=False,
    )
    logger.info(f"Evaluating baseline (no reranker) on {len(questions)} questions")

    details = []
    for i, q in enumerate(questions, start=1):
        logger.info(f"  [{i}/{len(questions)}] {q.get('id', '?')} - {q.get('query', '')[:70]}")
        details.append(_evaluate_query(retriever, q, ks))

    summary = _aggregate(details, ks)
    return {
        "reranker_enabled": False,
        "rerank_pool_size": 0,
        "summary": summary,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune reranker pool size")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="Ground-truth dataset JSON")
    parser.add_argument("--pools", nargs="+", type=int, default=DEFAULT_POOLS, help="Pool sizes to test")
    parser.add_argument("--k", nargs="+", type=int, default=DEFAULT_KS, help="K values (e.g. 5 10 20)")
    parser.add_argument("--verbose", action="store_true", help="Enable per-query logs")
    parser.add_argument("--skip-no-reranker", action="store_true", help="Skip no-reranker baseline")
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.getLogger().setLevel(level)
    logger.setLevel(level)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ks = sorted({k for k in args.k if k > 0})
    pools = sorted({p for p in args.pools if p > 0})
    if not ks:
        raise ValueError("Provide at least one positive K value.")
    if not pools:
        raise ValueError("Provide at least one positive pool size.")

    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in dataset: {dataset_path}")

    vector_store = BOFIPVectorStore()
    bm25_index = get_bm25_index()
    reranker = get_reranker()
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=reranker,
        use_reranker=True,
        rerank_pool_size=pools[0],
    )

    runs = []

    if not args.skip_no_reranker:
        runs.append(_run_no_reranker(questions, vector_store, bm25_index, ks))

    for pool_size in pools:
        runs.append(_run_one_setting(questions, retriever, ks, pool_size))

    reranker_runs = [r for r in runs if r["reranker_enabled"]]
    no_reranker_runs = [r for r in runs if not r["reranker_enabled"]]

    best_reranker_run = max(reranker_runs, key=lambda r: _score_candidate(r["summary"], primary_k=ks[0]))
    best_overall = best_reranker_run
    recommended_mode = "reranker"
    if no_reranker_runs:
        baseline_run = no_reranker_runs[0]
        if _score_candidate(baseline_run["summary"], primary_k=ks[0]) >= _score_candidate(
            best_reranker_run["summary"], primary_k=ks[0]
        ):
            best_overall = baseline_run
            recommended_mode = "no_reranker"

    out = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "questions_count": len(questions),
        "ks": ks,
        "candidates": pools,
        "runs": runs,
        "recommended_mode": recommended_mode,
        "recommended_rerank_pool_size": best_reranker_run["rerank_pool_size"],
        "recommended_overall_pool_size": best_overall["rerank_pool_size"],
        "recommendation_basis": f"max Recall@{ks[0]}, then Precision@{ks[0]}, then min avg latency",
    }

    out_path = Path(__file__).parent.parent / "data" / f"reranker_pool_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("RERANKER POOL TUNING SUMMARY")
    print("=" * 72)
    print(f"Dataset: {dataset_path} | Questions: {len(questions)}")
    print(f"K values: {ks}")
    for run in runs:
        label = "No reranker" if not run["reranker_enabled"] else f"Pool {run['rerank_pool_size']}"
        m = run["summary"]["metrics"][str(ks[0])]
        print(
            f"{label:>12} -> Recall@{ks[0]}: {m['avg_recall']:.1%} | "
            f"Precision@{ks[0]}: {m['avg_precision']:.1%} | "
            f"Latency: {run['summary']['avg_latency_s']:.2f}s"
        )
    print(f"Best reranker pool size: {out['recommended_rerank_pool_size']}")
    print(f"Recommended overall mode: {out['recommended_mode']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""
Retrieval-only evaluation (no LLM) with Recall@K / Precision@K.

Uses ground-truth expectations from scripts/test_questions.json.
This avoids false confidence from end-to-end keyword-only scoring.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --k 5 10 20 --no-reranker
    python scripts/evaluate_retrieval.py --dataset scripts/test_questions_expanded.json --k 5 10 20
    python scripts/evaluate_retrieval.py --dataset scripts/test_questions_expanded.json --verbose
"""

import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.bm25 import get_bm25_index
from src.retrieval.vector_store import BOFIPVectorStore
from src.retrieval.reranker import get_reranker

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KS = [5, 10, 20]
GROUND_TRUTH_FILE = Path(__file__).parent / "test_questions.json"


def _normalize_reference(reference: str) -> str:
    """
    Normalize BOI/CGI/LPF references for tolerant matching.

    - Collapse whitespace
    - Normalize unicode apostrophes/spaces
    - Uppercase
    - Strip trailing punctuation
    """
    ref = (reference or "").replace("\xa0", " ").replace("’", "'").strip()
    ref = re.sub(r"\s+", " ", ref)
    ref = ref.rstrip(".,;:")
    return ref.upper()


def _matches_expected(reference: str, expected_refs: List[str]) -> bool:
    """
    Prefix-based matching to handle dated BOI references and minor formatting variance.
    """
    ref = _normalize_reference(reference)
    if not ref:
        return False
    for expected in expected_refs:
        exp = _normalize_reference(expected)
        if not exp:
            continue
        if ref.startswith(exp):
            return True
    return False


def _evaluate_one_query(
    retriever: HybridRetriever,
    query_data: Dict[str, Any],
    ks: List[int],
) -> Dict[str, Any]:
    query = query_data["query"]
    expected = query_data.get("expected_boi", [])
    max_k = max(ks)

    results = retriever.search_simple(query, n_results=max_k)
    retrieved_refs = [r.get("metadata", {}).get("boi_reference", "") for r in results]

    per_k = {}
    first_hit_rank = None

    for idx, ref in enumerate(retrieved_refs, start=1):
        if _matches_expected(ref, expected):
            first_hit_rank = idx
            break

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
            "hits_count": hit_count,
        }

    return {
        "id": query_data.get("id", ""),
        "query": query,
        "domain": query_data.get("domain", ""),
        "expected_boi": expected,
        "retrieved_refs": retrieved_refs,
        "first_hit_rank": first_hit_rank,
        "metrics": per_k,
    }


def _aggregate(results: List[Dict[str, Any]], ks: List[int]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    n = len(results) if results else 1

    for k in ks:
        key = str(k)
        recall = sum(r["metrics"][key]["recall"] for r in results) / n
        precision = sum(r["metrics"][key]["precision"] for r in results) / n
        hit_rate = sum(r["metrics"][key]["hit"] for r in results) / n
        summary[key] = {
            "avg_recall": recall,
            "avg_precision": precision,
            "hit_rate": hit_rate,
        }

    return summary


def run(
    ks: List[int],
    use_reranker: bool,
    rerank_pool_size: int | None = None,
    dataset_path: Path = GROUND_TRUTH_FILE,
    vector_persist_dir: Path | None = None,
    vector_collection_name: str = "bofip_chunks",
    embedding_model_name: str | None = None,
) -> Tuple[Dict[str, Any], Path]:
    dataset_path = dataset_path.resolve()
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        test_data = json.load(f)

    if isinstance(test_data, dict):
        questions = test_data.get("questions", [])
    elif isinstance(test_data, list):
        questions = test_data
    else:
        questions = []

    if not questions:
        raise ValueError(f"No questions found in {dataset_path}")

    logger.info("Initializing retriever for retrieval-only evaluation...")
    retriever = HybridRetriever(
        vector_store=BOFIPVectorStore(
            persist_dir=vector_persist_dir,
            collection_name=vector_collection_name,
            embedding_model_name=embedding_model_name,
        ),
        bm25_index=get_bm25_index(),
        reranker=get_reranker() if use_reranker else None,
        use_reranker=use_reranker,
        rerank_pool_size=rerank_pool_size,
    )
    actual_pool_size = retriever.rerank_pool_size if use_reranker else 0

    detailed = []
    for i, q in enumerate(questions, start=1):
        logger.info(f"[{i}/{len(questions)}] {q.get('id', '?')} - {q.get('query', '')[:70]}")
        detailed.append(_evaluate_one_query(retriever, q, ks))

    summary = _aggregate(detailed, ks)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "questions_count": len(questions),
        "ks": ks,
        "use_reranker": use_reranker,
        "rerank_pool_size": actual_pool_size,
        "vector_persist_dir": str(vector_persist_dir) if vector_persist_dir else None,
        "vector_collection_name": vector_collection_name,
        "embedding_model_name": embedding_model_name,
        "summary": summary,
        "details": detailed,
    }

    out_path = Path(__file__).parent.parent / "data" / f"eval_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload, out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (Recall@K / Precision@K)")
    parser.add_argument("--k", nargs="+", type=int, default=DEFAULT_KS, help="K values, e.g. --k 5 10 20")
    parser.add_argument("--no-reranker", action="store_true", help="Disable reranker for pure retrieval baseline")
    parser.add_argument("--verbose", action="store_true", help="Enable per-query logs")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=GROUND_TRUTH_FILE,
        help="Path to retrieval dataset JSON (dict with `questions` list or plain list)",
    )
    parser.add_argument(
        "--rerank-pool-size",
        type=int,
        help="Override reranker candidate pool size for this run",
    )
    parser.add_argument(
        "--vector-persist-dir",
        type=Path,
        help="Optional Chroma persistence directory",
    )
    parser.add_argument(
        "--vector-collection-name",
        type=str,
        default="bofip_chunks",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        help="Embedding model name used by vector store/query embeddings",
    )
    args = parser.parse_args()

    ks = sorted({k for k in args.k if k > 0})
    if not ks:
        raise ValueError("Provide at least one positive K value.")

    level = logging.INFO if args.verbose else logging.WARNING
    logging.getLogger().setLevel(level)
    logger.setLevel(level)

    payload, out_path = run(
        ks,
        use_reranker=not args.no_reranker,
        rerank_pool_size=args.rerank_pool_size,
        dataset_path=args.dataset,
        vector_persist_dir=args.vector_persist_dir,
        vector_collection_name=args.vector_collection_name,
        embedding_model_name=args.embedding_model_name,
    )

    print("=" * 72)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 72)
    print(
        f"Questions: {payload['questions_count']} | Reranker: {payload['use_reranker']} "
        f"(pool={payload['rerank_pool_size']})"
    )
    for k in ks:
        m = payload["summary"][str(k)]
        print(
            f"@{k} -> Recall: {m['avg_recall']:.1%} | "
            f"Precision: {m['avg_precision']:.1%} | HitRate: {m['hit_rate']:.1%}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

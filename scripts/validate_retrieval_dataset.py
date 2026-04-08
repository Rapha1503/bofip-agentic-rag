"""
Validate retrieval dataset quality against current retriever behavior.

Main use case:
- Keep gold questions as-is.
- Auto-score silver questions (coverage + hit rank).
- Optionally write a filtered dataset (gold + auto-validated silver).

Usage:
    python scripts/validate_retrieval_dataset.py --dataset scripts/test_questions_expanded.json
    python scripts/validate_retrieval_dataset.py --dataset scripts/test_questions_expanded.json --out scripts/test_questions_validated.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
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


def _evaluate_question(
    retriever: HybridRetriever,
    question: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    query = question.get("query", "")
    expected = question.get("expected_boi", [])
    results = retriever.search_simple(query, n_results=top_k)
    retrieved_refs = [r.get("metadata", {}).get("boi_reference", "") for r in results]

    found_expected = []
    for exp in expected:
        if any(_matches_expected(ref, [exp]) for ref in retrieved_refs):
            found_expected.append(exp)

    first_hit_rank = None
    for idx, ref in enumerate(retrieved_refs, start=1):
        if _matches_expected(ref, expected):
            first_hit_rank = idx
            break

    expected_count = max(1, len(expected))
    coverage = len(found_expected) / expected_count

    return {
        "id": question.get("id", ""),
        "query": query,
        "expected_boi": expected,
        "retrieved_refs": retrieved_refs,
        "found_expected": found_expected,
        "expected_count": len(expected),
        "found_count": len(found_expected),
        "coverage": coverage,
        "first_hit_rank": first_hit_rank,
        "source": question.get("source", "gold"),
    }


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("questions", [])
    if isinstance(data, list):
        return data
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate retrieval dataset entries against current retriever")
    parser.add_argument("--dataset", type=Path, required=True, help="Input dataset JSON")
    parser.add_argument("--out", type=Path, help="Optional filtered dataset output path")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k retrieval depth for validation")
    parser.add_argument("--min-coverage", type=float, default=0.5, help="Minimum expected ref coverage for silver validation")
    parser.add_argument("--max-rank", type=int, default=20, help="Maximum first hit rank to validate silver question")
    parser.add_argument("--no-reranker", action="store_true", help="Validate with reranker disabled")
    parser.add_argument("--rerank-pool-size", type=int, help="Override reranker pool size")
    parser.add_argument("--verbose", action="store_true", help="Enable per-query logs")
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.getLogger().setLevel(level)
    logger.setLevel(level)

    dataset_path = args.dataset.resolve()
    questions = _load_questions(dataset_path)
    if not questions:
        raise ValueError(f"No questions found in dataset: {dataset_path}")

    retriever = HybridRetriever(
        vector_store=BOFIPVectorStore(),
        bm25_index=get_bm25_index(),
        reranker=get_reranker() if not args.no_reranker else None,
        use_reranker=not args.no_reranker,
        rerank_pool_size=args.rerank_pool_size,
    )

    evaluated = []
    for i, q in enumerate(questions, start=1):
        logger.info(f"[{i}/{len(questions)}] {q.get('id', '?')} - {q.get('query', '')[:70]}")
        evaluated.append(_evaluate_question(retriever, q, top_k=max(1, args.top_k)))

    validated_questions: List[Dict[str, Any]] = []
    silver_total = 0
    silver_valid = 0
    gold_total = 0

    by_id = {q.get("id"): q for q in questions}
    for row in evaluated:
        q = dict(by_id.get(row["id"], {}))
        source = q.get("source", "gold")
        is_silver = source == "llm_cache_silver"
        if is_silver:
            silver_total += 1
        else:
            gold_total += 1

        status = "gold_kept"
        if is_silver:
            valid = (
                row["coverage"] >= args.min_coverage
                and row["first_hit_rank"] is not None
                and row["first_hit_rank"] <= args.max_rank
            )
            status = "silver_validated" if valid else "silver_needs_review"
            if valid:
                silver_valid += 1

        q["validation"] = {
            "status": status,
            "coverage": row["coverage"],
            "first_hit_rank": row["first_hit_rank"],
            "found_count": row["found_count"],
            "expected_count": row["expected_count"],
            "top_k": args.top_k,
            "evaluated_at": datetime.now().isoformat(),
            "retriever_reranker": not args.no_reranker,
            "rerank_pool_size": retriever.rerank_pool_size if not args.no_reranker else 0,
        }

        if status in ("gold_kept", "silver_validated"):
            validated_questions.append(q)

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "questions_total": len(questions),
        "gold_total": gold_total,
        "silver_total": silver_total,
        "silver_validated": silver_valid,
        "silver_validation_rate": (silver_valid / silver_total) if silver_total else 0.0,
        "kept_total": len(validated_questions),
        "top_k": args.top_k,
        "min_coverage": args.min_coverage,
        "max_rank": args.max_rank,
        "reranker_enabled": not args.no_reranker,
        "rerank_pool_size": retriever.rerank_pool_size if not args.no_reranker else 0,
        "details": evaluated,
    }

    report_path = Path(__file__).parent.parent / "data" / f"retrieval_dataset_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if args.out:
        out_payload = {
            "version": "validated-1.0",
            "description": "Gold + auto-validated silver retrieval dataset",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "source_dataset": str(dataset_path),
            "validation_report": str(report_path),
            "stats": {
                "questions_total": len(questions),
                "kept_total": len(validated_questions),
                "gold_total": gold_total,
                "silver_total": silver_total,
                "silver_validated": silver_valid,
            },
            "questions": validated_questions,
        }
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
    else:
        out_path = None

    print("=" * 72)
    print("RETRIEVAL DATASET VALIDATION SUMMARY")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Total questions: {len(questions)}")
    print(f"Gold kept: {gold_total}")
    print(f"Silver total: {silver_total}")
    print(f"Silver validated: {silver_valid} ({report['silver_validation_rate']:.1%})")
    print(f"Kept total (gold + validated silver): {len(validated_questions)}")
    print(f"Validation report: {report_path}")
    if out_path:
        print(f"Filtered dataset: {out_path}")


if __name__ == "__main__":
    main()

"""
Embedding model benchmark on retrieval metrics.

Compares embedding models end-to-end on retrieval-only metrics using a shared dataset.
For each model:
1) Ensure a dedicated Chroma collection is indexed with that model.
2) Run retrieval evaluation (`scripts/evaluate_retrieval.py`) against that collection.
3) Save consolidated benchmark report.

Usage:
    python scripts/benchmark_embeddings.py
    python scripts/benchmark_embeddings.py --models intfloat/multilingual-e5-base intfloat/multilingual-e5-large
    python scripts/benchmark_embeddings.py --dataset scripts/test_questions_validated.json --k 5 10 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL
from scripts import evaluate_retrieval
from src.retrieval.embeddings import reset_embedding_models
from src.retrieval.vector_store import BOFIPVectorStore, index_chunks_from_file


logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
]


def _model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip().lower())
    return slug.strip("_") or "default"


def _collection_name(prefix: str, model_name: str) -> str:
    slug = _model_slug(model_name)
    max_slug_len = max(8, 63 - len(prefix) - 1)
    return f"{prefix}_{slug[:max_slug_len]}"


def _resolve_chunks_file(chunks_file: Path) -> Path:
    if chunks_file.is_absolute():
        return chunks_file
    if chunks_file.exists():
        return chunks_file.resolve()
    return (DATA_DIR / "processed" / chunks_file).resolve()


def _load_chunk_count(chunks_file: Path) -> int:
    with open(chunks_file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Chunks file must be a JSON list: {chunks_file}")
    return len(data)


def _ensure_model_index(
    model_name: str,
    expected_count: int,
    chunks_file: Path,
    persist_root: Path,
    collection_prefix: str,
    rebuild: bool,
    allow_index_build: bool,
) -> Tuple[Path, str, bool, float]:
    # Reuse production index for baseline model when counts already match.
    if not rebuild and model_name == EMBEDDING_MODEL:
        prod_store = BOFIPVectorStore(
            persist_dir=CHROMA_DB_DIR,
            collection_name="bofip_chunks",
            embedding_model_name=model_name,
        )
        if prod_store.get_count() == expected_count:
            logger.info("Reusing production Chroma index for baseline model=%s", model_name)
            return CHROMA_DB_DIR, "bofip_chunks", False, 0.0

    persist_dir = persist_root / _model_slug(model_name)
    collection_name = _collection_name(collection_prefix, model_name)
    persist_dir.mkdir(parents=True, exist_ok=True)

    store = BOFIPVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=model_name,
    )
    current_count = store.get_count()
    needs_rebuild = rebuild or current_count != expected_count
    index_seconds = 0.0

    if needs_rebuild:
        if not allow_index_build:
            raise RuntimeError(
                f"Index build required for model '{model_name}' (current={current_count}, expected={expected_count}) "
                "but --allow-index-build is not set."
            )
        logger.warning(
            "Rebuilding index for model=%s (current=%s, expected=%s)",
            model_name,
            current_count,
            expected_count,
        )
        start = time.perf_counter()
        index_chunks_from_file(
            chunks_file=str(chunks_file),
            clear_existing=True,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model_name=model_name,
        )
        index_seconds = time.perf_counter() - start
    else:
        logger.info("Reusing existing index for model=%s (count=%s)", model_name, current_count)

    return persist_dir, collection_name, needs_rebuild, index_seconds


def _run_one_model(
    model_name: str,
    dataset: Path,
    ks: List[int],
    use_reranker: bool,
    rerank_pool_size: int | None,
    expected_count: int,
    chunks_file: Path,
    persist_root: Path,
    collection_prefix: str,
    rebuild: bool,
    allow_index_build: bool,
) -> Dict[str, Any]:
    # Avoid keeping previous model in memory during A/B loop.
    reset_embedding_models()

    persist_dir, collection_name, indexed, index_seconds = _ensure_model_index(
        model_name=model_name,
        expected_count=expected_count,
        chunks_file=chunks_file,
        persist_root=persist_root,
        collection_prefix=collection_prefix,
        rebuild=rebuild,
        allow_index_build=allow_index_build,
    )

    eval_start = time.perf_counter()
    payload, eval_path = evaluate_retrieval.run(
        ks=ks,
        use_reranker=use_reranker,
        rerank_pool_size=rerank_pool_size,
        dataset_path=dataset,
        vector_persist_dir=persist_dir,
        vector_collection_name=collection_name,
        embedding_model_name=model_name,
    )
    eval_seconds = time.perf_counter() - eval_start

    return {
        "model_name": model_name,
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "indexed": indexed,
        "index_seconds": index_seconds,
        "evaluation_seconds": eval_seconds,
        "evaluation_output": str(eval_path),
        "summary": payload["summary"],
        "questions_count": payload["questions_count"],
        "use_reranker": payload["use_reranker"],
        "rerank_pool_size": payload["rerank_pool_size"],
    }


def _winner(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ""

    def score_tuple(item: Dict[str, Any]) -> Tuple[float, float, float]:
        summary = item.get("summary", {})
        r5 = summary.get("5", {}).get("avg_recall", 0.0)
        r10 = summary.get("10", {}).get("avg_recall", 0.0)
        p5 = summary.get("5", {}).get("avg_precision", 0.0)
        return (r5, r10, p5)

    best = max(results, key=score_tuple)
    return best["model_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding models on retrieval metrics.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Embedding model names to benchmark",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "test_questions_validated.json",
        help="Evaluation dataset path",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="K values for retrieval metrics",
    )
    parser.add_argument("--no-reranker", action="store_true", help="Disable reranker during benchmark.")
    parser.add_argument("--rerank-pool-size", type=int, help="Override reranker pool size.")
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("chunks.json"),
        help="Chunks JSON path (relative to data/processed by default).",
    )
    parser.add_argument(
        "--persist-root",
        type=Path,
        default=DATA_DIR / "chroma_db_models",
        help="Directory where per-model Chroma databases are stored.",
    )
    parser.add_argument(
        "--collection-prefix",
        type=str,
        default="bofip_chunks",
        help="Prefix used for model-specific Chroma collection names.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force index rebuild for all models.")
    parser.add_argument(
        "--allow-index-build",
        action="store_true",
        help="Allow automatic/forced index build (can be long). Disabled by default for safety.",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=5.0,
        help="Hard stop for benchmark runtime. Default: 5 minutes.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable info logs.")
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(level)

    ks = sorted({k for k in args.k if k > 0})
    if not ks:
        raise ValueError("Provide at least one positive K value.")

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    chunks_file = _resolve_chunks_file(args.chunks_file)
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    expected_count = _load_chunk_count(chunks_file)
    logger.info("Expected chunk count: %s", expected_count)

    results: List[Dict[str, Any]] = []
    benchmark_start = time.perf_counter()
    for model in args.models:
        elapsed_minutes = (time.perf_counter() - benchmark_start) / 60.0
        if elapsed_minutes >= args.max_runtime_minutes:
            logger.warning(
                "Stopping benchmark due to max runtime limit (%.1f minutes).",
                args.max_runtime_minutes,
            )
            break
        logger.warning("Benchmarking model: %s", model)
        result = _run_one_model(
            model_name=model,
            dataset=dataset_path,
            ks=ks,
            use_reranker=not args.no_reranker,
            rerank_pool_size=args.rerank_pool_size,
            expected_count=expected_count,
            chunks_file=chunks_file,
            persist_root=args.persist_root,
            collection_prefix=args.collection_prefix,
            rebuild=args.rebuild,
            allow_index_build=args.allow_index_build,
        )
        results.append(result)

    total_seconds = time.perf_counter() - benchmark_start
    winner = _winner(results)

    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "ks": ks,
        "models": args.models,
        "results": results,
        "winner_by_recall5_then_recall10_then_precision5": winner,
        "total_seconds": total_seconds,
    }

    out_path = DATA_DIR / f"eval_embedding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("EMBEDDING BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Questions: {results[0]['questions_count'] if results else 0}")
    print(f"Reranker: {not args.no_reranker}")
    for item in results:
        summary = item["summary"]
        metric_parts = []
        for k in ks:
            key = str(k)
            recall = summary.get(key, {}).get("avg_recall", 0.0)
            precision = summary.get(key, {}).get("avg_precision", 0.0)
            metric_parts.append(f"R@{k}={recall:.1%}")
            if k == 5:
                metric_parts.append(f"P@{k}={precision:.1%}")
        print(
            f"- {item['model_name']}\n"
            f"  {' | '.join(metric_parts)}\n"
            f"  Indexed={item['indexed']} ({item['index_seconds']:.1f}s) | Eval={item['evaluation_seconds']:.1f}s"
        )
    print(f"Winner: {winner or 'N/A'}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

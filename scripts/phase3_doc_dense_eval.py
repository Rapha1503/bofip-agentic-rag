from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.dense_retrieval import DEFAULT_DENSE_MODEL, DenseDocumentIndex, DenseEncoder
from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.models import raw_document_from_dict
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def _safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Document-level dense retrieval evaluation.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dense-mode", type=str, default="sections_firstpara", choices=["base", "sections", "sections_firstpara"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--cache-prefix", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    queries = read_jsonl(Path(args.queries))
    ks = sorted(set(args.top_k))

    print(f"[doc-dense] loading model: {args.model}")
    model_start = time.time()
    encoder = DenseEncoder(args.model, device=args.device)
    model_elapsed = time.time() - model_start
    print(f"[doc-dense] model loaded in {model_elapsed:.2f}s")

    cache_prefix = Path(args.cache_prefix).resolve() if args.cache_prefix else None
    cache_npy = cache_prefix.with_suffix(".npy") if cache_prefix else None
    cache_meta = cache_prefix.with_suffix(".json") if cache_prefix else None

    if cache_npy and cache_npy.exists():
        print(f"[doc-dense] loading cached embeddings: {cache_npy}")
        encode_docs_start = time.time()
        document_embeddings = np.load(cache_npy)
        doc_encode_elapsed = time.time() - encode_docs_start
    else:
        print(f"[doc-dense] encoding {len(documents)} documents")
        encode_docs_start = time.time()
        document_embeddings = encoder.encode_documents(
            documents,
            mode=args.dense_mode,
            batch_size=args.batch_size,
            show_progress_bar=args.show_progress,
        )
        doc_encode_elapsed = time.time() - encode_docs_start
        if cache_npy:
            cache_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_npy, document_embeddings)
            write_json(
                cache_meta,
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "raw_docs_path": str(Path(args.raw_docs).resolve()),
                    "model_name": args.model,
                    "dense_mode": args.dense_mode,
                    "embedding_shape": list(document_embeddings.shape),
                    "boi_references": [document.boi_reference for document in documents],
                },
            )
            print(f"[doc-dense] cached embeddings written to: {cache_npy}")
    print(f"[doc-dense] document embeddings ready in {doc_encode_elapsed:.2f}s")

    query_texts = [payload["query"] for payload in queries]
    print(f"[doc-dense] encoding {len(query_texts)} queries")
    encode_queries_start = time.time()
    query_embeddings = encoder.encode_queries(query_texts, batch_size=args.batch_size, show_progress_bar=args.show_progress)
    query_encode_elapsed = time.time() - encode_queries_start
    print(f"[doc-dense] query encoding done in {query_encode_elapsed:.2f}s")

    index = DenseDocumentIndex(documents, document_embeddings)
    hits_by_k = {k: 0 for k in ks}
    supported_query_count = 0
    unsupported_query_count = 0
    results: list[dict] = []

    search_start = time.time()
    for payload, query_embedding in zip(queries, query_embeddings):
        query = payload["query"]
        expected = payload.get("expected_boi")
        supported = bool(expected)
        if supported:
            supported_query_count += 1
        else:
            unsupported_query_count += 1

        hits = index.search_from_vector(query_embedding, top_k=max(ks))
        returned = [hit.boi_reference for hit in hits]
        row = {
            "id": payload["id"],
            "pattern": payload.get("pattern"),
            "query": query,
            "expected_boi": expected,
            "supported_query": supported,
            "returned_boi": returned,
            "top_hits": [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "boi_reference": hit.boi_reference,
                    "title": hit.document.title,
                }
                for hit in hits
            ],
        }
        if supported:
            for k in ks:
                matched = expected in returned[:k]
                row[f"hit@{k}"] = matched
                if matched:
                    hits_by_k[k] += 1
        results.append(row)
    search_elapsed = time.time() - search_start
    print(f"[doc-dense] scored {len(queries)} queries in {search_elapsed:.2f}s")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "model_name": args.model,
        "device": args.device,
        "dense_mode": args.dense_mode,
        "batch_size": args.batch_size,
        "document_count": len(documents),
        "query_count": len(queries),
        "supported_query_count": supported_query_count,
        "unsupported_query_count": unsupported_query_count,
        "timings": {
            "model_load_seconds": round(model_elapsed, 2),
            "document_encode_seconds": round(doc_encode_elapsed, 2),
            "query_encode_seconds": round(query_encode_elapsed, 2),
            "search_seconds": round(search_elapsed, 2),
        },
        "metrics": {
            f"hit@{k}": round(hits_by_k[k] / supported_query_count, 4) if supported_query_count else 0.0
            for k in ks
        },
        "results": results,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"phase3_doc_dense_eval_{Path(args.raw_docs).stem}__{Path(args.queries).stem}__{args.dense_mode}__{_safe_model_name(args.model)}.json"
    )
    write_json(report_path, summary)
    print(f"Document dense evaluation complete: {report_path}")
    for k in ks:
        print(f"hit@{k} = {summary['metrics'][f'hit@{k}']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

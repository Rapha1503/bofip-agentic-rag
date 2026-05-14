from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.dense_retrieval import DEFAULT_DENSE_MODEL, DenseDocumentIndex, DenseEncoder, DenseIndex
from bofip_cleanroom.hybrid_retrieval import (
    RankedDoc,
    compute_source_rank_profiles,
    confidence_weighted_reciprocal_rank_fuse,
    reciprocal_rank_fuse,
)
from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.lexical_retrieval import DocumentLexicalIndex, get_document_search_text_fn, tokenize
from bofip_cleanroom.models import chunk_node_from_dict, raw_document_from_dict
from bofip_cleanroom.specificity_rerank import SpecificityReranker
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs
from bofip_cleanroom.alias_expansion import build_acronym_expansion_map, expand_query_with_acronyms


def _safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _parse_lexical_mode_spec(spec: str) -> tuple[str, bool]:
    if spec.endswith("_stem"):
        return spec[:-5], True
    return spec, False


def _title_rerank_hits(
    *,
    query: str,
    hits,
    documents_by_ref: dict[str, object],
    rerank_top_n: int,
    rerank_weight: float,
    rerank_stem: bool,
    rank_constant: int,
):
    if rerank_top_n <= 1 or rerank_weight <= 0.0:
        return list(hits)

    candidate_hits = list(hits[:rerank_top_n])
    candidate_docs = [documents_by_ref[hit.boi_reference] for hit in candidate_hits if hit.boi_reference in documents_by_ref]
    if len(candidate_docs) <= 1:
        return list(hits)

    local_index = DocumentLexicalIndex(
        candidate_docs,
        search_text_fn=get_document_search_text_fn("title"),
        tokenize_fn=(lambda text: tokenize(text, stem=True)) if rerank_stem else None,
    )
    local_hits = local_index.search_documents(query, top_k=len(candidate_docs))
    local_rank_scores = {
        hit.boi_reference: 1.0 / (rank_constant + hit.rank)
        for hit in local_hits
    }
    reranked = []
    for hit in hits:
        local_bonus = rerank_weight * local_rank_scores.get(hit.boi_reference, 0.0)
        reranked.append((hit.score + local_bonus, local_rank_scores.get(hit.boi_reference, 0.0), hit))

    ordered = sorted(reranked, key=lambda item: (item[0], item[1]), reverse=True)
    return [
        type(hit)(
            rank=index + 1,
            boi_reference=hit.boi_reference,
            score=score,
            sources=hit.sources,
            ranks=hit.ranks,
        )
        for index, (score, _, hit) in enumerate(ordered)
    ]


def _report_filename(
    *,
    raw_docs: Path,
    queries: Path,
    modes_tag: str,
    dense_mode: str,
    fusion_tag: str,
    model_name: str,
    weights_tag: str,
    candidate_k: int,
) -> str:
    base_name = (
        f"phase3_doc_multiview_hybrid_eval_{raw_docs.stem}__{queries.stem}"
        f"__lex{modes_tag}__dense{dense_mode}__cand{candidate_k}"
        f"__{fusion_tag}__{model_name}__{weights_tag}.json"
    )
    if len(base_name) <= 220:
        return base_name

    config_string = "|".join([modes_tag, dense_mode, str(candidate_k), fusion_tag, model_name, weights_tag])
    digest = hashlib.sha1(config_string.encode("utf-8")).hexdigest()[:12]
    return (
        f"phase3_doc_multiview_hybrid_eval_{raw_docs.stem}__{queries.stem}"
        f"__cfg_{digest}.json"
    )


def _parse_weights(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value.strip())
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Document-level multiview hybrid evaluation.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chunk-dense-model", type=str, default="")
    parser.add_argument("--chunk-dense-device", type=str, default=None)
    parser.add_argument("--lexical-modes", type=str, default="base,sections")
    parser.add_argument("--stem-lexical", action="store_true")
    parser.add_argument("--dense-mode", type=str, default="sections_firstpara", choices=["base", "sections", "sections_firstpara"])
    parser.add_argument("--weights", type=str, default="base=1,sections=1,dense=2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--query-acronym-expansion", action="store_true")
    parser.add_argument("--query-acronym-max-expansions", type=int, default=3)
    parser.add_argument("--local-title-rerank-top-n", type=int, default=0)
    parser.add_argument("--local-title-rerank-weight", type=float, default=0.0)
    parser.add_argument("--local-title-rerank-stem", action="store_true")
    parser.add_argument("--specificity-rerank-top-n", type=int, default=0)
    parser.add_argument("--specificity-rerank-weight", type=float, default=0.0)
    parser.add_argument("--fusion-mode", type=str, default="rrf", choices=["rrf", "confidence"])
    parser.add_argument("--confidence-top-n", type=int, default=5)
    parser.add_argument("--confidence-alpha", type=float, default=1.0)
    parser.add_argument("--score-alpha", type=float, default=0.5)
    parser.add_argument("--cache-prefix", type=str, default="")
    parser.add_argument("--chunk-dense-cache", type=str, default="")
    parser.add_argument("--chunks", type=str, default="")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    documents_by_ref = {document.boi_reference: document for document in documents}
    specificity_reranker = SpecificityReranker(documents)
    queries = read_jsonl(Path(args.queries))
    ks = sorted(set(args.top_k))
    lexical_modes = [mode.strip() for mode in args.lexical_modes.split(",") if mode.strip()]
    weights = _parse_weights(args.weights)

    lexical_indexes = {}
    for mode_spec in lexical_modes:
        base_mode, mode_stem = _parse_lexical_mode_spec(mode_spec)
        lexical_indexes[mode_spec] = DocumentLexicalIndex(
            documents,
            search_text_fn=get_document_search_text_fn(base_mode),
            tokenize_fn=(lambda text: tokenize(text, stem=True)) if (args.stem_lexical or mode_stem) else None,
        )

    acronym_map = (
        build_acronym_expansion_map(documents)
        if args.query_acronym_expansion
        else {}
    )

    print(f"[doc-multiview] loading document-dense model: {args.model}")
    model_start = time.time()
    encoder = DenseEncoder(args.model, device=args.device)
    model_elapsed = time.time() - model_start
    print(f"[doc-multiview] document-dense model loaded in {model_elapsed:.2f}s")

    cache_prefix = Path(args.cache_prefix).resolve() if args.cache_prefix else None
    cache_npy = cache_prefix.with_suffix(".npy") if cache_prefix else None
    if cache_npy and cache_npy.exists():
        print(f"[doc-multiview] loading cached embeddings: {cache_npy}")
        encode_docs_start = time.time()
        document_embeddings = np.load(cache_npy)
        doc_encode_elapsed = time.time() - encode_docs_start
    else:
        print(f"[doc-multiview] encoding {len(documents)} documents")
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
            print(f"[doc-multiview] cached embeddings written to: {cache_npy}")
    print(f"[doc-multiview] document embeddings ready in {doc_encode_elapsed:.2f}s")

    query_texts = [payload["query"] for payload in queries]
    print(f"[doc-multiview] encoding {len(query_texts)} document-dense queries")
    encode_queries_start = time.time()
    query_embeddings = encoder.encode_queries(query_texts, batch_size=args.batch_size, show_progress_bar=args.show_progress)
    query_encode_elapsed = time.time() - encode_queries_start
    print(f"[doc-multiview] document-dense query encoding done in {query_encode_elapsed:.2f}s")

    dense_index = DenseDocumentIndex(documents, document_embeddings)
    chunk_dense_index = None
    chunk_query_embeddings = None
    chunk_query_encode_elapsed = 0.0
    chunk_dense_model_name = None
    chunk_dense_source_enabled = bool(args.chunk_dense_cache or args.chunks)
    if chunk_dense_source_enabled:
        if not args.chunk_dense_cache or not args.chunks:
            raise ValueError("Both --chunk-dense-cache and --chunks are required to enable chunk-dense source")
        print(f"[doc-multiview] loading chunk-dense cache: {args.chunk_dense_cache}")
        chunk_dense_embeddings = np.load(Path(args.chunk_dense_cache).resolve())
        chunk_nodes = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
        chunk_dense_index = DenseIndex(chunk_nodes, chunk_dense_embeddings)
        chunk_dense_model_name = args.chunk_dense_model or args.model
        if chunk_dense_model_name == args.model:
            chunk_query_embeddings = query_embeddings
        else:
            print(f"[doc-multiview] loading chunk-dense query model: {chunk_dense_model_name}")
            chunk_model_start = time.time()
            chunk_encoder = DenseEncoder(chunk_dense_model_name, device=args.chunk_dense_device or args.device)
            chunk_model_elapsed = time.time() - chunk_model_start
            print(f"[doc-multiview] chunk-dense query model loaded in {chunk_model_elapsed:.2f}s")
            print(f"[doc-multiview] encoding {len(query_texts)} chunk-dense queries")
            chunk_query_start = time.time()
            chunk_query_embeddings = chunk_encoder.encode_queries(
                query_texts,
                batch_size=args.batch_size,
                show_progress_bar=args.show_progress,
            )
            chunk_query_encode_elapsed = time.time() - chunk_query_start
            print(f"[doc-multiview] chunk-dense query encoding done in {chunk_query_encode_elapsed:.2f}s")

    hits_by_k = {k: 0 for k in ks}
    supported_query_count = 0
    unsupported_query_count = 0
    results: list[dict] = []

    search_start = time.time()
    for query_idx, (payload, query_embedding) in enumerate(zip(queries, query_embeddings)):
        query = payload["query"]
        lexical_query = query
        acronym_expansions: list[tuple[str, str]] = []
        if acronym_map:
            lexical_query, acronym_expansions = expand_query_with_acronyms(
                query,
                acronym_map,
                max_expansions_per_query=args.query_acronym_max_expansions,
            )
        expected = payload.get("expected_boi")
        supported = bool(expected)
        if supported:
            supported_query_count += 1
        else:
            unsupported_query_count += 1

        rankings = {
            mode: [
                RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source=mode)
                for hit in index.search_documents(lexical_query, top_k=args.candidate_k)
            ]
            for mode, index in lexical_indexes.items()
        }
        rankings["dense"] = [
            RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source="dense")
            for hit in dense_index.search_from_vector(query_embedding, top_k=args.candidate_k)
        ]
        if chunk_dense_index is not None:
            chunk_query_embedding = chunk_query_embeddings[query_idx]
            rankings["chunk_dense"] = [
                RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source="chunk_dense")
                for hit in chunk_dense_index.search_documents_from_vector(chunk_query_embedding, top_k=args.candidate_k)
            ]

        profiles = compute_source_rank_profiles(rankings, top_n=args.confidence_top_n)
        fused_top_k = max(ks)
        if args.local_title_rerank_top_n > 0:
            fused_top_k = max(fused_top_k, args.local_title_rerank_top_n)

        if args.fusion_mode == "confidence":
            fused = confidence_weighted_reciprocal_rank_fuse(
                rankings,
                top_k=fused_top_k,
                rank_constant=args.rank_constant,
                source_weights=weights,
                confidence_top_n=args.confidence_top_n,
                confidence_alpha=args.confidence_alpha,
                score_alpha=args.score_alpha,
            )
        else:
            fused = reciprocal_rank_fuse(
                rankings,
                top_k=fused_top_k,
                rank_constant=args.rank_constant,
                source_weights=weights,
            )
        if args.local_title_rerank_top_n > 0 and args.local_title_rerank_weight > 0.0:
            fused = _title_rerank_hits(
                query=query,
                hits=fused,
                documents_by_ref=documents_by_ref,
                rerank_top_n=args.local_title_rerank_top_n,
                rerank_weight=args.local_title_rerank_weight,
                rerank_stem=args.local_title_rerank_stem,
                rank_constant=args.rank_constant,
            )
        if args.specificity_rerank_top_n > 1 and args.specificity_rerank_weight > 0.0:
            fused = specificity_reranker.rerank_hits(
                query,
                fused,
                get_reference=lambda hit: hit.boi_reference,
                get_score=lambda hit: hit.score,
                clone_hit=lambda hit, rank, score: type(hit)(
                    rank=rank,
                    boi_reference=hit.boi_reference,
                    score=score,
                    sources=hit.sources,
                    ranks=hit.ranks,
                ),
                top_n=min(args.specificity_rerank_top_n, len(fused)),
                weight=args.specificity_rerank_weight,
            )
        returned = [hit.boi_reference for hit in fused]
        row = {
            "id": payload["id"],
            "pattern": payload.get("pattern"),
            "query": query,
            "lexical_query": lexical_query,
            "acronym_expansions": [
                {"acronym": acronym, "phrase": phrase}
                for acronym, phrase in acronym_expansions
            ],
            "expected_boi": expected,
            "supported_query": supported,
            "returned_boi": returned,
            "top_hits": [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "boi_reference": hit.boi_reference,
                    "sources": hit.sources,
                    "ranks": hit.ranks,
                }
                for hit in fused
            ],
            "source_confidences": {
                source_name: round(profile.confidence, 6)
                for source_name, profile in profiles.items()
            },
        }
        if supported:
            for k in ks:
                matched = expected in returned[:k]
                row[f"hit@{k}"] = matched
                if matched:
                    hits_by_k[k] += 1
        results.append(row)
    search_elapsed = time.time() - search_start
    print(f"[doc-multiview] scored {len(queries)} queries in {search_elapsed:.2f}s")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "model_name": args.model,
        "device": args.device,
        "lexical_modes": lexical_modes,
        "dense_mode": args.dense_mode,
        "stem_lexical": args.stem_lexical,
        "chunk_dense_enabled": chunk_dense_index is not None,
        "chunk_dense_model_name": chunk_dense_model_name,
        "chunk_dense_cache": str(Path(args.chunk_dense_cache).resolve()) if args.chunk_dense_cache else None,
        "chunks_path": str(Path(args.chunks).resolve()) if args.chunks else None,
        "weights": weights,
        "candidate_k": args.candidate_k,
        "rank_constant": args.rank_constant,
        "query_acronym_expansion": bool(acronym_map),
        "query_acronym_max_expansions": args.query_acronym_max_expansions,
        "acronym_map_size": len(acronym_map),
        "local_title_rerank_top_n": args.local_title_rerank_top_n,
        "local_title_rerank_weight": args.local_title_rerank_weight,
        "local_title_rerank_stem": args.local_title_rerank_stem,
        "specificity_rerank_top_n": args.specificity_rerank_top_n,
        "specificity_rerank_weight": args.specificity_rerank_weight,
        "fusion_mode": args.fusion_mode,
        "confidence_top_n": args.confidence_top_n,
        "confidence_alpha": args.confidence_alpha,
        "score_alpha": args.score_alpha,
        "document_count": len(documents),
        "query_count": len(queries),
        "supported_query_count": supported_query_count,
        "unsupported_query_count": unsupported_query_count,
        "timings": {
            "model_load_seconds": round(model_elapsed, 2),
            "document_encode_seconds": round(doc_encode_elapsed, 2),
            "query_encode_seconds": round(query_encode_elapsed, 2),
            "chunk_query_encode_seconds": round(chunk_query_encode_elapsed, 2),
            "search_seconds": round(search_elapsed, 2),
        },
        "metrics": {
            f"hit@{k}": round(hits_by_k[k] / supported_query_count, 4) if supported_query_count else 0.0
            for k in ks
        },
        "results": results,
    }

    modes_tag = "_".join(lexical_modes)
    weights_tag = "_".join(f"{name}{str(value).replace('.', 'p')}" for name, value in sorted(weights.items()))
    fusion_tag = args.fusion_mode
    if args.fusion_mode == "confidence":
        fusion_tag = (
            f"{fusion_tag}_top{args.confidence_top_n}"
            f"_ca{str(args.confidence_alpha).replace('.', 'p')}"
            f"_sa{str(args.score_alpha).replace('.', 'p')}"
        )
    if args.stem_lexical:
        fusion_tag = f"{fusion_tag}_stem"
    if args.query_acronym_expansion:
        fusion_tag = f"{fusion_tag}_acr{args.query_acronym_max_expansions}"
    if args.local_title_rerank_top_n > 0 and args.local_title_rerank_weight > 0.0:
        fusion_tag = (
            f"{fusion_tag}_ltr{args.local_title_rerank_top_n}"
            f"_w{str(args.local_title_rerank_weight).replace('.', 'p')}"
            f"{'_stem' if args.local_title_rerank_stem else ''}"
        )
    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / _report_filename(
            raw_docs=Path(args.raw_docs),
            queries=Path(args.queries),
            modes_tag=modes_tag,
            dense_mode=args.dense_mode,
            fusion_tag=fusion_tag,
            model_name=_safe_model_name(args.model),
            weights_tag=weights_tag,
            candidate_k=args.candidate_k,
        )
    )
    write_json(report_path, summary)
    print(f"Document multiview hybrid evaluation complete: {report_path}")
    for k in ks:
        print(f"hit@{k} = {summary['metrics'][f'hit@{k}']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

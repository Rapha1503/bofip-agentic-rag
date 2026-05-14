from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.family_guided_retrieval import FamilyGuidedRetriever, PriorDocumentHit
from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.models import chunk_node_from_dict, raw_document_from_dict
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate family-guided document rerank and chunk retrieval from a stage-1 report.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--stage1-report", type=str, required=True)
    parser.add_argument("--family-top-docs", type=int, default=1)
    parser.add_argument("--max-family-docs", type=int, default=25)
    parser.add_argument("--ancestor-expansion-levels", type=int, default=0)
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument("--chunks-per-doc", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--family-doc-mode", type=str, default="sections_leads")
    parser.add_argument("--no-family-doc-stem", action="store_true")
    parser.add_argument("--chunk-mode", type=str, default="body", choices=["full", "leaf", "body"])
    parser.add_argument("--family-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.25)
    parser.add_argument("--tail-weight", type=float, default=0.0)
    parser.add_argument("--rank-constant", type=int, default=20)
    parser.add_argument("--overview-weight", type=float, default=0.0)
    parser.add_argument("--overview-min-descendants", type=int, default=2)
    parser.add_argument("--overview-top-family-ranks", type=int, default=6)
    parser.add_argument("--preserve-stage1-top1", action="store_true")
    parser.add_argument("--specificity-rerank-top-n", type=int, default=0)
    parser.add_argument("--specificity-rerank-weight", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    stage1 = json.loads(Path(args.stage1_report).read_text(encoding="utf-8"))

    retriever = FamilyGuidedRetriever(
        documents,
        chunks,
        family_doc_mode=args.family_doc_mode,
        family_doc_stem=not args.no_family_doc_stem,
        local_chunk_mode=args.chunk_mode,
    )

    supported_rows = [row for row in stage1["results"] if row.get("supported_query")]
    doc_hits_by_k = {1: 0, 3: 0, 5: 0}
    chunk_hits_by_k = {1: 0, 3: 0, 5: 0}
    expected_in_family = 0
    stage1_exact_top1 = 0
    improved_top1 = 0
    worsened_top1 = 0
    rows: list[dict] = []

    for row in supported_rows:
        expected = row["expected_boi"]
        stage1_hits = [
            PriorDocumentHit(
                rank=hit["rank"],
                score=float(hit["score"]),
                boi_reference=hit["boi_reference"],
            )
            for hit in row.get("top_hits", [])
        ]
        if row.get("hit@1"):
            stage1_exact_top1 += 1

        result = retriever.search(
            row["query"],
            lexical_query=row.get("lexical_query"),
            stage1_hits=stage1_hits,
            family_top_docs=args.family_top_docs,
            max_family_docs=args.max_family_docs,
            ancestor_expansion_levels=args.ancestor_expansion_levels,
            top_docs=args.top_docs,
            chunks_per_doc=args.chunks_per_doc,
            max_chunks=args.max_chunks,
            family_weight=args.family_weight,
            prior_weight=args.prior_weight,
            tail_weight=args.tail_weight,
            rank_constant=args.rank_constant,
            overview_weight=args.overview_weight,
            overview_min_descendants=args.overview_min_descendants,
            overview_top_family_ranks=args.overview_top_family_ranks,
            preserve_stage1_top1=args.preserve_stage1_top1,
            specificity_rerank_top_n=args.specificity_rerank_top_n,
            specificity_rerank_weight=args.specificity_rerank_weight,
        )

        reranked_doc_refs = [hit.boi_reference for hit in result.document_hits]
        reranked_chunk_doc_refs = [hit.boi_reference for hit in result.chunk_hits]
        if expected in result.family_selection.members:
            expected_in_family += 1
        for k in doc_hits_by_k:
            if expected in reranked_doc_refs[:k]:
                doc_hits_by_k[k] += 1
            if expected in reranked_chunk_doc_refs[:k]:
                chunk_hits_by_k[k] += 1

        family_hit_at_1 = bool(reranked_doc_refs and reranked_doc_refs[0] == expected)
        if not row.get("hit@1") and family_hit_at_1:
            improved_top1 += 1
        if row.get("hit@1") and not family_hit_at_1:
            worsened_top1 += 1

        rows.append(
            {
                "id": row["id"],
                "pattern": row.get("pattern"),
                "query": row["query"],
                "lexical_query": row.get("lexical_query"),
                "expected_boi": expected,
                "stage1_top_hits": row.get("top_hits", [])[:5],
                "family_anchor_bois": result.family_selection.anchor_references,
                "family_prefixes": [list(prefix) for prefix in result.family_selection.prefixes],
                "family_size": len(result.family_selection.members),
                "expected_in_family": expected in result.family_selection.members,
                "family_document_hits": [
                    {
                        "rank": hit.rank,
                        "boi_reference": hit.boi_reference,
                        "combined_score": round(hit.combined_score, 6),
                        "family_rank": hit.family_rank,
                        "family_score": round(hit.family_score, 6),
                        "prior_rank": hit.prior_rank,
                        "prior_score": round(hit.prior_score, 6) if hit.prior_score is not None else None,
                        "tail_rank": hit.tail_rank,
                        "tail_score": round(hit.tail_score, 6) if hit.tail_score is not None else None,
                        "descendant_count": hit.descendant_count,
                        "descendant_support": round(hit.descendant_support, 6),
                        "title": hit.title,
                    }
                    for hit in result.document_hits
                ],
                "chunk_hits": [
                    {
                        "global_rank": hit.global_rank,
                        "document_rank": hit.document_rank,
                        "local_rank": hit.local_rank,
                        "document_score": round(hit.document_score, 6),
                        "local_score": round(hit.local_score, 6),
                        "boi_reference": hit.boi_reference,
                        "chunk_id": hit.chunk.chunk_id,
                        "chunk_kind": hit.chunk.chunk_kind,
                        "section_path": " > ".join(hit.chunk.section_path),
                        "text": hit.chunk.text[:500],
                    }
                    for hit in result.chunk_hits
                ],
                "stage1_hit@1": bool(row.get("hit@1")),
                "family_doc_hit@1": family_hit_at_1,
                "family_doc_hit@3": expected in reranked_doc_refs[:3],
                "family_doc_hit@5": expected in reranked_doc_refs[:5],
                "chunk_expected_doc_hit@1": expected in reranked_chunk_doc_refs[:1],
                "chunk_expected_doc_hit@3": expected in reranked_chunk_doc_refs[:3],
                "chunk_expected_doc_hit@5": expected in reranked_chunk_doc_refs[:5],
            }
        )

    supported_count = len(supported_rows)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "chunks_path": str(Path(args.chunks).resolve()),
        "stage1_report_path": str(Path(args.stage1_report).resolve()),
        "family_top_docs": args.family_top_docs,
        "max_family_docs": args.max_family_docs,
        "ancestor_expansion_levels": args.ancestor_expansion_levels,
        "family_doc_mode": args.family_doc_mode,
        "family_doc_stem": not args.no_family_doc_stem,
        "chunk_mode": args.chunk_mode,
        "family_weight": args.family_weight,
        "prior_weight": args.prior_weight,
        "tail_weight": args.tail_weight,
        "rank_constant": args.rank_constant,
        "overview_weight": args.overview_weight,
        "overview_min_descendants": args.overview_min_descendants,
        "overview_top_family_ranks": args.overview_top_family_ranks,
        "preserve_stage1_top1": args.preserve_stage1_top1,
        "specificity_rerank_top_n": args.specificity_rerank_top_n,
        "specificity_rerank_weight": args.specificity_rerank_weight,
        "top_docs": args.top_docs,
        "chunks_per_doc": args.chunks_per_doc,
        "max_chunks": args.max_chunks,
        "supported_query_count": supported_count,
        "stage1_exact_hit@1": round(stage1_exact_top1 / supported_count, 4) if supported_count else 0.0,
        "expected_in_family_rate": round(expected_in_family / supported_count, 4) if supported_count else 0.0,
        "metrics": {
            **{
                f"family_doc_hit@{k}": round(doc_hits_by_k[k] / supported_count, 4) if supported_count else 0.0
                for k in sorted(doc_hits_by_k)
            },
            **{
                f"chunk_expected_doc_hit@{k}": round(chunk_hits_by_k[k] / supported_count, 4) if supported_count else 0.0
                for k in sorted(chunk_hits_by_k)
            },
        },
        "improved_top1_count": improved_top1,
        "worsened_top1_count": worsened_top1,
        "rows": rows,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"phase4_family_guided_eval_{Path(args.stage1_report).stem}__ftop{args.family_top_docs}__cfg.json"
    )
    write_json(report_path, summary)
    print(f"Family-guided eval complete: {report_path}")
    for key, value in summary["metrics"].items():
        print(f"{key} = {value:.4f}")
    print(f"improved_top1_count = {summary['improved_top1_count']}")
    print(f"worsened_top1_count = {summary['worsened_top1_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

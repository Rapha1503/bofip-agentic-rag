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
from bofip_cleanroom.passage_eval import chunk_matches_passage_gold
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate document and passage retrieval against a manually curated passage gold set.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--stage1-report", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--family-top-docs", type=int, default=2)
    parser.add_argument("--max-family-docs", type=int, default=25)
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
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    stage1 = json.loads(Path(args.stage1_report).read_text(encoding="utf-8"))
    queries = read_jsonl(Path(args.queries))
    retriever = FamilyGuidedRetriever(
        documents,
        chunks,
        family_doc_mode=args.family_doc_mode,
        family_doc_stem=not args.no_family_doc_stem,
        local_chunk_mode=args.chunk_mode,
    )

    ks = sorted(set(args.top_k))
    stage1_doc_hits = {k: 0 for k in ks}
    family_doc_hits = {k: 0 for k in ks}
    chunk_doc_hits = {k: 0 for k in ks}
    passage_hits = {k: 0 for k in ks}
    expected_in_family = 0
    rows: list[dict] = []

    for gold_row in queries:
        source_query_id = gold_row.get("source_query_id")
        stage1_row = next((row for row in stage1["results"] if row["id"] == gold_row["id"]), None)
        if stage1_row is None and source_query_id:
            stage1_row = next((row for row in stage1["results"] if row["id"] == source_query_id), None)
        if stage1_row is None:
            raise KeyError(
                f"Stage-1 report does not contain matching query id for passage row {gold_row['id']}"
            )
        if not stage1_row.get("supported_query", True):
            raise ValueError(f"Passage gold row {gold_row['id']} points to unsupported query {source_query_id}")

        expected = gold_row["expected_boi"]
        stage1_hits = [
            PriorDocumentHit(
                rank=hit["rank"],
                score=float(hit["score"]),
                boi_reference=hit["boi_reference"],
            )
            for hit in stage1_row.get("top_hits", [])
        ]

        result = retriever.search(
            gold_row["query"],
            lexical_query=stage1_row.get("lexical_query"),
            stage1_hits=stage1_hits,
            family_top_docs=args.family_top_docs,
            max_family_docs=args.max_family_docs,
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
        )

        stage1_doc_refs = [hit.boi_reference for hit in stage1_hits]
        family_doc_refs = [hit.boi_reference for hit in result.document_hits]
        chunk_doc_refs = [hit.boi_reference for hit in result.chunk_hits]
        passage_match_ranks = [
            hit.global_rank
            for hit in result.chunk_hits
            if chunk_matches_passage_gold(hit.chunk, gold_row)
        ]
        if expected in result.family_selection.members:
            expected_in_family += 1

        for k in ks:
            if expected in stage1_doc_refs[:k]:
                stage1_doc_hits[k] += 1
            if expected in family_doc_refs[:k]:
                family_doc_hits[k] += 1
            if expected in chunk_doc_refs[:k]:
                chunk_doc_hits[k] += 1
            if any(rank <= k for rank in passage_match_ranks):
                passage_hits[k] += 1

        rows.append(
            {
                "id": gold_row["id"],
                "source_query_id": source_query_id,
                "pattern": gold_row.get("pattern"),
                "query": gold_row["query"],
                "expected_boi": expected,
                "stage1_top_hits": stage1_row.get("top_hits", [])[:5],
                "expected_in_family": expected in result.family_selection.members,
                "family_anchor_bois": result.family_selection.anchor_references,
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
                        "boi_reference": hit.boi_reference,
                        "chunk_id": hit.chunk.chunk_id,
                        "chunk_kind": hit.chunk.chunk_kind,
                        "section_path": " > ".join(hit.chunk.section_path),
                        "text": hit.chunk.text[:500],
                        "passage_match": chunk_matches_passage_gold(hit.chunk, gold_row),
                    }
                    for hit in result.chunk_hits
                ],
                **{f"stage1_doc_hit@{k}": expected in stage1_doc_refs[:k] for k in ks},
                **{f"family_doc_hit@{k}": expected in family_doc_refs[:k] for k in ks},
                **{f"chunk_expected_doc_hit@{k}": expected in chunk_doc_refs[:k] for k in ks},
                **{f"passage_hit@{k}": any(rank <= k for rank in passage_match_ranks) for k in ks},
                "first_passage_match_rank": min(passage_match_ranks) if passage_match_ranks else None,
            }
        )

    query_count = len(queries)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "chunks_path": str(Path(args.chunks).resolve()),
        "stage1_report_path": str(Path(args.stage1_report).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "family_top_docs": args.family_top_docs,
        "max_family_docs": args.max_family_docs,
        "top_docs": args.top_docs,
        "chunks_per_doc": args.chunks_per_doc,
        "max_chunks": args.max_chunks,
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
        "query_count": query_count,
        "expected_in_family_rate": round(expected_in_family / query_count, 4) if query_count else 0.0,
        "metrics": {
            **{f"stage1_doc_hit@{k}": round(stage1_doc_hits[k] / query_count, 4) if query_count else 0.0 for k in ks},
            **{f"family_doc_hit@{k}": round(family_doc_hits[k] / query_count, 4) if query_count else 0.0 for k in ks},
            **{f"chunk_expected_doc_hit@{k}": round(chunk_doc_hits[k] / query_count, 4) if query_count else 0.0 for k in ks},
            **{f"passage_hit@{k}": round(passage_hits[k] / query_count, 4) if query_count else 0.0 for k in ks},
        },
        "rows": rows,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"phase5_passage_gold_eval_{Path(args.stage1_report).stem}.json"
    )
    write_json(report_path, summary)
    print(f"Passage gold eval complete: {report_path}")
    for key, value in summary["metrics"].items():
        print(f"{key} = {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

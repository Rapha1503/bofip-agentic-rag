from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.models import chunk_node_from_dict, raw_document_from_dict
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs
from bofip_cleanroom.two_stage_retrieval import TwoStageLexicalRetriever
from bofip_cleanroom.lexical_retrieval import tokenize


def _supported_queries(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("expected_boi")]


def _overlap_metrics(query: str, text: str) -> tuple[int, float]:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens:
        return 0, 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap, overlap / len(query_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local retrieval strategies inside the selected BOFIP document.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--document-mode", type=str, default="sections", choices=["base", "sections", "sections_firstpara"])
    parser.add_argument("--chunk-mode", type=str, default="body", choices=["full", "leaf", "body"])
    parser.add_argument("--top-docs", type=int, default=3)
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--sections-per-doc", type=int, default=2)
    parser.add_argument("--chunks-per-section", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    queries = _supported_queries(read_jsonl(Path(args.queries)))

    strategies = {
        "chunk": TwoStageLexicalRetriever(
            documents,
            chunks,
            document_mode=args.document_mode,
            local_chunk_mode=args.chunk_mode,
            local_strategy="chunk",
        ),
        "section_then_chunk": TwoStageLexicalRetriever(
            documents,
            chunks,
            document_mode=args.document_mode,
            local_chunk_mode=args.chunk_mode,
            local_strategy="section_then_chunk",
        ),
    }

    summary: dict[str, dict] = {}
    examples: list[dict] = []

    for strategy_name, retriever in strategies.items():
        stats = {
            "query_count": 0,
            "doc_hit1_count": 0,
            "top_chunk_evaluated_count": 0,
            "title_only_section_top1": 0,
            "actualite_top1": 0,
            "avg_token_count_acc": 0,
            "avg_overlap_count_acc": 0,
            "avg_overlap_ratio_acc": 0.0,
        }
        per_query: dict[str, dict] = {}

        for payload in queries:
            stats["query_count"] += 1
            result = retriever.search(
                payload["query"],
                top_docs=args.top_docs,
                sections_per_doc=args.sections_per_doc,
                chunks_per_doc=args.chunks_per_doc,
                chunks_per_section=args.chunks_per_section,
                max_chunks=args.max_chunks,
            )
            top_doc = result.document_hits[0].boi_reference if result.document_hits else None
            doc_hit1 = top_doc == payload["expected_boi"]
            if doc_hit1:
                stats["doc_hit1_count"] += 1

            top_chunk = result.chunk_hits[0] if result.chunk_hits else None
            if top_chunk and doc_hit1:
                stats["top_chunk_evaluated_count"] += 1
                stats["avg_token_count_acc"] += top_chunk.chunk.token_count
                if len(top_chunk.chunk.section_path) == 1:
                    stats["title_only_section_top1"] += 1
                if top_chunk.chunk.text.strip().lower().startswith("actualite liee"):
                    stats["actualite_top1"] += 1
                overlap_count, overlap_ratio = _overlap_metrics(payload["query"], top_chunk.chunk.text)
                stats["avg_overlap_count_acc"] += overlap_count
                stats["avg_overlap_ratio_acc"] += overlap_ratio

            per_query[payload["id"]] = {
                "query": payload["query"],
                "expected_boi": payload["expected_boi"],
                "top_doc": top_doc,
                "doc_hit1": doc_hit1,
                "top_chunk": {
                    "boi_reference": top_chunk.boi_reference,
                    "chunk_id": top_chunk.chunk.chunk_id,
                    "chunk_kind": top_chunk.chunk.chunk_kind,
                    "section_rank": top_chunk.section_rank,
                    "local_rank": top_chunk.local_rank,
                    "section_path": " > ".join(top_chunk.chunk.section_path),
                    "text": top_chunk.chunk.text[:400],
                }
                if top_chunk
                else None,
            }

        top_chunk_count = stats["top_chunk_evaluated_count"] or 1
        summary[strategy_name] = {
            "query_count": stats["query_count"],
            "doc_hit1_rate": round(stats["doc_hit1_count"] / (stats["query_count"] or 1), 4),
            "top_chunk_evaluated_count": stats["top_chunk_evaluated_count"],
            "title_only_section_rate": round(stats["title_only_section_top1"] / top_chunk_count, 4),
            "actualite_rate": round(stats["actualite_top1"] / top_chunk_count, 4),
            "avg_token_count": round(stats["avg_token_count_acc"] / top_chunk_count, 1),
            "avg_overlap_count": round(stats["avg_overlap_count_acc"] / top_chunk_count, 2),
            "avg_overlap_ratio": round(stats["avg_overlap_ratio_acc"] / top_chunk_count, 4),
        }
        summary[strategy_name]["per_query"] = per_query

    for payload in queries:
        chunk_row = summary["chunk"]["per_query"][payload["id"]]
        section_row = summary["section_then_chunk"]["per_query"][payload["id"]]
        if chunk_row["top_chunk"] and section_row["top_chunk"] and chunk_row["top_chunk"]["chunk_id"] != section_row["top_chunk"]["chunk_id"]:
            examples.append(
                {
                    "id": payload["id"],
                    "query": payload["query"],
                    "expected_boi": payload["expected_boi"],
                    "chunk_strategy": chunk_row,
                    "section_then_chunk_strategy": section_row,
                }
            )
        if len(examples) >= 20:
            break

    for strategy_name in list(summary):
        summary[strategy_name].pop("per_query", None)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "chunks_path": str(Path(args.chunks).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "document_mode": args.document_mode,
        "chunk_mode": args.chunk_mode,
        "top_docs": args.top_docs,
        "chunks_per_doc": args.chunks_per_doc,
        "sections_per_doc": args.sections_per_doc,
        "chunks_per_section": args.chunks_per_section,
        "max_chunks": args.max_chunks,
        "summary": summary,
        "examples_where_top_chunk_differs": examples,
    }

    output_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"phase3_local_strategy_audit_{Path(args.queries).stem}_{args.document_mode}_{args.chunk_mode}.json"
    )
    write_json(output_path, report)
    print(f"Local strategy audit written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

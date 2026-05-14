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


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the two-stage BOFIP retrieval baseline.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--top-docs", type=int, default=3)
    parser.add_argument("--document-mode", type=str, default="base", choices=["base", "sections", "sections_firstpara"])
    parser.add_argument("--local-strategy", type=str, default="chunk", choices=["chunk", "section_then_chunk"])
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--sections-per-doc", type=int, default=2)
    parser.add_argument("--chunks-per-section", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--chunk-mode", type=str, default="body", choices=["full", "leaf", "body"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    queries = read_jsonl(Path(args.queries))[: args.limit]

    retriever = TwoStageLexicalRetriever(
        documents,
        chunks,
        document_mode=args.document_mode,
        local_chunk_mode=args.chunk_mode,
        local_strategy=args.local_strategy,
    )

    rows = []
    for payload in queries:
        result = retriever.search(
            payload["query"],
            top_docs=args.top_docs,
            sections_per_doc=args.sections_per_doc,
            chunks_per_doc=args.chunks_per_doc,
            chunks_per_section=args.chunks_per_section,
            max_chunks=args.max_chunks,
        )
        rows.append(
            {
                "id": payload["id"],
                "pattern": payload.get("pattern"),
                "query": payload["query"],
                "expected_boi": payload.get("expected_boi"),
                "document_hits": [
                    {
                        "rank": hit.rank,
                        "score": round(hit.score, 4),
                        "boi_reference": hit.boi_reference,
                        "title": hit.best_chunk.text,
                    }
                    for hit in result.document_hits
                ],
                "section_hits": [
                    {
                        "global_rank": hit.global_rank,
                        "document_rank": hit.document_rank,
                        "document_score": round(hit.document_score, 4),
                        "section_rank": hit.section_rank,
                        "section_score": round(hit.section_score, 4),
                        "boi_reference": hit.boi_reference,
                        "section_key": hit.section_key,
                        "section_path": " > ".join(hit.section_path),
                    }
                    for hit in result.section_hits
                ],
                "chunk_hits": [
                    {
                        "global_rank": hit.global_rank,
                        "document_rank": hit.document_rank,
                        "document_score": round(hit.document_score, 4),
                        "section_rank": hit.section_rank,
                        "section_score": round(hit.section_score, 4) if hit.section_score is not None else None,
                        "local_rank": hit.local_rank,
                        "local_score": round(hit.local_score, 4),
                        "boi_reference": hit.boi_reference,
                        "chunk_id": hit.chunk.chunk_id,
                        "chunk_kind": hit.chunk.chunk_kind,
                        "section_path": " > ".join(hit.chunk.section_path),
                        "text": hit.chunk.text[:600],
                    }
                    for hit in result.chunk_hits
                ],
            }
        )

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR
        / f"phase3_two_stage_probe_{Path(args.queries).stem}_{args.document_mode}_{args.local_strategy}_{args.chunk_mode}.json"
    )
    write_json(
        report_path,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "raw_docs_path": str(Path(args.raw_docs).resolve()),
            "chunks_path": str(Path(args.chunks).resolve()),
            "queries_path": str(Path(args.queries).resolve()),
            "top_docs": args.top_docs,
            "document_mode": args.document_mode,
            "local_strategy": args.local_strategy,
            "sections_per_doc": args.sections_per_doc,
            "chunks_per_doc": args.chunks_per_doc,
            "chunks_per_section": args.chunks_per_section,
            "max_chunks": args.max_chunks,
            "chunk_mode": args.chunk_mode,
            "rows": rows,
        },
    )
    print(f"Two-stage probe written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

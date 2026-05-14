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
from bofip_cleanroom.lexical_retrieval import (
    DocumentLexicalIndex,
    LexicalBM25Index,
    chunk_search_text,
    chunk_search_text_body,
    chunk_search_text_leaf,
)
from bofip_cleanroom.models import chunk_node_from_dict, raw_document_from_dict
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def _supported_queries(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not str(row.get("id", "")).startswith("u")]


def _top_chunk(index: LexicalBM25Index, query: str):
    hits = index.search(query, top_k=1)
    return hits[0] if hits else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit chunk ordering inside the correct top document.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()

    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    queries = _supported_queries(read_jsonl(Path(args.queries)))

    doc_index = DocumentLexicalIndex(documents)
    chunks_by_reference: dict[str, list] = {}
    for chunk in chunks:
        chunks_by_reference.setdefault(chunk.boi_reference, []).append(chunk)

    mode_fns = {
        "full": chunk_search_text,
        "leaf": chunk_search_text_leaf,
        "body": chunk_search_text_body,
    }

    mode_stats: dict[str, dict] = {
        mode: {
            "count": 0,
            "title_only_section_top1": 0,
            "actualite_top1": 0,
            "avg_token_count_acc": 0,
        }
        for mode in mode_fns
    }
    examples: list[dict] = []

    for payload in queries:
        doc_hits = doc_index.search_documents(payload["query"], top_k=1)
        if not doc_hits:
            continue
        top_doc = doc_hits[0].boi_reference
        if top_doc != payload["expected_boi"]:
            continue

        doc_chunks = chunks_by_reference.get(top_doc, [])
        if not doc_chunks:
            continue

        per_mode: dict[str, dict] = {}
        for mode, search_text_fn in mode_fns.items():
            top_hit = _top_chunk(LexicalBM25Index(doc_chunks, search_text_fn=search_text_fn), payload["query"])
            if top_hit is None:
                continue
            chunk = top_hit.chunk
            mode_stats[mode]["count"] += 1
            mode_stats[mode]["avg_token_count_acc"] += chunk.token_count
            if len(chunk.section_path) == 1:
                mode_stats[mode]["title_only_section_top1"] += 1
            if chunk.text.strip().lower().startswith("actualité liée"):
                mode_stats[mode]["actualite_top1"] += 1
            per_mode[mode] = {
                "chunk_id": chunk.chunk_id,
                "score": round(top_hit.score, 4),
                "chunk_kind": chunk.chunk_kind,
                "token_count": chunk.token_count,
                "section_path": " > ".join(chunk.section_path),
                "text": chunk.text[:320],
            }

        if (
            "full" in per_mode
            and "body" in per_mode
            and per_mode["full"]["chunk_id"] != per_mode["body"]["chunk_id"]
            and len(examples) < 15
        ):
            examples.append(
                {
                    "id": payload["id"],
                    "pattern": payload.get("pattern"),
                    "query": payload["query"],
                    "boi_reference": top_doc,
                    "full_top_chunk": per_mode["full"],
                    "body_top_chunk": per_mode["body"],
                }
            )

    summary = {}
    for mode, stats in mode_stats.items():
        count = stats["count"] or 1
        summary[mode] = {
            "query_count": stats["count"],
            "title_only_section_rate": round(stats["title_only_section_top1"] / count, 4),
            "actualite_rate": round(stats["actualite_top1"] / count, 4),
            "avg_token_count": round(stats["avg_token_count_acc"] / count, 1),
        }

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "chunks_path": str(Path(args.chunks).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "selection_rule": "supported queries where stage-1 document lexical retrieval is top1-correct",
        "summary": summary,
        "examples_where_body_differs_from_full": examples,
    }

    output_path = Path(args.output).resolve() if args.output else REPORTS_DIR / "phase3_chunk_order_audit_sample_1000.json"
    write_json(output_path, report)
    print(f"Chunk-order audit written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

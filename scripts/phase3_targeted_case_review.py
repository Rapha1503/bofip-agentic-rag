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


TARGET_IDS = ["q01", "q28", "q30"]


def _query_map(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


def _doc_title_map(raw_docs_path: Path) -> dict[str, str]:
    documents = [raw_document_from_dict(item) for item in read_jsonl(raw_docs_path)]
    return {document.boi_reference: document.title for document in documents}


def _classify_case(expected_ref: str, doc_hits: list[dict]) -> str:
    returned = [hit["boi_reference"] for hit in doc_hits]
    if expected_ref in returned[:1]:
        return "exact_top1"
    expected_core = expected_ref.split("-")[:-1]
    for returned_ref in returned[:3]:
        returned_core = returned_ref.split("-")[:-1]
        if expected_core == returned_core[: len(expected_core)] or returned_core == expected_core[: len(returned_core)]:
            return "family_confusion_or_specificity"
    return "true_retrieval_miss"


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted review for q01/q28/q30 on the current full-corpus baseline.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    query_map = _query_map(Path(args.queries))
    title_map = _doc_title_map(Path(args.raw_docs))

    retriever = TwoStageLexicalRetriever(
        documents,
        chunks,
        document_mode="base",
        local_chunk_mode="body",
        local_strategy="chunk",
    )

    rows = []
    for query_id in TARGET_IDS:
        payload = query_map[query_id]
        result = retriever.search(payload["query"], top_docs=5, chunks_per_doc=2, max_chunks=10)
        doc_hits = [
            {
                "rank": hit.rank,
                "score": round(hit.score, 4),
                "boi_reference": hit.boi_reference,
                "title": title_map.get(hit.boi_reference),
            }
            for hit in result.document_hits
        ]
        chunk_hits = [
            {
                "global_rank": hit.global_rank,
                "document_rank": hit.document_rank,
                "local_rank": hit.local_rank,
                "boi_reference": hit.boi_reference,
                "section_path": " > ".join(hit.chunk.section_path),
                "text": hit.chunk.text[:500],
            }
            for hit in result.chunk_hits
        ]
        rows.append(
            {
                "id": query_id,
                "query": payload["query"],
                "expected_boi": payload["expected_boi"],
                "expected_title": title_map.get(payload["expected_boi"]),
                "provisional_classification": _classify_case(payload["expected_boi"], doc_hits),
                "document_hits": doc_hits,
                "chunk_hits": chunk_hits,
            }
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "chunks_path": str(Path(args.chunks).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "baseline": {
            "document_mode": "base",
            "local_strategy": "chunk",
            "chunk_mode": "body",
            "top_docs": 5,
            "max_chunks": 10,
        },
        "rows": rows,
    }

    output_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"phase3_targeted_case_review_{Path(args.raw_docs).stem}.json"
    )
    write_json(output_path, report)
    print(f"Targeted case review written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

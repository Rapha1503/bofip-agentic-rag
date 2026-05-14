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

from bofip_cleanroom.direct_chunk_retrieval import DirectChunkRetriever, Stage1DocumentHit
from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.models import chunk_node_from_dict
from bofip_cleanroom.passage_eval import chunk_matches_passage_gold
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate direct stage-1-doc to local-chunk retrieval against passage gold.")
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--stage1-report", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument("--chunks-per-doc", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--chunk-mode", type=str, default="body", choices=["full", "leaf", "body"])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    stage1 = json.loads(Path(args.stage1_report).read_text(encoding="utf-8"))
    queries = read_jsonl(Path(args.queries))
    direct_retriever = DirectChunkRetriever(chunks, local_chunk_mode=args.chunk_mode)
    stage1_by_id = {row["id"]: row for row in stage1["results"]}

    ks = sorted(set(args.top_k))
    stage1_doc_hits = {k: 0 for k in ks}
    stage2_doc_hits = {k: 0 for k in ks}
    passage_hits = {k: 0 for k in ks}
    rows: list[dict] = []

    for gold_row in queries:
        stage1_row = stage1_by_id.get(gold_row["id"])
        if stage1_row is None:
            raise KeyError(f"Missing stage-1 row for passage query id {gold_row['id']}")

        expected = gold_row["expected_boi"]
        stage1_top_hits = stage1_row.get("top_hits", [])[: args.top_docs]
        lexical_query = stage1_row.get("lexical_query") or gold_row["query"]
        stage1_doc_refs = [hit["boi_reference"] for hit in stage1_top_hits]
        direct_result = direct_retriever.search(
            gold_row["query"],
            lexical_query=lexical_query,
            stage1_hits=[
                Stage1DocumentHit(
                    rank=hit["rank"],
                    score=float(hit["score"]),
                    boi_reference=hit["boi_reference"],
                )
                for hit in stage1_top_hits
            ],
            top_docs=args.top_docs,
            chunks_per_doc=args.chunks_per_doc,
            max_chunks=args.max_chunks,
        )
        chunk_doc_refs = [hit.boi_reference for hit in direct_result.chunk_hits]
        passage_match_ranks = [
            index + 1
            for index, hit in enumerate(direct_result.chunk_hits)
            if chunk_matches_passage_gold(hit.chunk, gold_row)
        ]

        for k in ks:
            if expected in stage1_doc_refs[:k]:
                stage1_doc_hits[k] += 1
            if expected in chunk_doc_refs[:k]:
                stage2_doc_hits[k] += 1
            if any(rank <= k for rank in passage_match_ranks):
                passage_hits[k] += 1

        rows.append(
            {
                "id": gold_row["id"],
                "pattern": gold_row.get("pattern"),
                "query": gold_row["query"],
                "expected_boi": expected,
                "stage1_top_hits": stage1_top_hits,
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
                    for hit in direct_result.chunk_hits
                ],
                **{f"stage1_doc_hit@{k}": expected in stage1_doc_refs[:k] for k in ks},
                **{f"stage2_doc_hit@{k}": expected in chunk_doc_refs[:k] for k in ks},
                **{f"passage_hit@{k}": any(rank <= k for rank in passage_match_ranks) for k in ks},
                "first_passage_match_rank": min(passage_match_ranks) if passage_match_ranks else None,
            }
        )

    query_count = len(queries)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "chunks_path": str(Path(args.chunks).resolve()),
        "stage1_report_path": str(Path(args.stage1_report).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "top_docs": args.top_docs,
        "chunks_per_doc": args.chunks_per_doc,
        "max_chunks": args.max_chunks,
        "chunk_mode": args.chunk_mode,
        "query_count": query_count,
        "metrics": {
            **{f"stage1_doc_hit@{k}": round(stage1_doc_hits[k] / query_count, 4) if query_count else 0.0 for k in ks},
            **{f"stage2_doc_hit@{k}": round(stage2_doc_hits[k] / query_count, 4) if query_count else 0.0 for k in ks},
            **{f"passage_hit@{k}": round(passage_hits[k] / query_count, 4) if query_count else 0.0 for k in ks},
        },
        "rows": rows,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"phase5_direct_chunk_eval_{Path(args.stage1_report).stem}.json"
    )
    write_json(report_path, summary)
    print(f"Direct chunk eval complete: {report_path}")
    for key, value in summary["metrics"].items():
        print(f"{key} = {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

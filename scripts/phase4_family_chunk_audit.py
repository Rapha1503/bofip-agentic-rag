from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.family_routing import collect_family_selection
from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.lexical_retrieval import LexicalBM25Index, get_chunk_search_text_fn
from bofip_cleanroom.models import chunk_node_from_dict
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit multi-doc family chunk retrieval from a stage-1 document report.")
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--stage1-report", type=str, required=True)
    parser.add_argument("--chunk-mode", type=str, default="body", choices=["full", "leaf", "body"])
    parser.add_argument("--family-top-docs", type=int, default=1)
    parser.add_argument("--max-family-docs", type=int, default=25)
    parser.add_argument("--chunk-top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--candidate-chunks", type=int, default=50)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    chunk_nodes = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    chunks_by_reference: dict[str, list] = defaultdict(list)
    for chunk in chunk_nodes:
        chunks_by_reference[chunk.boi_reference].append(chunk)
    all_references = sorted(chunks_by_reference.keys())
    stage1 = __import__("json").loads(Path(args.stage1_report).read_text(encoding="utf-8"))
    ks = sorted(set(args.chunk_top_k))
    hits_by_k = {k: 0 for k in ks}
    supported_count = 0
    expected_in_family = 0
    rows: list[dict] = []
    chunk_search_text_fn = get_chunk_search_text_fn(args.chunk_mode)

    for row in stage1["results"]:
        if not row.get("supported_query"):
            continue
        supported_count += 1
        top_doc_hits = row.get("top_hits", [])[: args.family_top_docs]
        anchor_references = [hit["boi_reference"] for hit in top_doc_hits if hit.get("boi_reference")]
        anchor_reference = anchor_references[0] if anchor_references else None
        family_rows = []
        chunk_hits = []
        family_members: list[str] = []
        family_prefix: list[str] = []
        family_prefixes: list[list[str]] = []

        if anchor_reference and anchor_reference in chunks_by_reference:
            seen_members: set[str] = set()
            for reference in anchor_references:
                if reference not in chunks_by_reference:
                    continue
                family = collect_family_selection(
                    reference,
                    all_references,
                    max_family_docs=args.max_family_docs,
                )
                family_prefixes.append(list(family.prefix))
                for member in family.members:
                    if member in seen_members:
                        continue
                    seen_members.add(member)
                    family_members.append(member)
            family_prefix = family_prefixes[0] if family_prefixes else []
            if row["expected_boi"] in family_members:
                expected_in_family += 1

            family_chunks = []
            for reference in family_members:
                family_chunks.extend(chunks_by_reference.get(reference, []))
            local_index = LexicalBM25Index(family_chunks, search_text_fn=chunk_search_text_fn)
            query_text = row.get("lexical_query") or row["query"]
            local_hits = local_index.search(query_text, top_k=min(args.candidate_chunks, len(family_chunks)))
            chunk_hits = [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "boi_reference": hit.chunk.boi_reference,
                    "chunk_id": hit.chunk.chunk_id,
                    "chunk_kind": hit.chunk.chunk_kind,
                    "section_path": " > ".join(hit.chunk.section_path),
                    "text": hit.chunk.text[:500],
                }
                for hit in local_hits[: max(ks)]
            ]
            for member in family_members:
                family_rows.append({"boi_reference": member})

            returned_chunk_docs = [hit["boi_reference"] for hit in chunk_hits]
            for k in ks:
                if row["expected_boi"] in returned_chunk_docs[:k]:
                    hits_by_k[k] += 1

        rows.append(
            {
                "id": row["id"],
                "pattern": row.get("pattern"),
                "query": row["query"],
                "lexical_query": row.get("lexical_query"),
                "expected_boi": row["expected_boi"],
                "stage1_top_hits": row.get("top_hits", [])[: max(ks)],
                "family_anchor_boi": anchor_reference,
                "family_anchor_bois": anchor_references,
                "family_prefix": family_prefix,
                "family_prefixes": family_prefixes,
                "family_size": len(family_members),
                "expected_in_family": row["expected_boi"] in family_members,
                "family_members": family_rows,
                "chunk_hits": chunk_hits,
                **{f"expected_doc_in_chunk_hits@{k}": row["expected_boi"] in [hit["boi_reference"] for hit in chunk_hits[:k]] for k in ks},
            }
        )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "chunks_path": str(Path(args.chunks).resolve()),
        "stage1_report_path": str(Path(args.stage1_report).resolve()),
        "chunk_mode": args.chunk_mode,
        "family_top_docs": args.family_top_docs,
        "max_family_docs": args.max_family_docs,
        "candidate_chunks": args.candidate_chunks,
        "supported_query_count": supported_count,
        "expected_in_family_rate": round(expected_in_family / supported_count, 4) if supported_count else 0.0,
        "metrics": {
            f"expected_doc_in_chunk_hits@{k}": round(hits_by_k[k] / supported_count, 4) if supported_count else 0.0
            for k in ks
        },
        "rows": rows,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR
        / f"phase4_family_chunk_audit_{Path(args.stage1_report).stem}__{args.chunk_mode}__topdocs{args.family_top_docs}.json"
    )
    write_json(report_path, summary)
    print(f"Family chunk audit complete: {report_path}")
    print(f"expected_in_family_rate = {summary['expected_in_family_rate']:.4f}")
    for k in ks:
        print(f"expected_doc_in_chunk_hits@{k} = {summary['metrics'][f'expected_doc_in_chunk_hits@{k}']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

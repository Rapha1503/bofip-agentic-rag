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
    get_document_search_text_fn,
    tokenize,
)
from bofip_cleanroom.models import raw_document_from_dict
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3A document-level lexical evaluation on BOFIP titles/metadata.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--doc-mode", type=str, choices=["base", "title", "title_tail", "sections", "sections_firstpara", "sections_leads"], default="base")
    parser.add_argument("--stem-lexical", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    queries = read_jsonl(Path(args.queries))
    index = DocumentLexicalIndex(
        documents,
        search_text_fn=get_document_search_text_fn(args.doc_mode),
        tokenize_fn=(lambda text: tokenize(text, stem=True)) if args.stem_lexical else None,
    )

    ks = sorted(set(args.top_k))
    hits_by_k = {k: 0 for k in ks}
    results: list[dict] = []
    supported_query_count = 0
    unsupported_query_count = 0

    for payload in queries:
        query = payload["query"]
        expected = payload.get("expected_boi")
        supported = bool(expected)
        if supported:
            supported_query_count += 1
        else:
            unsupported_query_count += 1
        hits = index.search_documents(query, top_k=max(ks))
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
                    "score": round(hit.score, 4),
                    "boi_reference": hit.boi_reference,
                    "chunk_id": hit.best_chunk.chunk_id,
                    "chunk_kind": hit.best_chunk.chunk_kind,
                    "section_path": " > ".join(hit.best_chunk.section_path),
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

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "queries_path": str(Path(args.queries).resolve()),
        "doc_mode": args.doc_mode,
        "stem_lexical": args.stem_lexical,
        "query_count": len(queries),
        "supported_query_count": supported_query_count,
        "unsupported_query_count": unsupported_query_count,
        "metrics": {
            f"hit@{k}": round(hits_by_k[k] / supported_query_count, 4) if supported_query_count else 0.0
            for k in ks
        },
        "results": results,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR
        / f"phase3_doc_lexical_eval_{Path(args.raw_docs).stem}__{Path(args.queries).stem}__{args.doc_mode}{'__stem' if args.stem_lexical else ''}.json"
    )
    write_json(report_path, summary)

    print(f"Document lexical evaluation complete: {report_path}")
    for k in ks:
        print(f"hit@{k} = {summary['metrics'][f'hit@{k}']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

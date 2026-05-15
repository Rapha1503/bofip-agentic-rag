from __future__ import annotations

import argparse
import statistics
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.chunking import SUPPORTED_STRATEGIES, build_chunks
from bofip_cleanroom.jsonio import read_jsonl, write_json, write_jsonl
from bofip_cleanroom.models import raw_document_from_dict
from bofip_cleanroom.settings import INTERIM_DIR, REPORTS_DIR, ensure_data_dirs


def _stats(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "avg": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(statistics.mean(values), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 BOFIP chunk builder.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True, choices=sorted(SUPPORTED_STRATEGIES))
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--min-tokens", type=int, default=40)
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    chunks = [
        chunk
        for document in documents
        for chunk in build_chunks(
            document,
            strategy=args.strategy,
            max_tokens=args.max_tokens,
            min_tokens=args.min_tokens,
        )
    ]

    sample_suffix = Path(args.raw_docs).stem.replace("raw_docs_", "")
    chunks_path = INTERIM_DIR / f"chunks_{args.strategy}_{sample_suffix}.jsonl"
    summary_path = REPORTS_DIR / f"phase2_chunks_{args.strategy}_{sample_suffix}.json"

    write_jsonl(chunks_path, [chunk.to_dict() for chunk in chunks])
    token_counts = [chunk.token_count for chunk in chunks]
    kind_counts = Counter(chunk.chunk_kind for chunk in chunks)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": args.strategy,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunk_kind_counts": dict(sorted(kind_counts.items())),
        "token_count_stats": _stats(token_counts),
        "too_long_count": sum(1 for chunk in chunks if chunk.token_count > args.max_tokens),
        "empty_text_count": sum(1 for chunk in chunks if not chunk.text.strip()),
        "chunks_path": str(chunks_path),
    }
    write_json(summary_path, summary)

    print(f"Chunk build complete: {len(chunks)} chunks")
    print(f"Strategy: {args.strategy}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

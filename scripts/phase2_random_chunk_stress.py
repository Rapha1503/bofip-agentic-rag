from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.chunking import SUPPORTED_STRATEGIES, build_chunks
from bofip_cleanroom.discovery import discover_content_documents
from bofip_cleanroom.document_builder import build_raw_document
from bofip_cleanroom.jsonio import write_json
from bofip_cleanroom.sampling import random_sample_documents
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs, get_raw_bofip_root


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
    parser = argparse.ArgumentParser(description="Random chunk stress test.")
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--content-type", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--min-tokens", type=int, default=40)
    args = parser.parse_args()

    ensure_data_dirs()
    raw_root = get_raw_bofip_root(args.raw_root)
    documents = discover_content_documents(raw_root)
    sampled = random_sample_documents(
        documents,
        args.sample_size,
        seed=args.seed,
        allowed_content_types=set(args.content_type) if args.content_type else None,
    )

    parsed_docs = [build_raw_document(doc) for doc in sampled]
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_root": str(raw_root),
        "sample_size": len(parsed_docs),
        "seed": args.seed,
        "content_types_filter": list(args.content_type),
        "strategies": {},
        "sample_docs": [
            {
                "boi_reference": doc.boi_reference,
                "content_type": doc.content_type,
                "title": doc.title,
                "sections": len(doc.sections),
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables),
            }
            for doc in parsed_docs
        ],
    }

    for strategy in sorted(SUPPORTED_STRATEGIES):
        all_chunks = [
            chunk
            for document in parsed_docs
            for chunk in build_chunks(
                document,
                strategy=strategy,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
            )
        ]
        kind_counts = Counter(chunk.chunk_kind for chunk in all_chunks)
        token_counts = [chunk.token_count for chunk in all_chunks]
        very_short = [
            {
                "boi_reference": chunk.boi_reference,
                "chunk_kind": chunk.chunk_kind,
                "token_count": chunk.token_count,
                "text": chunk.text[:120],
            }
            for chunk in sorted(all_chunks, key=lambda c: (c.token_count, len(c.text)))[:30]
        ]
        report["strategies"][strategy] = {
            "chunk_count": len(all_chunks),
            "chunk_kind_counts": dict(sorted(kind_counts.items())),
            "token_count_stats": _stats(token_counts),
            "too_long_count": sum(1 for chunk in all_chunks if chunk.token_count > args.max_tokens),
            "empty_text_count": sum(1 for chunk in all_chunks if not chunk.text.strip()),
            "very_short_count_le_5": sum(1 for chunk in all_chunks if chunk.token_count <= 5),
            "very_short_examples": very_short,
        }

    name_parts = ["phase2_random_chunk_stress", f"n{len(parsed_docs)}", f"seed{args.seed}"]
    if args.content_type:
        name_parts.append("filtered")
    output_path = REPORTS_DIR / ("_".join(name_parts) + ".json")
    write_json(output_path, report)
    print(f"Random chunk stress written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

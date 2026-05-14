from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl
from bofip_cleanroom.lexical_retrieval import LexicalBM25Index
from bofip_cleanroom.models import chunk_node_from_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3A lexical BM25 probe.")
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(args.chunks))]
    index = LexicalBM25Index(chunks)
    hits = index.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Hits: {len(hits)}")
    for hit in hits:
        excerpt = hit.chunk.text[:240].replace("\n", " ")
        section_path = " > ".join(hit.chunk.section_path)
        print("-" * 80)
        print(f"rank={hit.rank} score={hit.score:.4f} ref={hit.chunk.boi_reference} kind={hit.chunk.chunk_kind}")
        print(f"section_path={section_path}")
        print(excerpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

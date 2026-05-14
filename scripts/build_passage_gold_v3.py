from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl, write_jsonl
from bofip_cleanroom.settings import INTERIM_DIR


def _resolve_gold_chunks(
    queries_path: Path,
    chunks_path: Path,
    output_path: Path,
    max_chunks_per_doc: int = 5,
) -> None:
    queries = read_jsonl(queries_path)
    chunk_rows = read_jsonl(chunks_path)

    chunks_by_ref: dict[str, list[dict]] = {}
    for row in chunk_rows:
        ref = row["boi_reference"]
        chunks_by_ref.setdefault(ref, []).append(row)

    gold_entries = []
    for query in queries:
        gold_doc_refs = query.get("gold_doc_refs", [])
        gold_chunk_ids: list[str] = []

        for ref in gold_doc_refs:
            doc_chunks = chunks_by_ref.get(ref, [])
            if not doc_chunks:
                continue
            doc_chunks.sort(key=lambda c: (
                len(c.get("section_path", [])),
                c.get("paragraph_range", [])[0] if c.get("paragraph_range") else "",
            ))
            for chunk in doc_chunks[:max_chunks_per_doc]:
                gold_chunk_ids.append(chunk["chunk_id"])

        gold_entries.append({
            "query_id": query["query_id"],
            "query": query["query"],
            "category": query.get("category", ""),
            "gold_doc_refs": gold_doc_refs,
            "gold_chunk_ids": gold_chunk_ids,
            "gold_chunk_count": len(gold_chunk_ids),
            "note": query.get("note", ""),
        })

    write_jsonl(output_path, gold_entries)
    total_with_gold = sum(1 for e in gold_entries if e["gold_chunk_ids"])
    print(f"Passage gold written to: {output_path}")
    print(f"Total queries: {len(gold_entries)}")
    print(f"Queries with gold chunks: {total_with_gold}")
    print(f"Total gold chunk references: {sum(e['gold_chunk_count'] for e in gold_entries)}")


if __name__ == "__main__":
    queries_path = INTERIM_DIR / "eval_queries_v1.jsonl"
    chunks_path = INTERIM_DIR / "chunks_section_window_sample_5666.jsonl"
    output_path = INTERIM_DIR / "passage_gold_v3.jsonl"
    _resolve_gold_chunks(queries_path, chunks_path, output_path)

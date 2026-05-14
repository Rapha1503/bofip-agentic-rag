from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _support_rows(eval_report: dict) -> list[dict]:
    return [row for row in eval_report["results"] if row.get("expected_boi")]


def _top_doc_rank(document_hits: list[dict], expected_boi: str) -> int | None:
    for hit in document_hits:
        if hit["boi_reference"] == expected_boi:
            return hit["rank"]
    return None


def _top_chunk_rank(chunk_hits: list[dict], expected_boi: str) -> int | None:
    for hit in chunk_hits:
        if hit["boi_reference"] == expected_boi:
            return hit["global_rank"]
    return None


def _repairability(doc_rank: int | None, chunk_rank: int | None) -> str:
    if doc_rank is None:
        return "not_repairable_at_stage2"
    if chunk_rank is not None and chunk_rank <= 2:
        return "stage2_repairable_strong"
    if chunk_rank is not None:
        return "stage2_repairable_weak"
    return "doc_in_candidates_but_no_chunk"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the remaining strict document misses against stage-2 retrieval.")
    parser.add_argument("--eval-report", type=str, required=True)
    parser.add_argument("--two-stage-report", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    eval_report = _load_json(Path(args.eval_report))
    two_stage_report = _load_json(Path(args.two_stage_report))

    stage2_map = {row["id"]: row for row in two_stage_report["rows"]}
    misses = []

    for row in _support_rows(eval_report):
        if row.get("hit@1"):
            continue
        stage2_row = stage2_map[row["id"]]
        expected_boi = row["expected_boi"]
        doc_rank = _top_doc_rank(stage2_row["document_hits"], expected_boi)
        chunk_rank = _top_chunk_rank(stage2_row["chunk_hits"], expected_boi)
        misses.append(
            {
                "id": row["id"],
                "pattern": row.get("pattern"),
                "query": row["query"],
                "expected_boi": expected_boi,
                "stage1_top1_boi": row["returned_boi"][0] if row.get("returned_boi") else None,
                "stage1_hit@3": row.get("hit@3"),
                "expected_doc_rank_in_stage2_docs": doc_rank,
                "expected_doc_rank_in_stage2_chunks": chunk_rank,
                "repairability": _repairability(doc_rank, chunk_rank),
                "document_hits": stage2_row["document_hits"],
                "chunk_hits": stage2_row["chunk_hits"],
            }
        )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "eval_report": str(Path(args.eval_report).resolve()),
        "two_stage_report": str(Path(args.two_stage_report).resolve()),
        "miss_count": len(misses),
        "repairability_counts": {
            key: sum(1 for row in misses if row["repairability"] == key)
            for key in sorted({row["repairability"] for row in misses})
        },
        "rows": misses,
    }

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Remaining miss audit written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

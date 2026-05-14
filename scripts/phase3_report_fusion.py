from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.hybrid_retrieval import RankedDoc, reciprocal_rank_fuse
from bofip_cleanroom.jsonio import write_json
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def parse_weighted_report(value: str) -> tuple[str, Path, float]:
    try:
        name, report_path, weight = value.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected SOURCE=PATH=WEIGHT") from exc
    return name, Path(report_path), float(weight)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse existing retrieval reports offline with weighted RRF.")
    parser.add_argument("--report", action="append", type=parse_weighted_report, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--label", type=str, default="offline_fusion")
    args = parser.parse_args()

    ensure_data_dirs()
    ks = sorted(set(args.top_k))

    loaded = []
    for source_name, report_path, weight in args.report:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        loaded.append((source_name, report_path, weight, payload))

    base_results = loaded[0][3]["results"]
    result_ids = [row["id"] for row in base_results]
    hits_by_k = {k: 0 for k in ks}
    supported_query_count = 0
    unsupported_query_count = 0
    fused_results: list[dict] = []

    for result_id in result_ids:
        rows = {}
        for source_name, report_path, weight, payload in loaded:
            row = next(item for item in payload["results"] if item["id"] == result_id)
            rows[source_name] = row

        exemplar = next(iter(rows.values()))
        supported = bool(exemplar.get("expected_boi"))
        if supported:
            supported_query_count += 1
        else:
            unsupported_query_count += 1

        rankings: dict[str, list[RankedDoc]] = {}
        source_weights: dict[str, float] = {}
        for source_name, _, weight, _ in loaded:
            source_weights[source_name] = weight
            rankings[source_name] = [
                RankedDoc(
                    boi_reference=hit["boi_reference"],
                    score=float(hit["score"]),
                    rank=int(hit["rank"]),
                    source=source_name,
                )
                for hit in rows[source_name].get("top_hits", [])
            ]

        fused = reciprocal_rank_fuse(
            rankings,
            top_k=max(ks),
            rank_constant=args.rank_constant,
            source_weights=source_weights,
        )
        returned = [hit.boi_reference for hit in fused]

        row_out = {
            "id": exemplar["id"],
            "pattern": exemplar.get("pattern"),
            "query": exemplar["query"],
            "expected_boi": exemplar.get("expected_boi"),
            "supported_query": supported,
            "returned_boi": returned,
            "top_hits": [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 6),
                    "boi_reference": hit.boi_reference,
                    "sources": hit.sources,
                    "ranks": hit.ranks,
                }
                for hit in fused
            ],
        }
        if supported:
            for k in ks:
                matched = exemplar["expected_boi"] in returned[:k]
                row_out[f"hit@{k}"] = matched
                if matched:
                    hits_by_k[k] += 1
        fused_results.append(row_out)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "reports": [
            {"source": source_name, "path": str(report_path.resolve()), "weight": weight}
            for source_name, report_path, weight, _ in loaded
        ],
        "rank_constant": args.rank_constant,
        "query_count": len(fused_results),
        "supported_query_count": supported_query_count,
        "unsupported_query_count": unsupported_query_count,
        "metrics": {
            f"hit@{k}": round(hits_by_k[k] / supported_query_count, 4) if supported_query_count else 0.0
            for k in ks
        },
        "results": fused_results,
    }

    report_path = REPORTS_DIR / f"phase3_{args.label}.json"
    write_json(report_path, summary)
    print(f"Offline fusion complete: {report_path}")
    for k in ks:
        print(f"hit@{k} = {summary['metrics'][f'hit@{k}']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

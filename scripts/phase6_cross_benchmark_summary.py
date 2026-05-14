from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import write_json
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dataset_key(payload: dict) -> str:
    if "queries_path" in payload:
        return Path(payload["queries_path"]).stem
    stage1_report = payload.get("stage1_report_path")
    if stage1_report:
        stage1_payload = _load_json(stage1_report)
        return Path(stage1_payload["queries_path"]).stem
    raise KeyError("Cannot infer dataset key from payload")


def _build_row(dataset: str, stage1: dict, family: dict | None, direct: dict | None) -> dict:
    row = {
        "dataset": dataset,
        "query_count": stage1["query_count"],
        "supported_query_count": stage1["supported_query_count"],
        "unsupported_query_count": stage1["unsupported_query_count"],
        "stage1_doc_hit@1": stage1["metrics"]["hit@1"],
        "stage1_doc_hit@3": stage1["metrics"]["hit@3"],
        "stage1_doc_hit@5": stage1["metrics"]["hit@5"],
        "family_hit_rate": family["expected_in_family_rate"] if family else None,
        "family_doc_hit@1": family["metrics"]["family_doc_hit@1"] if family else None,
        "family_doc_hit@3": family["metrics"]["family_doc_hit@3"] if family else None,
        "family_doc_hit@5": family["metrics"]["family_doc_hit@5"] if family else None,
        "chunk_doc_hit@1": family["metrics"]["chunk_expected_doc_hit@1"] if family else None,
        "chunk_doc_hit@3": family["metrics"]["chunk_expected_doc_hit@3"] if family else None,
        "chunk_doc_hit@5": family["metrics"]["chunk_expected_doc_hit@5"] if family else None,
        "passage_hit@1": direct["metrics"]["passage_hit@1"] if direct else None,
        "passage_hit@3": direct["metrics"]["passage_hit@3"] if direct else None,
        "passage_hit@5": direct["metrics"]["passage_hit@5"] if direct else None,
    }
    row["has_passage_gold"] = direct is not None
    return row


def _gate_a(rows: list[dict]) -> dict:
    by_dataset = {row["dataset"]: row for row in rows}
    v3 = by_dataset.get("retrieval_queries_sample_1000_v3")
    v4 = by_dataset.get("retrieval_queries_full_v4")
    v5 = by_dataset.get("retrieval_queries_full_v5")
    pg2 = by_dataset.get("passage_gold_v2")

    checks = {
        "v3_no_regression_gt_3pts_vs_78_33_top1_baseline": bool(v3 and v3["stage1_doc_hit@1"] >= 0.7533),
        "v4_doc_hit@5_ge_0_80": bool(v4 and v4["stage1_doc_hit@5"] >= 0.80),
        "v5_doc_hit@5_ge_0_80": bool(v5 and v5["stage1_doc_hit@5"] >= 0.80),
        "v4_family_rate_ge_0_90": bool(v4 and v4["family_hit_rate"] is not None and v4["family_hit_rate"] >= 0.90),
        "v5_family_rate_ge_0_90": bool(v5 and v5["family_hit_rate"] is not None and v5["family_hit_rate"] >= 0.90),
        "passage_gold_v2_hit@3_ge_0_45": bool(pg2 and pg2["passage_hit@3"] is not None and pg2["passage_hit@3"] >= 0.45),
        "passage_gold_v2_hit@5_ge_0_65": bool(pg2 and pg2["passage_hit@5"] is not None and pg2["passage_hit@5"] >= 0.65),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize phase-6 document/family/passage metrics across benchmarks.")
    parser.add_argument("--stage1-reports", nargs="+", required=True)
    parser.add_argument("--family-reports", nargs="*", default=[])
    parser.add_argument("--direct-passage-reports", nargs="*", default=[])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    stage1_by_dataset = {_dataset_key(_load_json(path)): _load_json(path) for path in args.stage1_reports}
    family_by_dataset = {_dataset_key(_load_json(path)): _load_json(path) for path in args.family_reports}
    direct_by_dataset = {_dataset_key(_load_json(path)): _load_json(path) for path in args.direct_passage_reports}

    datasets = sorted(set(stage1_by_dataset))
    rows = [
        _build_row(dataset, stage1_by_dataset[dataset], family_by_dataset.get(dataset), direct_by_dataset.get(dataset))
        for dataset in datasets
    ]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": rows,
        "gate_a": _gate_a(rows),
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / "phase6_cross_benchmark_summary.json"
    )
    write_json(report_path, summary)
    print(f"Phase-6 cross benchmark summary written to: {report_path}")
    for row in rows:
        print(
            f"{row['dataset']}: "
            f"family={row['family_hit_rate']}, "
            f"doc@5={row['stage1_doc_hit@5']:.4f}, "
            f"passage@5={row['passage_hit@5']}"
        )
    print(f"Gate A passed: {summary['gate_a']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

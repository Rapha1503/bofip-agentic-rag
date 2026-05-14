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


def _dataset_name_from_path(path: str) -> str:
    return Path(path).stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize cross-benchmark family/doc/chunk metrics.")
    parser.add_argument("--stage1-reports", nargs="+", required=True)
    parser.add_argument("--family-reports", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    stage1_by_dataset = {}
    for path in args.stage1_reports:
        payload = _load_json(path)
        stage1_by_dataset[_dataset_name_from_path(payload["queries_path"])] = payload

    family_by_dataset = {}
    for path in args.family_reports:
        payload = _load_json(path)
        stage1_payload = _load_json(payload["stage1_report_path"])
        family_by_dataset[_dataset_name_from_path(stage1_payload["queries_path"])] = payload

    datasets = sorted(set(stage1_by_dataset) & set(family_by_dataset))
    rows = []
    for dataset in datasets:
        stage1 = stage1_by_dataset[dataset]
        family = family_by_dataset[dataset]
        rows.append(
            {
                "dataset": dataset,
                "query_count": stage1["query_count"],
                "supported_query_count": stage1["supported_query_count"],
                "unsupported_query_count": stage1["unsupported_query_count"],
                "stage1_doc_hit@1": stage1["metrics"]["hit@1"],
                "stage1_doc_hit@3": stage1["metrics"]["hit@3"],
                "stage1_doc_hit@5": stage1["metrics"]["hit@5"],
                "family_hit_rate": family["expected_in_family_rate"],
                "family_doc_hit@1": family["metrics"]["family_doc_hit@1"],
                "family_doc_hit@3": family["metrics"]["family_doc_hit@3"],
                "family_doc_hit@5": family["metrics"]["family_doc_hit@5"],
                "chunk_doc_hit@1": family["metrics"]["chunk_expected_doc_hit@1"],
                "chunk_doc_hit@3": family["metrics"]["chunk_expected_doc_hit@3"],
                "chunk_doc_hit@5": family["metrics"]["chunk_expected_doc_hit@5"],
                "passage_metric_available": False,
                "passage_metric_note": "No passage-level gold labels yet; current reliable proxy is whether returned chunks come from the expected document.",
            }
        )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": rows,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / "phase4_cross_benchmark_summary.json"
    )
    write_json(report_path, summary)
    print(f"Cross-benchmark summary written to: {report_path}")
    for row in rows:
        print(
            f"{row['dataset']}: "
            f"stage1@1={row['stage1_doc_hit@1']:.4f}, "
            f"family@1={row['family_doc_hit@1']:.4f}, "
            f"family@5={row['family_doc_hit@5']:.4f}, "
            f"chunk@5={row['chunk_doc_hit@5']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

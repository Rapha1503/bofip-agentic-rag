from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare family-guided and direct local chunk stage-2 reports.")
    parser.add_argument("--family-report", type=str, required=True)
    parser.add_argument("--direct-report", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    family = _load_json(args.family_report)
    direct = _load_json(args.direct_report)
    if Path(family["queries_path"]).resolve() != Path(direct["queries_path"]).resolve():
        raise ValueError("Family and direct reports must target the same query set")

    ks = sorted(
        {
            int(key.rsplit("@", 1)[1])
            for key in family["metrics"]
            if key.startswith("passage_hit@")
        }
    )
    family_rows = {row["id"]: row for row in family["rows"]}
    direct_rows = {row["id"]: row for row in direct["rows"]}

    improved = []
    regressed = []
    unchanged = []
    for row_id, family_row in family_rows.items():
        direct_row = direct_rows.get(row_id)
        if direct_row is None:
            continue
        family_rank = family_row.get("first_passage_match_rank")
        direct_rank = direct_row.get("first_passage_match_rank")
        if family_rank is None and direct_rank is None:
            unchanged.append(row_id)
        elif family_rank is None:
            improved.append(row_id)
        elif direct_rank is None:
            regressed.append(row_id)
        elif direct_rank < family_rank:
            improved.append(row_id)
        elif direct_rank > family_rank:
            regressed.append(row_id)
        else:
            unchanged.append(row_id)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "queries_path": str(Path(family["queries_path"]).resolve()),
        "family_report_path": str(Path(args.family_report).resolve()),
        "direct_report_path": str(Path(args.direct_report).resolve()),
        "family_metrics": family["metrics"],
        "direct_metrics": direct["metrics"],
        "deltas": {
            f"passage_hit@{k}": round(direct["metrics"][f"passage_hit@{k}"] - family["metrics"][f"passage_hit@{k}"], 4)
            for k in ks
        },
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_count": len(unchanged),
        "improved_ids": improved,
        "regressed_ids": regressed,
        "unchanged_ids": unchanged,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"phase5_stage2_comparison__{Path(family['queries_path']).stem}.json"
    )
    write_json(report_path, summary)
    print(f"Stage-2 comparison written to: {report_path}")
    print(f"improved={len(improved)} regressed={len(regressed)} unchanged={len(unchanged)}")
    for key, value in summary["deltas"].items():
        print(f"{key} delta = {value:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

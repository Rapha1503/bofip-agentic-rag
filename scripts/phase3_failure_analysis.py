from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")


def _ref_core(boi_reference: str | None) -> list[str]:
    if not boi_reference:
        return []
    parts = boi_reference.split("-")
    if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
        parts = parts[:-1]
    return parts


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token.lower() for token in TOKEN_RE.findall(text)}


def _prefix_len(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def classify_failure(expected_ref: str, top_ref: str, expected_title: str, top_title: str) -> str:
    expected_core = _ref_core(expected_ref)
    top_core = _ref_core(top_ref)
    prefix = _prefix_len(expected_core, top_core)

    if expected_core and top_core and (
        expected_core == top_core[: len(expected_core)] or top_core == expected_core[: len(top_core)]
    ):
        return "parent_child_family_confusion"

    expected_tokens = _tokenize(expected_title)
    top_tokens = _tokenize(top_title)
    overlap = len(expected_tokens & top_tokens)

    if prefix >= 4:
        return "same_family_neighbor"
    if prefix >= 2 and overlap >= 4:
        return "same_domain_neighbor"
    if overlap >= 6:
        return "title_equivalent_or_version_confusion"
    return "true_top1_miss"


def load_meta(raw_docs_path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for item in read_jsonl(raw_docs_path):
        meta[item["boi_reference"]] = {
            "title": item.get("title"),
            "content_type": item.get("content_type"),
            "publication_date": item.get("publication_date"),
        }
    return meta


def analyze_report(report_path: Path, meta: dict[str, dict]) -> dict:
    report = report_path.read_text(encoding="utf-8")
    import json

    payload = json.loads(report)
    misses = []
    by_category = Counter()
    by_pattern = Counter()
    supported = 0

    for row in payload["results"]:
        if not row.get("supported_query"):
            continue
        supported += 1
        top_hits = row.get("top_hits") or []
        top1 = top_hits[0] if top_hits else None
        if row.get("hit@1"):
            continue
        expected_ref = row["expected_boi"]
        top_ref = top1["boi_reference"] if top1 else None
        expected_title = meta.get(expected_ref, {}).get("title")
        top_title = meta.get(top_ref, {}).get("title")
        category = classify_failure(expected_ref, top_ref or "", expected_title or "", top_title or "")
        by_category[category] += 1
        by_pattern[row.get("pattern") or "unknown"] += 1
        misses.append(
            {
                "id": row["id"],
                "pattern": row.get("pattern"),
                "query": row["query"],
                "expected_boi": expected_ref,
                "expected_title": expected_title,
                "top1_boi": top_ref,
                "top1_title": top_title,
                "top1_section": top1.get("section_path") if top1 else None,
                "hit@3": row.get("hit@3"),
                "category": category,
            }
        )

    return {
        "report_path": str(report_path.resolve()),
        "metrics": payload.get("metrics", {}),
        "supported_query_count": supported,
        "miss_count": len(misses),
        "miss_categories": dict(by_category),
        "miss_patterns": dict(by_pattern),
        "misses": misses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze retrieval failures on BOFIP clean-room evaluation reports.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    meta = load_meta(Path(args.raw_docs))
    analyses = [analyze_report(Path(report_path), meta) for report_path in args.reports]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_docs_path": str(Path(args.raw_docs).resolve()),
        "reports": analyses,
    }

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / f"phase3_failure_analysis_{Path(args.reports[0]).stem}.json"
    )
    write_json(report_path, summary)

    print(f"Failure analysis complete: {report_path}")
    for analysis in analyses:
        metrics = analysis["metrics"]
        print(
            f"{Path(analysis['report_path']).name}: "
            f"hit@1={metrics.get('hit@1')} hit@3={metrics.get('hit@3')} hit@5={metrics.get('hit@5')} "
            f"misses={analysis['miss_count']} categories={analysis['miss_categories']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.discovery import discover_content_documents
from bofip_cleanroom.document_builder import build_raw_document
from bofip_cleanroom.jsonio import write_json
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
    parser = argparse.ArgumentParser(description="Full parse audit on all BOFIP content docs.")
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for faster dry runs.")
    args = parser.parse_args()

    ensure_data_dirs()
    raw_root = get_raw_bofip_root(args.raw_root)
    documents = discover_content_documents(raw_root)
    if args.limit and args.limit > 0:
        documents = documents[: args.limit]

    failures: list[dict] = []
    content_type_counts: Counter[str] = Counter()
    section_counts: list[int] = []
    paragraph_counts: list[int] = []
    table_counts: list[int] = []
    depth_counts: list[int] = []
    docs_with_zero_paragraphs: list[str] = []
    docs_with_zero_sections: list[str] = []
    docs_with_synthetic_root_only: list[str] = []
    docs_with_tables: list[str] = []
    docs_by_content_type_quality: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    total = len(documents)
    for idx, document in enumerate(documents, start=1):
        try:
            parsed = build_raw_document(document)
        except Exception as exc:
            failures.append(
                {
                    "document_id": document.document_id,
                    "xml_path": str(document.xml_path),
                    "html_path": str(document.html_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        content_type = parsed.content_type or "UNKNOWN"
        content_type_counts[content_type] += 1
        section_count = len(parsed.sections)
        paragraph_count = len(parsed.paragraphs)
        table_count = len(parsed.tables)
        max_depth = max((section.level for section in parsed.sections), default=-1)

        section_counts.append(section_count)
        paragraph_counts.append(paragraph_count)
        table_counts.append(table_count)
        depth_counts.append(max_depth)

        if paragraph_count == 0:
            docs_with_zero_paragraphs.append(parsed.boi_reference)
            docs_by_content_type_quality[content_type]["zero_paragraphs"].append(parsed.boi_reference)
        if section_count == 0:
            docs_with_zero_sections.append(parsed.boi_reference)
            docs_by_content_type_quality[content_type]["zero_sections"].append(parsed.boi_reference)
        if section_count == 1 and parsed.sections[0].level == 0:
            docs_with_synthetic_root_only.append(parsed.boi_reference)
            docs_by_content_type_quality[content_type]["synthetic_root_only"].append(parsed.boi_reference)
        if table_count > 0:
            docs_with_tables.append(parsed.boi_reference)
            docs_by_content_type_quality[content_type]["has_tables"].append(parsed.boi_reference)

        if idx % 1000 == 0 or idx == total:
            print(f"[parse-audit] processed {idx}/{total}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_root": str(raw_root),
        "documents_considered": total,
        "success_count": total - len(failures),
        "failure_count": len(failures),
        "failure_rate": round(len(failures) / total, 6) if total else 0.0,
        "content_type_counts": dict(sorted(content_type_counts.items())),
        "section_count_stats": _stats(section_counts),
        "paragraph_count_stats": _stats(paragraph_counts),
        "table_count_stats": _stats(table_counts),
        "max_section_depth_stats": _stats(depth_counts),
        "docs_with_zero_paragraphs_count": len(docs_with_zero_paragraphs),
        "docs_with_zero_sections_count": len(docs_with_zero_sections),
        "docs_with_synthetic_root_only_count": len(docs_with_synthetic_root_only),
        "docs_with_tables_count": len(docs_with_tables),
        "docs_with_zero_paragraphs_sample": docs_with_zero_paragraphs[:50],
        "docs_with_zero_sections_sample": docs_with_zero_sections[:50],
        "docs_with_synthetic_root_only_sample": docs_with_synthetic_root_only[:50],
        "docs_with_tables_sample": docs_with_tables[:50],
        "content_type_quality": {
            content_type: {name: values[:50] for name, values in buckets.items()}
            for content_type, buckets in sorted(docs_by_content_type_quality.items())
        },
        "failures": failures[:100],
    }

    output_path = REPORTS_DIR / "phase0_full_parse_audit.json"
    write_json(output_path, report)
    print(f"Full parse audit written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

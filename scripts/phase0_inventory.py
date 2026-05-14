from __future__ import annotations

import argparse
import statistics
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.discovery import discover_attachment_documents, discover_content_documents
from bofip_cleanroom.html_parser import parse_html_structure
from bofip_cleanroom.jsonio import write_json
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs, get_raw_bofip_root
from bofip_cleanroom.versioning import build_manifest, fingerprint_paths
from bofip_cleanroom.xml_parser import parse_document_xml


PARSER_VERSION = "0.1.0"


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
    parser = argparse.ArgumentParser(description="Phase 0 BOFIP raw inventory.")
    parser.add_argument("--raw-root", type=str, required=True, help="Path to BOFiP/documents root containing Contenu and Attachment.")
    parser.add_argument("--html-sample-size", type=int, default=25)
    args = parser.parse_args()

    ensure_data_dirs()
    raw_root = get_raw_bofip_root(args.raw_root)
    content_documents = discover_content_documents(raw_root)
    attachment_documents = discover_attachment_documents(raw_root)

    content_type_counts: Counter[str] = Counter()
    publication_year_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    relation_type_counts: Counter[str] = Counter()
    title_lengths: list[int] = []
    html_byte_sizes: list[int] = []
    xml_paths: list[Path] = []
    html_paths: list[Path] = []

    for document in content_documents:
        metadata = parse_document_xml(document.xml_path)
        content_type_counts[metadata.get("content_type") or "UNKNOWN"] += 1
        publication_date = metadata.get("publication_date") or document.publication_date
        publication_year_counts[(publication_date or "UNKNOWN")[:4]] += 1
        for subject in metadata.get("subjects", []):
            subject_counts[subject] += 1
        for relation in metadata.get("relations", []):
            relation_type_counts[relation.get("relation_type") or "UNKNOWN"] += 1
        title_lengths.append(len(metadata.get("title") or ""))
        html_byte_sizes.append(document.html_path.stat().st_size)
        xml_paths.append(document.xml_path)
        html_paths.append(document.html_path)

    sample_structures: list[dict] = []
    for document in content_documents[: max(0, args.html_sample_size)]:
        parsed = parse_html_structure(document.html_path, document_id=document.document_id)
        sample_structures.append(
            {
                "document_id": document.document_id,
                "sections": len(parsed["sections"]),
                "paragraphs": len(parsed["paragraphs"]),
                "tables": len(parsed["tables"]),
                "internal_links": len(parsed["internal_links"]),
                "legal_refs": len(parsed["legal_refs"]),
            }
        )

    inventory = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_root": str(raw_root),
        "content_documents": len(content_documents),
        "attachment_documents": len(attachment_documents),
        "html_documents": len(content_documents),
        "content_type_counts": dict(sorted(content_type_counts.items())),
        "publication_year_counts": dict(sorted(publication_year_counts.items())),
        "subject_counts_top20": dict(subject_counts.most_common(20)),
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
        "title_length_stats": _stats(title_lengths),
        "html_byte_size_stats": _stats(html_byte_sizes),
        "sample_content_documents": [str(item.xml_path) for item in content_documents[:10]],
        "sample_structure_estimates": {
            "sample_size": len(sample_structures),
            "section_stats": _stats([item["sections"] for item in sample_structures]),
            "paragraph_stats": _stats([item["paragraphs"] for item in sample_structures]),
            "table_stats": _stats([item["tables"] for item in sample_structures]),
            "internal_link_stats": _stats([item["internal_links"] for item in sample_structures]),
            "legal_ref_stats": _stats([item["legal_refs"] for item in sample_structures]),
        },
    }

    manifest = build_manifest(
        raw_root=raw_root,
        parser_version=PARSER_VERSION,
        extra={
            "generated_at": inventory["generated_at"],
            "content_docs_fingerprint": fingerprint_paths(xml_paths + html_paths),
            "content_document_count": len(content_documents),
            "attachment_document_count": len(attachment_documents),
        },
    )

    write_json(REPORTS_DIR / "raw_inventory.json", inventory)
    write_json(REPORTS_DIR / "manifest.json", manifest)

    print(f"Inventory complete: {len(content_documents)} content docs, {len(attachment_documents)} attachments")
    print(f"Top content types: {content_type_counts.most_common(5)}")
    print(f"HTML sample size: {len(sample_structures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

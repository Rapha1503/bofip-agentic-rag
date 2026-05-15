from __future__ import annotations

import argparse
import shutil
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.discovery import discover_content_documents
from bofip_cleanroom.document_builder import build_raw_document, raw_document_to_tree_rows
from bofip_cleanroom.jsonio import write_json, write_jsonl
from bofip_cleanroom.sampling import stratified_sample_documents
from bofip_cleanroom.settings import INTERIM_DIR, RAW_DIR, REPORTS_DIR, ensure_data_dirs, get_raw_bofip_root
from bofip_cleanroom.versioning import build_manifest, fingerprint_paths


PARSER_VERSION = "0.1.0"


def _copy_sample_raw(raw_root: Path, sampled_docs: list, sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for document in sampled_docs:
        for source in (document.xml_path, document.html_path):
            rel = source.relative_to(raw_root)
            destination = sample_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 BOFIP raw extraction.")
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--content-type", action="append", default=[])
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    ensure_data_dirs()
    raw_root = get_raw_bofip_root(args.raw_root)
    all_docs = discover_content_documents(raw_root)
    sampled_docs = stratified_sample_documents(
        all_docs,
        args.sample_size,
        seed=args.seed,
        allowed_content_types=set(args.content_type) if args.content_type else None,
    )

    raw_documents = [build_raw_document(document) for document in sampled_docs]
    tree_rows = [row for document in raw_documents for row in raw_document_to_tree_rows(document)]

    sample_tag = f"sample_{len(raw_documents)}"
    raw_docs_path = INTERIM_DIR / f"raw_docs_{sample_tag}.jsonl"
    tree_path = INTERIM_DIR / f"doc_tree_{sample_tag}.jsonl"
    summary_path = REPORTS_DIR / f"phase1_extract_summary_{sample_tag}.json"
    manifest_path = REPORTS_DIR / f"manifest_phase1_{sample_tag}.json"

    write_jsonl(raw_docs_path, [document.to_dict() for document in raw_documents])
    write_jsonl(tree_path, tree_rows)

    if args.copy_raw:
        _copy_sample_raw(raw_root, sampled_docs, RAW_DIR / sample_tag)

    content_type_counts = Counter(document.content_type or "UNKNOWN" for document in raw_documents)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_size": len(raw_documents),
        "seed": args.seed,
        "raw_root": str(raw_root),
        "raw_docs_path": str(raw_docs_path),
        "doc_tree_path": str(tree_path),
        "content_type_counts": dict(sorted(content_type_counts.items())),
        "documents": [
            {
                "document_id": document.document_id,
                "boi_reference": document.boi_reference,
                "content_type": document.content_type,
                "publication_date": document.publication_date,
                "sections": len(document.sections),
                "paragraphs": len(document.paragraphs),
                "tables": len(document.tables),
                "internal_links": len(document.internal_links),
                "legal_refs": len(document.legal_refs),
            }
            for document in raw_documents
        ],
    }
    write_json(summary_path, summary)

    manifest = build_manifest(
        raw_root=raw_root,
        parser_version=PARSER_VERSION,
        extra={
            "generated_at": summary["generated_at"],
            "sample_size": len(raw_documents),
            "sample_fingerprint": fingerprint_paths([document.xml_path for document in sampled_docs] + [document.html_path for document in sampled_docs]),
            "seed": args.seed,
            "content_types_filter": list(args.content_type),
        },
    )
    write_json(manifest_path, manifest)

    print(f"Phase 1 extraction complete: {len(raw_documents)} docs")
    print(f"Raw docs: {raw_docs_path}")
    print(f"Doc tree: {tree_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

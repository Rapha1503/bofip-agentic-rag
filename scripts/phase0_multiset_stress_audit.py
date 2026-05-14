from __future__ import annotations

import argparse
from collections import Counter
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
from bofip_cleanroom.sampling import stratified_sample_documents
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


def _summarize_set(name: str, sample_docs: list) -> dict:
    raw_docs = [build_raw_document(doc) for doc in sample_docs]
    content_type_counts = Counter(doc.content_type or "UNKNOWN" for doc in raw_docs)
    section_depths = [max((section.level for section in doc.sections), default=-1) for doc in raw_docs]
    synthetic_root_only = [
        doc.boi_reference
        for doc in raw_docs
        if len(doc.sections) == 1 and doc.sections[0].level == 0
    ]
    return {
        "name": name,
        "sample_size": len(raw_docs),
        "content_type_counts": dict(sorted(content_type_counts.items())),
        "section_count_stats": _stats([len(doc.sections) for doc in raw_docs]),
        "paragraph_count_stats": _stats([len(doc.paragraphs) for doc in raw_docs]),
        "table_count_stats": _stats([len(doc.tables) for doc in raw_docs]),
        "max_section_depth_stats": _stats(section_depths),
        "docs_with_zero_sections": [doc.boi_reference for doc in raw_docs if len(doc.sections) == 0],
        "docs_with_zero_paragraphs": [doc.boi_reference for doc in raw_docs if len(doc.paragraphs) == 0],
        "docs_with_synthetic_root_only": synthetic_root_only,
        "docs_with_legal_refs": [doc.boi_reference for doc in raw_docs if doc.legal_refs],
        "sample_docs": [
            {
                "boi_reference": doc.boi_reference,
                "content_type": doc.content_type,
                "title": doc.title,
                "sections": len(doc.sections),
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables),
                "max_depth": max((section.level for section in doc.sections), default=-1),
            }
            for doc in raw_docs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-set structural stress audit for BOFIP clean-room.")
    parser.add_argument("--raw-root", type=str, required=True)
    args = parser.parse_args()

    ensure_data_dirs()
    raw_root = get_raw_bofip_root(args.raw_root)
    documents = discover_content_documents(raw_root)

    set_specs = [
        ("diverse_10_seed_0", 10, 0, None),
        ("diverse_10_seed_17", 10, 17, None),
        ("commentary_20_seed_7", 20, 7, {"Commentaire"}),
        ("commentary_20_seed_23", 20, 23, {"Commentaire"}),
        ("noncommentary_15_seed_5", 15, 5, {"Autres annexes", "Barème", "Cartographie", "Formulaire", "Lettre Type / Modèle"}),
    ]

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_root": str(raw_root),
        "sets": [],
    }

    for name, size, seed, allowed_types in set_specs:
        sample = stratified_sample_documents(documents, size, seed=seed, allowed_content_types=allowed_types)
        report["sets"].append(_summarize_set(name, sample))

    output_path = REPORTS_DIR / "phase0_multiset_stress_audit.json"
    write_json(output_path, report)
    print(f"Multi-set stress audit written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

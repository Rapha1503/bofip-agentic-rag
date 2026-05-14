from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl
from bofip_cleanroom.models import raw_document_from_dict
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1.5 fidelity report.")
    parser.add_argument("--raw-docs", type=str, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    ensure_data_dirs()
    documents = [raw_document_from_dict(item) for item in read_jsonl(Path(args.raw_docs))]
    target = documents[: max(0, args.limit)]

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"phase15_fidelity_report_{timestamp}.md"

    lines: list[str] = []
    lines.append("# Phase 1.5 Fidelity Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(UTC).isoformat()}")
    lines.append(f"Documents reviewed: {len(target)}")
    lines.append("")

    for document in target:
        lines.append(f"## {document.boi_reference} ({document.document_id})")
        lines.append("")
        lines.append(f"- Title: {document.title}")
        lines.append(f"- Content type: {document.content_type}")
        lines.append(f"- Publication date: {document.publication_date}")
        lines.append(f"- Source URL: {document.source_url}")
        lines.append(f"- Sections: {len(document.sections)}")
        lines.append(f"- Paragraphs: {len(document.paragraphs)}")
        lines.append(f"- Tables: {len(document.tables)}")
        lines.append(f"- Internal links: {len(document.internal_links)}")
        lines.append(f"- Legal refs: {len(document.legal_refs)}")
        lines.append("")
        lines.append("### Section Preview")
        for section in document.sections[:8]:
            path = " > ".join(section.path) if section.path else section.title
            lines.append(f"- L{section.level}: {path}")
        if not document.sections:
            lines.append("- No explicit section headings captured")
        lines.append("")
        lines.append("### Paragraph Preview")
        for paragraph in document.paragraphs[:8]:
            excerpt = paragraph.text[:300].replace("\n", " ")
            lines.append(f"- `{paragraph.paragraph_id}`: {excerpt}")
        if not document.paragraphs:
            lines.append("- No paragraphs captured")
        lines.append("")
        lines.append("### Table Preview")
        for table in document.tables[:3]:
            excerpt = table.linearized_text[:300].replace("\n", " / ")
            lines.append(f"- `{table.table_id}`: {excerpt}")
        if not document.tables:
            lines.append("- No tables captured")
        lines.append("")
        lines.append("### Manual Checklist")
        lines.append("- [ ] Structure coherent with source HTML")
        lines.append("- [ ] Texte fidèle et non tronqué")
        lines.append("- [ ] Tableaux lisibles")
        lines.append("- [ ] Remarques / encadrés visibles si présents")
        lines.append("- [ ] Renvois internes encore interprétables")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Fidelity report written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

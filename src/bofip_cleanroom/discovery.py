from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceDocumentPaths:
    document_id: str
    publication_date: str
    category_path: list[str]
    xml_path: Path
    html_path: Path


def discover_content_documents(raw_root: Path) -> list[SourceDocumentPaths]:
    content_root = raw_root / "Contenu"
    if not content_root.exists():
        raise FileNotFoundError(f"Missing Contenu directory under {raw_root}")

    docs: list[SourceDocumentPaths] = []
    for xml_path in sorted(content_root.rglob("document.xml")):
        html_path = xml_path.with_name("data.html")
        if not html_path.exists():
            html_path = xml_path.with_name("data.htm")
        if not html_path.exists():
            continue
        date_dir = xml_path.parent
        doc_dir = date_dir.parent
        document_id = doc_dir.name
        publication_date = date_dir.name
        category_path = list(doc_dir.relative_to(content_root).parts[:-1])
        docs.append(
            SourceDocumentPaths(
                document_id=document_id,
                publication_date=publication_date,
                category_path=category_path,
                xml_path=xml_path,
                html_path=html_path,
            )
        )
    return docs


def discover_attachment_documents(raw_root: Path) -> list[Path]:
    attachment_root = raw_root / "Attachment"
    if not attachment_root.exists():
        return []
    return sorted(attachment_root.rglob("document.xml"))

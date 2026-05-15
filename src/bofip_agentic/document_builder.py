from __future__ import annotations

from .discovery import SourceDocumentPaths
from .html_parser import parse_html_structure
from .models import (
    RawDocument,
    RawLink,
    RawParagraph,
    RawRelation,
    RawSectionNode,
    RawTable,
)
from .xml_parser import parse_document_xml


def build_raw_document(paths: SourceDocumentPaths) -> RawDocument:
    xml_payload = parse_document_xml(paths.xml_path)
    html_payload = parse_html_structure(paths.html_path, document_id=paths.document_id)

    sections = [RawSectionNode(**item) for item in html_payload["sections"]]
    paragraphs = [
        RawParagraph(
            paragraph_id=item["paragraph_id"],
            section_id=item.get("section_id"),
            order_index=item["order_index"],
            html_tag=item["html_tag"],
            anchor=item.get("anchor"),
            paragraph_number=item.get("paragraph_number"),
            text=item["text"],
            legal_refs=list(item.get("legal_refs", [])),
            links=[RawLink(**link) for link in item.get("links", [])],
        )
        for item in html_payload["paragraphs"]
    ]
    tables = [RawTable(**item) for item in html_payload["tables"]]
    internal_links = [RawLink(**item) for item in html_payload["internal_links"]]
    relations = [RawRelation(**item) for item in xml_payload["relations"]]

    raw_text_length = sum(len(item.text) for item in paragraphs) + sum(len(item.linearized_text) for item in tables)

    return RawDocument(
        document_id=xml_payload["document_id"] or paths.document_id,
        boi_reference=xml_payload["boi_reference"],
        title=xml_payload["title"] or html_payload["html_title"] or paths.document_id,
        document_type=xml_payload["document_type"] or "Contenu",
        content_type=xml_payload.get("content_type"),
        publication_date=xml_payload.get("publication_date") or paths.publication_date,
        source_url=xml_payload.get("source_url"),
        language=xml_payload.get("language"),
        subjects=list(xml_payload.get("subjects", [])),
        identifiers=list(xml_payload.get("identifiers", [])),
        relations=relations,
        category_path=list(paths.category_path),
        raw_xml_path=str(paths.xml_path),
        raw_html_path=str(paths.html_path),
        version_status=xml_payload.get("version_status"),
        sections=sections,
        paragraphs=paragraphs,
        tables=tables,
        internal_links=internal_links,
        legal_refs=list(html_payload.get("legal_refs", [])),
        html_title=html_payload.get("html_title"),
        raw_text_length=raw_text_length,
    )


def raw_document_to_tree_rows(document: RawDocument) -> list[dict]:
    rows: list[dict] = []
    for section in document.sections:
        rows.append(
            {
                "document_id": document.document_id,
                "boi_reference": document.boi_reference,
                "node_type": "section",
                "node_id": section.section_id,
                "parent_node_id": section.parent_section_id,
                "order_index": section.order_index,
                "title": section.title,
                "path": section.path,
            }
        )
    for paragraph in document.paragraphs:
        rows.append(
            {
                "document_id": document.document_id,
                "boi_reference": document.boi_reference,
                "node_type": "paragraph",
                "node_id": paragraph.paragraph_id,
                "parent_node_id": paragraph.section_id,
                "order_index": paragraph.order_index,
                "title": None,
                "path": None,
                "paragraph_number": paragraph.paragraph_number,
                "text": paragraph.text,
                "legal_refs": paragraph.legal_refs,
            }
        )
    for table in document.tables:
        rows.append(
            {
                "document_id": document.document_id,
                "boi_reference": document.boi_reference,
                "node_type": "table",
                "node_id": table.table_id,
                "parent_node_id": table.section_id,
                "order_index": table.order_index,
                "title": table.caption,
                "path": None,
                "text": table.linearized_text,
            }
        )
    return rows

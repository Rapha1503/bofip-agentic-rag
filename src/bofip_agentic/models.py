from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RawRelation:
    value: str
    relation_type: str | None = None


@dataclass
class RawLink:
    text: str
    href: str
    is_internal: bool
    source_anchor: str | None = None


@dataclass
class RawTable:
    table_id: str
    section_id: str | None
    order_index: int
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str | None = None
    linearized_text: str = ""


@dataclass
class RawParagraph:
    paragraph_id: str
    section_id: str | None
    order_index: int
    html_tag: str
    anchor: str | None
    paragraph_number: str | None
    text: str
    legal_refs: list[str] = field(default_factory=list)
    links: list[RawLink] = field(default_factory=list)


@dataclass
class RawSectionNode:
    section_id: str
    parent_section_id: str | None
    level: int
    order_index: int
    title: str
    anchor: str | None
    path: list[str] = field(default_factory=list)


@dataclass
class RawDocument:
    document_id: str
    boi_reference: str
    title: str
    document_type: str
    content_type: str | None
    publication_date: str | None
    source_url: str | None
    language: str | None
    subjects: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    relations: list[RawRelation] = field(default_factory=list)
    category_path: list[str] = field(default_factory=list)
    raw_xml_path: str = ""
    raw_html_path: str = ""
    version_status: str | None = None
    sections: list[RawSectionNode] = field(default_factory=list)
    paragraphs: list[RawParagraph] = field(default_factory=list)
    tables: list[RawTable] = field(default_factory=list)
    internal_links: list[RawLink] = field(default_factory=list)
    legal_refs: list[str] = field(default_factory=list)
    html_title: str | None = None
    raw_text_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkNode:
    chunk_id: str
    source_type: str
    document_id: str
    boi_reference: str
    doc_version: str | None
    strategy: str
    section_id: str | None
    parent_chunk_id: str | None
    section_path: list[str]
    paragraph_range: list[str]
    text: str
    token_count: int
    chunk_kind: str
    legal_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def raw_link_from_dict(payload: dict[str, Any]) -> RawLink:
    return RawLink(**payload)


def raw_relation_from_dict(payload: dict[str, Any]) -> RawRelation:
    return RawRelation(**payload)


def raw_table_from_dict(payload: dict[str, Any]) -> RawTable:
    return RawTable(**payload)


def raw_paragraph_from_dict(payload: dict[str, Any]) -> RawParagraph:
    return RawParagraph(
        paragraph_id=payload["paragraph_id"],
        section_id=payload.get("section_id"),
        order_index=payload["order_index"],
        html_tag=payload["html_tag"],
        anchor=payload.get("anchor"),
        paragraph_number=payload.get("paragraph_number"),
        text=payload["text"],
        legal_refs=list(payload.get("legal_refs", [])),
        links=[raw_link_from_dict(item) for item in payload.get("links", [])],
    )


def raw_section_from_dict(payload: dict[str, Any]) -> RawSectionNode:
    return RawSectionNode(
        section_id=payload["section_id"],
        parent_section_id=payload.get("parent_section_id"),
        level=payload["level"],
        order_index=payload["order_index"],
        title=payload["title"],
        anchor=payload.get("anchor"),
        path=list(payload.get("path", [])),
    )


def raw_document_from_dict(payload: dict[str, Any]) -> RawDocument:
    return RawDocument(
        document_id=payload["document_id"],
        boi_reference=payload["boi_reference"],
        title=payload["title"],
        document_type=payload["document_type"],
        content_type=payload.get("content_type"),
        publication_date=payload.get("publication_date"),
        source_url=payload.get("source_url"),
        language=payload.get("language"),
        subjects=list(payload.get("subjects", [])),
        identifiers=list(payload.get("identifiers", [])),
        relations=[raw_relation_from_dict(item) for item in payload.get("relations", [])],
        category_path=list(payload.get("category_path", [])),
        raw_xml_path=payload.get("raw_xml_path", ""),
        raw_html_path=payload.get("raw_html_path", ""),
        version_status=payload.get("version_status"),
        sections=[raw_section_from_dict(item) for item in payload.get("sections", [])],
        paragraphs=[raw_paragraph_from_dict(item) for item in payload.get("paragraphs", [])],
        tables=[raw_table_from_dict(item) for item in payload.get("tables", [])],
        internal_links=[raw_link_from_dict(item) for item in payload.get("internal_links", [])],
        legal_refs=list(payload.get("legal_refs", [])),
        html_title=payload.get("html_title"),
        raw_text_length=payload.get("raw_text_length", 0),
    )


def chunk_node_from_dict(payload: dict[str, Any]) -> ChunkNode:
    return ChunkNode(
        chunk_id=payload["chunk_id"],
        source_type=payload["source_type"],
        document_id=payload["document_id"],
        boi_reference=payload["boi_reference"],
        doc_version=payload.get("doc_version"),
        strategy=payload["strategy"],
        section_id=payload.get("section_id"),
        parent_chunk_id=payload.get("parent_chunk_id"),
        section_path=list(payload.get("section_path", [])),
        paragraph_range=list(payload.get("paragraph_range", [])),
        text=payload["text"],
        token_count=payload["token_count"],
        chunk_kind=payload["chunk_kind"],
        legal_refs=list(payload.get("legal_refs", [])),
    )

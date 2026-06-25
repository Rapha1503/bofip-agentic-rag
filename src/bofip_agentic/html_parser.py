from __future__ import annotations

from pathlib import Path
import re

from bs4 import BeautifulSoup, Tag

from .models import RawLink, RawParagraph, RawSectionNode, RawTable
from .text_utils import extract_legal_refs, normalize_whitespace, slugify


HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
TEXT_BLOCK_TAGS = {"p", "li", "blockquote"}
PARAGRAPH_NUMBER_RE = re.compile(r"^[§]?\d+(?:\s?[A-Za-z])?$")


def _tag_text(tag: Tag) -> str:
    return normalize_whitespace(tag.get_text(" ", strip=True))


def _anchor(tag: Tag) -> str | None:
    value = normalize_whitespace(tag.get("name", "") or tag.get("id", ""))
    return value or None


def _build_section_id(document_id: str, order_index: int, title: str) -> str:
    return f"{document_id}__section__{order_index:04d}__{slugify(title)}"


def parse_html_structure(html_path: Path, *, document_id: str) -> dict:
    return parse_html_content(html_path.read_text(encoding="utf-8", errors="replace"), document_id=document_id)


def parse_html_content(html: str, *, document_id: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    html_title = normalize_whitespace(soup.title.get_text(" ", strip=True) if soup.title else "") or None

    sections: list[RawSectionNode] = []
    paragraphs: list[RawParagraph] = []
    tables: list[RawTable] = []
    internal_links: list[RawLink] = []
    legal_refs: list[str] = []

    current_section_id: str | None = None
    section_stack: list[RawSectionNode] = []
    section_counter = 0
    paragraph_counter = 0
    table_counter = 0
    pending_paragraph_number: str | None = None

    for tag in body.find_all(list(HEADING_TAGS | TEXT_BLOCK_TAGS | {"table"})):
        if not isinstance(tag, Tag):
            continue

        if tag.name in HEADING_TAGS:
            title = _tag_text(tag)
            if not title:
                continue
            level = int(tag.name[1])
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()
            parent_section_id = section_stack[-1].section_id if section_stack else None
            path = [node.title for node in section_stack] + [title]
            section_id = _build_section_id(document_id, section_counter, title)
            section = RawSectionNode(
                section_id=section_id,
                parent_section_id=parent_section_id,
                level=level,
                order_index=section_counter,
                title=title,
                anchor=_anchor(tag),
                path=path,
            )
            sections.append(section)
            section_stack.append(section)
            current_section_id = section_id
            section_counter += 1
            continue

        if tag.name == "table":
            rows: list[list[str]] = []
            headers: list[str] = []
            for tr in tag.find_all("tr"):
                cells = [_tag_text(cell) for cell in tr.find_all(["th", "td"])]
                cells = [cell for cell in cells if cell]
                if not cells:
                    continue
                if tr.find("th") and not headers:
                    headers = cells
                else:
                    rows.append(cells)
            linearized_parts: list[str] = []
            if headers:
                linearized_parts.append(" | ".join(headers))
            for row in rows:
                linearized_parts.append(" | ".join(row))
            linearized_text = "\n".join(linearized_parts)
            if linearized_text:
                tables.append(
                    RawTable(
                        table_id=f"{document_id}__table__{table_counter:04d}",
                        section_id=current_section_id,
                        order_index=table_counter,
                        headers=headers,
                        rows=rows,
                        caption=None,
                        linearized_text=linearized_text,
                    )
                )
                legal_refs.extend(extract_legal_refs(linearized_text))
                table_counter += 1
            continue

        text = _tag_text(tag)
        if not text:
            continue

        if tag.name == "p" and PARAGRAPH_NUMBER_RE.fullmatch(text):
            pending_paragraph_number = text
            continue

        links: list[RawLink] = []
        for anchor_tag in tag.find_all("a", href=True):
            link = RawLink(
                text=_tag_text(anchor_tag),
                href=anchor_tag.get("href", ""),
                is_internal=anchor_tag.get("href", "").startswith("#"),
                source_anchor=_anchor(tag),
            )
            links.append(link)
            internal_links.append(link)

        paragraph = RawParagraph(
            paragraph_id=f"{document_id}__paragraph__{paragraph_counter:05d}",
            section_id=current_section_id,
            order_index=paragraph_counter,
            html_tag=tag.name,
            anchor=_anchor(tag),
            paragraph_number=pending_paragraph_number,
            text=text,
            legal_refs=extract_legal_refs(text),
            links=links,
        )
        paragraphs.append(paragraph)
        legal_refs.extend(paragraph.legal_refs)
        paragraph_counter += 1
        pending_paragraph_number = None

    deduped_refs: list[str] = []
    seen = set()
    for ref in legal_refs:
        key = ref.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_refs.append(ref)

    orphan_paragraphs = [paragraph for paragraph in paragraphs if paragraph.section_id is None]
    orphan_tables = [table for table in tables if table.section_id is None]
    if orphan_paragraphs or orphan_tables:
        synthetic_title = html_title or document_id
        synthetic_section_id = _build_section_id(document_id, section_counter, synthetic_title)
        synthetic_section = RawSectionNode(
            section_id=synthetic_section_id,
            parent_section_id=None,
            level=0,
            order_index=section_counter,
            title=synthetic_title,
            anchor=None,
            path=[synthetic_title],
        )
        sections.insert(0, synthetic_section)
        for paragraph in orphan_paragraphs:
            paragraph.section_id = synthetic_section_id
        for table in orphan_tables:
            table.section_id = synthetic_section_id

    return {
        "html_title": html_title,
        "sections": [section.__dict__ for section in sections],
        "paragraphs": [
            {
                **paragraph.__dict__,
                "links": [link.__dict__ for link in paragraph.links],
            }
            for paragraph in paragraphs
        ],
        "tables": [table.__dict__ for table in tables],
        "internal_links": [link.__dict__ for link in internal_links],
        "legal_refs": deduped_refs,
    }

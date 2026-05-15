from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .models import RawRelation
from .text_utils import normalize_whitespace


DC_NS = {"dc": "http://purl.org/dc/elements/1.1"}
BOFIP_NS = {"bofip": "https://bofip.impots.gouv.fr"}


def _findall_text(root: ET.Element, xpath: str, namespaces: dict[str, str]) -> list[str]:
    values: list[str] = []
    for node in root.findall(xpath, namespaces):
        text = normalize_whitespace(node.text or "")
        if text:
            values.append(text)
    return values


def parse_document_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()

    identifiers = _findall_text(root, ".//dc:identifier", DC_NS)
    urls = [value for value in identifiers if value.startswith("http")]
    opaque_ids = [value for value in identifiers if not value.startswith("http")]

    relations: list[RawRelation] = []
    for node in root.findall(".//dc:relation", DC_NS):
        value = normalize_whitespace(node.text or "")
        if not value:
            continue
        relations.append(
            RawRelation(
                value=value,
                relation_type=normalize_whitespace(node.attrib.get("type", "")) or None,
            )
        )

    content_type_node = root.find(".//bofip:contenu_type", BOFIP_NS)
    content_id_node = root.find(".//bofip:contenu_id", BOFIP_NS)
    title_node = root.find(".//dc:title", DC_NS)
    date_node = root.find(".//dc:date", DC_NS)
    language_node = root.find(".//dc:language", DC_NS)
    data_ref_node = root.find(".//parts/part/dataRef")

    title = normalize_whitespace(title_node.text if title_node is not None else "")
    publication_date = normalize_whitespace(date_node.text if date_node is not None else "") or None
    content_type = normalize_whitespace(content_type_node.text if content_type_node is not None else "") or None
    boi_reference = normalize_whitespace(content_id_node.text if content_id_node is not None else "") or title

    return {
        "document_type": normalize_whitespace(root.attrib.get("type", "")) or None,
        "title": title,
        "boi_reference": boi_reference,
        "publication_date": publication_date,
        "language": normalize_whitespace(language_node.text if language_node is not None else "") or None,
        "source_url": urls[0] if urls else None,
        "subjects": _findall_text(root, ".//dc:subject", DC_NS),
        "identifiers": identifiers,
        "document_id": opaque_ids[0] if opaque_ids else path.parent.parent.name,
        "relations": [relation.__dict__ for relation in relations],
        "content_type": content_type,
        "data_ref": normalize_whitespace(data_ref_node.text if data_ref_node is not None else "") or None,
        "version_status": None,
    }

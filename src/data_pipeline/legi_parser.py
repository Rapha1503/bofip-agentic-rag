"""
LEGI XML parser (minimal, KISS).

Parses local LEGI XML article files into BOFIPChunk-compatible records.
This is a non-breaking addition to start migrating away from PDF parsing.
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from src.data_pipeline.chunker import BOFIPChunk

logger = logging.getLogger(__name__)


def _local_name(tag: str) -> str:
    """Return XML local name without namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _iter_text_values(root: ET.Element, names: set[str]) -> List[str]:
    values = []
    names_upper = {n.upper() for n in names}
    for elem in root.iter():
        if _local_name(elem.tag).upper() in names_upper:
            text = "".join(elem.itertext()).strip()
            if text:
                values.append(text)
    return values


def _first_text(root: ET.Element, names: set[str]) -> Optional[str]:
    vals = _iter_text_values(root, names)
    return vals[0] if vals else None


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


class LEGIArticleParser:
    """Parse LEGI XML files into chunks."""

    # Field names are intentionally broad to support schema variants.
    ARTICLE_NUM_FIELDS = {"NUM", "NUM_ARTICLE", "NUMERO", "ARTICLE_NUM"}
    ARTICLE_ID_FIELDS = {"ID", "ID_ARTI", "ID_ARTICLE", "CID"}
    TITLE_FIELDS = {"TITRE", "TITRE_TM", "TITLE", "INTITULE"}
    DATE_FIELDS = {"DATE_DEBUT", "DATE_FIN", "DATE_MAJ", "DATE_VERSION", "VIGUEUR"}
    CODE_NAME_FIELDS = {"NATURE", "NOM_CODE", "CODE", "TITRE_TA", "TITRE_CODE"}
    BODY_FIELDS = {"BLOC_TEXTUEL", "CONTENU", "TEXTE", "ALINEA", "PARAGRAPHE", "P"}

    def parse_xml_file(self, xml_path: Path) -> Optional[BOFIPChunk]:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as exc:
            logger.warning(f"Failed to parse XML {xml_path}: {exc}")
            return None

        article_num = _first_text(root, self.ARTICLE_NUM_FIELDS)
        article_id = _first_text(root, self.ARTICLE_ID_FIELDS) or xml_path.stem
        section_title = _first_text(root, self.TITLE_FIELDS)
        publication_date = _first_text(root, self.DATE_FIELDS) or ""
        code_name = _first_text(root, self.CODE_NAME_FIELDS) or ""

        body_blocks = _iter_text_values(root, self.BODY_FIELDS)
        body_text = _clean_text(" ".join(body_blocks))
        if not body_text:
            # Fallback: use full document text if no dedicated body fields found.
            body_text = _clean_text(" ".join(root.itertext()))

        if not body_text:
            logger.warning(f"Skipping XML without text: {xml_path}")
            return None

        content_type, source = self._classify_source(code_name, body_text)
        normalized_article = self._normalize_article_number(article_num)
        boi_reference = self._build_reference(content_type, normalized_article, article_id)
        doc_id = f"{content_type}-{article_id}" if content_type in ("CGI", "LPF") else f"LEGI-{article_id}"

        context = []
        if section_title:
            context.append(f"[{section_title}]")
        if normalized_article:
            context.append(f"Article {normalized_article}")
        context.append(body_text)
        text_with_context = "\n".join(context)

        chunk_hash = hashlib.md5((article_id + body_text).encode("utf-8")).hexdigest()[:8]
        chunk_id = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{doc_id}_{normalized_article or 'article'}_{chunk_hash}")

        source_url = ""
        if article_id.startswith("LEGIARTI"):
            source_url = f"https://www.legifrance.gouv.fr/codes/article_lc/{article_id}"

        return BOFIPChunk(
            chunk_id=chunk_id,
            text=body_text,
            text_with_context=text_with_context,
            boi_reference=boi_reference,
            doc_id=doc_id,
            series=[],
            section_title=section_title,
            paragraph_number=None,
            publication_date=publication_date,
            source_url=source_url,
            content_type=content_type,
            contains_table=False,
            is_header=False,
            token_count=_estimate_tokens(body_text),
            source=source,
        )

    def parse_directory(self, input_dir: Path) -> List[BOFIPChunk]:
        chunks: List[BOFIPChunk] = []
        xml_files = sorted(input_dir.rglob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML files in {input_dir}")

        for xml_path in xml_files:
            chunk = self.parse_xml_file(xml_path)
            if chunk:
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} LEGI chunks")
        return chunks

    @staticmethod
    def _normalize_article_number(article_num: Optional[str]) -> Optional[str]:
        if not article_num:
            return None
        ref = _clean_text(article_num).rstrip(".")
        match = re.match(r"^([LRA])(\*)?\.?\s*(.+)$", ref, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            star = match.group(2) or ""
            rest = match.group(3).strip()
            return f"{letter}{star}. {rest}"
        return ref

    @staticmethod
    def _classify_source(code_name: str, body_text: str) -> Tuple[str, str]:
        sample = f"{code_name} {body_text[:400]}".lower()
        if "proc" in sample and "fiscal" in sample:
            return "LPF", "LPF"
        if "impot" in sample or "imp\u00f4t" in sample:
            return "CGI", "CGI"
        return "LEGI", "LEGI"

    @staticmethod
    def _build_reference(content_type: str, article_num: Optional[str], article_id: str) -> str:
        if article_num:
            if content_type in ("CGI", "LPF"):
                return f"{content_type} Art. {article_num}"
            return f"LEGI Art. {article_num}"
        return f"{content_type} {article_id}"

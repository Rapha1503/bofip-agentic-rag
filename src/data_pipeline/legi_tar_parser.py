"""
Parse LEGI tar archives for targeted French legal codes (CGI / LPF).

This parser is designed for DILA LEGI dumps:
- daily archives: LEGI_YYYYMMDD-HHMMSS.tar.gz
- full archive:   Freemium_legi_global_YYYYMMDD-HHMMSS.tar.gz

It extracts article XML files directly from the tar stream and keeps one
"as-of" version per article number.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.data_pipeline.chunker import BOFIPChunk

logger = logging.getLogger(__name__)


# Canonical LEGI text IDs for tax codes.
DEFAULT_CODE_IDS: Dict[str, str] = {
    "CGI": "LEGITEXT000006069577",
    "LPF": "LEGITEXT000006069583",
}


@dataclass
class _ArticleCandidate:
    source: str
    code_id: str
    article_id: str
    article_num: str
    article_num_normalized: str
    state: str
    date_start: Optional[dt.date]
    date_end: Optional[dt.date]
    code_title: str
    section_path: List[str]
    body_text: str


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _iter_by_local_name(root: ET.Element, name: str) -> Iterable[ET.Element]:
    target = name.upper()
    for elem in root.iter():
        if _local_name(elem.tag).upper() == target:
            yield elem


def _first_text(root: ET.Element, names: Iterable[str]) -> str:
    wanted = {n.upper() for n in names}
    for elem in root.iter():
        if _local_name(elem.tag).upper() in wanted:
            text = _normalize_space("".join(elem.itertext()))
            if text:
                return text
    return ""


def _normalize_article_number(article_num: str) -> str:
    cleaned = _normalize_space(article_num).rstrip(".")
    # LPF refs often appear as "L64", "R57-1", "L. 64", "R*57-1".
    m = re.match(r"^([LRA])(\*)?\.?\s*([0-9].+)$", cleaned, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        star = m.group(2) or ""
        rest = _normalize_space(m.group(3))
        return f"{letter}{star}. {rest}"
    return cleaned


def _extract_code_title(root: ET.Element) -> str:
    # Prefer title nodes with id_txt LEGITEXT..., but keep fallback generic.
    for elem in _iter_by_local_name(root, "TITRE_TXT"):
        text = _normalize_space("".join(elem.itertext()))
        if text:
            return text
    return ""


def _extract_section_path(root: ET.Element) -> List[str]:
    titles = []
    for elem in _iter_by_local_name(root, "TITRE_TM"):
        text = _normalize_space("".join(elem.itertext()))
        if text:
            titles.append(text)
    return titles


def _extract_body_text(root: ET.Element) -> str:
    blocks: List[str] = []
    for bloc in _iter_by_local_name(root, "BLOC_TEXTUEL"):
        for contenu in _iter_by_local_name(bloc, "CONTENU"):
            text = _normalize_space("".join(contenu.itertext()))
            if text:
                blocks.append(text)

    if not blocks:
        # Fallback for schema variations.
        for contenu in _iter_by_local_name(root, "CONTENU"):
            text = _normalize_space("".join(contenu.itertext()))
            if text:
                blocks.append(text)

    return _normalize_space("\n".join(blocks))


def _is_as_of(candidate: _ArticleCandidate, as_of: dt.date) -> bool:
    start = candidate.date_start or dt.date.min
    end = candidate.date_end or dt.date.max
    if candidate.state == "ABROGE":
        return False
    return start <= as_of <= end


def _select_best_candidate(
    candidates: List[_ArticleCandidate],
    as_of: dt.date,
    include_future: bool,
) -> Optional[_ArticleCandidate]:
    if not candidates:
        return None

    current = [c for c in candidates if _is_as_of(c, as_of)]
    if current:
        # If multiple match, choose the one that starts most recently.
        return sorted(
            current,
            key=lambda c: (c.date_start or dt.date.min, c.date_end or dt.date.max, c.article_id),
            reverse=True,
        )[0]

    if include_future:
        future = [
            c
            for c in candidates
            if c.state != "ABROGE" and c.date_start and c.date_start > as_of
        ]
        if future:
            # Nearest upcoming version.
            return sorted(
                future,
                key=lambda c: (c.date_start, c.article_id),
            )[0]

    return None


class LEGITarCodeParser:
    """
    Parse targeted code articles from LEGI tar archives.

    Output chunks are BOFIPChunk-compatible with:
    - source/content_type in {"CGI", "LPF"}
    - boi_reference as "CGI Art. X" / "LPF Art. L. 64"
    """

    def __init__(self, code_ids: Optional[Dict[str, str]] = None):
        self.code_ids = dict(DEFAULT_CODE_IDS)
        if code_ids:
            self.code_ids.update(code_ids)

    def parse_archive(
        self,
        archive_path: Path,
        target_sources: Iterable[str] = ("CGI", "LPF"),
        as_of: Optional[dt.date] = None,
        include_future: bool = False,
    ) -> Dict[str, List[BOFIPChunk]]:
        if not archive_path.exists():
            raise FileNotFoundError(f"LEGI archive not found: {archive_path}")

        as_of = as_of or dt.date.today()
        target_sources = [s.upper() for s in target_sources]
        unknown = [s for s in target_sources if s not in self.code_ids]
        if unknown:
            raise ValueError(f"Unknown target source(s): {unknown}")

        markers = {
            source: f"/{self.code_ids[source]}/article/LEGI/ARTI/"
            for source in target_sources
        }
        stats_scanned = {source: 0 for source in target_sources}
        candidates: Dict[str, Dict[str, List[_ArticleCandidate]]] = {
            source: {} for source in target_sources
        }

        logger.info(f"Parsing LEGI archive: {archive_path}")
        logger.info(f"Target sources: {target_sources} | as_of={as_of} | include_future={include_future}")

        with tarfile.open(archive_path, mode="r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".xml"):
                    continue

                source = None
                for s, marker in markers.items():
                    if marker in member.name:
                        source = s
                        break
                if not source:
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    continue

                try:
                    xml_bytes = extracted.read()
                    candidate = self._parse_article_xml(
                        xml_bytes=xml_bytes,
                        source=source,
                        code_id=self.code_ids[source],
                    )
                except Exception as exc:
                    logger.warning(f"Failed to parse article XML {member.name}: {exc}")
                    continue

                stats_scanned[source] += 1
                if not candidate:
                    continue

                bucket = candidates[source].setdefault(candidate.article_num_normalized, [])
                bucket.append(candidate)

        result: Dict[str, List[BOFIPChunk]] = {source: [] for source in target_sources}
        for source in target_sources:
            selected = 0
            skipped = 0
            for article_num, versions in candidates[source].items():
                best = _select_best_candidate(versions, as_of=as_of, include_future=include_future)
                if not best:
                    skipped += 1
                    continue
                result[source].append(self._candidate_to_chunk(best))
                selected += 1

            result[source].sort(
                key=lambda c: c.boi_reference
            )
            logger.info(
                f"{source}: scanned_xml={stats_scanned[source]} | "
                f"unique_articles={len(candidates[source])} | selected={selected} | skipped={skipped}"
            )

        return result

    def _parse_article_xml(self, xml_bytes: bytes, source: str, code_id: str) -> Optional[_ArticleCandidate]:
        root = ET.fromstring(xml_bytes)

        article_id = _first_text(root, {"ID", "ID_ARTI", "ID_ARTICLE"})
        article_num = _first_text(root, {"NUM", "NUM_ARTICLE", "NUMERO"})
        if not article_id or not article_num:
            return None

        article_num_normalized = _normalize_article_number(article_num)
        state = _first_text(root, {"ETAT"}).upper()
        date_start = _parse_date(_first_text(root, {"DATE_DEBUT"}))
        date_end = _parse_date(_first_text(root, {"DATE_FIN"}))
        code_title = _extract_code_title(root)
        section_path = _extract_section_path(root)
        body_text = _extract_body_text(root)
        if not body_text:
            return None

        return _ArticleCandidate(
            source=source,
            code_id=code_id,
            article_id=article_id,
            article_num=article_num,
            article_num_normalized=article_num_normalized,
            state=state,
            date_start=date_start,
            date_end=date_end,
            code_title=code_title,
            section_path=section_path,
            body_text=body_text,
        )

    def _candidate_to_chunk(self, candidate: _ArticleCandidate) -> BOFIPChunk:
        boi_reference = f"{candidate.source} Art. {candidate.article_num_normalized}"
        section_title = candidate.section_path[-1] if candidate.section_path else None

        context_parts: List[str] = []
        if candidate.code_title:
            context_parts.append(f"[{candidate.code_title}]")
        if candidate.section_path:
            context_parts.append(" > ".join(candidate.section_path))
        context_parts.append(f"Article {candidate.article_num_normalized}")
        context_parts.append(candidate.body_text)
        text_with_context = "\n".join(context_parts)

        publication_date = candidate.date_start.isoformat() if candidate.date_start else ""
        source_url = f"https://www.legifrance.gouv.fr/codes/article_lc/{candidate.article_id}"

        return BOFIPChunk(
            chunk_id=f"{candidate.source}_{candidate.article_id}",
            text=candidate.body_text,
            text_with_context=text_with_context,
            boi_reference=boi_reference,
            doc_id=f"{candidate.source}-{candidate.code_id}",
            series=[],
            section_title=section_title,
            paragraph_number=None,
            publication_date=publication_date,
            source_url=source_url,
            content_type=candidate.source,
            contains_table=False,
            is_header=False,
            token_count=_estimate_tokens(candidate.body_text),
            source=candidate.source,
        )

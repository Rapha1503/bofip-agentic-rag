from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

from .models import ChunkNode, RawDocument, RawParagraph, RawTable
from .text_utils import estimate_token_count, extract_legal_refs, normalize_whitespace, slugify


SUPPORTED_STRATEGIES = {"paragraph_preserving", "section_window", "parent_child"}
SOFT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\:\;])\s+|\n+")
TRIVIAL_FRAGMENT_RE = re.compile(
    r"^(?:"
    r"[\W_]+|"
    r"\(\d+(?:[-–]\d+)?\)|"
    r"\d+(?:[-–]\d+)?|"
    r"N(?:[+-]\d+)?|"
    r"Article\s+\d+(?:\s+et\s+\d+)?|"
    r"Cf\..*|"
    r"Remarques?\s*:?"
    r")$",
    re.IGNORECASE,
)


@dataclass
class ParagraphUnit:
    section_id: str | None
    paragraph_ids: list[str]
    text: str
    legal_refs: list[str]


def _section_lookup(document: RawDocument) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for section in document.sections:
        lookup[section.section_id] = list(section.path)
    return lookup


def _section_path(document: RawDocument, section_id: str | None) -> list[str]:
    if not section_id:
        return [document.title]
    lookup = _section_lookup(document)
    return lookup.get(section_id, [document.title])


def _dedupe_refs(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _is_trivial_fragment(text: str) -> bool:
    token_count = estimate_token_count(text)
    if token_count > 5:
        return False
    return bool(TRIVIAL_FRAGMENT_RE.fullmatch(text) or len(text) <= 12)


def _build_paragraph_units(paragraphs: list[RawParagraph]) -> list[ParagraphUnit]:
    units: list[ParagraphUnit] = []
    pending_ids: list[str] = []
    pending_texts: list[str] = []
    pending_refs: list[str] = []

    for paragraph in paragraphs:
        text = normalize_whitespace(paragraph.text)
        if not text:
            continue
        if _is_trivial_fragment(text):
            pending_ids.append(paragraph.paragraph_id)
            pending_texts.append(text)
            pending_refs.extend(paragraph.legal_refs)
            continue

        paragraph_ids = pending_ids + [paragraph.paragraph_id]
        text_parts = pending_texts + [text]
        legal_refs = _dedupe_refs(pending_refs + list(paragraph.legal_refs))
        units.append(
            ParagraphUnit(
                section_id=paragraph.section_id,
                paragraph_ids=paragraph_ids,
                text="\n\n".join(text_parts),
                legal_refs=legal_refs,
            )
        )
        pending_ids = []
        pending_texts = []
        pending_refs = []

    if pending_texts:
        if units:
            last = units[-1]
            last.paragraph_ids.extend(pending_ids)
            last.text = "\n\n".join([last.text] + pending_texts)
            last.legal_refs = _dedupe_refs(last.legal_refs + pending_refs)
        else:
            units.append(
                ParagraphUnit(
                    section_id=paragraphs[0].section_id if paragraphs else None,
                    paragraph_ids=list(pending_ids),
                    text="\n\n".join(pending_texts),
                    legal_refs=_dedupe_refs(pending_refs),
                )
            )

    return units


def _paragraph_groups(document: RawDocument) -> list[tuple[str | None, list[RawParagraph]]]:
    grouped: dict[str | None, list[RawParagraph]] = defaultdict(list)
    order: list[str | None] = []
    for paragraph in document.paragraphs:
        if paragraph.section_id not in grouped:
            order.append(paragraph.section_id)
        grouped[paragraph.section_id].append(paragraph)
    return [(section_id, grouped[section_id]) for section_id in order]


def _split_text_for_limit(text: str, max_tokens: int) -> list[str]:
    normalized = normalize_whitespace(text)
    if estimate_token_count(normalized) <= max_tokens:
        return [normalized]

    pieces = [piece.strip() for piece in SOFT_SPLIT_RE.split(normalized) if piece.strip()]
    if not pieces:
        pieces = normalized.split()

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if current and estimate_token_count(candidate) > max_tokens:
            chunks.append(current)
            current = piece
        else:
            current = candidate

    if current:
        chunks.append(current)

    final_chunks: list[str] = []
    for chunk in chunks:
        words = chunk.split()
        if estimate_token_count(chunk) <= max_tokens:
            final_chunks.append(chunk)
            continue
        start = 0
        while start < len(words):
            current_words: list[str] = []
            while start < len(words):
                candidate_words = current_words + [words[start]]
                candidate = " ".join(candidate_words)
                if current_words and estimate_token_count(candidate) > max_tokens:
                    break
                current_words = candidate_words
                start += 1
            final_chunks.append(" ".join(current_words))
    return final_chunks


def _make_chunk_base_id(document: RawDocument, strategy: str, section_id: str | None, paragraph_ids: list[str]) -> str:
    first_id = paragraph_ids[0] if paragraph_ids else "none"
    last_id = paragraph_ids[-1] if paragraph_ids else "none"
    return (
        f"{document.document_id}__{strategy}__"
        f"{slugify(first_id)}__{slugify(last_id)}__{slugify('|'.join(_section_path(document, section_id)))}"
    )


def _make_table_chunk_base_id(document: RawDocument, strategy: str, table: RawTable) -> str:
    return (
        f"{document.document_id}__{strategy}__"
        f"{slugify(table.table_id)}__{slugify('|'.join(_section_path(document, table.section_id)))}"
    )


def _emit_chunks(
    document: RawDocument,
    *,
    strategy: str,
    section_id: str | None,
    paragraph_ids: list[str],
    text: str,
    legal_refs: list[str],
    chunk_kind: str,
    parent_chunk_id: str | None,
    max_tokens: int,
    chunk_id_base: str | None = None,
) -> list[ChunkNode]:
    parts = _split_text_for_limit(text, max_tokens=max_tokens)
    base_id = chunk_id_base or _make_chunk_base_id(document, strategy, section_id, paragraph_ids)
    emitted: list[ChunkNode] = []
    for idx, part in enumerate(parts):
        emitted.append(
            ChunkNode(
                chunk_id=base_id if len(parts) == 1 else f"{base_id}__part{idx:03d}",
                source_type="BOFIP",
                document_id=document.document_id,
                boi_reference=document.boi_reference,
                doc_version=document.publication_date,
                strategy=strategy,
                section_id=section_id,
                parent_chunk_id=parent_chunk_id,
                section_path=_section_path(document, section_id),
                paragraph_range=list(paragraph_ids),
                text=part,
                token_count=estimate_token_count(part),
                chunk_kind=chunk_kind if len(parts) == 1 else f"{chunk_kind}_split",
                legal_refs=list(legal_refs),
            )
        )
    return emitted


def _emit_table_chunks(document: RawDocument, *, strategy: str, max_tokens: int) -> list[ChunkNode]:
    chunks: list[ChunkNode] = []
    for table in sorted(document.tables, key=lambda item: item.order_index):
        text = normalize_whitespace(table.linearized_text)
        if not text:
            continue
        chunks.extend(
            _emit_chunks(
                document,
                strategy=strategy,
                section_id=table.section_id,
                paragraph_ids=[table.table_id],
                text=text,
                legal_refs=extract_legal_refs(text),
                chunk_kind="table",
                parent_chunk_id=None,
                max_tokens=max_tokens,
                chunk_id_base=_make_table_chunk_base_id(document, strategy, table),
            )
        )
    return chunks


def _merge_chunk_pair(left: ChunkNode, right: ChunkNode) -> ChunkNode:
    combined_text = "\n\n".join([left.text, right.text])
    return ChunkNode(
        chunk_id=f"{left.chunk_id}__merge__{right.chunk_id.split('__')[-1]}",
        source_type=left.source_type,
        document_id=left.document_id,
        boi_reference=left.boi_reference,
        doc_version=left.doc_version,
        strategy=left.strategy,
        section_id=left.section_id,
        parent_chunk_id=None,
        section_path=list(left.section_path),
        paragraph_range=list(left.paragraph_range) + list(right.paragraph_range),
        text=combined_text,
        token_count=estimate_token_count(combined_text),
        chunk_kind="paragraph_window",
        legal_refs=_dedupe_refs(left.legal_refs + right.legal_refs),
    )


def _can_merge_chunk_pair(left: ChunkNode, right: ChunkNode) -> bool:
    return (
        left.document_id == right.document_id
        and left.boi_reference == right.boi_reference
        and left.section_id == right.section_id
        and left.chunk_kind != "table"
        and right.chunk_kind != "table"
    )


def _should_merge_small_chunk(chunk: ChunkNode, min_tokens: int) -> bool:
    return chunk.token_count < min_tokens or _is_trivial_fragment(chunk.text)


def _merge_small_chunks(chunks: list[ChunkNode], *, max_tokens: int, min_tokens: int) -> list[ChunkNode]:
    if not chunks:
        return []

    forward_merged: list[ChunkNode] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        if (
            _should_merge_small_chunk(current, min_tokens)
            and index + 1 < len(chunks)
            and current.token_count + chunks[index + 1].token_count <= max_tokens
            and _can_merge_chunk_pair(current, chunks[index + 1])
        ):
            forward_merged.append(_merge_chunk_pair(current, chunks[index + 1]))
            index += 2
            continue
        forward_merged.append(current)
        index += 1

    final_merged: list[ChunkNode] = []
    for chunk in forward_merged:
        if (
            final_merged
            and _should_merge_small_chunk(chunk, min_tokens)
            and final_merged[-1].token_count + chunk.token_count <= max_tokens
            and _can_merge_chunk_pair(final_merged[-1], chunk)
        ):
            final_merged[-1] = _merge_chunk_pair(final_merged[-1], chunk)
        else:
            final_merged.append(chunk)
    return final_merged


def _build_parent_child_chunks(document: RawDocument, *, max_tokens: int) -> list[ChunkNode]:
    chunks: list[ChunkNode] = []

    for section_id, paragraphs in _paragraph_groups(document):
        if not paragraphs:
            continue
        section_path = _section_path(document, section_id)
        units = _build_paragraph_units(paragraphs)
        parent_windows: list[list[ParagraphUnit]] = []
        current_window: list[ParagraphUnit] = []
        for unit in units:
            candidate = current_window + [unit]
            candidate_text = "\n\n".join(item.text for item in candidate if normalize_whitespace(item.text))
            if current_window and estimate_token_count(candidate_text) > max_tokens:
                parent_windows.append(current_window)
                current_window = [unit]
            else:
                current_window = candidate
        if current_window:
            parent_windows.append(current_window)

        for window_index, window_paragraphs in enumerate(parent_windows):
            paragraph_ids = [pid for unit in window_paragraphs for pid in unit.paragraph_ids]
            parent_text = "\n\n".join(item.text for item in window_paragraphs if normalize_whitespace(item.text))
            parent_legal_refs = _dedupe_refs([ref for paragraph in window_paragraphs for ref in paragraph.legal_refs])
            section_slug = slugify("|".join(section_path)) or "root"
            parent_base_id = (
                f"{document.document_id}__parent_child__parent__{section_slug}"
                f"__{slugify(paragraph_ids[0])}__{slugify(paragraph_ids[-1])}__w{window_index:03d}"
            )
            parent_chunks = _emit_chunks(
                document,
                strategy="parent_child",
                section_id=section_id,
                paragraph_ids=paragraph_ids,
                text=parent_text,
                legal_refs=parent_legal_refs,
                chunk_kind="parent_section",
                parent_chunk_id=None,
                max_tokens=max_tokens,
                chunk_id_base=parent_base_id,
            )
            chunks.extend(parent_chunks)
            resolved_parent_chunk_id = parent_chunks[0].chunk_id if parent_chunks else None
            for unit in window_paragraphs:
                chunks.extend(
                    _emit_chunks(
                        document,
                        strategy="parent_child",
                        section_id=section_id,
                        paragraph_ids=unit.paragraph_ids,
                        text=unit.text,
                        legal_refs=unit.legal_refs,
                        chunk_kind="child_paragraph",
                        parent_chunk_id=resolved_parent_chunk_id,
                        max_tokens=max_tokens,
                    )
                )
    return chunks


def build_chunks(
    document: RawDocument,
    *,
    strategy: str,
    max_tokens: int = 350,
    min_tokens: int = 40,
) -> list[ChunkNode]:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    if strategy == "paragraph_preserving":
        chunks: list[ChunkNode] = []
        for section_id, paragraphs in _paragraph_groups(document):
            for unit in _build_paragraph_units(paragraphs):
                chunks.extend(
                    _emit_chunks(
                        document,
                        strategy=strategy,
                        section_id=section_id,
                        paragraph_ids=unit.paragraph_ids,
                        text=unit.text,
                        legal_refs=unit.legal_refs,
                        chunk_kind="paragraph",
                        parent_chunk_id=None,
                        max_tokens=max_tokens,
                    )
                )
        return chunks + _emit_table_chunks(document, strategy=strategy, max_tokens=max_tokens)

    if strategy == "section_window":
        chunks: list[ChunkNode] = []
        for section_id, paragraphs in _paragraph_groups(document):
            units = _build_paragraph_units(paragraphs)
            window: list[ParagraphUnit] = []
            for unit in units:
                candidate = window + [unit]
                candidate_text = "\n\n".join(item.text for item in candidate if normalize_whitespace(item.text))
                if window and estimate_token_count(candidate_text) > max_tokens:
                    previous = window
                    merged_text = "\n\n".join(item.text for item in previous)
                    merged_ids = [pid for item in previous for pid in item.paragraph_ids]
                    merged_refs = _dedupe_refs([ref for item in previous for ref in item.legal_refs])
                    chunks.extend(
                        _emit_chunks(
                            document,
                            strategy=strategy,
                            section_id=section_id,
                            paragraph_ids=merged_ids,
                            text=merged_text,
                            legal_refs=merged_refs,
                            chunk_kind="paragraph_window" if len(previous) > 1 else "paragraph",
                            parent_chunk_id=None,
                            max_tokens=max_tokens,
                        )
                    )
                    window = [unit]
                else:
                    window = candidate
            if window:
                merged_text = "\n\n".join(item.text for item in window)
                merged_ids = [pid for item in window for pid in item.paragraph_ids]
                merged_refs = _dedupe_refs([ref for item in window for ref in item.legal_refs])
                chunks.extend(
                    _emit_chunks(
                        document,
                        strategy=strategy,
                        section_id=section_id,
                        paragraph_ids=merged_ids,
                        text=merged_text,
                        legal_refs=merged_refs,
                        chunk_kind="paragraph_window" if len(window) > 1 else "paragraph",
                        parent_chunk_id=None,
                        max_tokens=max_tokens,
                    )
                )

        merged = _merge_small_chunks(chunks, max_tokens=max_tokens, min_tokens=min_tokens)
        return merged + _emit_table_chunks(document, strategy=strategy, max_tokens=max_tokens)

    return _build_parent_child_chunks(document, max_tokens=max_tokens) + _emit_table_chunks(
        document, strategy=strategy, max_tokens=max_tokens
    )


def build_chunks_for_documents(
    documents: list[RawDocument],
    *,
    strategy: str,
    max_tokens: int = 350,
    min_tokens: int = 40,
) -> list[ChunkNode]:
    chunks: list[ChunkNode] = []
    for document in documents:
        chunks.extend(build_chunks(document, strategy=strategy, max_tokens=max_tokens, min_tokens=min_tokens))
    return chunks

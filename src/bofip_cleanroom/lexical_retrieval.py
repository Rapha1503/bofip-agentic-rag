from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from nltk.stem.snowball import FrenchStemmer
from rank_bm25 import BM25Okapi

from .models import ChunkNode, RawDocument
from .text_utils import normalize_whitespace, strip_accents


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\-_\.]*")
STEMMER = FrenchStemmer()


def tokenize(text: str, *, stem: bool = False) -> list[str]:
    normalized = strip_accents(text).lower()
    tokens = TOKEN_RE.findall(normalized)
    if not stem:
        return tokens
    return [STEMMER.stem(token) for token in tokens]


def chunk_search_text(chunk: ChunkNode) -> str:
    parts = [
        chunk.boi_reference,
        " ".join(chunk.section_path),
        " ".join(chunk.legal_refs),
        chunk.text,
    ]
    return "\n".join(part for part in parts if normalize_whitespace(part))


def chunk_search_text_body(chunk: ChunkNode) -> str:
    return chunk.text


def chunk_search_text_leaf(chunk: ChunkNode) -> str:
    parts = [
        chunk.section_path[-1] if chunk.section_path else "",
        " ".join(chunk.legal_refs),
        chunk.text,
    ]
    return "\n".join(part for part in parts if normalize_whitespace(part))


def document_search_text(document: RawDocument) -> str:
    parts = [
        document.boi_reference,
        document.title,
        document.html_title or "",
        " ".join(document.category_path),
        " ".join(document.subjects),
    ]
    return "\n".join(part for part in parts if normalize_whitespace(part))


def document_search_text_title(document: RawDocument) -> str:
    parts = [
        document.boi_reference,
        document.title,
        document.html_title or "",
    ]
    return "\n".join(part for part in parts if normalize_whitespace(part))


def _title_segments(document: RawDocument) -> list[str]:
    normalized = normalize_whitespace(document.title)
    if not normalized:
        return []
    return [segment.strip() for segment in normalized.split(" - ") if segment.strip()]


def document_search_text_title_tail(document: RawDocument) -> str:
    segments = _title_segments(document)
    if not segments:
        return document_search_text_title(document)
    tail = " - ".join(segments[-2:]) if len(segments) >= 2 else segments[0]
    parts = [
        document.boi_reference,
        tail,
    ]
    return "\n".join(part for part in parts if normalize_whitespace(part))


def document_search_text_sections(document: RawDocument) -> str:
    section_titles = " ".join(section.title for section in document.sections[:12])
    parts = [
        document_search_text(document),
        section_titles,
    ]
    return "\n".join(part for part in parts if normalize_whitespace(part))


def document_search_text_sections_firstpara(document: RawDocument) -> str:
    first_paragraphs = " ".join(paragraph.text for paragraph in document.paragraphs[:3])
    parts = [
        document_search_text_sections(document),
        first_paragraphs,
    ]
    return "\n".join(part for part in parts if normalize_whitespace(part))


def _first_sentence_snippet(text: str) -> str:
    normalized = normalize_whitespace(text)
    if not normalized:
        return ""
    for separator in [". ", " ; ", " : "]:
        if separator in normalized:
            return normalized.split(separator, 1)[0][:220]
    return normalized[:220]


def document_search_text_sections_leads(document: RawDocument) -> str:
    paragraphs_by_section: dict[str, list[str]] = {}
    for paragraph in document.paragraphs:
        section_key = paragraph.section_id or "__root__"
        paragraphs_by_section.setdefault(section_key, []).append(paragraph.text)

    section_leads: list[str] = []
    for section in document.sections[:8]:
        texts = paragraphs_by_section.get(section.section_id, [])
        if texts:
            section_leads.append(_first_sentence_snippet(texts[0]))

    root_paragraphs = paragraphs_by_section.get("__root__", [])
    if root_paragraphs:
        section_leads.append(_first_sentence_snippet(root_paragraphs[0]))

    parts = [
        document_search_text_sections(document),
        " ".join(section_leads[:8]),
    ]
    return "\n".join(part for part in parts if normalize_whitespace(part))


def get_chunk_search_text_fn(mode: str) -> Callable[[ChunkNode], str]:
    modes = {
        "full": chunk_search_text,
        "leaf": chunk_search_text_leaf,
        "body": chunk_search_text_body,
    }
    if mode not in modes:
        raise ValueError(f"Unsupported chunk search text mode: {mode}")
    return modes[mode]


def get_document_search_text_fn(mode: str) -> Callable[[RawDocument], str]:
    modes = {
        "base": document_search_text,
        "title": document_search_text_title,
        "title_tail": document_search_text_title_tail,
        "sections": document_search_text_sections,
        "sections_firstpara": document_search_text_sections_firstpara,
        "sections_leads": document_search_text_sections_leads,
    }
    if mode not in modes:
        raise ValueError(f"Unsupported document search text mode: {mode}")
    return modes[mode]


@dataclass
class RetrievalHit:
    rank: int
    score: float
    chunk: ChunkNode


@dataclass
class DocumentRetrievalHit:
    rank: int
    score: float
    boi_reference: str
    best_chunk: ChunkNode


class LexicalBM25Index:
    def __init__(
        self,
        chunks: list[ChunkNode],
        *,
        search_text_fn: Callable[[ChunkNode], str] | None = None,
        tokenize_fn: Callable[[str], list[str]] | None = None,
    ):
        self.chunks = list(chunks)
        self.search_text_fn = search_text_fn or chunk_search_text
        self.tokenize_fn = tokenize_fn or tokenize
        self.search_texts = [self.search_text_fn(chunk) for chunk in self.chunks]
        tokenized = [self.tokenize_fn(text) for text in self.search_texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        if not self.bm25 or not self.chunks:
            return []
        query_tokens = self.tokenize_fn(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            RetrievalHit(rank=rank + 1, score=float(scores[idx]), chunk=self.chunks[idx])
            for rank, idx in enumerate(ranked_indices)
        ]

    def search_documents(self, query: str, *, top_k: int = 5) -> list[DocumentRetrievalHit]:
        if not self.bm25 or not self.chunks:
            return []
        query_tokens = self.tokenize_fn(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        docs: list[DocumentRetrievalHit] = []
        seen: set[str] = set()
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            boi_reference = chunk.boi_reference
            if boi_reference in seen:
                continue
            seen.add(boi_reference)
            docs.append(
                DocumentRetrievalHit(
                    rank=len(docs) + 1,
                    score=float(scores[idx]),
                    boi_reference=boi_reference,
                    best_chunk=chunk,
                )
            )
            if len(docs) >= top_k:
                break
        return docs


class DocumentLexicalIndex:
    def __init__(
        self,
        documents: list[RawDocument],
        *,
        search_text_fn: Callable[[RawDocument], str] | None = None,
        tokenize_fn: Callable[[str], list[str]] | None = None,
    ):
        self.documents = list(documents)
        self.search_text_fn = search_text_fn or document_search_text
        self.tokenize_fn = tokenize_fn or tokenize
        self.search_texts = [self.search_text_fn(document) for document in self.documents]
        tokenized = [self.tokenize_fn(text) for text in self.search_texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search_documents(self, query: str, *, top_k: int = 5) -> list[DocumentRetrievalHit]:
        if not self.bm25 or not self.documents:
            return []
        query_tokens = self.tokenize_fn(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            DocumentRetrievalHit(
                rank=rank + 1,
                score=float(scores[idx]),
                boi_reference=self.documents[idx].boi_reference,
                best_chunk=ChunkNode(
                    chunk_id=f"{self.documents[idx].document_id}__document_title",
                    source_type="BOFIP",
                    document_id=self.documents[idx].document_id,
                    boi_reference=self.documents[idx].boi_reference,
                    doc_version=self.documents[idx].publication_date,
                    strategy="document_lexical",
                    section_id=None,
                    parent_chunk_id=None,
                    section_path=[self.documents[idx].title],
                    paragraph_range=[],
                    text=self.documents[idx].title,
                    token_count=len(tokenize(self.documents[idx].title)),
                    chunk_kind="document_title",
                    legal_refs=list(self.documents[idx].legal_refs),
                ),
            )
            for rank, idx in enumerate(ranked_indices)
        ]

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .lexical_retrieval import (
    DocumentLexicalIndex,
    DocumentRetrievalHit,
    LexicalBM25Index,
    RetrievalHit,
    get_chunk_search_text_fn,
    get_document_search_text_fn,
    tokenize,
)
from .models import ChunkNode, RawDocument


@dataclass
class TwoStageSectionHit:
    global_rank: int
    document_rank: int
    document_score: float
    section_rank: int
    section_score: float
    boi_reference: str
    section_key: str
    section_path: list[str]


@dataclass
class TwoStageChunkHit:
    global_rank: int
    document_rank: int
    document_score: float
    section_rank: int | None
    section_score: float | None
    local_rank: int
    local_score: float
    boi_reference: str
    chunk: ChunkNode


@dataclass
class TwoStageResult:
    document_hits: list[DocumentRetrievalHit]
    section_hits: list[TwoStageSectionHit]
    chunk_hits: list[TwoStageChunkHit]


@dataclass
class SectionCandidate:
    section_key: str
    section_path: list[str]
    chunks: list[ChunkNode]
    search_text: str


@dataclass
class SectionRetrievalHit:
    rank: int
    score: float
    candidate: SectionCandidate


class SectionLexicalIndex:
    def __init__(self, candidates: list[SectionCandidate]):
        self.candidates = list(candidates)
        tokenized = [tokenize(candidate.search_text) for candidate in self.candidates]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, *, top_k: int = 3) -> list[SectionRetrievalHit]:
        if not self.bm25 or not self.candidates:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            SectionRetrievalHit(rank=rank + 1, score=float(scores[idx]), candidate=self.candidates[idx])
            for rank, idx in enumerate(ranked_indices)
        ]


class TwoStageLexicalRetriever:
    def __init__(
        self,
        documents: list[RawDocument],
        chunks: list[ChunkNode],
        *,
        document_mode: str = "base",
        local_chunk_mode: str = "body",
        local_strategy: str = "chunk",
    ):
        self.documents = list(documents)
        self.chunks = list(chunks)
        self.document_mode = document_mode
        self.document_search_text_fn = get_document_search_text_fn(document_mode)
        self.document_index = DocumentLexicalIndex(self.documents, search_text_fn=self.document_search_text_fn)
        self.chunks_by_reference: dict[str, list[ChunkNode]] = {}
        for chunk in self.chunks:
            self.chunks_by_reference.setdefault(chunk.boi_reference, []).append(chunk)
        self._chunk_indexes_by_reference: dict[str, LexicalBM25Index] = {}
        self._chunk_indexes_by_reference_and_section: dict[tuple[str, str], LexicalBM25Index] = {}
        self._section_indexes_by_reference: dict[str, SectionLexicalIndex] = {}
        self.local_chunk_mode = local_chunk_mode
        self.local_chunk_search_text_fn = get_chunk_search_text_fn(local_chunk_mode)
        if local_strategy not in {"chunk", "section_then_chunk"}:
            raise ValueError(f"Unsupported local strategy: {local_strategy}")
        self.local_strategy = local_strategy

    def _chunk_index_for_reference(self, boi_reference: str) -> LexicalBM25Index:
        if boi_reference not in self._chunk_indexes_by_reference:
            self._chunk_indexes_by_reference[boi_reference] = LexicalBM25Index(
                self.chunks_by_reference.get(boi_reference, []),
                search_text_fn=self.local_chunk_search_text_fn,
            )
        return self._chunk_indexes_by_reference[boi_reference]

    @staticmethod
    def _section_key(chunk: ChunkNode) -> str:
        if chunk.section_id:
            return chunk.section_id
        if chunk.section_path:
            return "::".join(chunk.section_path)
        return chunk.chunk_id

    def _section_candidates_for_reference(self, boi_reference: str) -> list[SectionCandidate]:
        section_groups: dict[str, list[ChunkNode]] = {}
        for chunk in self.chunks_by_reference.get(boi_reference, []):
            section_groups.setdefault(self._section_key(chunk), []).append(chunk)

        candidates: list[SectionCandidate] = []
        for section_key, group in section_groups.items():
            ordered_group = sorted(group, key=lambda item: item.chunk_id)
            section_path = ordered_group[0].section_path
            parts = [" > ".join(section_path)] if section_path else []
            parts.extend(self.local_chunk_search_text_fn(chunk) for chunk in ordered_group)
            search_text = "\n".join(part for part in parts if part.strip())
            candidates.append(
                SectionCandidate(
                    section_key=section_key,
                    section_path=list(section_path),
                    chunks=ordered_group,
                    search_text=search_text,
                )
            )
        candidates.sort(key=lambda candidate: candidate.section_key)
        return candidates

    def _section_index_for_reference(self, boi_reference: str) -> SectionLexicalIndex:
        if boi_reference not in self._section_indexes_by_reference:
            self._section_indexes_by_reference[boi_reference] = SectionLexicalIndex(
                self._section_candidates_for_reference(boi_reference)
            )
        return self._section_indexes_by_reference[boi_reference]

    def _chunk_index_for_reference_and_section(self, boi_reference: str, section_key: str) -> LexicalBM25Index:
        cache_key = (boi_reference, section_key)
        if cache_key not in self._chunk_indexes_by_reference_and_section:
            section_chunks = [
                chunk
                for chunk in self.chunks_by_reference.get(boi_reference, [])
                if self._section_key(chunk) == section_key
            ]
            self._chunk_indexes_by_reference_and_section[cache_key] = LexicalBM25Index(
                section_chunks,
                search_text_fn=self.local_chunk_search_text_fn,
            )
        return self._chunk_indexes_by_reference_and_section[cache_key]

    @staticmethod
    def _token_overlap_count(query: str, text: str) -> int:
        query_tokens = set(tokenize(query))
        text_tokens = set(tokenize(text))
        return len(query_tokens & text_tokens)

    def _rerank_sections(self, query: str, hits: list[SectionRetrievalHit]) -> list[SectionRetrievalHit]:
        ordered = sorted(
            hits,
            key=lambda hit: (
                -self._token_overlap_count(query, hit.candidate.search_text),
                -hit.score,
                hit.rank,
            ),
        )
        return [
            SectionRetrievalHit(rank=index + 1, score=hit.score, candidate=hit.candidate)
            for index, hit in enumerate(ordered)
        ]

    def _rerank_chunks(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        ordered = sorted(
            hits,
            key=lambda hit: (
                -self._token_overlap_count(query, self.local_chunk_search_text_fn(hit.chunk)),
                -hit.score,
                hit.rank,
            ),
        )
        return [
            RetrievalHit(rank=index + 1, score=hit.score, chunk=hit.chunk)
            for index, hit in enumerate(ordered)
        ]

    def search(
        self,
        query: str,
        *,
        top_docs: int = 3,
        sections_per_doc: int = 2,
        chunks_per_doc: int = 3,
        chunks_per_section: int = 2,
        max_chunks: int = 6,
    ) -> TwoStageResult:
        document_hits = self.document_index.search_documents(query, top_k=top_docs)

        section_hits: list[TwoStageSectionHit] = []
        chunk_hits: list[TwoStageChunkHit] = []
        for document_hit in document_hits:
            if self.local_strategy == "section_then_chunk":
                section_candidates = self._section_candidates_for_reference(document_hit.boi_reference)
                local_section_hits = self._section_index_for_reference(document_hit.boi_reference).search(
                    query, top_k=len(section_candidates)
                )
                local_section_hits = self._rerank_sections(query, local_section_hits)[:sections_per_doc]
                for section_hit in local_section_hits:
                    section_hits.append(
                        TwoStageSectionHit(
                            global_rank=0,
                            document_rank=document_hit.rank,
                            document_score=document_hit.score,
                            section_rank=section_hit.rank,
                            section_score=section_hit.score,
                            boi_reference=document_hit.boi_reference,
                            section_key=section_hit.candidate.section_key,
                            section_path=list(section_hit.candidate.section_path),
                        )
                    )
                    chunk_index = self._chunk_index_for_reference_and_section(
                        document_hit.boi_reference,
                        section_hit.candidate.section_key,
                    )
                    local_hits = chunk_index.search(query, top_k=len(section_hit.candidate.chunks))
                    local_hits = self._rerank_chunks(query, local_hits)[:chunks_per_section]
                    for local_hit in local_hits:
                        chunk_hits.append(
                            TwoStageChunkHit(
                                global_rank=0,
                                document_rank=document_hit.rank,
                                document_score=document_hit.score,
                                section_rank=section_hit.rank,
                                section_score=section_hit.score,
                                local_rank=local_hit.rank,
                                local_score=local_hit.score,
                                boi_reference=document_hit.boi_reference,
                                chunk=local_hit.chunk,
                            )
                        )
            else:
                doc_chunks = self.chunks_by_reference.get(document_hit.boi_reference, [])
                chunk_index = self._chunk_index_for_reference(document_hit.boi_reference)
                local_hits = chunk_index.search(query, top_k=len(doc_chunks))
                local_hits = self._rerank_chunks(query, local_hits)[:chunks_per_doc]
                for local_hit in local_hits:
                    chunk_hits.append(
                        TwoStageChunkHit(
                            global_rank=0,
                            document_rank=document_hit.rank,
                            document_score=document_hit.score,
                            section_rank=None,
                            section_score=None,
                            local_rank=local_hit.rank,
                            local_score=local_hit.score,
                            boi_reference=document_hit.boi_reference,
                            chunk=local_hit.chunk,
                        )
                    )

        section_hits = sorted(
            section_hits,
            key=lambda hit: (hit.document_rank, hit.section_rank, -hit.section_score),
        )
        for index, hit in enumerate(section_hits, start=1):
            hit.global_rank = index

        # Global chunk ordering is round-robin across document-local winners:
        # keep the best chunk from the top document first, but do not let
        # documents ranked 2/3 disappear behind the 2nd/3rd chunk of doc 1.
        chunk_hits = sorted(
            chunk_hits,
            key=lambda hit: (
                hit.local_rank,
                hit.document_rank,
                hit.section_rank if hit.section_rank is not None else 9999,
                -hit.local_score,
            ),
        )[:max_chunks]
        for index, hit in enumerate(chunk_hits, start=1):
            hit.global_rank = index

        return TwoStageResult(document_hits=document_hits, section_hits=section_hits, chunk_hits=chunk_hits)

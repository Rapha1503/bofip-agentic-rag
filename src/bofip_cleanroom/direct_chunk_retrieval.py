from __future__ import annotations

from dataclasses import dataclass

from .lexical_retrieval import LexicalBM25Index, get_chunk_search_text_fn
from .models import ChunkNode


@dataclass(frozen=True)
class Stage1DocumentHit:
    rank: int
    score: float
    boi_reference: str


@dataclass
class DirectChunkHit:
    global_rank: int
    document_rank: int
    local_rank: int
    local_score: float
    boi_reference: str
    chunk: ChunkNode


@dataclass
class DirectChunkResult:
    chunk_hits: list[DirectChunkHit]


class DirectChunkRetriever:
    def __init__(self, chunks: list[ChunkNode], *, local_chunk_mode: str = "full"):
        self.chunks = list(chunks)
        self.local_chunk_mode = local_chunk_mode
        self.local_chunk_search_text_fn = get_chunk_search_text_fn(local_chunk_mode)
        self.chunks_by_reference: dict[str, list[ChunkNode]] = {}
        for chunk in self.chunks:
            self.chunks_by_reference.setdefault(chunk.boi_reference, []).append(chunk)
        self._chunk_indexes_by_reference: dict[str, LexicalBM25Index] = {}

    def _chunk_index_for_reference(self, boi_reference: str) -> LexicalBM25Index:
        if boi_reference not in self._chunk_indexes_by_reference:
            self._chunk_indexes_by_reference[boi_reference] = LexicalBM25Index(
                self.chunks_by_reference.get(boi_reference, []),
                search_text_fn=self.local_chunk_search_text_fn,
            )
        return self._chunk_indexes_by_reference[boi_reference]

    def search(
        self,
        query: str,
        *,
        lexical_query: str | None = None,
        stage1_hits: list[Stage1DocumentHit],
        top_docs: int = 5,
        chunks_per_doc: int = 2,
        max_chunks: int = 6,
    ) -> DirectChunkResult:
        effective_query = lexical_query or query
        chunk_hits: list[DirectChunkHit] = []
        for doc_hit in stage1_hits[:top_docs]:
            boi_reference = doc_hit.boi_reference
            doc_chunks = self.chunks_by_reference.get(boi_reference, [])
            if not doc_chunks:
                continue
            local_hits = self._chunk_index_for_reference(boi_reference).search(
                effective_query,
                top_k=len(doc_chunks),
            )[:chunks_per_doc]
            for local_hit in local_hits:
                chunk_hits.append(
                    DirectChunkHit(
                        global_rank=0,
                        document_rank=doc_hit.rank,
                        local_rank=local_hit.rank,
                        local_score=local_hit.score,
                        boi_reference=boi_reference,
                        chunk=local_hit.chunk,
                    )
                )

        chunk_hits.sort(
            key=lambda hit: (
                hit.local_rank,
                hit.document_rank,
                -hit.local_score,
                hit.chunk.chunk_id,
            )
        )
        chunk_hits = chunk_hits[:max_chunks]
        for index, hit in enumerate(chunk_hits, start=1):
            hit.global_rank = index
        return DirectChunkResult(chunk_hits=chunk_hits)

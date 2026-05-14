from __future__ import annotations

from dataclasses import dataclass

from .family_routing import FamilyUnionSelection, collect_family_union, reference_core
from .lexical_retrieval import (
    DocumentLexicalIndex,
    LexicalBM25Index,
    RetrievalHit,
    get_chunk_search_text_fn,
    get_document_search_text_fn,
    tokenize,
)
from .models import ChunkNode, RawDocument
from .specificity_rerank import SpecificityReranker


@dataclass(frozen=True)
class PriorDocumentHit:
    rank: int
    score: float
    boi_reference: str


@dataclass
class FamilyDocumentHit:
    rank: int
    boi_reference: str
    combined_score: float
    family_rank: int
    family_score: float
    prior_rank: int | None
    prior_score: float | None
    title: str
    tail_rank: int | None = None
    tail_score: float | None = None
    descendant_count: int = 0
    descendant_support: float = 0.0


@dataclass
class FamilyChunkHit:
    global_rank: int
    document_rank: int
    document_score: float
    local_rank: int
    local_score: float
    boi_reference: str
    chunk: ChunkNode


@dataclass
class FamilyGuidedResult:
    family_selection: FamilyUnionSelection
    document_hits: list[FamilyDocumentHit]
    chunk_hits: list[FamilyChunkHit]


class FamilyGuidedRetriever:
    def __init__(
        self,
        documents: list[RawDocument],
        chunks: list[ChunkNode],
        *,
        family_doc_mode: str = "sections_leads",
        family_doc_stem: bool = True,
        local_chunk_mode: str = "body",
    ):
        self.documents = list(documents)
        self.chunks = list(chunks)
        self.documents_by_reference = {document.boi_reference: document for document in self.documents}
        self.all_references = sorted(self.documents_by_reference)
        self.specificity_reranker = SpecificityReranker(self.documents)
        self.family_doc_mode = family_doc_mode
        self.family_doc_search_text_fn = get_document_search_text_fn(family_doc_mode)
        self.family_doc_tokenize_fn = (
            (lambda text: tokenize(text, stem=True))
            if family_doc_stem
            else tokenize
        )
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

    def _family_doc_hits(
        self,
        query: str,
        stage1_hits: list[PriorDocumentHit],
        *,
        family_top_docs: int,
        max_family_docs: int,
        ancestor_expansion_levels: int,
        top_docs: int,
        family_weight: float,
        prior_weight: float,
        tail_weight: float,
        rank_constant: int,
        overview_weight: float,
        overview_min_descendants: int,
        overview_top_family_ranks: int,
    ) -> tuple[FamilyUnionSelection, list[FamilyDocumentHit]]:
        anchor_references = [hit.boi_reference for hit in stage1_hits[:family_top_docs]]
        family_selection = collect_family_union(
            anchor_references,
            self.all_references,
            max_family_docs=max_family_docs,
            ancestor_expansion_levels=ancestor_expansion_levels,
        )

        family_documents = [
            self.documents_by_reference[reference]
            for reference in family_selection.members
            if reference in self.documents_by_reference
        ]
        if not family_documents:
            return family_selection, []

        family_index = DocumentLexicalIndex(
            family_documents,
            search_text_fn=self.family_doc_search_text_fn,
            tokenize_fn=self.family_doc_tokenize_fn,
        )
        family_hits = family_index.search_documents(query, top_k=len(family_documents))
        tail_hits_by_reference: dict[str, object] = {}
        tail_strengths: dict[str, float] = {}
        if tail_weight > 0.0:
            tail_index = DocumentLexicalIndex(
                family_documents,
                search_text_fn=get_document_search_text_fn("title_tail"),
                tokenize_fn=(lambda text: tokenize(text, stem=True)),
            )
            tail_hits = tail_index.search_documents(query, top_k=len(family_documents))
            tail_hits_by_reference = {hit.boi_reference: hit for hit in tail_hits}
            tail_scores = [hit.score for hit in tail_hits]
            if tail_scores:
                tail_min_score = min(tail_scores)
                tail_span = max(tail_scores) - tail_min_score
                tail_strengths = {
                    hit.boi_reference: ((hit.score - tail_min_score) / tail_span if tail_span > 1e-9 else 1.0)
                    for hit in tail_hits
                }
        prior_hits_by_reference = {hit.boi_reference: hit for hit in stage1_hits}

        family_scores = [hit.score for hit in family_hits]
        if family_scores:
            min_score = min(family_scores)
            max_score = max(family_scores)
            score_span = max_score - min_score
        else:
            min_score = 0.0
            score_span = 0.0

        scored_hits: list[FamilyDocumentHit] = []
        for family_hit in family_hits:
            prior_hit = prior_hits_by_reference.get(family_hit.boi_reference)
            if score_span > 1e-9:
                family_strength = (family_hit.score - min_score) / score_span
            else:
                family_strength = 1.0
            family_component = family_strength + (1.0 / (rank_constant + family_hit.rank))
            combined_score = family_weight * family_component
            tail_hit = tail_hits_by_reference.get(family_hit.boi_reference)
            if tail_hit is not None:
                tail_component = tail_strengths.get(family_hit.boi_reference, 0.0) + (1.0 / (rank_constant + tail_hit.rank))
                combined_score += tail_weight * tail_component
            if prior_hit is not None:
                combined_score += prior_weight * (1.0 / (rank_constant + prior_hit.rank))
            document = self.documents_by_reference[family_hit.boi_reference]
            scored_hits.append(
                FamilyDocumentHit(
                    rank=0,
                    boi_reference=family_hit.boi_reference,
                    combined_score=combined_score,
                    family_rank=family_hit.rank,
                    family_score=family_hit.score,
                    prior_rank=prior_hit.rank if prior_hit is not None else None,
                    prior_score=prior_hit.score if prior_hit is not None else None,
                    title=document.title,
                    tail_rank=tail_hit.rank if tail_hit is not None else None,
                    tail_score=tail_hit.score if tail_hit is not None else None,
                )
            )

        if overview_weight > 0.0:
            family_hits_by_reference = {hit.boi_reference: hit for hit in family_hits}
            reference_cores = {
                reference: reference_core(reference)
                for reference in family_selection.members
            }
            candidate_refs = [
                hit.boi_reference
                for hit in family_hits[: max(1, overview_top_family_ranks)]
            ]
            family_strengths = {
                hit.boi_reference: ((hit.score - min_score) / score_span if score_span > 1e-9 else 1.0)
                for hit in family_hits
            }
            descendant_rank_score = {
                hit.boi_reference: (1.0 / (rank_constant + hit.rank))
                for hit in family_hits
            }

            for scored_hit in scored_hits:
                core = reference_cores.get(scored_hit.boi_reference, ())
                descendants = [
                    candidate_ref
                    for candidate_ref in candidate_refs
                    if candidate_ref != scored_hit.boi_reference
                    and len(reference_cores.get(candidate_ref, ())) > len(core)
                    and reference_cores[candidate_ref][: len(core)] == core
                ]
                if len(descendants) < overview_min_descendants:
                    continue
                descendant_support = sum(
                    family_strengths.get(candidate_ref, 0.0)
                    + descendant_rank_score.get(candidate_ref, 0.0)
                    for candidate_ref in descendants
                )
                if descendant_support <= 0.0:
                    continue
                scored_hit.descendant_count = len(descendants)
                scored_hit.descendant_support = descendant_support
                scored_hit.combined_score += overview_weight * descendant_support

        scored_hits.sort(
            key=lambda hit: (
                -hit.combined_score,
                -hit.descendant_support,
                hit.family_rank,
                hit.prior_rank if hit.prior_rank is not None else 9999,
                hit.boi_reference,
            )
        )
        for index, hit in enumerate(scored_hits, start=1):
            hit.rank = index
        return family_selection, scored_hits

    @staticmethod
    def _rerank_local_hits(query: str, hits: list[RetrievalHit], search_text_fn) -> list[RetrievalHit]:
        query_tokens = set(tokenize(query))

        def token_overlap(hit: RetrievalHit) -> int:
            return len(query_tokens & set(tokenize(search_text_fn(hit.chunk))))

        ordered = sorted(
            hits,
            key=lambda hit: (
                -token_overlap(hit),
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
        lexical_query: str | None = None,
        stage1_hits: list[PriorDocumentHit],
        family_top_docs: int = 1,
        max_family_docs: int = 25,
        ancestor_expansion_levels: int = 0,
        top_docs: int = 5,
        chunks_per_doc: int = 2,
        max_chunks: int = 6,
        family_weight: float = 1.0,
        prior_weight: float = 0.25,
        tail_weight: float = 0.0,
        rank_constant: int = 20,
        overview_weight: float = 0.0,
        overview_min_descendants: int = 2,
        overview_top_family_ranks: int = 6,
        preserve_stage1_top1: bool = False,
        specificity_rerank_top_n: int = 0,
        specificity_rerank_weight: float = 0.0,
    ) -> FamilyGuidedResult:
        family_query = lexical_query or query
        family_selection, document_hits = self._family_doc_hits(
            family_query,
            stage1_hits,
            family_top_docs=family_top_docs,
            max_family_docs=max_family_docs,
            ancestor_expansion_levels=ancestor_expansion_levels,
            top_docs=top_docs,
            family_weight=family_weight,
            prior_weight=prior_weight,
            tail_weight=tail_weight,
            rank_constant=rank_constant,
            overview_weight=overview_weight,
            overview_min_descendants=overview_min_descendants,
            overview_top_family_ranks=overview_top_family_ranks,
        )

        if preserve_stage1_top1 and stage1_hits:
            stage1_top1_ref = stage1_hits[0].boi_reference
            anchored_hits: list[FamilyDocumentHit] = []
            seen_refs: set[str] = set()
            top1_hit = next((hit for hit in document_hits if hit.boi_reference == stage1_top1_ref), None)
            if top1_hit is not None:
                anchored_hits.append(top1_hit)
                seen_refs.add(top1_hit.boi_reference)
            for hit in document_hits:
                if hit.boi_reference in seen_refs:
                    continue
                anchored_hits.append(hit)
                seen_refs.add(hit.boi_reference)
            document_hits = anchored_hits[:top_docs]
            for index, hit in enumerate(document_hits, start=1):
                hit.rank = index
        else:
            document_hits = document_hits[:top_docs]
            for index, hit in enumerate(document_hits, start=1):
                hit.rank = index

        if specificity_rerank_top_n > 1 and specificity_rerank_weight > 0.0:
            document_hits = self.specificity_reranker.rerank_hits(
                family_query,
                document_hits,
                get_reference=lambda hit: hit.boi_reference,
                get_score=lambda hit: hit.combined_score,
                clone_hit=lambda hit, rank, score: FamilyDocumentHit(
                    rank=rank,
                    boi_reference=hit.boi_reference,
                    combined_score=score,
                    family_rank=hit.family_rank,
                    family_score=hit.family_score,
                    prior_rank=hit.prior_rank,
                    prior_score=hit.prior_score,
                    title=hit.title,
                    tail_rank=hit.tail_rank,
                    tail_score=hit.tail_score,
                    descendant_count=hit.descendant_count,
                    descendant_support=hit.descendant_support,
                ),
                top_n=min(specificity_rerank_top_n, len(document_hits)),
                weight=specificity_rerank_weight,
            )[:top_docs]
            for index, hit in enumerate(document_hits, start=1):
                hit.rank = index

        chunk_hits: list[FamilyChunkHit] = []
        for document_hit in document_hits:
            chunk_index = self._chunk_index_for_reference(document_hit.boi_reference)
            doc_chunks = self.chunks_by_reference.get(document_hit.boi_reference, [])
            local_hits = chunk_index.search(family_query, top_k=len(doc_chunks))
            local_hits = self._rerank_local_hits(family_query, local_hits, self.local_chunk_search_text_fn)
            for local_hit in local_hits[:chunks_per_doc]:
                chunk_hits.append(
                    FamilyChunkHit(
                        global_rank=0,
                        document_rank=document_hit.rank,
                        document_score=document_hit.combined_score,
                        local_rank=local_hit.rank,
                        local_score=local_hit.score,
                        boi_reference=document_hit.boi_reference,
                        chunk=local_hit.chunk,
                    )
                )

        chunk_hits.sort(
            key=lambda hit: (
                hit.document_rank,
                hit.local_rank,
                -hit.local_score,
                hit.chunk.chunk_id,
            )
        )
        for index, hit in enumerate(chunk_hits[:max_chunks], start=1):
            hit.global_rank = index

        return FamilyGuidedResult(
            family_selection=family_selection,
            document_hits=document_hits,
            chunk_hits=chunk_hits[:max_chunks],
        )

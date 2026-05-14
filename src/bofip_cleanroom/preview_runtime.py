from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .alias_expansion import build_acronym_expansion_map, expand_query_with_acronyms
from .dense_retrieval import DenseDocumentIndex, DenseEncoder, DenseIndex
from .direct_chunk_retrieval import DirectChunkRetriever, Stage1DocumentHit
from .family_routing import FamilyUnionSelection, collect_family_union
from .hybrid_retrieval import (
    RankedDoc,
    compute_source_rank_profiles,
    confidence_weighted_reciprocal_rank_fuse,
)
from .jsonio import read_jsonl
from .lexical_retrieval import DocumentLexicalIndex, get_document_search_text_fn, tokenize
from .models import ChunkNode, RawDocument, chunk_node_from_dict, raw_document_from_dict
from .specificity_rerank import SpecificityReranker


DEFAULT_PREVIEW_CORPUS = "commentary"
DEFAULT_DOC_MODEL = str(Path(__file__).resolve().parents[2] / "data" / "models" / "intfloat--multilingual-e5-large")
DEFAULT_CHUNK_MODEL = "intfloat/multilingual-e5-base"


@dataclass(frozen=True)
class PreviewStage1Hit:
    rank: int
    score: float
    boi_reference: str
    title: str
    sources: list[str]
    ranks: dict[str, int]


@dataclass(frozen=True)
class PreviewChunk:
    citation_id: int
    boi_reference: str
    title: str
    section_path: str
    chunk_id: str
    chunk_kind: str
    text: str
    publication_date: str | None


@dataclass(frozen=True)
class PreviewRetrievalResult:
    query: str
    lexical_query: str
    acronym_expansions: list[dict[str, str]]
    stage1_hits: list[PreviewStage1Hit]
    family_selection: dict
    stage2_chunks: list[PreviewChunk]
    source_confidences: dict[str, float]


class Phase8bPreviewRuntime:
    def __init__(
        self,
        *,
        documents: list[RawDocument],
        chunks: list[ChunkNode],
        doc_encoder: DenseEncoder,
        chunk_encoder: DenseEncoder,
        document_embeddings: np.ndarray,
        chunk_embeddings: np.ndarray,
    ):
        self.documents = documents
        self.documents_by_ref = {document.boi_reference: document for document in documents}
        self.chunks = chunks
        self.doc_encoder = doc_encoder
        self.chunk_encoder = chunk_encoder
        self.document_embeddings = document_embeddings
        self.chunk_embeddings = chunk_embeddings

        self.acronym_map = build_acronym_expansion_map(documents)
        self.lexical_indexes = {
            "base": DocumentLexicalIndex(documents, search_text_fn=get_document_search_text_fn("base")),
            "sections_leads": DocumentLexicalIndex(documents, search_text_fn=get_document_search_text_fn("sections_leads")),
            "sections_leads_stem": DocumentLexicalIndex(
                documents,
                search_text_fn=get_document_search_text_fn("sections_leads"),
                tokenize_fn=(lambda text: tokenize(text, stem=True)),
            ),
        }
        self.doc_dense_index = DenseDocumentIndex(documents, document_embeddings)
        self.chunk_dense_index = DenseIndex(chunks, chunk_embeddings)
        self.specificity_reranker = SpecificityReranker(documents)
        self.direct_chunk_retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")

    @classmethod
    def from_local_corpus(
        cls,
        *,
        corpus: str = DEFAULT_PREVIEW_CORPUS,
        project_root: Path | None = None,
        doc_model: str = DEFAULT_DOC_MODEL,
        chunk_model: str = DEFAULT_CHUNK_MODEL,
        device: str = "cpu",
    ) -> "Phase8bPreviewRuntime":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        config = {
            "commentary": {
                "raw_docs": root / "data" / "interim" / "raw_docs_sample_5666.jsonl",
                "chunks": root / "data" / "interim" / "chunks_section_window_sample_5666.jsonl",
                "doc_dense_cache": root / "data" / "interim" / "doc_dense_cache_5666_sections_firstpara_e5large.npy",
                "chunk_dense_cache": root / "data" / "interim" / "chunk_dense_cache_5666_full_e5.npy",
            },
            "mixed": {
                "raw_docs": root / "data" / "interim" / "raw_docs_sample_6295.jsonl",
                "chunks": root / "data" / "interim" / "chunks_section_window_sample_6295.jsonl",
                "doc_dense_cache": root / "data" / "interim" / "doc_dense_cache_6295_sections_firstpara_e5.npy",
                "chunk_dense_cache": None,
            },
        }
        if corpus not in config:
            raise ValueError(f"Unsupported preview corpus: {corpus}")
        cfg = config[corpus]
        documents = [raw_document_from_dict(item) for item in read_jsonl(cfg["raw_docs"])]
        chunks = [chunk_node_from_dict(item) for item in read_jsonl(cfg["chunks"])]
        document_embeddings = np.load(cfg["doc_dense_cache"])
        if cfg["chunk_dense_cache"] is None:
            raise ValueError("Preview runtime currently supports commentary corpus only because chunk-dense cache is required")
        chunk_embeddings = np.load(cfg["chunk_dense_cache"])
        return cls(
            documents=documents,
            chunks=chunks,
            doc_encoder=DenseEncoder(doc_model, device=device),
            chunk_encoder=DenseEncoder(chunk_model, device=device),
            document_embeddings=document_embeddings,
            chunk_embeddings=chunk_embeddings,
        )

    def _build_rankings(self, query: str, lexical_query: str) -> tuple[dict[str, list[RankedDoc]], dict[str, float]]:
        rankings = {
            mode: [
                RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source=mode)
                for hit in index.search_documents(lexical_query, top_k=20)
            ]
            for mode, index in self.lexical_indexes.items()
        }
        doc_query_embedding = self.doc_encoder.encode_queries([query])[0]
        rankings["dense"] = [
            RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source="dense")
            for hit in self.doc_dense_index.search_from_vector(doc_query_embedding, top_k=20)
        ]
        chunk_query_embedding = self.chunk_encoder.encode_queries([query])[0]
        rankings["chunk_dense"] = [
            RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source="chunk_dense")
            for hit in self.chunk_dense_index.search_documents_from_vector(chunk_query_embedding, top_k=20)
        ]
        profiles = compute_source_rank_profiles(rankings, top_n=5)
        confidences = {name: round(profile.confidence, 6) for name, profile in profiles.items()}
        return rankings, confidences

    def retrieve(
        self,
        query: str,
        *,
        top_docs: int = 5,
        chunks_per_doc: int = 3,
        max_chunks: int = 8,
    ) -> PreviewRetrievalResult:
        lexical_query, expansions = expand_query_with_acronyms(
            query,
            self.acronym_map,
            max_expansions_per_query=3,
        )
        rankings, confidences = self._build_rankings(query, lexical_query)
        fused = confidence_weighted_reciprocal_rank_fuse(
            rankings,
            top_k=20,
            rank_constant=60,
            source_weights={"base": 1.0, "sections_leads": 2.0, "sections_leads_stem": 1.0, "dense": 1.0, "chunk_dense": 2.0},
            confidence_top_n=5,
            confidence_alpha=1.0,
            score_alpha=0.5,
        )
        fused = self.specificity_reranker.rerank_hits(
            query,
            fused,
            get_reference=lambda hit: hit.boi_reference,
            get_score=lambda hit: hit.score,
            clone_hit=lambda hit, rank, score: type(hit)(
                rank=rank,
                boi_reference=hit.boi_reference,
                score=score,
                sources=hit.sources,
                ranks=hit.ranks,
            ),
            top_n=min(5, len(fused)),
            weight=0.05,
        )
        top_stage1 = fused[:top_docs]
        direct = self.direct_chunk_retriever.search(
            query,
            lexical_query=lexical_query,
            stage1_hits=[
                Stage1DocumentHit(rank=hit.rank, score=hit.score, boi_reference=hit.boi_reference)
                for hit in top_stage1
            ],
            top_docs=top_docs,
            chunks_per_doc=chunks_per_doc,
            max_chunks=max_chunks,
        )

        family_selection: FamilyUnionSelection = collect_family_union(
            [hit.boi_reference for hit in top_stage1[:2]],
            sorted(self.documents_by_ref),
            max_family_docs=25,
        )
        stage1_hits = [
            PreviewStage1Hit(
                rank=hit.rank,
                score=hit.score,
                boi_reference=hit.boi_reference,
                title=self.documents_by_ref[hit.boi_reference].title,
                sources=hit.sources,
                ranks=hit.ranks,
            )
            for hit in top_stage1
        ]
        preview_chunks = []
        for idx, hit in enumerate(direct.chunk_hits, start=1):
            document = self.documents_by_ref[hit.boi_reference]
            preview_chunks.append(
                PreviewChunk(
                    citation_id=idx,
                    boi_reference=hit.boi_reference,
                    title=document.title,
                    section_path=" > ".join(hit.chunk.section_path),
                    chunk_id=hit.chunk.chunk_id,
                    chunk_kind=hit.chunk.chunk_kind,
                    text=hit.chunk.text,
                    publication_date=document.publication_date,
                )
            )

        return PreviewRetrievalResult(
            query=query,
            lexical_query=lexical_query,
            acronym_expansions=[{"acronym": acronym, "phrase": phrase} for acronym, phrase in expansions],
            stage1_hits=stage1_hits,
            family_selection={
                "anchor_references": family_selection.anchor_references,
                "prefixes": [list(prefix) for prefix in family_selection.prefixes],
                "members": family_selection.members,
            },
            stage2_chunks=preview_chunks,
            source_confidences=confidences,
        )

    @staticmethod
    def build_context_chunks(result: PreviewRetrievalResult) -> list[dict]:
        return [
            {
                "citation_id": chunk.citation_id,
                "boi_reference": chunk.boi_reference,
                "title": chunk.title,
                "publication_date": chunk.publication_date,
                "section_path": chunk.section_path,
                "chunk_id": chunk.chunk_id,
                "chunk_kind": chunk.chunk_kind,
                "text": chunk.text,
            }
            for chunk in result.stage2_chunks
        ]

    @staticmethod
    def as_dict(result: PreviewRetrievalResult) -> dict:
        return {
            "query": result.query,
            "lexical_query": result.lexical_query,
            "acronym_expansions": result.acronym_expansions,
            "source_confidences": result.source_confidences,
            "stage1_hits": [asdict(hit) for hit in result.stage1_hits],
            "family_selection": result.family_selection,
            "stage2_chunks": [asdict(chunk) for chunk in result.stage2_chunks],
        }

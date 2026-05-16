from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .dense_retrieval import DenseDocumentIndex, DenseEncoder, DenseIndex
from .direct_chunk_retrieval import DirectChunkRetriever, Stage1DocumentHit
from .hybrid_retrieval import (
    RankedDoc,
    compute_source_rank_profiles,
    confidence_weighted_reciprocal_rank_fuse,
)
from .jsonio import read_jsonl
from .lexical_retrieval import DocumentLexicalIndex, get_document_search_text_fn, tokenize
from .models import ChunkNode, RawDocument, chunk_node_from_dict, raw_document_from_dict
from .reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL


DEFAULT_CORPUS = "commentary"
DATA_ROOT_CACHE = None


def _get_data_root() -> Path:
    global DATA_ROOT_CACHE
    if DATA_ROOT_CACHE is None:
        from .settings import BOFIP_DATA_ROOT
        DATA_ROOT_CACHE = BOFIP_DATA_ROOT
    return DATA_ROOT_CACHE


def _resolve_path(root: Path, paths: dict, fallbacks: dict, key: str) -> Path:
    """Resolve corpus path with fallback to generic filenames."""
    primary = root / paths.get(key, "")
    if primary.exists():
        return primary
    for alt in fallbacks.get(key, []):
        alt_path = root / alt
        if alt_path.exists():
            return alt_path
    return primary  # return original even if missing — let FileNotFoundError surface


DEFAULT_DOC_MODEL = str(_get_data_root() / "data" / "models" / "intfloat--multilingual-e5-large")
DEFAULT_CHUNK_MODEL = DEFAULT_DOC_MODEL
STAGE2_CANDIDATES_PER_DOC = 8

CORPUS_PATHS: dict[str, dict[str, str]] = {
    "commentary": {
        "raw_docs": "data/interim/raw_docs_sample_5666.jsonl",
        "chunks": "data/interim/chunks_section_window_sample_5666.jsonl",
        "doc_dense_cache": "data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy",
        "chunk_dense_cache": "data/interim/chunk_dense_cache_5666_full_e5large.npy",
    },
}

# Fallback to generic filenames (produced by setup.py)
_CORPUS_FALLBACKS: dict[str, list[str]] = {
    "raw_docs": ["data/interim/raw_docs.jsonl"],
    "chunks": ["data/interim/chunks.jsonl"],
    "doc_dense_cache": ["data/interim/doc_dense_cache.npy"],
    "chunk_dense_cache": ["data/interim/chunk_dense_cache.npy"],
}

DEFAULT_RANK_CONSTANT = 60
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "base": 0.5,
    "sections_leads": 0.5,
    "sections_leads_stem": 0.5,
    "dense": 2.0,
    "chunk_dense": 2.0,
}


@dataclass(frozen=True)
class RagChunkHit:
    rank: int
    boi_reference: str
    title: str
    section_path: str
    chunk_id: str
    chunk_kind: str
    text: str
    publication_date: str | None
    score: float


@dataclass(frozen=True)
class RagStage1Hit:
    rank: int
    score: float
    boi_reference: str
    title: str


@dataclass(frozen=True)
class RagResult:
    query: str
    stage1_hits: list[RagStage1Hit]
    stage2_chunks: list[RagChunkHit]
    source_confidences: dict[str, float]
    pipeline_log: dict = field(default_factory=dict)


class RagRuntime:
    def __init__(
        self,
        *,
        documents: list[RawDocument],
        chunks: list[ChunkNode],
        doc_encoder: DenseEncoder,
        chunk_encoder: DenseEncoder | None = None,
        document_embeddings: np.ndarray,
        chunk_embeddings: np.ndarray,
        reranker: CrossEncoderReranker,
    ):
        self.documents = documents
        self.documents_by_ref = {d.boi_reference: d for d in documents}
        self.chunks = chunks
        self.doc_encoder = doc_encoder
        self.chunk_encoder = chunk_encoder if chunk_encoder is not None else doc_encoder
        self.document_embeddings = document_embeddings
        self.chunk_embeddings = chunk_embeddings
        self.reranker = reranker

        self.lexical_indexes = self._init_lexical(documents)
        self.doc_dense_index = DenseDocumentIndex(documents, document_embeddings)
        self.chunk_dense_index = DenseIndex(chunks, chunk_embeddings)
        self.chunk_retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")

    def _init_lexical(self, documents: list[RawDocument]) -> dict[str, DocumentLexicalIndex]:
        """Load BM25 indexes from cache if available, otherwise build + save."""
        indexes = {}
        cache_dir = Path.home() / ".cache" / "bofip_rag"
        cache_dir.mkdir(parents=True, exist_ok=True)
        modes = {
            "base": ("base", None),
            "sections_leads": ("sections_leads", None),
            "sections_leads_stem": ("sections_leads", lambda text: tokenize(text, stem=True)),
        }
        for mode_key, (search_mode, tok_fn) in modes.items():
            cache_path = cache_dir / f"bm25_cache_5666_{mode_key}.pkl"
            try:
                if cache_path.exists():
                    indexes[mode_key] = DocumentLexicalIndex.load(
                        cache_path,
                        search_text_fn=get_document_search_text_fn(search_mode),
                        tokenize_fn=tok_fn,
                    )
                    continue
            except (pickle.PickleError, OSError, EOFError, ImportError):
                pass  # Corrupted or incompatible cache — rebuild
            indexes[mode_key] = DocumentLexicalIndex(
                documents,
                search_text_fn=get_document_search_text_fn(search_mode),
                tokenize_fn=tok_fn,
            )
            try:
                indexes[mode_key].save(cache_path)
            except (pickle.PickleError, OSError):
                pass  # Can't persist — non-fatal
        return indexes

    @classmethod
    def from_local_corpus(
        cls,
        *,
        corpus: str = DEFAULT_CORPUS,
        project_root: Path | None = None,
        raw_docs_path: Path | str | None = None,
        chunks_path: Path | str | None = None,
        doc_dense_path: Path | str | None = None,
        chunk_dense_path: Path | str | None = None,
        doc_model: str = DEFAULT_DOC_MODEL,
        chunk_model: str = DEFAULT_CHUNK_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        device: str = "cuda",
    ) -> "RagRuntime":
        root = (project_root or _get_data_root()).resolve()
        paths = CORPUS_PATHS.get(corpus, {})
        raw = raw_docs_path or _resolve_path(root, paths, _CORPUS_FALLBACKS, "raw_docs")
        chk = chunks_path or _resolve_path(root, paths, _CORPUS_FALLBACKS, "chunks")
        doc_dense = doc_dense_path or _resolve_path(root, paths, _CORPUS_FALLBACKS, "doc_dense_cache")
        chunk_dense = chunk_dense_path or _resolve_path(root, paths, _CORPUS_FALLBACKS, "chunk_dense_cache")

        documents = [raw_document_from_dict(item) for item in read_jsonl(Path(raw))]
        chunks = [chunk_node_from_dict(item) for item in read_jsonl(Path(chk))]
        document_embeddings = np.load(str(doc_dense))
        chunk_embeddings = np.load(str(chunk_dense))

        return cls(
            documents=documents,
            chunks=chunks,
            doc_encoder=DenseEncoder(doc_model, device=device),
            document_embeddings=document_embeddings,
            chunk_embeddings=chunk_embeddings,
            reranker=CrossEncoderReranker(reranker_model, device=device),
        )

    def _build_rankings(self, query: str, lexical_query: str, *, branch_hint: str | None = None) -> tuple[dict[str, list[RankedDoc]], dict[str, float]]:
        rankings = {
            mode: [
                RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source=mode)
                for hit in index.search_documents(lexical_query, top_k=20)
            ]
            for mode, index in self.lexical_indexes.items()
        }
        doc_emb = self.doc_encoder.encode_queries([query])[0]
        rankings["dense"] = [
            RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source="dense")
            for hit in self.doc_dense_index.search_from_vector(doc_emb, top_k=20)
        ]
        chunk_emb = self.chunk_encoder.encode_queries([query])[0]
        rankings["chunk_dense"] = [
            RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source="chunk_dense")
            for hit in self.chunk_dense_index.search_documents_from_vector(chunk_emb, top_k=20)
        ]

        # Soft branch hint boost (does NOT filter, only boosts)
        if branch_hint:
            for source, docs in rankings.items():
                for doc in docs:
                    if doc.boi_reference.startswith(branch_hint):
                        doc.score *= 1.15

        profiles = compute_source_rank_profiles(rankings, top_n=5)
        confidences = {name: round(profile.confidence, 6) for name, profile in profiles.items()}
        return rankings, confidences

    def retrieve(
        self,
        query: str,
        *,
        top_docs: int = 8,
        chunks_per_doc: int = STAGE2_CANDIDATES_PER_DOC,
        max_chunks: int = 8,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
        source_weights: dict[str, float] | None = None,
        use_lexical: bool = True,
        use_dense: bool = True,
        use_chunk_dense: bool = True,
        use_anchor_filter: bool = True,
        use_reranker: bool = True,
        branch_hint: str | None = None,
    ) -> RagResult:
        rankings, confidences = self._build_rankings(query, query, branch_hint=branch_hint)
        if source_weights is None:
            source_weights = dict(DEFAULT_SOURCE_WEIGHTS)

        if not use_lexical:
            for source in ("base", "sections_leads", "sections_leads_stem"):
                rankings.pop(source, None)
                source_weights.pop(source, None)
        if not use_dense:
            rankings.pop("dense", None)
            source_weights.pop("dense", None)
        if not use_chunk_dense:
            rankings.pop("chunk_dense", None)
            source_weights.pop("chunk_dense", None)

        # Dense-anchor: lexical sources may only re-rank documents that dense
        # already found semantically relevant. This prevents term-level matches
        # on common phrases (e.g. "resident fiscal") from flooding the fusion
        # with irrelevant conventions et al.
        if use_anchor_filter and rankings.get("dense"):
            dense_anchor_refs = {doc.boi_reference for doc in rankings.get("dense", [])[:20]}
            for source in list(rankings):
                if source in ("dense", "chunk_dense"):
                    continue
                rankings[source] = [
                    doc for doc in rankings[source]
                    if doc.boi_reference in dense_anchor_refs
                ]

        fused = confidence_weighted_reciprocal_rank_fuse(
            rankings,
            top_k=20,
            rank_constant=rank_constant,
            source_weights=source_weights,
            confidence_top_n=5,
            confidence_alpha=1.0,
            score_alpha=0.5,
        )

        stage1_hits = fused[:top_docs]

        direct_result = self.chunk_retriever.search(
            query,
            lexical_query=query,
            stage1_hits=[
                Stage1DocumentHit(rank=h.rank, score=h.score, boi_reference=h.boi_reference)
                for h in stage1_hits
            ],
            top_docs=top_docs,
            chunks_per_doc=chunks_per_doc,
            max_chunks=chunks_per_doc * top_docs,
        )

        candidates = direct_result.chunk_hits
        log = {}

        if use_reranker:
            # Get all scores for diversity pool (top_48 from reranker)
            ranked_all = self.reranker.rerank(
                query,
                candidates,
                get_text=lambda hit: " > ".join(hit.chunk.section_path) + "\n" + hit.chunk.text,
                top_k=min(32, len(candidates)),
            )
            reranked_pool = [(r.item, float(r.score)) for r in ranked_all]
            selected = self._select_diverse(reranked_pool, max_chunks=max_chunks)

            # Log: all reranked scores + selection decisions
            log["reranker_scores"] = [{"chunk_id": r.item.chunk.chunk_id[:60], "doc": r.item.boi_reference,
                "raw_score": float(r.score), "selected": r.item in selected} for r in ranked_all]
            log["diversity_selected"] = len(selected)
        else:
            chunk_items = [(c, float(c.local_score)) for c in candidates[:max_chunks]]
            selected = [c for c, _ in chunk_items]

        preview_chunks = []
        for idx, hit in enumerate(selected, start=1):
            doc = self.documents_by_ref[hit.boi_reference]
            preview_chunks.append(
                RagChunkHit(
                    rank=idx,
                    boi_reference=hit.boi_reference,
                    title=doc.title,
                    section_path=" > ".join(hit.chunk.section_path),
                    chunk_id=hit.chunk.chunk_id,
                    chunk_kind=hit.chunk.chunk_kind,
                    text=hit.chunk.text,
                    publication_date=doc.publication_date,
                    score=float(hit.local_score) if not use_reranker else next((s for c, s in reranked_pool if c is hit), 0.0),
                )
            )

        # Build pipeline log
        stage1_refs = [h.boi_reference for h in stage1_hits]
        final_refs = [c.boi_reference for c in preview_chunks]
        final_docs = set(final_refs)
        from collections import Counter
        doc_dist = dict(Counter(final_refs))
        log.update({
            "stage1_docs_found": len(stage1_hits),
            "stage1_docs_dropped": [ref for ref in stage1_refs if ref not in final_docs],
            "stage2_candidates": len(candidates),
            "final_chunks": len(preview_chunks),
            "unique_docs_final": len(final_docs),
            "max_chunks_per_doc": max(doc_dist.values()) if doc_dist else 0,
            "doc_distribution_final": {k[:30]: v for k, v in doc_dist.items()},
        })

        stage1_out = [
            RagStage1Hit(
                rank=h.rank,
                score=h.score,
                boi_reference=h.boi_reference,
                title=self.documents_by_ref[h.boi_reference].title,
            )
            for h in stage1_hits
        ]

        return RagResult(
            query=query,
            stage1_hits=stage1_out,
            stage2_chunks=preview_chunks,
            source_confidences=confidences,
            pipeline_log=log,
        )

    # ── Diversity selection ────────────────────────────────────────

    @staticmethod
    def _diversity_penalty(candidate, selected: list) -> float:
        """Compute penalty for adding candidate given already-selected chunks."""
        doc = candidate.boi_reference
        doc_count = sum(1 for c in selected if c.boi_reference == doc)
        if doc_count >= 3:
            return 999.0
        penalty = 0.0
        if doc_count == 2:
            penalty = 0.30
        elif doc_count == 1:
            penalty = 0.15
        for sel in selected:
            if sel.boi_reference == doc:
                if sel.chunk.section_path == candidate.chunk.section_path:
                    penalty += 0.15
                elif len(sel.chunk.section_path) > 0 and len(candidate.chunk.section_path) > 0:
                    if sel.chunk.section_path[:-1] == candidate.chunk.section_path[:-1]:
                        penalty += 0.05
        return penalty

    @staticmethod
    def _select_diverse(chunks_and_scores: list, max_chunks: int = 8) -> list:
        """Greedy selection by recalculated marginal utility at each step."""
        remaining = list(chunks_and_scores)  # (DirectChunkHit, float_score)
        if not remaining:
            return remaining

        # Normalize scores to [0, 1]
        scores = [s for _, s in remaining]
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-9:
            return [c for c, _ in remaining[:max_chunks]]
        normed = [(c, (s - mn) / (mx - mn)) for (c, s), s in zip(remaining, scores)]

        # Step 1: strict diversity
        selected = []
        for _ in range(max_chunks):
            if not normed:
                break
            best_score = -float("inf")
            best_idx = -1
            for i, (c, ns) in enumerate(normed):
                penalty = RagRuntime._diversity_penalty(c, selected)
                adjusted = ns - penalty
                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = i
            if best_idx < 0 or best_score < 0:
                break
            selected.append(normed[best_idx][0])
            normed.pop(best_idx)

        # Step 2: if not enough, relax — allow 3rd chunk per doc
        if len(selected) < max_chunks:
            for _ in range(max_chunks - len(selected)):
                if not normed:
                    break
                best_score = -float("inf")
                best_idx = -1
                for i, (c, ns) in enumerate(normed):
                    doc_count = sum(1 for s in selected if s.boi_reference == c.boi_reference)
                    if doc_count >= 4:
                        continue
                    adjusted = ns - (0.10 if doc_count >= 3 else 0)
                    if adjusted > best_score:
                        best_score = adjusted
                        best_idx = i
                if best_idx < 0:
                    break
                selected.append(normed[best_idx][0])
                normed.pop(best_idx)

        # Step 3: last resort — fill by raw score
        if len(selected) < max_chunks:
            remaining_raw = [(c, s) for c, s in remaining if c not in selected]
            remaining_raw.sort(key=lambda x: x[1], reverse=True)
            for c, _ in remaining_raw[:max_chunks - len(selected)]:
                selected.append(c)

        return selected

    @staticmethod
    def as_dict(result: RagResult) -> dict:
        return {
            "query": result.query,
            "source_confidences": result.source_confidences,
            "stage1_hits": [asdict(h) for h in result.stage1_hits],
            "stage2_chunks": [asdict(chunk) for chunk in result.stage2_chunks],
            "pipeline_log": result.pipeline_log,
        }

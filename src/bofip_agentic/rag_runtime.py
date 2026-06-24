from __future__ import annotations

import hashlib
import pickle
import re
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
from .lexical_retrieval import LexicalIndex, get_document_search_text_fn, tokenize
from .models import ChunkNode, RawDocument, chunk_node_from_dict, raw_document_from_dict
from .reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL
from .text_utils import normalize_whitespace, strip_accents


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


def _reference_matches_prefix(boi_reference: str, prefix: str) -> bool:
    normalized = prefix.strip().lower().removeprefix("boi-").strip("-")
    if not normalized:
        return False
    ref = boi_reference.lower()
    return (
        ref.startswith(normalized)
        or ref.startswith(f"boi-{normalized}")
        or ref.startswith(f"boi-res-{normalized}")
    )


BOI_REFERENCE_RE = re.compile(r"\bBOI-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", re.IGNORECASE)


def _normalize_boi_reference(value: str) -> str:
    return value.strip().upper().removeprefix("BOI-").strip("-")


def _query_boi_references(query: str) -> list[str]:
    references: list[str] = []
    for match in BOI_REFERENCE_RE.finditer(query):
        reference = "BOI-" + _normalize_boi_reference(match.group(0))
        if reference not in references:
            references.append(reference)
    return references[:5]


def _is_exact_or_child_reference(boi_reference: str, target_reference: str) -> bool:
    ref = _normalize_boi_reference(boi_reference)
    target = _normalize_boi_reference(target_reference)
    return bool(target and (ref == target or ref.startswith(target + "-")))


def _document_text_for_query_score(document: RawDocument) -> str:
    return " ".join(
        part
        for part in (
            document.boi_reference,
            document.title,
            document.html_title or "",
            " ".join(document.category_path),
            " ".join(document.subjects),
        )
        if part
    )


def _significant_query_tokens(query: str) -> tuple[set[str], set[tuple[str, str]]]:
    tokens = [token for token in tokenize(query) if len(token) >= 4]
    return set(tokens), set(zip(tokens, tokens[1:]))


def _token_overlap_score(text: str, query_terms: set[str], query_bigrams: set[tuple[str, str]]) -> float:
    if not query_terms:
        return 0.0
    text_tokens = [token for token in tokenize(text) if len(token) >= 4]
    text_terms = set(text_tokens)
    text_bigrams = set(zip(text_tokens, text_tokens[1:]))
    return float(len(query_terms & text_terms)) + 2.0 * float(len(query_bigrams & text_bigrams))


def _document_query_overlap_score(
    query: str,
    document: RawDocument,
    chunks_by_reference: dict[str, list[ChunkNode]] | None = None,
) -> float:
    query_terms, query_bigrams = _significant_query_tokens(query)
    doc_score = _token_overlap_score(_document_text_for_query_score(document), query_terms, query_bigrams)
    chunk_score = 0.0
    if chunks_by_reference:
        for chunk in chunks_by_reference.get(document.boi_reference, []):
            chunk_text = " ".join(chunk.section_path) + "\n" + " ".join(chunk.legal_refs) + "\n" + chunk.text
            chunk_score = max(chunk_score, _token_overlap_score(chunk_text, query_terms, query_bigrams))
    return min(30.0, doc_score + chunk_score)


def _reference_match_depth(boi_reference: str, prefix: str) -> int:
    ref_parts = _normalize_boi_reference(boi_reference).split("-")
    prefix_parts = _normalize_boi_reference(prefix).split("-")
    depth = 0
    for ref_part, prefix_part in zip(ref_parts, prefix_parts):
        if ref_part != prefix_part:
            break
        depth += 1
    return depth


def _normalized_text(value: str) -> str:
    return strip_accents(value or "").lower()


def _document_navigation_text(
    document: RawDocument,
    chunks_by_reference: dict[str, list[ChunkNode]] | None = None,
) -> str:
    section_paths: list[str] = []
    if chunks_by_reference:
        for chunk in chunks_by_reference.get(document.boi_reference, [])[:12]:
            section_paths.extend(chunk.section_path)
    return _normalized_text(
        "\n".join(
            part
            for part in (
                document.boi_reference,
                document.title,
                document.html_title or "",
                " ".join(document.category_path),
                " ".join(section_paths),
            )
            if part
        )
    )


def _has_general_rule_intent(query: str) -> bool:
    normalized = _normalized_text(query)
    return (
        "regle generale" in normalized
        or "regles generales" in normalized
        or "droit commun" in normalized
    )


def _has_exception_intent(query: str) -> bool:
    normalized = _normalized_text(query)
    return any(
        marker in normalized
        for marker in (
            "derogation",
            "derogations",
            "exception",
            "exceptions",
            "cas particulier",
            "regle particuliere",
            "regle specifique",
        )
    )


def _has_rescript_intent(query: str) -> bool:
    normalized = _normalized_text(query)
    return "rescrit" in normalized or "boi-res" in normalized


def _document_navigation_bias(
    query: str,
    document: RawDocument,
    chunks_by_reference: dict[str, list[ChunkNode]] | None = None,
) -> float:
    haystack = _document_navigation_text(document, chunks_by_reference)
    bias = 0.0

    if _has_general_rule_intent(query):
        has_derogation = "derogation" in haystack or "derogations" in haystack
        exception_intent = _has_exception_intent(query)
        if (
            not has_derogation
            and ("regle generale" in haystack or "regles generales" in haystack or "droit commun" in haystack)
        ):
            bias += 10.0
        if has_derogation and not exception_intent:
            bias -= 10.0

    if document.boi_reference.upper().startswith("BOI-RES-") and not _has_rescript_intent(query):
        title_score = _token_overlap_score(
            document.title,
            *_significant_query_tokens(query),
        )
        bias -= 3.0 if title_score >= 8.0 else 8.0

    return bias


def _prefix_overlap_rankings(
    query: str,
    prefix: str,
    documents: list[RawDocument],
    chunks_by_reference: dict[str, list[ChunkNode]] | None = None,
) -> list[RankedDoc]:
    if not prefix:
        return []

    detailed_prefix = _normalize_boi_reference(prefix).count("-") >= 2
    candidates: list[tuple[float, str, RankedDoc]] = []
    for document in documents:
        if not _reference_matches_prefix(document.boi_reference, prefix):
            continue
        overlap = _document_query_overlap_score(query, document, chunks_by_reference)
        if overlap <= 0 and not detailed_prefix:
            continue
        score = (
            overlap
            + 0.25 * _reference_match_depth(document.boi_reference, prefix)
            + _document_navigation_bias(query, document, chunks_by_reference)
        )
        candidates.append(
            (
                -score,
                document.boi_reference,
                RankedDoc(
                    boi_reference=document.boi_reference,
                    score=score,
                    rank=0,
                    source="prefix_overlap",
                ),
            )
        )

    candidates.sort()
    ranked = [item[2] for item in candidates[:20]]
    for idx, doc in enumerate(ranked, start=1):
        doc.rank = idx
    return ranked


def _exact_reference_rankings(
    query: str,
    documents: list[RawDocument],
    chunks_by_reference: dict[str, list[ChunkNode]] | None = None,
) -> list[RankedDoc]:
    query_refs = _query_boi_references(query)
    if not query_refs:
        return []

    candidates: list[tuple[float, str, RankedDoc]] = []
    seen: set[str] = set()
    for target in query_refs:
        target_norm = _normalize_boi_reference(target)
        for document in documents:
            if not _is_exact_or_child_reference(document.boi_reference, target):
                continue
            if document.boi_reference in seen:
                continue
            ref_norm = _normalize_boi_reference(document.boi_reference)
            extra_depth = max(0, ref_norm.count("-") - target_norm.count("-"))
            relationship_score = 120.0 if ref_norm == target_norm else 105.0 - min(extra_depth, 6)
            overlap_score = _document_query_overlap_score(query, document, chunks_by_reference)
            score = relationship_score + overlap_score
            candidates.append(
                (
                    -score,
                    document.boi_reference,
                    RankedDoc(
                        boi_reference=document.boi_reference,
                        score=score,
                        rank=0,
                        source="exact_reference",
                    ),
                )
            )
            seen.add(document.boi_reference)

    candidates.sort()
    ranked = [item[2] for item in candidates[:20]]
    for idx, doc in enumerate(ranked, start=1):
        doc.rank = idx
    return ranked


def _prepend_ranked_docs(primary: list[RankedDoc], rest: list[RankedDoc]) -> list[RankedDoc]:
    if not primary:
        return rest
    seen: set[str] = set()
    merged: list[RankedDoc] = []
    for doc in primary + rest:
        if doc.boi_reference in seen:
            continue
        seen.add(doc.boi_reference)
        merged.append(doc)
    for idx, doc in enumerate(merged, start=1):
        doc.rank = idx
    return merged


DEFAULT_DOC_MODEL = str(_get_data_root() / "data" / "models" / "intfloat--multilingual-e5-large")
DEFAULT_CHUNK_MODEL = DEFAULT_DOC_MODEL
STAGE2_CANDIDATES_PER_DOC = 8
DEFAULT_RERANKER_CANDIDATE_LIMIT = 16
DEFAULT_RERANKER_TEXT_LIMIT = 900

CORPUS_PATHS: dict[str, dict[str, str]] = {
    "commentary": {
        "raw_docs": "data/interim/raw_docs.jsonl",
        "chunks": "data/interim/chunks.jsonl",
        "doc_dense_cache": "data/interim/doc_dense_cache.npy",
        "chunk_dense_cache": "data/interim/chunk_dense_cache.npy",
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


def _reranker_text(hit, *, limit: int = DEFAULT_RERANKER_TEXT_LIMIT) -> str:
    section = normalize_whitespace(" > ".join(hit.chunk.section_path))
    body = normalize_whitespace(hit.chunk.text)
    text = f"{section}\n{body}" if section else body
    return text[:limit]


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
        doc_encoder: DenseEncoder | None,
        chunk_encoder: DenseEncoder | None = None,
        document_embeddings: np.ndarray,
        chunk_embeddings: np.ndarray,
        reranker: CrossEncoderReranker | None,
        dense_error: str | None = None,
    ):
        self.documents = documents
        self.documents_by_id = {d.document_id: d for d in documents}
        self.documents_by_ref = {}
        for document in documents:
            self.documents_by_ref.setdefault(document.boi_reference, document)
        self.chunks = chunks
        self.doc_encoder = doc_encoder
        self.chunk_encoder = chunk_encoder if chunk_encoder is not None else doc_encoder
        self.document_embeddings = document_embeddings
        self.chunk_embeddings = chunk_embeddings
        self.reranker = reranker
        self.dense_error = dense_error

        self.lexical_indexes = self._init_lexical(documents)
        self.doc_dense_index = DenseDocumentIndex(documents, document_embeddings)
        self.chunk_dense_index = DenseIndex(chunks, chunk_embeddings)
        self.chunk_retriever = DirectChunkRetriever(chunks, local_chunk_mode="full")
        self.chunks_by_reference = self.chunk_retriever.chunks_by_reference

    def _init_lexical(self, documents: list[RawDocument]) -> dict[str, LexicalIndex]:
        """Load BM25 indexes from cache if available, otherwise build + save."""
        indexes = {}
        cache_dir = Path.home() / ".cache" / "bofip_rag"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.md5()
        for doc in documents:
            digest.update(doc.boi_reference.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update((doc.publication_date or "").encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(doc.title.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        doc_hash = digest.hexdigest()[:12]
        modes = {
            "base": ("base", None),
            "sections_leads": ("sections_leads", None),
            "sections_leads_stem": ("sections_leads", lambda text: tokenize(text, stem=True)),
        }
        for mode_key, (search_mode, tok_fn) in modes.items():
            cache_path = cache_dir / f"bm25_cache_{doc_hash}_{mode_key}.pkl"
            try:
                if cache_path.exists():
                    indexes[mode_key] = LexicalIndex.load(
                        cache_path,
                        search_text_fn=get_document_search_text_fn(search_mode),
                        tokenize_fn=tok_fn,
                    )
                    continue
            except (pickle.PickleError, OSError, EOFError, ImportError, MemoryError):
                pass  # Corrupted or incompatible cache — rebuild
            indexes[mode_key] = LexicalIndex(
                documents,
                search_text_fn=get_document_search_text_fn(search_mode),
                tokenize_fn=tok_fn,
                document_mode=True,
            )
            try:
                indexes[mode_key].save(cache_path)
            except (pickle.PickleError, OSError, MemoryError):
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
        load_reranker: bool = True,
        load_dense: bool = True,
        allow_lexical_fallback: bool = True,
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
        document_embeddings = np.load(str(doc_dense), mmap_mode="r")
        chunk_embeddings = np.load(str(chunk_dense), mmap_mode="r")
        if not load_dense:
            if len(document_embeddings) != len(documents):
                document_embeddings = np.zeros((len(documents), 1), dtype=np.float32)
            if len(chunk_embeddings) != len(chunks):
                chunk_embeddings = np.zeros((len(chunks), 1), dtype=np.float32)

        dense_error = None
        if load_dense:
            try:
                doc_encoder = DenseEncoder(doc_model, device=device)
            except (OSError, RuntimeError, MemoryError) as exc:
                if not allow_lexical_fallback:
                    raise
                doc_encoder = None
                dense_error = f"{exc.__class__.__name__}: {exc}"
        else:
            doc_encoder = None
            dense_error = "dense disabled by configuration"

        reranker = None
        if load_reranker:
            try:
                reranker = CrossEncoderReranker(reranker_model, device=device)
            except (OSError, RuntimeError, MemoryError) as exc:
                if not allow_lexical_fallback:
                    raise
                reranker_error = f"reranker {exc.__class__.__name__}: {exc}"
                dense_error = f"{dense_error} | {reranker_error}" if dense_error else reranker_error

        return cls(
            documents=documents,
            chunks=chunks,
            doc_encoder=doc_encoder,
            document_embeddings=document_embeddings,
            chunk_embeddings=chunk_embeddings,
            reranker=reranker,
            dense_error=dense_error,
        )

    def _build_rankings(
        self,
        query: str,
        lexical_query: str,
        *,
        use_dense: bool = True,
        use_chunk_dense: bool = True,
    ) -> tuple[dict[str, list[RankedDoc]], dict[str, float]]:
        rankings = {
            mode: [
                RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source=mode)
                for hit in index.search_documents(lexical_query, top_k=20)
            ]
            for mode, index in self.lexical_indexes.items()
        }
        if use_dense and self.doc_encoder is not None:
            doc_emb = self.doc_encoder.encode_queries([query])[0]
            rankings["dense"] = [
                RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source="dense")
                for hit in self.doc_dense_index.search_from_vector(doc_emb, top_k=20)
            ]
        if use_chunk_dense and self.chunk_encoder is not None:
            chunk_emb = self.chunk_encoder.encode_queries([query])[0]
            rankings["chunk_dense"] = [
                RankedDoc(boi_reference=hit.boi_reference, score=float(hit.score), rank=hit.rank, source="chunk_dense")
                for hit in self.chunk_dense_index.search_documents_from_vector(chunk_emb, top_k=20)
            ]
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
        boost_prefix: str = "",
        chunk_query: str | None = None,
    ) -> RagResult:
        stage2_query = chunk_query or query
        dense_enabled = use_dense and self.doc_encoder is not None
        chunk_dense_enabled = use_chunk_dense and self.chunk_encoder is not None
        rankings, confidences = self._build_rankings(
            query,
            query,
            use_dense=dense_enabled,
            use_chunk_dense=chunk_dense_enabled,
        )
        if source_weights is None:
            source_weights = dict(DEFAULT_SOURCE_WEIGHTS)
        exact_ranked = _exact_reference_rankings(query, self.documents, self.chunks_by_reference)
        if exact_ranked:
            rankings["exact_reference"] = exact_ranked
            source_weights["exact_reference"] = 5.0

        if not use_lexical:
            for source in ("base", "sections_leads", "sections_leads_stem"):
                rankings.pop(source, None)
                source_weights.pop(source, None)
        if not dense_enabled:
            rankings.pop("dense", None)
            source_weights.pop("dense", None)
        if not chunk_dense_enabled:
            rankings.pop("chunk_dense", None)
            source_weights.pop("chunk_dense", None)

        # Dense-anchor: lexical sources may only re-rank documents that dense
        # already found semantically relevant. This prevents term-level matches
        # on common phrases from flooding the fusion with irrelevant documents.
        # When a boost_prefix is provided, documents matching that prefix
        # bypass the anchor filter so that domain-specific documents are not
        # unfairly excluded when the dense embeddings don't surface them.
        if use_anchor_filter and rankings.get("dense"):
            dense_anchor_refs = {doc.boi_reference for doc in rankings.get("dense", [])[:20]}
            if boost_prefix:
                for d in self.documents:
                    if _reference_matches_prefix(d.boi_reference, boost_prefix):
                        dense_anchor_refs.add(d.boi_reference)
            for source in list(rankings):
                if source in ("dense", "chunk_dense"):
                    continue
                rankings[source] = [
                    doc for doc in rankings[source]
                    if doc.boi_reference in dense_anchor_refs
                ]

        prefix_ranked = _prefix_overlap_rankings(
            stage2_query,
            boost_prefix,
            self.documents,
            self.chunks_by_reference,
        )
        if prefix_ranked:
            rankings["prefix_overlap"] = prefix_ranked
            source_weights["prefix_overlap"] = 2.0

        fused = confidence_weighted_reciprocal_rank_fuse(
            rankings,
            top_k=20,
            rank_constant=rank_constant,
            source_weights=source_weights,
            confidence_top_n=5,
            confidence_alpha=1.0,
            score_alpha=0.5,
        )
        fused = _prepend_ranked_docs(exact_ranked, fused)

        stage1_hits = fused[:top_docs]

        direct_result = self.chunk_retriever.search(
            query,
            lexical_query=stage2_query,
            stage1_hits=[
                Stage1DocumentHit(rank=h.rank, score=h.score, boi_reference=h.boi_reference)
                for h in stage1_hits
            ],
            top_docs=top_docs,
            chunks_per_doc=chunks_per_doc,
            max_candidates=chunks_per_doc * top_docs,
        )

        candidates = direct_result.chunk_hits
        log = {}
        if self.doc_encoder is None:
            log["dense_status"] = "unavailable"
            log["dense_error"] = self.dense_error or "dense encoder not loaded"
        else:
            log["dense_status"] = "active"
        reranker_enabled = use_reranker and self.reranker is not None
        reranked_pool = []

        if reranker_enabled:
            reranker_candidates = candidates[:DEFAULT_RERANKER_CANDIDATE_LIMIT]
            ranked_all = self.reranker.rerank(
                stage2_query,
                reranker_candidates,
                get_text=_reranker_text,
                top_k=min(max_chunks * 3, len(reranker_candidates)),
            )
            reranked_pool = [(r.item, float(r.score)) for r in ranked_all]
            selected = self._select_diverse(reranked_pool, max_chunks=max_chunks)

            # Log: all reranked scores + selection decisions
            log["reranker_scores"] = [{"chunk_id": r.item.chunk.chunk_id[:60], "doc": r.item.boi_reference,
                "raw_score": float(r.score), "selected": r.item in selected} for r in ranked_all]
            log["diversity_selected"] = len(selected)
            log["reranker_candidate_limit"] = DEFAULT_RERANKER_CANDIDATE_LIMIT
            log["reranker_candidates_scored"] = len(reranker_candidates)
            log["reranker_text_limit"] = DEFAULT_RERANKER_TEXT_LIMIT
        else:
            chunk_items = [(c, float(c.local_score)) for c in candidates[:max_chunks]]
            selected = [c for c, _ in chunk_items]
            if use_reranker:
                log["reranker_skipped"] = "not_loaded"

        preview_chunks = []
        for idx, hit in enumerate(selected, start=1):
            doc = self.documents_by_id.get(hit.chunk.document_id) or self.documents_by_ref[hit.boi_reference]
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
                    score=float(hit.local_score) if not reranker_enabled else next((s for c, s in reranked_pool if c is hit), 0.0),
                )
            )

        # Build pipeline log
        stage1_refs = [h.boi_reference for h in stage1_hits]
        candidate_chunk_ids = [hit.chunk.chunk_id for hit in candidates]
        candidate_doc_refs = [hit.boi_reference for hit in candidates]
        final_chunk_ids = [chunk.chunk_id for chunk in preview_chunks]
        final_refs = [c.boi_reference for c in preview_chunks]
        final_docs = set(final_refs)
        from collections import Counter
        doc_dist = dict(Counter(final_refs))
        log.update({
            "stage1_doc_refs": stage1_refs,
            "stage2_candidate_chunk_ids": candidate_chunk_ids,
            "stage2_candidate_doc_refs": candidate_doc_refs,
            "final_chunk_ids": final_chunk_ids,
            "final_doc_refs": final_refs,
            "stage1_docs_found": len(stage1_hits),
            "stage1_docs_dropped": [ref for ref in stage1_refs if ref not in final_docs],
            "stage2_candidates": len(candidates),
            "stage2_query": stage2_query if stage2_query != query else "",
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

    def retrieve_within_documents(
        self,
        query: str,
        boi_references: list[str],
        *,
        chunks_per_doc: int = 6,
        max_chunks: int = 8,
    ) -> RagResult:
        """Search locally inside already identified BOFiP documents."""
        resolved_refs = self._resolve_intra_document_references(query, boi_references, limit=8)

        stage1_hits = [
            Stage1DocumentHit(rank=rank, score=1.0 / rank, boi_reference=ref)
            for rank, ref in enumerate(resolved_refs[:8], start=1)
        ]
        if not stage1_hits:
            return RagResult(
                query=query,
                stage1_hits=[],
                stage2_chunks=[],
                source_confidences={},
                pipeline_log={"retrieval_scope": "intra_document", "searched_documents": []},
            )

        direct_result = self.chunk_retriever.search(
            query,
            lexical_query=query,
            stage1_hits=stage1_hits,
            top_docs=len(stage1_hits),
            chunks_per_doc=chunks_per_doc,
            max_candidates=chunks_per_doc * len(stage1_hits),
        )
        selected = sorted(
            direct_result.chunk_hits,
            key=lambda hit: (
                -hit.local_score,
                hit.document_rank,
                hit.local_rank,
                hit.chunk.chunk_id,
            ),
        )[:max_chunks]

        preview_chunks = []
        for idx, hit in enumerate(selected, start=1):
            doc = self.documents_by_id.get(hit.chunk.document_id) or self.documents_by_ref[hit.boi_reference]
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
                    score=float(hit.local_score),
                )
            )

        stage1_out = [
            RagStage1Hit(
                rank=hit.rank,
                score=hit.score,
                boi_reference=hit.boi_reference,
                title=self.documents_by_ref[hit.boi_reference].title,
            )
            for hit in stage1_hits
        ]
        return RagResult(
            query=query,
            stage1_hits=stage1_out,
            stage2_chunks=preview_chunks,
            source_confidences={},
            pipeline_log={
                "retrieval_scope": "intra_document",
                "searched_documents": resolved_refs[:8],
                "stage1_doc_refs": resolved_refs[:8],
                "stage2_candidate_chunk_ids": [hit.chunk.chunk_id for hit in direct_result.chunk_hits],
                "stage2_candidate_doc_refs": [hit.boi_reference for hit in direct_result.chunk_hits],
                "final_chunk_ids": [chunk.chunk_id for chunk in preview_chunks],
                "final_doc_refs": [chunk.boi_reference for chunk in preview_chunks],
                "stage2_candidates": len(direct_result.chunk_hits),
                "final_chunks": len(preview_chunks),
            },
        )

    def _resolve_intra_document_references(
        self,
        query: str,
        boi_references: list[str],
        *,
        limit: int,
    ) -> list[str]:
        """Resolve broad BOFiP parents into query-ranked child documents."""
        candidates: dict[str, tuple[float, int]] = {}
        order = 0

        def add(ref: str, score: float) -> None:
            nonlocal order
            if not ref:
                return
            previous = candidates.get(ref)
            if previous is None:
                candidates[ref] = (score, order)
                order += 1
                return
            old_score, old_order = previous
            candidates[ref] = (max(old_score, score), old_order)

        for reference_order, reference in enumerate(boi_references):
            if not reference:
                continue
            target_norm = _normalize_boi_reference(reference)
            if not target_norm:
                continue
            target_is_detailed = target_norm.count("-") >= 2

            def is_broad_parent_doc(doc_ref: str) -> bool:
                if target_is_detailed or _normalize_boi_reference(doc_ref) != target_norm:
                    return False
                return any(
                    other_ref != doc_ref and _is_exact_or_child_reference(other_ref, reference)
                    for other_ref in self.documents_by_ref
                )

            for ranked in _prefix_overlap_rankings(
                query,
                reference,
                self.documents,
                self.chunks_by_reference,
            ):
                if is_broad_parent_doc(ranked.boi_reference):
                    continue
                score = float(ranked.score) + max(0.0, 2.0 - reference_order * 0.1)
                if target_is_detailed and _normalize_boi_reference(ranked.boi_reference) == target_norm:
                    score += 10.0
                add(ranked.boi_reference, score)

            for doc_ref, document in self.documents_by_ref.items():
                if not _is_exact_or_child_reference(doc_ref, reference):
                    continue
                if is_broad_parent_doc(doc_ref):
                    continue
                doc_ref_norm = _normalize_boi_reference(doc_ref)
                score = _document_query_overlap_score(query, document, self.chunks_by_reference)
                if target_is_detailed and doc_ref_norm == target_norm:
                    score += 8.0
                score += max(0.0, 1.0 - reference_order * 0.05)
                add(doc_ref, score)

        ranked_refs = sorted(candidates.items(), key=lambda item: (-item[1][0], item[1][1], item[0]))
        return [ref for ref, _ in ranked_refs[:limit]]

    @staticmethod
    def _select_diverse(chunks_and_scores: list, max_chunks: int = 8) -> list:
        """Greedy selection by recalculated marginal utility at each step."""
        remaining = list(chunks_and_scores)
        if not remaining:
            return remaining

        scores = [s for _, s in remaining]
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-9:
            return [c for c, _ in remaining[:max_chunks]]
        normed = [(c, (s - mn) / (mx - mn)) for (c, s), s in zip(remaining, scores)]

        selected = []
        doc_counts: dict[str, int] = {}
        doc_section_paths: dict[str, set] = {}
        doc_parent_paths: dict[str, set] = {}

        # Step 1: strict diversity — max 2 per doc
        for _ in range(max_chunks):
            if not normed:
                break
            best_score = -float("inf")
            best_idx = -1
            for i, (c, ns) in enumerate(normed):
                doc = c.boi_reference
                cnt = doc_counts.get(doc, 0)
                if cnt >= 3:
                    continue
                penalty = 0.0
                if cnt == 2:
                    penalty = 0.30
                elif cnt == 1:
                    penalty = 0.15

                sp = tuple(c.chunk.section_path)
                if doc in doc_section_paths and sp in doc_section_paths[doc]:
                    penalty += 0.15
                parent = tuple(c.chunk.section_path[:-1]) if c.chunk.section_path else ()
                if doc in doc_parent_paths and parent in doc_parent_paths[doc]:
                    penalty += 0.05

                adjusted = ns - penalty
                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = i
            if best_idx < 0 or best_score < 0:
                break
            best = normed[best_idx][0]
            selected.append(best)
            doc = best.boi_reference
            doc_counts[doc] = doc_counts.get(doc, 0) + 1
            sp = tuple(best.chunk.section_path)
            doc_section_paths.setdefault(doc, set()).add(sp)
            parent = tuple(best.chunk.section_path[:-1]) if best.chunk.section_path else ()
            doc_parent_paths.setdefault(doc, set()).add(parent)
            normed.pop(best_idx)

        # Step 2: if not enough, relax — allow 3rd chunk per doc
        if len(selected) < max_chunks:
            for _ in range(max_chunks - len(selected)):
                if not normed:
                    break
                best_score = -float("inf")
                best_idx = -1
                for i, (c, ns) in enumerate(normed):
                    doc = c.boi_reference
                    cnt = doc_counts.get(doc, 0)
                    if cnt >= 4:
                        continue
                    adjusted = ns - (0.10 if cnt >= 3 else 0)
                    if adjusted > best_score:
                        best_score = adjusted
                        best_idx = i
                if best_idx < 0:
                    break
                best = normed[best_idx][0]
                selected.append(best)
                doc_counts[best.boi_reference] = doc_counts.get(best.boi_reference, 0) + 1
                normed.pop(best_idx)

        # Step 3: last resort — fill from normed leftovers
        if len(selected) < max_chunks and normed:
            leftovers = sorted(normed, key=lambda x: x[1], reverse=True)
            for c, _ in leftovers[:max_chunks - len(selected)]:
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

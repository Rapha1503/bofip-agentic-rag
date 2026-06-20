from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import ChunkNode, RawDocument
from .text_utils import normalize_whitespace


DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"


def _resolve_local_model_path(model_name: str) -> str:
    direct_path = Path(model_name)
    if direct_path.exists():
        return str(direct_path)

    if "/" not in model_name:
        return model_name

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not cache_root.exists():
        return model_name

    snapshots = [path for path in cache_root.iterdir() if path.is_dir()]
    if not snapshots:
        return model_name

    latest_snapshot = max(snapshots, key=lambda path: path.stat().st_mtime)
    return str(latest_snapshot)


def build_dense_chunk_text(chunk: ChunkNode, *, mode: str = "full") -> str:
    if mode == "body":
        parts = [chunk.text]
    elif mode == "leaf":
        parts = [
            chunk.section_path[-1] if chunk.section_path else "",
            " ".join(chunk.legal_refs),
            chunk.text,
        ]
    elif mode == "full":
        parts = [
            chunk.boi_reference,
            " ".join(chunk.section_path),
            " ".join(chunk.legal_refs),
            chunk.text,
        ]
    else:
        raise ValueError(f"Unsupported dense chunk mode: {mode}")
    return "\n".join(part for part in parts if normalize_whitespace(part))


def build_dense_document_text(document: RawDocument, *, mode: str = "sections_firstpara") -> str:
    section_titles = " ".join(section.title for section in document.sections[:12])
    first_paragraphs = " ".join(paragraph.text for paragraph in document.paragraphs[:3])
    parts = [
        document.boi_reference,
        document.title,
        document.html_title or "",
        " ".join(document.category_path),
        " ".join(document.subjects),
    ]
    if mode in {"sections", "sections_firstpara"}:
        parts.append(section_titles)
    if mode == "sections_firstpara":
        parts.append(first_paragraphs)
    if mode not in {"base", "sections", "sections_firstpara"}:
        raise ValueError(f"Unsupported dense document mode: {mode}")
    return "\n".join(part for part in parts if normalize_whitespace(part))


def _dense_prompt_style(model_name: str) -> str:
    normalized = model_name.lower()
    if "e5" in normalized:
        return "e5"
    return "plain"


def _query_text(query: str, *, prompt_style: str) -> str:
    normalized = normalize_whitespace(query)
    if prompt_style == "e5":
        return f"query: {normalized}"
    return normalized


def _passage_text(text: str, *, prompt_style: str) -> str:
    normalized = normalize_whitespace(text)
    if prompt_style == "e5":
        return f"passage: {normalized}"
    return normalized


@dataclass
class DenseRetrievalHit:
    rank: int
    score: float
    chunk: ChunkNode


@dataclass
class DenseDocumentRetrievalHit:
    rank: int
    score: float
    boi_reference: str
    best_chunk: ChunkNode


class DenseEncoder:
    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL, *, device: str | None = None):
        self.model_name = model_name
        self.model_path = _resolve_local_model_path(model_name)
        self.prompt_style = _dense_prompt_style(model_name)
        self.device = device
        model_is_local = Path(self.model_path).exists()
        local_files_only = os.environ.get("BOFIP_LOCAL_FILES_ONLY", "").strip().lower() in {"1", "true", "yes"}
        self.model = SentenceTransformer(
            self.model_path,
            local_files_only=local_files_only or model_is_local,
            device=device,
            model_kwargs={"torch_dtype": "auto"},
        )

    def encode_queries(self, queries: list[str], *, batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                [_query_text(query, prompt_style=self.prompt_style) for query in queries],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
            )
        )

    def encode_chunks(
        self,
        chunks: list[ChunkNode],
        *,
        mode: str = "full",
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        texts = [_passage_text(build_dense_chunk_text(chunk, mode=mode), prompt_style=self.prompt_style) for chunk in chunks]
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
            )
        )

    def encode_documents(
        self,
        documents: list[RawDocument],
        *,
        mode: str = "sections_firstpara",
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        texts = [_passage_text(build_dense_document_text(document, mode=mode), prompt_style=self.prompt_style) for document in documents]
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
            )
        )


class DenseIndex:
    def __init__(self, chunks: list[ChunkNode], chunk_embeddings: np.ndarray):
        if len(chunks) != len(chunk_embeddings):
            raise ValueError("chunks and chunk_embeddings must have the same length")
        self.chunks = list(chunks)
        self.chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

    def search_from_vector(self, query_embedding: np.ndarray, *, top_k: int = 5) -> list[DenseRetrievalHit]:
        if not len(self.chunks):
            return []
        scores = np.dot(self.chunk_embeddings, np.asarray(query_embedding, dtype=np.float32))
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            DenseRetrievalHit(rank=rank + 1, score=float(scores[idx]), chunk=self.chunks[int(idx)])
            for rank, idx in enumerate(ranked_indices)
        ]

    def search_documents_from_vector(self, query_embedding: np.ndarray, *, top_k: int = 5) -> list[DenseDocumentRetrievalHit]:
        if not len(self.chunks):
            return []
        scores = np.dot(self.chunk_embeddings, np.asarray(query_embedding, dtype=np.float32))
        ranked_indices = np.argsort(scores)[::-1]

        docs: list[DenseDocumentRetrievalHit] = []
        seen: set[str] = set()
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)]
            boi_reference = chunk.boi_reference
            if boi_reference in seen:
                continue
            seen.add(boi_reference)
            docs.append(
                DenseDocumentRetrievalHit(
                    rank=len(docs) + 1,
                    score=float(scores[int(idx)]),
                    boi_reference=boi_reference,
                    best_chunk=chunk,
                )
            )
            if len(docs) >= top_k:
                break
        return docs

    def search(self, encoder: DenseEncoder, query: str, *, top_k: int = 5, batch_size: int = 32) -> list[DenseRetrievalHit]:
        query_embedding = encoder.encode_queries([query], batch_size=batch_size)[0]
        return self.search_from_vector(query_embedding, top_k=top_k)


@dataclass
class DenseRawDocumentHit:
    rank: int
    score: float
    document: RawDocument

    @property
    def boi_reference(self) -> str:
        return self.document.boi_reference


class DenseDocumentIndex:
    def __init__(self, documents: list[RawDocument], document_embeddings: np.ndarray):
        if len(documents) != len(document_embeddings):
            raise ValueError("documents and document_embeddings must have the same length")
        self.documents = list(documents)
        self.document_embeddings = np.asarray(document_embeddings, dtype=np.float32)

    def search_from_vector(self, query_embedding: np.ndarray, *, top_k: int = 5) -> list[DenseRawDocumentHit]:
        if not len(self.documents):
            return []
        scores = np.dot(self.document_embeddings, np.asarray(query_embedding, dtype=np.float32))
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [
            DenseRawDocumentHit(rank=rank + 1, score=float(scores[idx]), document=self.documents[int(idx)])
            for rank, idx in enumerate(ranked_indices)
        ]

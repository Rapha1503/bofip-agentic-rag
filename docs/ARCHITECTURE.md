# BOFiP Agentic RAG Architecture

This document describes the current cleanroom architecture for the BOFiP RAG prototype by Rapha1503.

## Runtime Flow

```text
User question
  -> optional LLM rewrite and facet detection
  -> per-facet retrieval
  -> document-stage hybrid retrieval
  -> local chunk retrieval inside selected documents
  -> cross-encoder reranking
  -> diversity-capped context selection
  -> coverage-aware answer prompt
  -> cited JSON answer
```

The shared retrieval engine is `src/bofip_cleanroom/rag_runtime.py`. The Streamlit app currently owns extra orchestration for query rewriting, multi-facet retrieval, computation-aware facets, and post-facet merging. A planned Phase 2 refactor will move that logic into a shared package module so CLI, app, and evaluation use the same RAG contract.

## Data Flow

```text
Local BOFiP export
  -> discovery.py
  -> xml_parser.py + html_parser.py
  -> document_builder.py
  -> RawDocument JSONL
  -> chunking.py
  -> ChunkNode JSONL
  -> dense_retrieval.py embedding caches
  -> rag_runtime.py in-memory indexes
```

Current full commentary runtime:

| Layer | Artifact |
| --- | --- |
| Raw documents | `data/interim/raw_docs_sample_5666.jsonl` |
| Chunks | `data/interim/chunks_section_window_sample_5666.jsonl` |
| Document embeddings | `data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy` |
| Chunk embeddings | `data/interim/chunk_dense_cache_5666_full_e5large.npy` |

The active embedding caches use E5-large 1024-dimensional vectors for both document and chunk retrieval.

## Core Modules

| Module | Responsibility |
| --- | --- |
| `models.py` | Dataclasses for raw documents, sections, paragraphs, tables, and chunks. |
| `discovery.py` | Finds local BOFiP XML/HTML document pairs. |
| `xml_parser.py` | Extracts metadata, identifiers, BOI references, dates, relations, and source URLs. |
| `html_parser.py` | Extracts section trees, paragraphs, links, legal refs, and tables. |
| `document_builder.py` | Combines XML and HTML payloads into `RawDocument`. |
| `chunking.py` | Builds section-window, paragraph-preserving, and parent-child chunks. |
| `lexical_retrieval.py` | BM25 indexes and French tokenization/stemming helpers. |
| `dense_retrieval.py` | E5 encoding and in-memory dense search over precomputed arrays. |
| `hybrid_retrieval.py` | Reciprocal rank fusion and confidence-weighted source scoring. |
| `direct_chunk_retrieval.py` | Local chunk retrieval inside selected Stage 1 documents. |
| `reranker.py` | Cross-encoder reranking. |
| `rag_runtime.py` | Main retrieval runtime and result contract. |
| `prompt_utils.py` | Coverage-aware answer prompt with citation constraints. |
| `eval_harness.py` | doc@k, passage@k, MRR, NDCG, and per-query diagnostics. |

## Public Interfaces

- `app.py`: Streamlit UI for interactive questions and batch testing.
- `scripts/preview_answer.py`: CLI answer preview with retrieval plus LLM generation.
- `scripts/evaluate.py`: standardized retrieval evaluation.
- `scripts/ablation.py`: retrieval component ablation.

Historical phase scripts and notebooks are intentionally excluded from the public cleanroom surface.

## Evaluation Assets

The tracked evaluation set contains 50 questions across direct lookup, paraphrase, cross-document, edge-case, and unsupported categories:

- `data/interim/eval_queries_v1.jsonl`
- `data/interim/passage_gold_v3.jsonl`

Current evaluation is strongest at retrieval level. Answer-level scoring for citation faithfulness, abstention quality, table-heavy cases, and calculations is planned.

## Known Technical Risks

- Query rewrite/facet logic is not yet shared by all interfaces.
- Some BOI references are duplicated across distinct documents; retrieval identity should move to stable `document_id`.
- Tables are parsed but not yet chunked as first-class retrievable evidence.
- Runtime artifacts are large and must be managed outside Git with a manifest and checksums.
- The project is a research prototype and must not be presented as tax advice.

# BOFiP Agentic RAG Architecture

This document describes the current architecture for the BOFiP Agentic RAG prototype by Raphael Ifergan.

## Runtime Flow

```text
User question
  -> AgenticRAG._classify_domain()
  -> RagRuntime.retrieve(boost_prefix=...)
  -> LLM answer with inline coverage self-evaluation
  -> if partial or insufficient: targeted BOFiP reformulation
  -> second retrieval over missing axes
  -> merged cited answer with trace
```

The shared retrieval engine is `src/bofip_agentic/rag_runtime.py`. The agentic orchestration is centralized in `src/bofip_agentic/agent_rag.py`, so the Streamlit app calls the same reusable pipeline instead of owning separate retrieval logic.

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
  -> agent_rag.py controlled multi-pass loop
```

Current full commentary runtime:

| Layer | Artifact |
| --- | --- |
| Raw documents | `data/interim/raw_docs_sample_5666.jsonl` |
| Chunks | `data/interim/chunks_section_window_sample_5666.jsonl` |
| Document embeddings | `data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy` |
| Chunk embeddings | `data/interim/chunk_dense_cache_5666_full_e5large.npy` |

## Core Modules

| Module | Responsibility |
| --- | --- |
| `agent_rag.py` | Controlled agent loop: classify, retrieve, answer, evaluate, reformulate, retry. |
| `rag_runtime.py` | Hybrid retrieval runtime and result contract. |
| `prompt_utils.py` | Citation-constrained answer prompt and coverage schema. |
| `providers.py` | Provider/model dropdown configuration for BYOK usage. |
| `artifact_download.py` | Runtime artifact download and manifest validation. |
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
| `reranker.py` | Optional cross-encoder reranking. |
| `eval_harness.py` | doc@k, passage@k, MRR, NDCG, and per-query diagnostics. |

## Public Interfaces

- `app.py`: Streamlit BYOK UI with visible agent trace.
- `scripts/setup.py`: first-time corpus build.
- `scripts/sync.py`: corpus refresh pipeline.
- `scripts/eval_full.py`: 50-query agentic evaluation.
- `scripts/eval_agent.py`: focused agent benchmark helper.
- `scripts/check_setup.py`: local artifact and model preflight check.
- `scripts/download_artifacts.py`: release artifact downloader.

## Evaluation Assets

The tracked evaluation set contains 50 questions across direct lookup, paraphrase, cross-document, edge-case, and unsupported categories:

- `data/interim/eval_queries_v1.jsonl`
- `data/interim/passage_gold_v3.jsonl`

## Known Technical Risks

- Some BOI references are duplicated across distinct documents; retrieval identity should move to stable `document_id`.
- Tables are parsed but not yet chunked as first-class retrievable evidence.
- Runtime artifacts are large and must be managed outside Git with a manifest and checksums.
- The hosted CPU demo disables the cross-encoder reranker by default; local GPU runs can enable it.
- The project is a research prototype and must not be presented as tax advice.

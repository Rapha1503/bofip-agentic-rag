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
data.economie.gouv.fr bofip-vigueur API
  -> scripts/sync.py no-filter source download
  -> stable source identity from BOFiP permalink / PGP id
  -> html_parser.py
  -> RawDocument JSONL
  -> chunking.py
  -> ChunkNode JSONL
  -> dense_retrieval.py embedding caches
  -> rag_runtime.py in-memory indexes
  -> agent_rag.py controlled multi-pass loop
```

Current full source runtime:

| Layer | Artifact |
| --- | --- |
| Parsed runtime documents | `data/interim/raw_docs.jsonl` |
| Chunks | `data/interim/chunks.jsonl` |
| Document embeddings | `data/interim/doc_dense_cache.npy` |
| Chunk embeddings | `data/interim/chunk_dense_cache.npy` |

Current active counts:

| Metric | Value |
| --- | ---: |
| BOFiP source rows | 9,048 |
| Stable document IDs | 9,048 |
| Base BOI references | 9,025 |
| Section-window chunks | 79,160 |
| Document embeddings | `(9048, 1024)` |
| Chunk embeddings | `(79160, 1024)` |

## Core Modules

| Module | Responsibility |
| --- | --- |
| `agent_rag.py` | Controlled agent loop: classify, retrieve, answer, evaluate, reformulate, retry. |
| `rag_runtime.py` | Retrieval runtime and result contract. BM25 full-corpus is the public default; E5 hybrid mode is optional. |
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
- `scripts/sync.py`: authoritative full-corpus refresh pipeline from the public API.
- `scripts/setup.py`: legacy local parser/chunker helper; it should not be used to copy artifacts from older projects.
- `scripts/eval_full.py`: 50-query agentic evaluation.
- `scripts/eval_agent.py`: focused agent benchmark helper.
- `scripts/check_setup.py`: local artifact and model preflight check.
- `scripts/download_artifacts.py`: release artifact downloader.

## Evaluation Assets

The tracked evaluation set contains 50 questions across direct lookup, paraphrase, cross-document, edge-case, and unsupported categories:

- `data/interim/eval_queries_v1.jsonl`
- `data/interim/passage_gold_v3.jsonl`

## Known Technical Risks

- Some BOI references are duplicated across distinct documents; runtime identity uses stable `document_id` while `boi_reference` remains the citation/routing reference.
- Tables are parsed and emitted as retrievable chunks, but table-heavy answer evaluation is still thin.
- Runtime artifacts are large and must be managed outside Git with a manifest and checksums.
- The hosted CPU demo disables the cross-encoder reranker by default; local GPU runs can enable it.
- The project is a research prototype and must not be presented as tax advice.

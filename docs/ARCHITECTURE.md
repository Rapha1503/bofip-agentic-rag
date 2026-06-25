# Architecture

BOFiP Agentic RAG is a full-corpus retrieval and answer-generation prototype for French BOFiP tax doctrine.

## Global Flow

```text
User question
  -> provider/model selection (BYOK)
  -> optional query rewrite
  -> fiscal facet detection
  -> document retrieval
  -> local chunk retrieval inside selected documents
  -> optional dense retrieval / reranking
  -> diversity-capped context selection
  -> source-aware answer prompt
  -> cited answer with explicit limits
```

## Data Pipeline

```text
BOFiP XML/HTML export
  -> metadata and section parsing
  -> RawDocument JSONL
  -> section-window chunking
  -> ChunkNode JSONL
  -> BM25 index
  -> E5 embedding caches
  -> Streamlit runtime
```

## Runtime Artifacts

| Artifact | Role |
|---|---|
| `data/interim/raw_docs_sample_5666.jsonl` | parsed BOFiP commentary documents |
| `data/interim/chunks_section_window_sample_5666.jsonl` | section-window passages used for retrieval |
| `data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy` | document-level E5 embeddings |
| `data/interim/chunk_dense_cache_5666_full_e5large.npy` | chunk-level E5 embeddings |
| `docs/full_corpus_manifest.json` | public artifact contract with counts and checksums |

The large runtime artifacts are intentionally kept outside Git and downloaded from release assets.

## Main Modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, provider selection, BYOK session handling |
| `src/bofip_cleanroom/rag_runtime.py` | full retrieval runtime |
| `src/bofip_cleanroom/lexical_retrieval.py` | BM25 retrieval and French tokenization |
| `src/bofip_cleanroom/dense_retrieval.py` | E5 encoding and dense search |
| `src/bofip_cleanroom/direct_chunk_retrieval.py` | chunk search inside selected BOFiP documents |
| `src/bofip_cleanroom/hybrid_retrieval.py` | reciprocal-rank fusion |
| `src/bofip_cleanroom/reranker.py` | optional cross-encoder reranking |
| `src/bofip_cleanroom/llm_preview.py` | answer prompting, parsing, and fallbacks |
| `src/bofip_cleanroom/eval_harness.py` | evaluation utilities |

## Design Choices

- **Full corpus, no reduced demo:** latency is handled with prebuilt artifacts and caches, not by shrinking the BOFiP scope.
- **Traceable answers:** the interface exposes retained BOFiP sources and limitations instead of hiding uncertainty.
- **BYOK model access:** provider keys are supplied by the user at runtime and are not committed to the repository.
- **Optional quality layers:** dense retrieval and reranking can improve quality locally, but the public CPU demo keeps latency constraints visible.

## Known Constraints

- The hosted demo is slower than a one-pass RAG because it prioritizes source traceability.
- Some BOFiP tables are parsed but are not yet first-class retrieval units.
- BOFiP updates require rebuilding the artifact bundle and manifest.

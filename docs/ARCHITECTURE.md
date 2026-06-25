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
  -> source review and answer prompt
  -> cited answer with explicit limits
```

## Data Pipeline

```text
BOFiP public API snapshot
  -> metadata and content normalization
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
| `data/interim/raw_docs.jsonl` | parsed BOFiP commentary documents |
| `data/interim/chunks.jsonl` | section-window passages used for retrieval |
| `data/interim/doc_dense_cache.npy` | document-level E5 embeddings |
| `data/interim/chunk_dense_cache.npy` | chunk-level E5 embeddings |
| `docs/full_corpus_manifest.json` | public artifact contract with counts and checksums |

The large runtime artifacts are intentionally kept outside Git and downloaded from release assets.

## Main Modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, provider selection, BYOK session handling |
| `src/bofip_agentic/rag_runtime.py` | full retrieval runtime |
| `src/bofip_agentic/lexical_retrieval.py` | BM25 retrieval and French tokenization |
| `src/bofip_agentic/dense_retrieval.py` | E5 encoding and dense search |
| `src/bofip_agentic/direct_chunk_retrieval.py` | chunk search inside selected BOFiP documents |
| `src/bofip_agentic/hybrid_retrieval.py` | reciprocal-rank fusion |
| `src/bofip_agentic/reranker.py` | optional cross-encoder reranking |
| `src/bofip_agentic/agent_rag.py` | planner, source critic, relaunch loop, and final answer |
| `src/bofip_agentic/eval_harness.py` | evaluation utilities |

## Design Choices

- **Full corpus, no reduced demo:** latency is handled with prebuilt artifacts and caches, not by shrinking the BOFiP scope.
- **Traceable answers:** the interface exposes retained BOFiP sources and limitations instead of hiding uncertainty.
- **BYOK model access:** provider keys are supplied by the user at runtime and are not committed to the repository.
- **Optional quality layers:** dense retrieval and reranking can improve quality locally, but the public CPU demo keeps latency constraints visible.

## Known Constraints

- The hosted demo is slower than a one-pass RAG because it prioritizes source traceability.
- Some BOFiP tables are parsed but are not yet first-class retrieval units.
- BOFiP updates require rebuilding the artifact bundle and manifest.

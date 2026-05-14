# BOFIP RAG — Architecture

**Single source of truth for pipeline design, data flow, and component reference.**

## 1. Overview

BOFIP RAG is a retrieval-augmented generation pipeline for the *Bulletin Officiel des Finances Publiques* — the official French tax authority doctrine. It answers accountant-style fiscal questions with cited evidence from 5,666 BOFIP commentary documents.

**Stack**: Python 3.11 · Sentence-Transformers · rank-bm25 · OpenAI SDK · Streamlit · DeepSeek

## 2. Pipeline Architecture

```
User Query (str, natural language French)
    │
    ▼ QUERY REWRITING (optional, via LLM)
    └── Expands acronyms, formalizes vocabulary, adds CGI/LPF references
        → rewritten_query (str)
    │
    ▼ STAGE 1 — Multi-View Hybrid Document Retrieval
    ├── Lexical BM25 [base]          → top-20 doc_ids + scores
    ├── Lexical BM25 [sections_leads]→ top-20 doc_ids + scores
    ├── Lexical BM25 [sections_stem] → top-20 doc_ids + scores
    ├── Dense E5-large (doc embed)   → top-20 doc_ids + scores
    ├── Dense E5-base (chunk embed)  → top-20 doc_ids + scores (max-pooled)
    │
    ├── DENSE-ANCHOR FILTER
    │   └── Lexical sources may only rank documents also found by dense top-20.
    │       Prevents "convention spam" on queries with "résident fiscal".
    │
    └── Confidence-weighted RRF fusion
        → top-8 document_ids (ranked, scored)
    │
    ▼ STAGE 2 — Local Chunk Retrieval
    └── BM25 inside each top-8 document
        → 8 chunks per document → 64 candidate (chunk_id, score) tuples
    │
    ▼ RERANKER — Cross-Encoder bge-reranker-v2-m3 (GPU)
    └── Section path prepended to chunk text
        → scores each (query, chunk_text) pair
        → top-8 chunks (final evidence)
    │
    ▼ LLM — Coverage-Aware Answer Generation
    └── Prompt: decompose into axes_requis → check coverage → produce:
        {
          "answer_status":  "supported" | "partial" | "insufficient_evidence",
          "axes_requis":     ["axe1", "axe2", ...],
          "axes_couverts":   ["axe1"],
          "axes_manquants":  ["axe2"],
          "conclusion":      "phrase courte (≤30 mots)",
          "justification_bullets": ["puce avec citation [n]", ...],
          "limits":          "axes manquants ou conditions non couvertes"
        }
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_docs` | 8 | Documents retrieved in Stage 1 |
| `chunks_per_doc` | 8 | Chunks per doc in Stage 2 |
| `max_chunks` | 8 | Final chunks after reranker |
| `rank_constant` | 60 | RRF fusion parameter |
| `source_weights` | dense=2.0, chunk_dense=2.0, base=0.5, sections_leads=0.5, sections_stem=0.5 | Fusion source weighting |

## 3. Component Reference

### Core Module (`src/bofip_cleanroom/`)

| File | Role | Key Classes |
|------|------|-------------|
| `rag_runtime.py` | Main retrieval runtime | `RagRuntime`, `CORPUS_PATHS`, `DEFAULT_SOURCE_WEIGHTS` |
| `reranker.py` | Cross-encoder reranker | `CrossEncoderReranker` (bge-reranker-v2-m3) |
| `dense_retrieval.py` | Dense embeddings (E5) | `DenseEncoder`, `DenseIndex`, `DenseDocumentIndex` |
| `lexical_retrieval.py` | BM25 with French stemming | `LexicalBM25Index`, `DocumentLexicalIndex`, `tokenize()` |
| `hybrid_retrieval.py` | RRF + confidence-weighted fusion | `confidence_weighted_reciprocal_rank_fuse()`, `RankedDoc`, `HybridDocHit` |
| `direct_chunk_retrieval.py` | Local BM25 chunk search | `DirectChunkRetriever`, `Stage1DocumentHit` |
| `chunking.py` | Document → chunks | 3 strategies: section_window, paragraph, parent_child |
| `models.py` | Data structures | `RawDocument` (26 fields), `ChunkNode` (13 fields) |
| `eval_harness.py` | Retrieval metrics | `EvalMetrics`, `QueryGold`, `evaluate()` (doc@k, passage@k, MRR, NDCG) |
| `env_utils.py` | .env loader | `load_default_env_files()` (utf-8-sig, BOM-safe) |
| `preview_runtime.py` | ⚠️ LEGACY Phase8b | Superseded by `rag_runtime.py` |
| `llm_preview.py` | ⚠️ LEGACY LLM interface | Superseded by `scripts/preview_answer.py` + `app.py` |

### Scripts

| File | Role |
|------|------|
| `scripts/preview_answer.py` | CLI: single-query, batch, resume. Query rewriting + retrieval + LLM. |
| `scripts/evaluate.py` | CLI: standardized retrieval eval (phase8b or rag runtime) |
| `app.py` | Streamlit UI: multi-provider, single + batch, full pipeline transparency |
| `scripts/ablation.py` | Component ablation testing (BM25 only, Dense only, Full, +reranker) |

## 4. Data Flow & Storage

```
RAW BOFIP FILES (6295 documents, XML + HTML)
    │
    ▼ Phase 0-1 (discovery.py, xml_parser.py, html_parser.py, document_builder.py)
    └── raw_docs_sample_5666.jsonl  (RawDocument records)
    │
    ▼ Phase 2 (chunking.py, strategy: section_window)
    └── chunks_section_window_sample_5666.jsonl  (66,289 chunks)
    │
    ▼ Dense encoding (dense_retrieval.py)
    ├── doc_dense_cache_5666_sections_firstpara_e5large.npy  (5666 × 1024 float32)
    └── chunk_dense_cache_5666_full_e5.npy  (66289 × 768 float32)
    │
    ▼ Runtime (rag_runtime.py)
    ├── 3 × DocumentLexicalIndex (BM25, in-memory, ~500 MB)
    ├── DenseDocumentIndex + DenseIndex (in-memory, embeddings loaded from .npy)
    └── DirectChunkRetriever (lazy per-document BM25 indices)
    │
    ▼ Models (downloaded/cached)
    ├── intfloat/multilingual-e5-large  (~2 GB, 1024-dim)
    ├── intfloat/multilingual-e5-base   (~1 GB, 768-dim)
    └── BAAI/bge-reranker-v2-m3         (~2 GB, cross-encoder)
```

## 5. Evaluation

**Benchmark**: `data/interim/eval_queries_v1.jsonl` — 50 queries in 5 categories:
- `direct` (15): Direct lookup of known BOFIP fact
- `paraphrase` (15): Same fact, different wording
- `cross_document` (10): Requires 2+ BOFIP documents
- `edge_case` (5): False premises, over-specific, overly vague
- `unsupported` (5): Topics BOFIP doesn't cover

**Gold annotations**: `data/interim/passage_gold_v3.jsonl` — aligned to 5666-doc section_window chunks.

**Metrics**: doc@1/3/5/8, passage@1/3/5/8, MRR (doc + passage), NDCG@1/3/5/8.

**Latest results** (15-query benchmark): 12 correct, 2 partial, 1 honest, 0 wrong.

## 6. Providers

Supported LLM providers (via OpenAI-compatible API):

| Provider | Default Model | API Key Env |
|----------|--------------|-------------|
| DeepSeek | deepseek-chat | DEEPSEEK_API_KEY |
| OpenAI | gpt-4o-mini | OPENAI_API_KEY |
| Anthropic | claude-3-5-haiku | ANTHROPIC_API_KEY |
| Mistral | mistral-small-latest | MISTRAL_API_KEY |
| Google (Gemini) | gemini-2.5-flash | GEMINI_API_KEY |
| Groq | llama-4-scout | GROQ_API_KEY |
| Together | Llama-4-Maverick | TOGETHER_API_KEY |

API keys can be set via `.env.local`, environment variable, or the Streamlit sidebar (password field, not saved).

## 7. Deployment

### Local
```powershell
streamlit run app.py
```

### HuggingFace Spaces (free)
1. Push to GitHub (public repo)
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces), create a new Space
3. Select "Streamlit" as SDK
4. Connect your GitHub repo
5. Set `DEEPSEEK_API_KEY` as a Secret in Space settings
6. The Space auto-deploys on push

**Note**: The app auto-detects available corpus. If the full 5666-doc corpus is absent (as on Spaces with limited disk), it falls back to the 200-doc demo corpus (~25 MB). To deploy the full corpus, use Git LFS or download on startup.

## 8. Known Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| 30s/query latency | Poor UX for real-time use | Use GPU, reduce rewrite tokens |
| 47s cold start | Annoying on first load | `st.cache_resource` persists runtime |
| "dictionnaire" → "livres" gap | Can't find BOFIP book rules for dictionaries | LLM query rewriting helps other cases |
| Gold annotations approximate | Passage recall undercounted | Review gold for v4 |
| French-only | Doesn't work with other tax authorities | Architecture is language-agnostic, data is French |

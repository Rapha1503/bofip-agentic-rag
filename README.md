# BOFIP RAG Cleanroom

RAG pipeline for French fiscal documents (BOFIP — Bulletin Officiel des Finances Publiques) with hybrid retrieval, cross-encoder reranking, LLM query rewriting, and coverage-aware generation.

## Architecture

```
user question
    │
    ▼ QUERY REWRITING (DeepSeek: informal → BOFIP vocabulary)
    │
    ▼ STAGE 1 — DOCUMENT RETRIEVAL
    ├── Lexical BM25 (base, sections_leads, sections_leads_stem)
    ├── Dense E5-large (document embeddings)
    ├── Chunk-dense E5-base (aggregated to documents)
    └── Dense-anchor filter → blocks lexical pollution from INT conventions
        └── Confidence-weighted RRF fusion → top-8 docs
    │
    ▼ STAGE 2 — CHUNK CANDIDATES
    └── Local BM25 inside each top doc → 8 chunks/doc → 64 candidates
    │
    ▼ RERANKER — Cross-encoder bge-reranker-v2-m3 (GPU)
    └── Section path prepended to chunk text → top-8 chunks
    │
    ▼ LLM — DeepSeek-chat (coverage-aware prompt)
    └── Identifies required fiscal axes → checks coverage
        → status: supported | partial | insufficient_evidence
        → cited justification with gap analysis
```

## Key Metrics

**15-query benchmark** (realistic accountant questions across BOFIP domains):  
12 correct · 2 partial · 1 honest · 0 wrong — report: `data/reports/batch_final_v1.json`

**102 unit tests** passing.

## Corpus

- **5666 commentary** BOFIP documents
- **66289 chunks** (section_window strategy)
- Dense embeddings: E5-large for docs, E5-base for chunks

## Commands

```powershell
# Single query
$env:PYTHONPATH='src'
python scripts/preview_answer.py --query "votre question fiscale"

# Batch (with resume support)
python scripts/preview_answer.py --input data/interim/batch_final.jsonl --output data/reports/batch.json

# Resume failed batch
python scripts/preview_answer.py --input data/interim/batch.jsonl --resume data/reports/batch.json --output data/reports/batch_v2.json

# Standardized evaluation
python scripts/evaluate.py --runtime rag --case-ids q001,q002 --limit 10

# Tests
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure DeepSeek API key in `.env.local`:
```
DEEPSEEK_API_KEY=sk-...
```

## Project Layout

```
src/bofip_cleanroom/
├── rag_runtime.py          # Clean retrieval runtime (dense-anchor, configurable weights)
├── reranker.py              # CrossEncoderReranker (bge-reranker-v2-m3)
├── dense_retrieval.py       # DenseEncoder, DenseIndex, DenseDocumentIndex
├── lexical_retrieval.py     # BM25 with French stemming
├── hybrid_retrieval.py      # Confidence-weighted RRF fusion
├── direct_chunk_retrieval.py# Local chunk BM25 inside top docs
├── chunking.py              # section_window, paragraph_preserving, parent_child
├── models.py                # RawDocument, ChunkNode, dataclasses
├── eval_harness.py          # EvalMetrics, QueryGold, evaluate()
├── env_utils.py             # .env.local / .env loader with BOM safety
├── jsonio.py                # JSON/JSONL read/write
├── settings.py              # Project paths
├── text_utils.py            # Normalization, token counting
├── html_parser.py           # BOFIP HTML parsing
├── xml_parser.py            # BOFIP XML metadata extraction
├── document_builder.py      # RawDocument assembly
├── discovery.py             # BOFIP file discovery
├── sampling.py              # Stratified sampling
├── llm_preview.py           # Legacy LLM interface (Gemini/OpenAI/DeepSeek)
├── pre_llm_verification.py  # Stack integrity checks
└── versioning.py            # Manifest builder

scripts/
├── preview_answer.py        # Main entry point: single-query, batch, resume
├── evaluate.py              # Standardized eval with eval_harness
└── phase*.py                # Historical pipeline scripts (Phases 0-8b)

data/
├── interim/
│   ├── eval_queries_v1.jsonl    # 50 diverse queries (5 categories)
│   ├── passage_gold_v3.jsonl    # Gold passage annotations
│   ├── batch_final.jsonl        # 15-query benchmark input
│   ├── raw_docs_sample_5666.jsonl
│   ├── chunks_section_window_sample_5666.jsonl
│   ├── doc_dense_cache_5666_*.npy
│   └── chunk_dense_cache_5666_*.npy
├── models/
│   └── intfloat--multilingual-e5-large/
└── reports/
    └── batch_final_v1.json      # 15-query benchmark results
```

# BOFIP-RAG - CV Project Dossier

## 1) Project Identity

**Project name:** BOFIP-RAG  
**Type:** Domain-specific Retrieval-Augmented Generation (RAG) assistant  
**Domain:** French tax and accounting law (BOFiP, CGI, LPF)  
**Primary users:** Accounting firms and tax professionals (expert-comptables)  
**Product form:** Streamlit conversational assistant (`app.py`)

---

## 2) Why This Project Exists

French tax professionals need fast, reliable, and source-backed answers across large and frequently updated legal/doctrinal corpora.  
This project addresses:

- Search friction across fragmented fiscal/legal sources
- Hallucination risk in generic LLM chatbots
- Need for traceable answers with legal references
- Time pressure in accounting workflows

**Core positioning:** research assistant, not legal advice replacement.

---

## 3) Problem Solved and Stake

### Operational stake
- Reduce time-to-answer for fiscal questions
- Improve research consistency across teams

### Risk/compliance stake
- Keep answer traceability (sources + references)
- Favor abstention when evidence is insufficient (faithfulness guardrail)
- Preserve legal hierarchy: law texts (CGI/LPF) prevail over commentary

### Business stake
- Foundation for a production-grade legal/accounting copilot
- Strong differentiator for accounting firms handling complex tax topics

---

## 4) End-to-End Pipeline (Input -> Output)

## A. Offline ingestion/indexing pipeline

1. **Raw inputs**
- BOFiP open-data extracted corpus (`data/raw/bofip_extracted/...`)
- LEGI legal archives (`data/raw/legi/*.tar.gz`)
- Optional PDF fallback for CGI/LPF (`data/raw/pdfs/*.pdf`)

2. **Parsing and chunking**
- BOFiP XML/HTML parsing (`src/data_pipeline/parser.py`)
- Semantic chunking by fiscal rule (`src/data_pipeline/semantic_chunker.py`)
- LEGI archive parser with as-of article versioning (`src/data_pipeline/legi_tar_parser.py`)
- Optional PDF legal parser (`src/data_pipeline/pdf_parser.py`)

3. **Structured outputs**
- Unified chunk store: `data/processed/chunks.json`
- Legal delta chunks: `data/processed/legi_chunks.json`
- BM25 index: `data/processed/bm25_index.pkl`
- Chroma vector index: `data/chroma_db/`

## B. Online question-answering pipeline

1. User question (chat UI)
2. Hybrid retrieval (`src/retrieval/hybrid.py`)
- BM25 lexical search
- Vector semantic search (E5 embeddings + ChromaDB)
- Legal diversification (CGI/LPF-only retrieval)
- Cross-reference injection from BOFiP -> CGI/LPF
- Value-aware boosts for rates/amounts
- Cross-encoder reranking
3. Context budget selection for LLM
4. LLM generation with fiscal analysis prompt (`src/generation/prompts.py`)
5. Faithfulness verification (LLM verifier + heuristic fallback)
6. Final output: answer + sources + guardrail status

---

## 5) Current Data and System Snapshot (Workspace)

From `data/processed/chunks.json` and state artifacts:

- **Total indexed chunks:** 86,045
- **BOFiP commentary chunks:** 82,653
- **CGI chunks:** 2,416
- **LPF chunks:** 976

Index synchronization evidence (`data/processed/legi_refresh_state.json`):
- `chunks.json`: 86,045
- BM25: 86,045
- Chroma: 86,045
- Last recorded refresh date in state file: **2026-02-10**

---

## 6) Repository Structure (Functional View)

## Product and config
- `app.py`: Streamlit chat app and interaction flow
- `config.py`: paths, model settings, retrieval and guardrail parameters

## Data pipeline (`src/data_pipeline/`)
- `parser.py`: BOFiP metadata/content extraction
- `semantic_chunker.py`: rule-centric chunking
- `legi_tar_parser.py`: structured legal archive ingestion (CGI/LPF)
- `pdf_parser.py`: fallback legal PDF parsing
- `process.py`: BOFiP processing orchestration

## Retrieval (`src/retrieval/`)
- `bm25.py`: French tokenizer + sparse index/search
- `embeddings.py`: E5 embeddings + model-scoped caching
- `vector_store.py`: Chroma persistence and vector operations
- `reranker.py`: cross-encoder reranker
- `hybrid.py`: production retrieval logic (`search_simple`)

## Generation (`src/generation/`)
- `prompts.py`: fiscal system/user/verifier prompts
- `llm.py`: Groq client, model fallback chain, caching, faithfulness guardrail

## Ops and evaluation scripts (`scripts/`)
- Legal ingestion/sync: `process_legi_archive.py`, `refresh_legi_daily.py`, `sync_legal_chunks.py`
- Index rebuild: `reindex_semantic.py`, `reindex_with_e5.py`
- Retrieval evaluation/tuning: `evaluate_retrieval.py`, `tune_reranker_pool.py`
- Dataset lifecycle: `build_retrieval_dataset_from_cache.py`, `validate_retrieval_dataset.py`
- Embedding A/B benchmark: `benchmark_embeddings.py`

---

## 7) Inputs and Outputs (Concrete)

## Inputs
- User natural-language fiscal questions (French)
- BOFiP XML/HTML corpus
- LEGI legal archives for CGI/LPF
- Optional local PDFs for legal fallback

## Outputs
- Final answer text in French
- Source list (BOI / CGI / LPF references + URLs when available)
- Guardrail status (`faithfulness`: pass/fail/mode/reason)
- Evaluation JSON reports (`data/eval_*.json`, `data/eval_retrieval_*.json`)

---

## 8) Measurable Performance and Evaluation Culture

The project includes both end-to-end and retrieval-only evaluation workflows.

Examples from repository artifacts:
- Expanded retrieval dataset built from production cache signals
- Validation pipeline separating gold and silver questions
- Reranker pool tuning script with recall/precision/latency trade-off
- Embedding benchmark workflow with isolated Chroma collections per model

Representative recorded values:
- Retrieval validation set size: 60 questions (`scripts/test_questions_validated.json`)
- Silver validation rate: ~90.7% (`retrieval_dataset_validation_20260210_180235.json`)
- Reranker pool recommendation: 30 (`reranker_pool_tuning_20260210_173443.json`)

---

## 9) Skills and Competencies Demonstrated

## AI/ML engineering
- RAG system design (hybrid retrieval + reranking + guardrails)
- Embedding lifecycle management and A/B benchmarking
- Cross-encoder ranking integration
- Hallucination mitigation through groundedness checks

## Data engineering
- Multi-source legal/document ingestion pipelines
- XML/HTML/PDF parsing for noisy real-world corpora
- Semantic chunking and metadata normalization
- Incremental refresh and delta upsert workflows

## Search engineering
- BM25 + dense retrieval fusion
- Domain-aware French tokenization
- Query-to-reference cross-link enrichment
- Score calibration and diversification strategies

## MLOps / reliability
- Persistent local index management (BM25 + Chroma consistency)
- Operational scripts for daily refresh and recovery
- Metrics-driven tuning and reproducible benchmark outputs

## Product engineering
- Usable chat interface for professionals
- Transparent source display and disclaimer-first UX
- Pragmatic guardrails for high-stakes domain answers

---

## 10) For Whom and In Which Context

- **Primary users:** accounting firms, tax consultants, expert-comptables
- **Use cases:** quick fiscal rule lookup, legal reference verification, preliminary case analysis
- **Value proposition:** faster research with traceable legal/doctrinal grounding

---

## 11) Interview / CV Ready Summary

## One-line version
Built a production-style French tax RAG assistant combining BOFiP doctrine and CGI/LPF legal texts with hybrid retrieval, reranking, and faithfulness guardrails.

## 3-bullet version
- Designed and implemented an end-to-end legal/fiscal RAG pipeline (ingestion, semantic chunking, indexing, retrieval, generation) over **86k+** chunks.  
- Engineered hybrid retrieval (BM25 + vector + legal cross-reference injection + reranker) and evaluation tooling (recall/precision/hitrate, pool tuning, embedding A/B tests).  
- Delivered operational reliability features: daily legal delta refresh, index sync checks, source-traceable answers, and abstention-based guardrails.

## Impact-style version
Transformed large, heterogeneous French tax/legal corpora into a queryable assistant with measurable retrieval tuning workflows and auditable outputs for accounting professionals.

---

## 12) Current Limits and Next Steps

Observed roadmap themes in project docs/scripts:
- Continue embedding benchmark comparison under realistic compute budgets
- Further calibrate faithfulness thresholds (reduce false abstentions/false passes)
- Expand and harden gold evaluation set with more human-reviewed questions
- Potential vector infrastructure evolution for larger-scale production


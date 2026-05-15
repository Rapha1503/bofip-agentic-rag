# BOFIP RAG — Assistant Fiscal 🇫🇷

RAG pipeline for French fiscal documents (BOFIP — *Bulletin Officiel des Finances Publiques*) with hybrid retrieval, cross-encoder reranking, LLM query rewriting, and coverage-aware generation.

[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()

## Architecture

```
user question
    │
    ▼ QUERY ANALYSIS (LLM: rewrite + facet + computation detection)
    │
    ▼ MULTI-FACET RETRIEVAL (auto-detected per query)
    ├── BM25 (base, sections_leads, sections_leads_stem)
    ├── Dense embeddings (E5-large docs, E5-base chunks)
    ├── Dense-anchor filter — blocks convention spam
    └── Confidence-weighted RRF fusion → top docs
    │
    ▼ CHUNK MERGE + DIVERSITY (sort by score, max 3/doc, dedup)
    │
    ▼ RERANKER — bge-reranker-v2-m3 cross-encoder
    │
    ▼ LLM — Accountant-style answer (DeepSeek, OpenAI, Anthropic, Mistral, Google)
    └── axes_requis/couverts/manquants → supported | partial | insufficient_evidence
        → cited, step-by-step answer with legal reasoning
```

## Key Features

- **Multi-facet retrieval** — auto-detects complex questions, splits into sub-queries per legal axis
- **Dynamic diversity selection** — prevents document domination (max 3 chunks per doc)
- **Computation-aware detection** — finds missing taux/rate sections for calculation questions
- **Chunk deduplication** — merges multi-facet results, keeps best by score
- **Accountant-style answers** — structured: ANSWER + Analyse détaillée with cited steps
- **7 LLM providers** — DeepSeek, OpenAI, Anthropic, Mistral, Google, Groq, Together

## Quick Start

```powershell
git clone https://github.com/Rapha1503/bofip-rag.git
cd bofip-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

echo DEEPSEEK_API_KEY=sk-... > .env.local

# Place data files in data/interim/
$env:PYTHONPATH='src'; streamlit run app.py
```

## Data

Place these files in `data/interim/`:

| File | Description |
|------|-------------|
| `raw_docs_sample_5666.jsonl` | 5,666 parsed BOFIP documents |
| `chunks_section_window_sample_5666.jsonl` | 66,289 chunks |
| `doc_dense_cache_5666_sections_firstpara_e5large.npy` | Doc embeddings |
| `chunk_dense_cache_5666_full_e5.npy` | Chunk embeddings |

Models auto-download from HuggingFace on first use.

## Commands

```powershell
# UI
$env:PYTHONPATH='src'; streamlit run app.py

# CLI
$env:PYTHONPATH='src'; python scripts/preview_answer.py --query "votre question"

# Batch
python scripts/preview_answer.py --input data/interim/batch.jsonl --output data/reports/batch.json
python scripts/preview_answer.py --input ... --resume data/reports/batch.json

# Profile
$env:PYTHONPATH='src'; python scripts/profile.py

# Tests
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## Providers

| Provider | Env Key | Default Model |
|----------|---------|--------------|
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-v4-flash |
| OpenAI | `OPENAI_API_KEY` | gpt-4.1-mini |
| Anthropic | `ANTHROPIC_API_KEY` | claude-haiku-4-5 |
| Mistral | `MISTRAL_API_KEY` | mistral-small-4 |
| Google | `GEMINI_API_KEY` | gemini-3.1-flash |
| Groq | `GROQ_API_KEY` | llama-4-scout |
| Together | `TOGETHER_API_KEY` | Llama-4-Maverick |

API keys can be set via `.env.local` or entered in the sidebar (password field, never saved).

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full pipeline design, data flow, component reference

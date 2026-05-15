# BOFIP RAG — Assistant Fiscal 🇫🇷

RAG pipeline for French fiscal documents (BOFIP — *Bulletin Officiel des Finances Publiques*) with hybrid retrieval, cross-encoder reranking, LLM query rewriting, and coverage-aware generation.

[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Quick Start

```powershell
# Clone + setup
git clone https://github.com/Rapha1503/bofip-rag.git
cd bofip-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Set API key (any provider: DeepSeek, OpenAI, Anthropic, Mistral, Google, Groq, Together)
echo DEEPSEEK_API_KEY=sk-... > .env.local

# Place data files in data/interim/ (see Data Setup below)
# Then run the Streamlit UI:
$env:PYTHONPATH='src'; streamlit run app.py
```

## Data Setup

The pipeline needs BOFIP corpus files in `data/interim/`. These are NOT included in the repo (too large). You need:

| File | Size | Description |
|------|------|-------------|
| `raw_docs_sample_5666.jsonl` | ~300 MB | 5,666 parsed BOFIP documents |
| `chunks_section_window_sample_5666.jsonl` | ~200 MB | 66,289 section-window chunks |
| `doc_dense_cache_5666_sections_firstpara_e5large.npy` | ~23 MB | Document embeddings (E5-large) |
| `chunk_dense_cache_5666_full_e5.npy` | ~200 MB | Chunk embeddings (E5-base) |

**To generate from BOFIP source**: Set `RAW_BOFIP_ROOT` to your BOFIP documents directory and run the extraction pipeline (scripts in git history).

**Models** are downloaded automatically from HuggingFace on first use: `intfloat/multilingual-e5-large`, `intfloat/multilingual-e5-base`, `BAAI/bge-reranker-v2-m3` (~4 GB total).

## Architecture

```
user question
    │
    ▼ QUERY REWRITING (LLM: informal → legal vocabulary)
    │
    ▼ STAGE 1 — HYBRID DOC RETRIEVAL
    ├── BM25 (base, sections_leads, sections_leads_stem)
    ├── Dense embeddings (E5-large docs, E5-base chunks)
    ├── Dense-anchor filter — blocks convention spam
    └── Confidence-weighted RRF fusion → top-8 docs
    │
    ▼ STAGE 2 — CHUNK CANDIDATES
    └── Local BM25 inside top-8 docs → 64 candidates
    │
    ▼ RERANKER — bge-reranker-v2-m3 cross-encoder
    └── Section-path-aware scoring → top-8 chunks
    │
    ▼ LLM — Coverage-aware generation
    └── axes_requis/couverts/manquants
        → supported | partial | insufficient_evidence
        → cited bullets with gap analysis
```

## Providers

Supports 7 LLM providers via OpenAI-compatible API. Enter your API key in the Streamlit sidebar (password field, never saved).

| Provider | Key Env | Default Model |
|----------|---------|--------------|
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini |
| Anthropic | `ANTHROPIC_API_KEY` | claude-3-5-haiku |
| Mistral | `MISTRAL_API_KEY` | mistral-small-latest |
| Google | `GEMINI_API_KEY` | gemini-2.5-flash |
| Groq | `GROQ_API_KEY` | llama-4-scout |
| Together | `TOGETHER_API_KEY` | Llama-4-Maverick |

## Commands

```powershell
# Streamlit UI (multi-provider, single + batch)
$env:PYTHONPATH='src'; streamlit run app.py

# CLI single query
$env:PYTHONPATH='src'; python scripts/preview_answer.py --query "votre question"

# CLI batch with resume
python scripts/preview_answer.py --input data/interim/batch_final.jsonl --output data/reports/batch.json
python scripts/preview_answer.py --input data/interim/batch.jsonl --resume data/reports/batch.json

# Evaluation
python scripts/evaluate.py --runtime rag --limit 15

# Profile (per-stage timing)
$env:PYTHONPATH='src'; python scripts/profile.py

# Tests
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## Key Metrics

**Benchmark**: 15 realistic accountant questions → 12 correct, 2 partial, 1 honest, 0 wrong.

**Ablation** (15 queries, 4 configs): reranker improves MRR_pass from 0.22→0.26.

**102 unit tests** passing.

## Docs

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full pipeline design, component reference, data flow
- [ROADMAP.md](ROADMAP.md) — Phase history and delivered features
- [NEXT_CODEX_START_HERE.md](NEXT_CODEX_START_HERE.md) — Session handoff guide

## Known Limitations

- 47s cold start (runtime loads 4 models + builds BM25 indexes — cached after first run)
- ~30s/query latency (rewrite + retrieval + reranker + LLM)
- French-only (BOFIP is French tax doctrine)
- "dictionnaire en ligne" → "livres numériques" terminology gap

## Deployment

Run locally with GPU (RTX 3060+ recommended). For sharing, use HuggingFace Spaces (free, CPU-only — slower but functional).


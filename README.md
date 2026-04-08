# BOFIP-RAG — French Tax Law RAG Assistant

> Production-style Retrieval-Augmented Generation chatbot over French tax doctrine
> (BOFiP) and law (CGI / LPF). Hybrid retrieval, cross-encoder reranking,
> faithfulness guardrails, fully sourced answers.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-4A90E2.svg)](https://www.trychroma.com/)
[![LLM](https://img.shields.io/badge/LLM-Groq_Llama_3.3_70B-orange.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why this project

French tax professionals need fast, source-backed answers across large and frequently
updated legal corpora. Generic LLM chatbots hallucinate. This project bridges the gap
with a domain-specific RAG pipeline grounded in official sources.

- **86,045 chunks indexed** — 82,653 BOFiP commentary + 2,416 CGI articles + 976 LPF articles
- **100% on three benchmark suites** — 7Q regression / 8Q user / 10Q anti-overfitting
- **Hybrid retrieval** — BM25 + dense E5 embeddings + cross-encoder reranker
- **Faithfulness guardrail** — LLM verifier with heuristic fallback prevents hallucinations
- **Fully sourced** — every answer cites BOI / CGI / LPF references with URLs

---

## Architecture

```
                ┌─────────────────────────────────────────┐
                │     Public sources (open data)          │
                │   BOFiP stock + LEGI archive (DILA)     │
                └───────────────────┬─────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │  Ingestion pipeline (scripts/bootstrap.py) │
              │  • XML/HTML parser (BOFiP)                 │
              │  • LEGI tar parser with as-of versioning   │
              │  • Semantic chunker (1 fiscal rule = 1 chunk)
              └─────────────────────┬──────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
          ┌───────────────┐                  ┌───────────────┐
          │   ChromaDB    │                  │   BM25 index  │
          │  (E5 dense)   │                  │  (FR sparse)  │
          └───────┬───────┘                  └───────┬───────┘
                  └─────────────────┬─────────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │   Hybrid retrieval      │
                       │   + cross-encoder rerank│
                       │   + legal cross-refs    │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │  Groq Llama 3.3 70B     │
                       │  + faithfulness guardrail
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │  Streamlit chat (app.py)│
                       │  Answer + sources       │
                       └─────────────────────────┘
```

Full pipeline documentation: [docs/SYSTEM.md](docs/SYSTEM.md)

---

## Quick start (~10 minutes, sample mode)

```bash
# 1. Clone
git clone https://github.com/Rapha1503/bofip-rag.git
cd bofip-rag

# 2. Install dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Configure your Groq API key (free tier is enough)
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...

# 4. Bootstrap a 500-document sample (~10 min)
python scripts/bootstrap.py --sample 500

# 5. Launch the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and ask a fiscal question in French.

## Full setup (full BOFiP + CGI + LPF, several hours)

```bash
python scripts/bootstrap.py --full
```

This downloads the complete BOFiP open-data stock (~116 MB), the latest LEGI archive
from DILA, parses everything, embeds 86k+ chunks with multilingual E5, and builds both
the dense (ChromaDB) and sparse (BM25) indexes.

---

## Tech stack

| Component       | Tech                                                 |
|-----------------|------------------------------------------------------|
| LLM             | Groq Llama 3.3 70B (with 8B + Llama 4 Scout fallback)|
| Embeddings      | `intfloat/multilingual-e5-base` (768d)               |
| Vector DB       | ChromaDB (persistent, local)                         |
| Sparse retrieval| BM25 with custom French tokenizer                    |
| Reranker        | `mmarco-mMiniLMv2-L12-H384-v1` (multilingual)        |
| Frontend        | Streamlit                                            |
| Data sources    | BOFiP open data + LEGI freemium archive (DILA)       |

---

## Key engineering choices

- **Semantic chunking by fiscal rule, not by token window.** Each numbered BOFiP
  paragraph becomes one chunk so retrieval returns complete, atomic rules instead of
  half-sentences cut at arbitrary boundaries.
- **Hybrid retrieval with legal cross-references.** BM25 catches exact article
  citations, dense vectors handle paraphrases, and a cross-link injector pulls in the
  underlying CGI/LPF article whenever a BOFiP chunk references one.
- **Faithfulness guardrail.** A second LLM call verifies the answer is grounded in
  retrieved sources, with a heuristic fallback when the verifier itself fails. Forces
  abstention over hallucination on under-evidenced questions.
- **A/B benchmarked embeddings.** E5-base was chosen over `paraphrase-multilingual-MiniLM`
  through a recall benchmark on a 60-question validated retrieval set.
- **Reranker pool tuning.** Pool size of 30 was selected via a recall/precision/latency
  trade-off study (`scripts/tune_reranker_pool.py`).
- **Reproducible delta refresh.** `scripts/refresh_legi_daily.py` ingests daily DILA
  deltas and upserts only the changed legal articles, keeping BOFiP embeddings untouched.

---

## Project structure

```
bofip-rag/
├── app.py                          # Streamlit chat UI
├── config.py                       # Paths, models, retrieval params
├── scripts/
│   ├── bootstrap.py                # One-command full pipeline build
│   ├── reindex_semantic.py         # BOFiP semantic chunking + indexing
│   ├── process_legi_archive.py     # LEGI tar archive ingestion (CGI/LPF)
│   ├── refresh_legi_daily.py       # Daily delta refresh
│   ├── sync_legal_chunks.py        # Legal chunk sync into ChromaDB
│   ├── evaluate.py                 # E2E benchmark suite
│   ├── evaluate_retrieval.py       # Retrieval-only metrics
│   ├── tune_reranker_pool.py       # Reranker pool size sweep
│   └── benchmark_embeddings.py     # Embedding model A/B test
├── src/
│   ├── data_pipeline/              # Parsers + chunkers (BOFiP, LEGI, PDF)
│   ├── retrieval/                  # BM25, embeddings, vector store, hybrid, reranker
│   └── generation/                 # LLM client, prompts, faithfulness guardrail
└── docs/
    ├── SYSTEM.md                   # Full architecture & pipeline reference
    └── LESSONS.md                  # 29 documented mistakes & anti-patterns
```

---

## Documentation

- [docs/SYSTEM.md](docs/SYSTEM.md) — full architecture, pipeline, every file/function, prompts, benchmarks
- [docs/LESSONS.md](docs/LESSONS.md) — **29 documented mistakes** with context, anti-patterns, and rules
- [docs/CV_PROJECT_DOSSIER.md](docs/CV_PROJECT_DOSSIER.md) — interview-ready project dossier

---

## Disclaimer

Cet outil est une aide à la recherche fiscale. Il **ne remplace pas** l'avis d'un
expert-comptable ou d'un avocat fiscaliste. Les réponses doivent toujours être
vérifiées contre les sources officielles citées avant toute utilisation
professionnelle.

---

## License

MIT — see [LICENSE](LICENSE).

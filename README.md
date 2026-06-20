# BOFiP Agentic RAG

Full-corpus retrieval-augmented generation prototype for French BOFiP doctrine, created by **Rapha1503**.

The project explores how to answer tax questions from official BOFiP commentary with cited evidence, hybrid retrieval, dense embeddings, cross-encoder reranking, and coverage-aware generation.

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)]()

## Why This Project Exists

French tax doctrine is broad, dense, and citation-sensitive. A useful RAG system cannot rely on a single vector search over loose chunks: it needs structured parsing, legal metadata, robust retrieval under paraphrase, citation traceability, and honest abstention when evidence is missing.

This repository is designed as a portfolio-grade cleanroom version of that work.

## Pipeline

```text
BOFiP XML/HTML export
  -> structured RawDocument records
  -> section-aware chunks
  -> BM25 document indexes
  -> dense document and chunk indexes
  -> confidence-weighted RRF fusion
  -> local chunk retrieval
  -> cross-encoder reranking
  -> cited JSON answer with coverage status
```

Current runtime corpus:

| Artifact | Local path | Notes |
| --- | --- | --- |
| Raw documents | `data/interim/raw_docs_sample_5666.jsonl` | 5,666 BOFiP commentary documents |
| Chunks | `data/interim/chunks_section_window_sample_5666.jsonl` | 66,289 section-window chunks |
| Document embeddings | `data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy` | E5-large, shape `(5666, 1024)` |
| Chunk embeddings | `data/interim/chunk_dense_cache_5666_full_e5large.npy` | E5-large, shape `(66289, 1024)` |
| Evaluation queries | `data/interim/eval_queries_v1.jsonl` | 50 test questions |
| Passage gold | `data/interim/passage_gold_v3.jsonl` | passage-level labels where available |

Large corpus/model artifacts are intentionally not committed to Git. See [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Retrieval Stack

- Multi-view BM25 over document text, section leads, and stemmed section leads.
- Dense document retrieval with multilingual E5-large.
- Dense chunk retrieval for semantic anchors.
- Dense-anchor filtering to limit lexical false positives.
- Confidence-weighted reciprocal rank fusion.
- Local per-document chunk retrieval.
- Cross-encoder reranking with `BAAI/bge-reranker-v2-m3`.
- Diversity selection to prevent one document from dominating context.

The current app also performs query rewriting, multi-facet expansion, and computation-aware facet injection before retrieval. A Phase 2 cleanup will move that orchestration out of `app.py` into a shared runtime module.

## Quick Start

```powershell
git clone https://github.com/Rapha1503/bofip-agentic-rag.git
cd bofip-agentic-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env.local
```

Add one provider key to `.env.local`, then place the required BOFiP artifacts in `data/interim/`.

Run the Streamlit app:

```powershell
$env:PYTHONPATH='src'
streamlit run app.py
```

Run a CLI answer preview:

```powershell
$env:PYTHONPATH='src'
python scripts/preview_answer.py --query "Quel taux de TVA pour une pompe a chaleur ?"
```

Run retrieval evaluation:

```powershell
$env:PYTHONPATH='src'
python scripts/evaluate.py --runtime rag --device cpu --limit 5
```

Run unit tests:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

## LLM Providers

The Streamlit app currently exposes these providers:

| Provider | Env key |
| --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |

API keys can be loaded from `.env.local` or entered in the Streamlit sidebar. Keys must not be committed or logged.

## Full-Corpus Deployment Principle

The live demo should not use a reduced corpus. If a user asks about a BOFiP family removed from the demo, the RAG system becomes misleading.

Deployment optimization must preserve full corpus coverage:

- prebuilt data artifacts;
- memory-mapped or cached embeddings;
- startup preflight checks;
- optional reranker or cheaper reranker mode;
- explicit latency and freshness limits;
- clear BYOK warning for user-provided API keys.

GitHub Pages should host the static portfolio page. A Python host such as Hugging Face Spaces is the better fit for the Streamlit runtime. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Limitations

- Research prototype, not tax advice.
- Local corpus max publication date observed during audit: `2026-01-28`.
- Official BOFiP may contain newer publications.
- The runtime currently indexes the 5,666-document commentary corpus, not every BOFiP content type.
- Some table content is parsed but not yet first-class in chunk retrieval.
- Some BOI references are duplicated across different documents; Phase 2 will key retrieval by stable document identity.
- End-to-end answer grading is not yet as complete as retrieval grading.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data card](docs/DATA_CARD.md)
- [Deployment notes](docs/DEPLOYMENT.md)
- [Roadmap](docs/ROADMAP.md)

## Author

Created and maintained by **Rapha1503**.

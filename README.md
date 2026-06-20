# BOFiP Agentic RAG

Full-corpus retrieval-augmented generation prototype for French BOFiP doctrine, created by **Rapha1503**.

The project explores how to answer tax questions from official BOFiP commentary with cited evidence, hybrid retrieval, dense embeddings, cross-encoder reranking, and coverage-aware generation.

[![CI](https://github.com/Rapha1503/bofip-agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Rapha1503/bofip-agentic-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-architecture%20%7C%20data%20%7C%20demo-blueviolet)](docs/)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](docs/ROADMAP.md)

## At a Glance

| Topic | Status |
| --- | --- |
| Core idea | Full-corpus BOFiP RAG with cited answers |
| Corpus coverage | 5,666 BOFiP commentary documents, not a reduced demo corpus |
| Fresh clone | Code, tests, docs, and small eval files are included |
| Required local artifacts | Full BOFiP JSONL, embedding caches, and local E5 model directory |
| Demo path | Local Streamlit now; hosted full-corpus demo planned |
| Current proof | Unit tests + setup checker + documented retrieval evaluation harness |
| Main caveat | Research prototype, not tax advice |

Example question:

```text
Quel taux de TVA pour la pose d'une pompe a chaleur chez un particulier ?
```

Expected answer shape: a JSON-backed response rendered in Streamlit with `supported`, `partial`, or `insufficient_evidence` status, cited BOFiP chunks, and a technical retrieval trace.

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

Large corpus/model artifacts are intentionally not committed to Git. See [docs/DATA_CARD.md](docs/DATA_CARD.md) and [docs/full_corpus_manifest.json](docs/full_corpus_manifest.json).

## Retrieval Stack

- Multi-view BM25 over document text, section leads, and stemmed section leads.
- Dense document retrieval with multilingual E5-large.
- Dense chunk retrieval for semantic anchors.
- Dense-anchor filtering to limit lexical false positives.
- Confidence-weighted reciprocal rank fusion.
- Local per-document chunk retrieval.
- Optional cross-encoder reranking with `BAAI/bge-reranker-v2-m3`.
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

For the Streamlit app, add one provider key to `.env.local`, then place the required BOFiP artifacts in `data/interim/`. A fresh public clone can run code checks and unit tests, but the full app needs the large local artifact bundle described in the manifest.

Check local artifact readiness:

```powershell
python scripts/check_setup.py
python scripts/check_setup.py --deep
```

The reranker is an optional quality layer. If `data/models/BAAI--bge-reranker-v2-m3/` is absent, the Streamlit app can still run full-corpus retrieval with the reranker disabled.

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

The CLI preview currently uses DeepSeek and requires `DEEPSEEK_API_KEY`. The Streamlit app exposes a BYOK interface for OpenAI-compatible providers, with an editable model ID field.

Run retrieval evaluation:

```powershell
$env:PYTHONPATH='src'
python scripts/evaluate.py --runtime rag --device cpu --limit 5
```

Run component ablations:

```powershell
$env:PYTHONPATH='src'
python scripts/ablation.py --device cpu --limit 15
```

Run unit tests:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

## LLM Providers

The Streamlit app currently exposes OpenAI-compatible endpoints for these providers:

| Provider | Env key |
| --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |

API keys can be loaded from `.env.local` or entered in the Streamlit sidebar. Model IDs are editable because provider model names evolve. Keys must not be committed or logged.

## Evaluation

The repository includes the reusable evaluation harness and the 50-query eval set:

- `data/interim/eval_queries_v1.jsonl`
- `data/interim/passage_gold_v3.jsonl`
- `scripts/evaluate.py`
- `scripts/ablation.py`

Tracked public metrics are still pending because the full local artifact bundle is not committed. The next portfolio milestone is a small, versioned evaluation report tied to a corpus manifest.

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
- [Local demo guide](docs/DEMO.md)
- [Deployment notes](docs/DEPLOYMENT.md)
- [Full-corpus manifest](docs/full_corpus_manifest.json)
- [References](docs/REFERENCES.md)
- [Roadmap](docs/ROADMAP.md)

## Author

Created and maintained by **Rapha1503**.

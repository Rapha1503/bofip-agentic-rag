# BOFiP Agentic RAG

Full-corpus RAG prototype for French BOFiP tax doctrine, created by **Raphael Ifergan**.

The project combines hybrid retrieval with a controlled agentic loop: domain classification, BOFiP retrieval, structured answer generation, coverage self-evaluation, targeted reformulation when evidence is missing, and cited output with explicit limits.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-50%20passing-brightgreen)](tests/)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](docs/ROADMAP.md)
[![Demo](https://img.shields.io/badge/demo-Hugging%20Face-yellow)](https://rapha1503-bofip-agentic-rag.hf.space/)

## What It Does

```text
Question utilisateur
  -> classification domaine BOFiP
  -> retrieval hybride BM25 + E5 + fusion RRF
  -> réponse JSON sourcée + auto-évaluation des axes couverts
  -> si preuve insuffisante: reformulation cibl?e + second retrieval
  -> réponse finale avec sources et limites
```

The live app uses the full commentary corpus. No reduced demo corpus is used.

| Layer | Current state |
| --- | --- |
| Corpus | 5,666 BOFiP commentary documents observed through `2026-01-28` |
| Index | 66,289 section-window passages |
| Retrieval | BM25 variants, E5-large dense retrieval, confidence-weighted RRF, per-document chunk selection |
| Agent | `AgenticRAG`: classify, retrieve, answer, self-evaluate, reformulate, retry |
| Output | `supported`, `partial`, or `insufficient_evidence` with visible BOFiP sources |
| Hosting | Streamlit BYOK app on Hugging Face Spaces |

## Quick Start

```powershell
git clone https://github.com/Rapha1503/bofip-agentic-rag.git
cd bofip-agentic-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env.local
```

Add at least one provider key to `.env.local`, for example:

```text
DEEPSEEK_API_KEY=sk-...
```

Download or place the full-corpus runtime artifacts:

```powershell
python scripts/download_artifacts.py
python scripts/check_setup.py --deep --skip-models
```

Run the app:

```powershell
streamlit run app.py
```

Run tests:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

## Runtime Artifacts

Large artifacts are intentionally not committed to Git. They are tracked by `docs/full_corpus_manifest.json` and can be downloaded from the project release.

| Artifact | Path |
| --- | --- |
| Raw documents | `data/interim/raw_docs_sample_5666.jsonl` |
| Chunks | `data/interim/chunks_section_window_sample_5666.jsonl` |
| Document embeddings | `data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy` |
| Chunk embeddings | `data/interim/chunk_dense_cache_5666_full_e5large.npy` |

Small evaluation files are versioned:

- `data/interim/eval_queries_v1.jsonl`
- `data/interim/passage_gold_v3.jsonl`

## LLM Providers

The Streamlit UI exposes provider and model dropdowns. Current BYOK defaults:

| Provider | Env key | Default model |
| --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |
| OpenAI | `OPENAI_API_KEY` | `gpt-5.4-mini` |
| Mistral | `MISTRAL_API_KEY` | `mistral-small-latest` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-3.5-flash` |

The hosted demo keeps the reranker off by default for free CPU hosting. The agentic loop remains active.

## Project Structure

```text
src/bofip_agentic/
  agent_rag.py              Agent loop and trace
  rag_runtime.py            Hybrid retrieval runtime
  prompt_utils.py           Citation and coverage prompt
  providers.py              BYOK provider/model config
  artifact_download.py      Release artifact downloader
  lexical_retrieval.py      BM25 and French tokenization
  dense_retrieval.py        E5 embedding search
  direct_chunk_retrieval.py Stage-2 chunk retrieval
  reranker.py               Optional cross-encoder reranker

scripts/
  setup.py                  Build corpus from source
  sync.py                   Refresh corpus safely
  eval_full.py              50-query agentic evaluation
  eval_agent.py             Agent benchmark helper
  check_setup.py            Artifact preflight
  download_artifacts.py     Download runtime artifacts
```

## Documentation

- [Agentic architecture](docs/AGENTIC.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Data card](docs/DATA_CARD.md)
- [Demo guide](docs/DEMO.md)
- [Deployment notes](docs/DEPLOYMENT.md)
- [Evaluation results](docs/RESULTS.md)
- [Roadmap](docs/ROADMAP.md)

## Limits

This is a research prototype, not tax advice. BOFiP may contain newer publications than the indexed corpus. The app surfaces cited passages and limits so the user can inspect evidence before relying on the answer.

## Author

Created and maintained by **Raphael Ifergan**.

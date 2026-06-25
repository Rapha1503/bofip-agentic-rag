# BOFiP Agentic RAG

Agentic RAG prototype for querying the French BOFiP tax doctrine with cited sources, source review, and full-corpus retrieval.

Built by **Raphael Ifergan**.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-7A1832)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Benchmark-45%2F50%20correct-brightgreen)](docs/evaluation/official50_portfolio_final_45_2026_06_25.md)
[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://rapha1503-bofip-agentic-rag.hf.space/)

## Live Demo

Try the hosted demo on Hugging Face Spaces:

**https://rapha1503-bofip-agentic-rag.hf.space/**

The app uses a BYOK model: bring your own provider API key, enter it in the interface, and query the BOFiP corpus. No API key is committed to this repository.

## Why This Project

French tax doctrine is dense, fragmented, and sensitive to wording. A generic chatbot can give a plausible answer while missing the relevant BOFiP section, confusing nearby tax regimes, or ignoring exceptions.

This project explores a more controlled workflow:

```text
User question
  -> fiscal planning
  -> retrieval by tax axis
  -> source review
  -> targeted relaunch if evidence is weak
  -> sourced answer with explicit limits
```

The goal is not to replace tax advice. The goal is to make BOFiP retrieval and answer grounding more transparent.

## What It Does

BOFiP Agentic RAG indexes the BOFiP doctrine corpus and answers French fiscal questions with:

- full-corpus BOFiP retrieval;
- cited source passages;
- fiscal-axis planning;
- source criticism before final answer;
- targeted relaunch when evidence is missing;
- visible limitations when the answer is uncertain.

Current corpus:

| Layer | Count |
|---|---:|
| BOFiP source rows | 9,048 |
| Section-window passages | 79,160 |
| Embedding dimension | 1,024 |
| Corpus mode | Full corpus, no reduced demo |

## Benchmark

Latest portfolio evaluation:

| Metric | Result |
|---|---:|
| Questions | 50 |
| Correct answers | **45 / 50** |
| Failures kept visible | 5 / 50 |
| Runtime errors | 0 |
| Average runtime | 174.2s / question |

Reports:

- [Markdown report](docs/evaluation/official50_portfolio_final_45_2026_06_25.md)
- [HTML report](docs/evaluation/official50_portfolio_final_45_2026_06_25.html)
- [CSV report](docs/evaluation/official50_portfolio_final_45_2026_06_25.csv)

The benchmark sends only the user question to the runtime. Expected answers and BOFiP references are used only after generation for evaluation.

## Architecture

```text
BOFiP public data
  -> full-corpus parsing and section chunking
  -> retrieval over the complete local corpus
  -> agentic source review and targeted relaunch
  -> sourced answer with visible limits
  -> Streamlit BYOK interface
```

Core modules:

| Module | Role |
|---|---|
| `agent_rag.py` | fiscal planner, source review, relaunch, final answer |
| `rag_runtime.py` | retrieval runtime |
| `lexical_retrieval.py` | BM25 and French tokenization |
| `dense_retrieval.py` | optional E5 embeddings |
| `direct_chunk_retrieval.py` | local section/chunk search |
| `eval_runner.py` | benchmark runner and report generation |
| `app.py` | Streamlit interface |

## Design Trade-Offs

The current public version prioritizes **source traceability** over raw latency.

Instead of returning the first plausible answer, the pipeline plans the fiscal question, retrieves BOFiP passages, reviews source coverage, and can relaunch a targeted search before generation. This makes the demo slower than a one-pass RAG, but it keeps the answer auditable and preserves full-corpus coverage.

## Supported Providers

The app is provider-agnostic and uses BYOK configuration.

| Provider | Environment variable | Notes |
|---|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` | Used for the published benchmark |
| OpenAI | `OPENAI_API_KEY` | Supported through provider config |
| Mistral | `MISTRAL_API_KEY` | Supported through provider config |
| Google Gemini | `GEMINI_API_KEY` | Supported through provider config |

For the hosted Hugging Face demo, enter the provider key directly in the UI.

## Quick Start

```powershell
git clone https://github.com/Rapha1503/bofip-agentic-rag.git
cd bofip-agentic-rag

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env.local
```

Add at least one provider key to `.env.local`:

```text
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
MISTRAL_API_KEY=
GEMINI_API_KEY=
```

Download runtime artifacts:

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
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

## Runtime Artifacts

Large full-corpus artifacts are not committed to Git.

Expected files:

```text
data/interim/raw_docs.jsonl
data/interim/chunks.jsonl
data/interim/doc_dense_cache.npy
data/interim/chunk_dense_cache.npy
```

They are tracked through `docs/full_corpus_manifest.json` and can be downloaded from the release artifacts.

## Hugging Face Deployment

The public demo is designed for Hugging Face Spaces. The Space loads full-corpus artifacts at startup and exposes provider/model selection in the UI.

Deployment principles:

- no model API key is hardcoded in the repository;
- reranking stays off by default on free CPU hosting;
- prompt/debug views stay hidden unless explicitly enabled;
- the app keeps full-corpus coverage instead of shipping a reduced demo corpus.

## Limitations

This is a research prototype, not tax advice.

Known limitations:

- the traceability-first runtime is slower than a one-pass RAG;
- some narrow BOFiP branches still fail retrieval;
- source review improves traceability but adds LLM calls;
- BOFiP updates require artifact refresh.

## Roadmap

- Improve source-review latency.
- Add a faster cited-answer path for interactive demos.
- Expand evaluation with more cross-domain fiscal cases.
- Add deployment health checks for Hugging Face.

## Author

Created and maintained by **Raphael Ifergan**.

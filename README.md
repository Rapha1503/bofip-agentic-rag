# BOFIP Agentic RAG 🇫🇷

**Self-evaluating retrieval agent for French tax doctrine — 96% question coverage, 7s latency.**

[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen)](https://github.com/Rapha1503/bofip-agentic-rag)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Problem

French tax accountants navigate **5,666 BOFIP doctrinal documents** from the Direction Generale des Finances Publiques. Most search tools use keyword matching — useless when the BOFIP terminology ("redevable de la TVA", "cession de titres de participation") doesn't match how accountants actually ask questions ("dois-je facturer la TVA?", "je vends ma societe").

Standard RAG pipelines fail on this vocabulary gap. The user gets surface-level results about *taxe sur les vehicules* instead of *deduction TVA vehicules*.

## Solution

A **self-evaluating Agentic RAG** that iterates until it has sufficient evidence:

```
User question (natural language)
  → Retrieve (BM25 + Dense E5-large + Cross-encoder reranker)
  → Answer + Self-evaluate (LLM reports: supported / partial / insufficient)
  → IF partial: Reformulate missing axes → Retrieve again → Merge → Final answer
  → Full audit trace (retrieval queries, docs found, coverage decisions)
```

## Architecture

```
┌─────────────┐    ┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│  User query │ →  │ Agent         │ →  │ Hybrid         │ →  │ Cross-encoder │
│  (natural)  │    │ Planner       │    │ Retrieval      │    │ Reranker      │
└─────────────┘    │ (LLM, 1 call) │    │ BM25 + Dense   │    │ bge-reranker  │
                   └───────┬───────┘    │ E5-large 1024d │    │ v2-m3         │
                           │            └───────┬────────┘    └───────┬───────┘
                   ┌───────▼───────┐            │                     │
                   │ Self-evaluate │ ◄─────────┘                     │
                   │ (same LLM     │                                  │
                   │  call)        │           ┌──────────────────────┘
                   │ • supported?  │           │
                   │ • axes req?   │ ◄─────────┘
                   │ • missing?    │
                   └───┬───────────┘
                       │
              ┌────────▼────────┐
              │ IF partial:     │
              │ Reformulate →   │──→ retrieve again → merge → final answer
              │ new search      │
              └─────────────────┘
```

**LLM calls:** 1 (first pass sufficient) or 2 (reformulation).  
**Cost per query:** ~$0.003 with DeepSeek V4 Flash.

## Key Metrics

| Metric | Value |
|---|---|
| **Answer quality** | **96%** supported (48/50) |
| Partial (honest) | 4% (2/50) |
| Insufficient evidence | 0% |
| Avg coverage (self-reported) | **96%** |
| Avg latency (GPU) | **7s** |
| Avg iterations | 1.2 |
| VRAM (GPU) | 3.4 GB / 6 GB |

Evaluated on 50 realistic French tax questions across 10 themes (TVA, BIC, CF, IS, IR, ENR, IF, PAT, Sanctions, Mixte). Full results in [`docs/RESULTS.md`](docs/RESULTS.md).

## Quick Start

### Prerequisites

- Python 3.10+ with CUDA 12.x (GPU recommended, CPU supported)
- NVIDIA GPU with ≥6 GB VRAM (RTX 3060 tested)
- BOFIP corpus files (download from [BOFIP](https://bofip.impots.gouv.fr))

### Setup

```powershell
# Clone
git clone https://github.com/Rapha1503/bofip-agentic-rag.git
cd bofip-agentic-rag

# Install
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Copy BOFIP corpus to data/interim/ (from your local BOFIP download)
# Required files:
#   data/interim/raw_docs_sample_5666.jsonl
#   data/interim/chunks_section_window_sample_5666.jsonl
#   data/interim/doc_dense_cache_5666_sections_firstpara_e5large.npy
#   data/interim/chunk_dense_cache_5666_full_e5large.npy

# Copy or download models to data/models/
#   intfloat/multilingual-e5-large (~2.1 GB)
#   BAAI/bge-reranker-v2-m3 (~2.2 GB)
# Or use HuggingFace: the pipeline auto-downloads on first run
```

### Run evaluation

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="sk-..."

# Quick 3-query benchmark
python scripts/benchmark_agentic.py

# Full 50-query evaluation
python scripts/eval_agent.py

# Resume if interrupted
python scripts/eval_agent.py --resume
```

## Commands

| Command | Description |
|---|---|
| `python scripts/eval_agent.py` | 50-query evaluation (incremental save, supports --resume) |
| `python scripts/eval_agent.py --limit 10` | Run first 10 queries only |
| `python scripts/benchmark_agentic.py` | 3-query comparison: baseline vs agent |
| `pytest tests/ -q` | Run test suite |

## Files

| Path | Role |
|---|---|
| `src/bofip_agentic/agent_rag.py` | Agent orchestration (plan → retrieve → evaluate → reformulate → answer) |
| `src/bofip_agentic/rag_runtime.py` | Hybrid retrieval (BM25 + Dense E5-large + Reranker) |
| `src/bofip_agentic/prompt_utils.py` | LLM prompt builder |
| `src/bofip_agentic/dense_retrieval.py` | Semantic embedding retrieval |
| `src/bofip_agentic/lexical_retrieval.py` | BM25 keyword retrieval |
| `src/bofip_agentic/reranker.py` | Cross-encoder reranker (bge-reranker-v2-m3) |
| `src/bofip_agentic/hybrid_retrieval.py` | Reciprocal-rank fusion |
| `scripts/eval_agent.py` | Evaluation harness |
| `data/eval/tax_eval_50.jsonl` | 50-question evaluation benchmark |

## Design Decisions

| Decision | Rationale |
|---|---|
| **Self-evaluating loop** instead of separate judge LLM | Saves 1 LLM call per query. The agent's `build_prompt()` already returns coverage info. |
| **Pragmatic coverage filter** | LLMs tend to nitpick (want exact BOFIP references, edge cases). A regex filter catches non-substantive missing axes. |
| **Hybrid retrieval** (BM25 + Dense) | Dense alone misses terminology gaps; BM25 alone misses semantic matches. Fusion with RRF gives best of both. |
| **Single E5-large encoder** for doc + chunk | Saves 1 GB VRAM vs separate E5-base for chunks. 1024-dim embeddings provide better semantic matching for French legal text. |
| **fp16 precision** | Reduces VRAM from 6.9 GB (overflow) to 3.4 GB. Enables reliable GPU inference on consumer hardware. |
| **Max 2 iterations** | Second pass catches most retrieval gaps. Third iteration has diminishing returns for this corpus size. |

## Limitations

| Issue | Mitigation |
|---|---|
| Requires pre-built BOFIP corpus (not included) | Data download instructions in setup |
| GPU required for sub-10s latency | CPU mode available (~30s/query) |
| French-only | Multilingual embedding model could handle other languages |
| Cross-category questions (Mixte) weakest (85%) | GraphRAG with document cross-references planned |
| No streaming or chat history | Single-question mode; chat context could improve reformulation |

## Roadmap

- [ ] **GraphRAG integration** — use BOFIP `relations` + `internal_links` for document graph expansion
- [ ] **Streamlit UI** — interactive agent with trace visualization  
- [ ] **Multi-provider support** — pluggable LLM backends (Anthropic, OpenAI, Mistral)
- [ ] **Chat history** — context-aware reformulation for multi-turn conversations
- [ ] **Fine-tuned embeddings** — domain-specific e5 model for French fiscal text

## License

MIT © 2026

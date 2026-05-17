# BOFIP Agentic RAG

**Self-evaluating retrieval agent for French tax doctrine.**

Answers accountant-style French tax questions through a controlled agentic workflow: BOFIP domain classification, hybrid retrieval, cross-encoder reranking, coverage-aware answer generation, and targeted reformulation when evidence is insufficient. Most queries complete in 2 LLM calls; harder cases trigger an additional reformulation pass (2-4 calls total).

```
User question (natural French)
  → Classify domain (LLM: BOFIP family + sub-family prefix)
  → Retrieve (BM25 + Dense E5-large + Taxonomy ranking + bge-reranker-v2-m3)
  → Answer + Self-evaluate (LLM: supported / partial / insufficient)
  → IF partial: Reformulate (LLM: generate technical BOFIP search query)
  → Retrieve again → Merge → Final answer
```

**Cost:** ~$0.003/query. **VRAM:** 3.4 GB (fp16, RTX 3060+). **Tests:** 47/47 passing. **Latency:** p50 14s, p95 47s (research assistant, not instant chat).

---

## Quick Start

### Prerequisites

- Python 3.11+, CUDA 12.x (GPU recommended, CPU works)
- NVIDIA GPU ≥ 6 GB VRAM
- API key for any supported LLM provider (DeepSeek, OpenAI, Anthropic, Mistral, Google)

### 1. Install

```powershell
git clone https://github.com/Rapha1503/bofip-agentic-rag.git
cd bofip-agentic-rag
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the corpus

**Option A — Build from scratch (~20 min)**

```powershell
python scripts/setup.py
```

Downloads BOFIP from `data.economie.gouv.fr`, parses HTML, chunks, embeds.

**Option B — Copy from another machine**

Place these files in `data/interim/`:

| File | Size |
|---|---|
| `raw_docs_sample_5666.jsonl` | ~208 MB |
| `chunks_section_window_sample_5666.jsonl` | ~131 MB |
| `doc_dense_cache_5666_sections_firstpara_e5large.npy` | ~23 MB |
| `chunk_dense_cache_5666_full_e5large.npy` | ~272 MB |

And models in `data/models/`:

| Directory | Size |
|---|---|
| `intfloat--multilingual-e5-large/` | ~12 GB |
| `BAAI--bge-m3/` | ~2 GB |

Or `python scripts/setup.py --copy-from <source-project>`.

### 3. Set your API key

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```
Or create `.env.local`:
```
DEEPSEEK_API_KEY=sk-...
```

### 4. Run

```powershell
streamlit run app.py        # no PYTHONPATH needed
```

Or double-click `run.bat` on Windows.

### Commands (no PYTHONPATH required)

| Command | What it does |
|---|---|
| `streamlit run app.py` | Web UI |
| `python scripts/eval_full.py` | 50-query evaluation |
| `python scripts/eval_full.py --limit 3` | Pilot eval |
| `python scripts/eval_full.py --resume` | Resume interrupted |
| `python scripts/sync.py` | Update corpus |
| `python scripts/sync.py --check` | Preview changes |
| `python scripts/setup.py` | First-time corpus build |
| `pytest tests/ -v` | Run tests |

All scripts auto-inject `src/` into the path. No `$env:PYTHONPATH` needed.

---

## Keeping the corpus up to date

BOFIP is updated weekly by the DGFIP. `scripts/sync.py` handles the full lifecycle:

```powershell
python scripts/sync.py              # Full sync
python scripts/sync.py --check      # Preview changes
python scripts/sync.py --force      # Skip confirmation
```

**How it works**

1. Downloads all BOFIP records (~9000, ~30s)
2. Content-level diff against last snapshot (hash-based, not date-based)
3. Reports exactly which documents changed: `+N new / ~N updated / -N removed`
4. Builds fresh corpus in `data/interim_tmp/` — never touches the live corpus
5. Validates integrity (≥5000 docs, NPY shapes match, embedding dim = 1024)
6. Backs up current corpus to `data/backup_YYYYMMDD_HHMMSS/`
7. Atomically swaps new corpus in place
8. Clears BM25 cache (rebuilds on next query, ~3s)

**Rollback:** `copy data/backup_*/ data/interim/`

**Rebuild from scratch:** `python scripts/setup.py`

---

## Evaluation Results

50 French tax questions, 10 themes, 3 difficulties. DeepSeek V4 Flash, GPU.
Results are pipeline self-evaluated — not audited tax accuracy.

| Metric | Value |
|---|---|
| **Self-reported supported** | **45/50 (90%)** |
| Partial | 4/50 (8%) |
| Insufficient evidence | 1/50 (2%) |
| Self-reported avg coverage | 97.2% |
| Avg iterations | 1.3 |
| Reformulated | 17/50 (34%) |
| Latency p50 | 14.4s |
| Latency p95 | 47.4s |
| Doc family recall@5 | 52% |
| Hallucination check (must_not_include) | 98.9% |

| Theme | Queries | Supported | Coverage |
|---|---:|---:|---:|
| BIC | 8 | 100% | 100% |
| CF | 6 | 100% | 100% |
| IS | 5 | 100% | 93% |
| TVA | 10 | 90% | 98% |
| Sanctions | 6 | 100% | 83% |
| IR | 5 | 60% | 120% |
| Mixte | 6 | 67% | 86% |

<details>
<summary>Full per-theme, per-difficulty breakdown →</summary>

| Difficulty | Queries | Supported | Coverage | Avg time |
|---|---:|---:|---:|---:|
| Easy | 18 | 94% | 100% | 16s |
| Medium | 20 | 85% | 97% | 21s |
| Hard | 12 | 92% | 93% | 23s |

| Type | Queries | Supported | Coverage |
|---|---:|---:|---:|
| Direct | 17 | 94% | 100% |
| Nuanced | 13 | 92% | 108% |
| Procedure | 8 | 88% | 97% |
| Calculation | 6 | 100% | 78% |
| Multi-source | 6 | 67% | 86% |
</details>

Full results: `docs/RESULTS.md`. Architecture: `docs/AGENTIC.md`.

---

## Supported LLM Providers

| Provider | Models | Env Key |
|---|---|---|
| **DeepSeek** | deepseek-v4-flash, deepseek-v4-pro | `DEEPSEEK_API_KEY` |
| **OpenAI** | gpt-5.5, gpt-5.4-mini, gpt-4.1 | `OPENAI_API_KEY` |
| **Anthropic** | claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7 | `ANTHROPIC_API_KEY` |
| **Mistral** | mistral-small-4, mistral-large-3 | `MISTRAL_API_KEY` |
| **Google** | gemini-3.1-flash, gemini-3.1-pro | `GEMINI_API_KEY` |

All providers use the OpenAI-compatible API. Configure in the Streamlit sidebar.

---

## Data Sources

| Source | Type | Via | Status |
|---|---|---|---|
| **BOFIP** | Doctrinal commentary | `data.economie.gouv.fr` open API | ✅ Synced |
| **CGI** (Code Général des Impôts) | Tax code | Legifrance / PISTE (`piste.gouv.fr`) | ❌ Requires registration |
| **LPF** (Livre des Procédures Fiscales) | Procedural law | Legifrance / PISTE | ❌ Requires registration |

BOFIP is the tax authority's doctrinal interpretation. CGI and LPF are the actual laws. BOFIP cites CGI/LPF articles inline (`legal_refs` field). For full CGI/LPF text, integrate the Legifrance API via PISTE.

**Adding a new data source:** implement `parse_<source>() → list[dict]` (RawDocument format), call from `sync.py`. The chunk → embed → swap pipeline is data-agnostic.

---

## Project Structure

```
src/bofip_agentic/
  agent_rag.py                 Agent loop: classify → retrieve → answer → evaluate → reformulate
  rag_runtime.py               Hybrid retrieval (BM25 + Dense + Taxonomy + RRF + Reranker)
  prompt_utils.py              Prompt builder: number extraction, computation forcing, coverage schema
  models.py                    Data classes: RawDocument, ChunkNode, RawSection, etc.
  settings.py                  Paths, env config, data directories
  ── Retrieval ──
  lexical_retrieval.py         BM25 with French Snowball stemming (unified LexicalIndex)
  dense_retrieval.py           E5-large semantic embeddings (1024-dim, fp16)
  hybrid_retrieval.py          Confidence-weighted RRF fusion + taxonomy ranking
  reranker.py                  Cross-encoder reranker (bge-reranker-v2-m3, fp16)
  direct_chunk_retrieval.py    Stage-2 per-document chunk selection
  ── Data pipeline ──
  chunking.py                  3 strategies: section_window, paragraph_preserving, parent_child
  xml_parser.py                BOFIP XML metadata parser
  html_parser.py               BOFIP HTML structure parser (sections, paragraphs, tables, links)
  document_builder.py          RawDocument assembler (XML + HTML → structured document)
  discovery.py                 Filesystem discovery of BOFIP document pairs
  sampling.py                  Random + stratified document sampling
  versioning.py                Pipeline version tracking + file fingerprinting
  ── Utilities ──
  eval_harness.py              IR metrics: Hit@K, MRR, NDCG
  jsonio.py                    JSON/JSONL read/write
  text_utils.py                Whitespace normalization, legal ref extraction, accent stripping
  env_utils.py                 .env.local loader

scripts/
  setup.py                     First-time corpus builder
  sync.py                      Corpus updater (download + diff + rebuild + atomic swap)
  eval_full.py                 Comprehensive evaluation harness (50 queries)
  eval_agent.py                Legacy agent evaluation

tests/                         47 unit/integration tests
data/
  interim/                     Corpus: JSONL documents + chunks, NPY embeddings
  models/                      E5-large + bge-reranker-v2-m3 (auto-downloaded by HuggingFace)
  eval/                        Evaluation datasets (tax_eval_50.jsonl)
  reports/                     Evaluation reports
  raw/                         Raw API downloads + sync snapshots
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **LLM-based domain classification** | No hardcoded keyword banks. LLM maps question to BOFIP prefix (e.g., `RPPM-PVBMI-20-10-40`). Designed to generalize across tax domains through LLM-based family classification. |
| **Taxonomy-aware retrieval** | Domain prefix boosts matching documents + bypasses dense-anchor filter. Prevents domain mismatch (BIC docs for individual questions). |
| **Number extraction + computation** | Generic regex finds numeric values in any question, injects them into prompt. Forces LLM to produce step-by-step calculation. |
| **Self-evaluating loop** | Coverage analysis embedded in the answer prompt. Saves 1 LLM call vs separate judge. |
| **Pragmatic coverage filter** | LLMs nitpick about missing reference numbers. Regex filters non-substantive concerns before deciding status. |
| **BM25 + Dense hybrid** | BM25 catches terminology gaps ("compte titre" vs "portefeuille-titres"), E5 catches semantics. RRF fusion combines both. |
| **Single E5-large** (doc + chunk) | Saves 1 GB VRAM vs separate E5-base. 1024-dim better for French legal text. |
| **fp16 everywhere** | 6.9 GB → 3.4 GB VRAM. Runs on consumer GPUs (RTX 3060 tested). |
| **Max 2 iterations** | Second pass catches most retrieval gaps. Third has diminishing returns. |

---

## Troubleshooting

**"No module named bofip_agentic"**
→ Set `$env:PYTHONPATH = "src"` before running any script.

**App crashes on start: "raw_docs_sample_5666.jsonl not found"**
→ Run `python scripts/setup.py` first to build the corpus.

**GPU out of memory**
→ Run with `--device cpu`: `python scripts/eval_full.py --device cpu` (slower but works).

**API key not found**
→ Set `$env:DEEPSEEK_API_KEY` or create `.env.local` with the key. Or use the sidebar in the Streamlit app.

**Models not downloading**
→ HuggingFace auto-downloads on first run (~4 GB compressed). Requires internet. If blocked, pre-download models to `data/models/`.

---

## Roadmap

- [ ] **GraphRAG** — expand retrieval along `relations` edges for cross-domain queries (Mixte: 67% → target 85%)
- [ ] **CGI/LPF integration** — Legifrance API via PISTE for full legal text
- [ ] **Incremental embedding** — speed up sync by reusing unchanged document embeddings
- [ ] **Multi-turn chat** — conversation history for follow-up questions
- [ ] **Answer grounding score** — separate metric for how well the answer applies rules to the specific case

## License

MIT

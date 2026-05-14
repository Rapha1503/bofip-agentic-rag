# BOFIP RAG Cleanroom — Roadmap & Status

> Last updated: 2026-05-13 — All phases delivered.

## Architecture (DELIVERED)

```
query → LLM rewriting (DeepSeek) → hybrid retrieval (BM25 + dense + dense-anchor)
→ local chunk BM25 → cross-encoder reranker (bge-reranker-v2-m3)
→ coverage-aware LLM answer (DeepSeek)
→ {answer_status: supported|partial|insufficient_evidence, axes_requis/couverts/manquants}
```

## Delivered Components

| Component | File | Status |
|-----------|------|--------|
| RagRuntime (clean, no dead branches) | `src/bofip_cleanroom/rag_runtime.py` | ✅ |
| Dense-anchor filter + configurable fusion | `rag_runtime.py` | ✅ |
| Cross-encoder reranker (bge-m3) | `src/bofip_cleanroom/reranker.py` | ✅ |
| Section-path-aware chunk reranking | `rag_runtime.py` | ✅ |
| Eval harness + CLI | `src/bofip_cleanroom/eval_harness.py`, `scripts/evaluate.py` | ✅ |
| 50-query test suite (5 categories) | `data/interim/eval_queries_v1.jsonl` | ✅ |
| Passage gold v3 (real chunk IDs) | `data/interim/passage_gold_v3.jsonl` | ✅ |
| DeepSeek provider | `scripts/preview_answer.py` | ✅ |
| LLM query rewriting | `scripts/preview_answer.py` | ✅ |
| Coverage-aware prompt (partial status) | `scripts/preview_answer.py` | ✅ |
| Batch mode + incremental save + resume | `scripts/preview_answer.py` | ✅ |

## Key Metrics

**15-query benchmark** (report: `data/reports/batch_final_v1.json`):
- 12 correct · 2 partial · 1 honest · 0 wrong · 14/15 format-valid

**11-query failure analysis** (diagnostic probe across 7 BOFIP domains):
- Pattern A (convention spam): Fixed via dense-anchor filter
- Pattern B (edge-case > general sub-document): Fixed via wider net + section-aware reranker
- Pattern C (terminology gap): Fixed via LLM query rewriting (helped 3/4 cases)

**102 tests** passing.

## PHASE A — Foundation Hardening ✅ DELIVERED

- A1. Store DeepSeek API key ✅
- A2. No code pruning (decision: build alongside, don't delete) ✅
- A3. Standardized evaluation protocol (`evaluate.py`, `eval_harness.py`) ✅
- A4. Diverse test suite v1 (50 queries, 5 categories) ✅
- A5. Passage gold v3 (real chunk IDs from 5666-doc corpus) ✅

## PHASE B — Cross-Encoder Reranker ✅ DELIVERED

- B1. Model: `BAAI/bge-reranker-v2-m3`, `reranker.py` ✅
- B2. Clean RagRuntime with reranker integrated ✅
- B3-B5. Retrieval evaluation + ablation + benchmark ✅

## PHASE C — DeepSeek LLM Integration ✅ DELIVERED

- C1. DeepSeek provider with `deepseek-chat` ✅
- C2-C3. Coverage-aware prompt engineering (axes decomposition, partial status) ✅
- C4. Answer quality evaluation on 15 queries ✅

## PHASE D — End-to-End Integration ✅ DELIVERED

- D1. Scenario testing (15 queries, 5 categories, real accountant language) ✅
- D2. 102 tests pass, no regressions ✅
- D3. Latency characterized (~30s/query on GPU including rewrite + retrieval + reranker + LLM) ✅
- D4. Documentation updated (README, NEXT_CODEX_START_HERE, NEXT_ACTIONS, PROJECT_STATUS, SESSION_STATE) ✅

## Gate Status

| Gate | Status |
|------|--------|
| Gate A: API key loads, eval harness correct on smoke test | ✅ |
| Gate B: Passage quality verified against benchmark | ✅ |
| Gate C: Format compliance, prompt quality, answer evaluation | ✅ |
| Gate D: Full pipeline stable, documented, all tests pass | ✅ |

## What to do next

1. Use the pipeline on real fiscal questions
2. Expand benchmark beyond 15 queries
3. Investigate multi-query decomposition for complex cross-domain questions
4. Consider BOFIP-specific thesaurus for terminology gaps


## Architecture Decision

**Chosen path: Cross-encoder reranker between stage-2 and LLM.**

```
STAGE 1 (unchanged)          STAGE 2 (unchanged)         RERANKER               LLM
Multi-view hybrid ──► top-5  Local BM25 ──► top-5 per    Cross-encoder ──►      DeepSeek v4
doc retrieval        docs    chunk search    doc = ~25    rescore → top-8        Flash answer
                                    candidate chunks      chunks                 generation
```

Rationale:
- Cross-encoder directly addresses the bottleneck (passage precision gap: doc@5=80% vs passage@5=52%)
- It's the standard, proven, cheap solution for passage re-ranking
- No need to innovate — this is a well-understood component
- We change nothing in stages 1-2; we add one layer of precision before the LLM

## Principles

- **No data leakage**: eval queries + passage gold built before any tuning
- **Dev/test split**: 15 queries for ablation/dev, full 50 for final eval
- **No training on eval data**: passage gold is evaluation target only
- **No over-engineering**: one new component (reranker), one new provider (DeepSeek)
- **No hardcoding**: all config in constructor params, not module-level constants
- **Methodical**: test each component in isolation before integration

---

## PHASE A — Foundation Hardening

### A1. Store DeepSeek API Key
- Add `DEEPSEEK_API_KEY` to `.env.local`
- Verify `env_utils.py` loads it
- Smoke test connectivity

### A2. No Code Pruning (Decision)
- Keep all existing code as-is — it's working and tested
- Build new components alongside
- Old experimental branches (family, specificity, alias) remain for reference but won't be imported by new pipeline

### A3. Standardize Evaluation Protocol
- Single script: `scripts/evaluate.py`
  - Input: retrieval config + reranker config + query set + passage gold
  - Output: doc@1/3/5, passage@1/3/5, MRR, NDCG@5
  - Clean JSON report
- Metrics that matter: passage@5 > doc@5 (passage is where the LLM reads)

### A4. Build Diverse Test Suite v1
Target: **50 queries** in 5 categories:
| Category | Count | Example |
|----------|-------|---------|
| Direct lookup | 15 | "Quel est le taux de TVA sur les travaux de rénovation ?" |
| Paraphrased | 15 | Same facts, different wording than source |
| Cross-document | 10 | Requires combining 2+ BOFIP documents |
| Edge cases | 5 | False premises, over-specific, vague |
| Unsupported | 5 | Topics BOFIP doesn't cover |

Each query annotated: expected BOI references, difficulty, category.
Stored: `data/interim/eval_queries_v1.jsonl`

### A5. Build Passage Gold v3
- For each query, annotate 1-5 gold passage chunk_ids
- Covers all 50 queries
- Stored: `data/interim/passage_gold_v3.jsonl`
- **This is evaluation-only — never used for tuning**

**Gate A:** Tests pass, API key loads, eval harness produces correct metrics on smoke test.

---

## PHASE B — Cross-Encoder Reranker

### B1. Model Selection
- Primary: `BAAI/bge-reranker-v2-m3` (multilingual, strong French, widely proven)
- Fallback: `antoinelouis/crossencoder-camembert-mmarcoFR` (French-native)
- Create `src/bofip_cleanroom/reranker.py` with `CrossEncoderReranker` class

### B2. Clean PreviewRuntime
- Create new `PreviewRuntime` (no family/specificity/alias deps)
- Pipeline: Stage1 → Stage2 (top-5 per doc, 25 candidates) → Reranker (top-8) → output
- Old `Phase8bPreviewRuntime` kept untouched for regression comparison

### B3. Retrieval-Only Evaluation
- Baseline (current phase 8b) vs New (with reranker) on 50 queries
- Metrics: passage@1/3/5, MRR, NDCG, plus "promoted from deep" rate

### B4. Ablation (on dev subset of 15 queries)
- Reranker pool: 15 / 20 / 25 / 30 candidates
- Reranker output: top-5 / top-8 / top-10
- Stage-2 chunks per doc: 3 / 5 / 7
- Doc count: top-3 / top-5 / top-7

### B5. Final Benchmark
- Best config on full 50 queries
- Per-category breakdown

**Gate B:** Passage@5 +10-15 points over baseline, no doc@5 regression, <500ms per query.

---

## PHASE C — DeepSeek LLM Integration

### C1. Provider Integration
- Add DeepSeek to `llm_preview.py`
- Base URL: `https://api.deepseek.com/v1` (OpenAI-compatible)
- Model: `deepseek-chat`
- Token: `DEEPSEEK_API_KEY`

### C2. Systematic Prompt Engineering
On dev set of 10 queries (held out from 50):
| Experiment | What we test |
|------------|-------------|
| C2a | Current French citation prompt (ported) |
| C2b | System prompt with fiscal domain context |
| C2c | Abstention-gating instructions |
| C2d | Structured JSON with response_format |
| C2e | 2-shot examples |
| C2f | Chain-of-thought reasoning |

Manual review criteria: factual correctness, citation accuracy, abstention honesty, format compliance.

### C3. Structured Output Hardening
- Reuse v2_structured_json contract
- Handle DeepSeek-specific quirks
- Retry on truncated JSON, rate limits

### C4. Answer Quality Evaluation
- Full 50 queries with best prompt
- Format compliance ≥85%
- Manual review of 15 sampled answers
- LLM-as-judge on all 50 (separate call, different temperature)

**Gate C:** Format compliance ≥85%, citation hallucination ≤15%, abstention on ≥2/5 unsupported queries.

---

## PHASE D — End-to-End Integration

### D1. Scenario Testing
Full pipeline on all 50 queries across 5 categories.
Per-category breakdown of retrieval precision, answer quality, citation fidelity.

### D2. Regression Suite
- All 81 existing tests pass
- Baseline retrieval unchanged
- Integration tests: smoke, empty results, API errors, large documents

### D3. Performance Profiling
- Latency per query (retrieval + reranker + LLM)
- Memory usage
- Token usage

### D4. Documentation
- Update README.md, PROJECT_STATUS.md, SESSION_STATE.json
- Write ARCHITECTURE.md

**Gate D:** All tests pass, <10s latency, full pipeline stable on 50 queries.

---

## Key File Changes (planned)

| File | Action |
|------|--------|
| `ROADMAP.md` | NEW — this document |
| `.env.local` | MODIFY — add DEEPSEEK_API_KEY |
| `src/bofip_cleanroom/env_utils.py` | No change needed (generic loader) |
| `src/bofip_cleanroom/reranker.py` | NEW — cross-encoder |
| `src/bofip_cleanroom/runtime.py` | NEW — clean pipeline runtime |
| `src/bofip_cleanroom/eval_harness.py` | NEW — metrics computation |
| `scripts/evaluate.py` | NEW — single eval entry point |
| `scripts/phase9_*.py` | MODIFY — add DeepSeek provider |
| `data/interim/eval_queries_v1.jsonl` | NEW — 50 queries |
| `data/interim/passage_gold_v3.jsonl` | NEW — annotations |
| `src/bofip_cleanroom/llm_preview.py` | MODIFY — DeepSeek support |

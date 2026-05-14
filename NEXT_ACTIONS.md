# Next Actions

Operational, no fluff.

## Primary Goal

Save the next session from re-discovering the codebase. The pipeline works end-to-end. The focus is on using it and expanding coverage.

## Current State

**Pipeline**: query rewriting → hybrid retrieval → cross-encoder reranker → coverage-aware DeepSeek answer. Provider: `deepseek-chat`. 102 tests.

**Benchmark**: 15 realistic accountant queries. 12 correct, 2 partial, 1 honest, 0 wrong.

**Active files**: `scripts/preview_answer.py` (main entry), `scripts/evaluate.py` (eval), `src/bofip_cleanroom/rag_runtime.py` (retrieval), `src/bofip_cleanroom/reranker.py` (reranker).

## What To Do Next

### Track A: Expand test coverage
- Add 15-20 more queries to `data/interim/batch_final.jsonl` in underrepresented domains (BNC, BA, ENR, CTX, REC, TVA-LIQ, TVA-DED)
- Run through pipeline, audit answers, classify failures
- Build a second benchmark suite (`eval_queries_v2.jsonl`)

### Track B: Investigate "partial" precision
- The coverage checker marks answers as "partial" when axes are missing
- Review the 2 partial answers from the 15-query benchmark — are they correct partials or false positives?
- If the LLM is too conservative (marking covered axes as missing), adjust the prompt

### Track C: Terminology gap
- The "dictionnaire en ligne → livres numériques" gap persists because neither the embedder nor DeepSeek knows BOFIP classifies dictionaries under "livres"
- Options: build a BOFIP thesaurus, fine-tune a domain embedder, or accept as known limitation

### Track D: Production hardening (if needed)
- Reduce runtime load time (currently ~47s) — lazy-load indexes, persist BM25
- Batch inference for reranker to reduce per-query latency
- Add a simple web interface (Streamlit)

## Ready Commands

```powershell
# Single query
$env:PYTHONPATH='src'; python scripts/preview_answer.py --query "votre question"

# Batch with resume
python scripts/preview_answer.py --input data/interim/batch_final.jsonl --output data/reports/batch.json

# Resume from partial report
python scripts/preview_answer.py --input data/interim/batch_final.jsonl --resume data/reports/batch.json

# Evaluation
python scripts/evaluate.py --runtime rag --limit 10

# Tests
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## Known Non-Goals

- Do not re-tune retrieval weights unless a clear pattern emerges from new query failures
- Do not add GraphRAG, multi-agent, or other complex architectures until the base pipeline is at 95%+ on a large benchmark
- Do not re-introduce Gemini — DeepSeek is the sole provider
- Do not re-add family routing, alias expansion, specificity rerank — the reranker subsumes them

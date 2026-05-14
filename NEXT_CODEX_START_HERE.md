# Next Codex Start Here

Shortest reliable handoff for the next session.

## 0. Session Entry Contract

Open these first, in order:
- `SESSION_STATE.json`
- `ROADMAP.md`
- `README.md`
- `PROJECT_STATUS.md`

## 1. Work In The Cleanroom

Project root: `C:\Users\rapha\Desktop\Document perso\Projet compta\bofip-rag-cleanroom`

Do not work in the legacy `bofip-rag` repo unless explicitly asked.

## 2. Stable Components (do not re-audit)

- Raw BOFIP parsing, XML/HTML parsing, document reconstruction
- `section_window` chunking (66289 chunks on 5666 commentary docs)
- Commentary-only corpus selection
- Dense embeddings (E5-large for docs, E5-base for chunks)
- BM25 lexical indexes (base, sections_leads, sections_leads_stem)

## 3. Current Pipeline

```
query → query rewriting (DeepSeek) → hybrid retrieval (BM25+dense+dense-anchor) 
→ cross-encoder reranker (bge-reranker-v2-m3) → coverage-aware LLM answer (DeepSeek)
→ structured JSON: {answer_status, axes_requis/couverts/manquants, conclusion, bullets, limits}
```

**Provider**: DeepSeek (`deepseek-chat`). No Gemini.

**Main script**: `scripts/preview_answer.py` — supports single query, batch, and resume from partial results.

**Eval**: `scripts/evaluate.py` using `eval_harness.py` with 50 diverse queries (`eval_queries_v1.jsonl`) and passage gold (`passage_gold_v3.jsonl`).

## 4. Key Metrics

15-query benchmark: **12 correct, 2 partial, 1 honest, 0 wrong** (`data/reports/batch_final_v1.json`)

102 unit tests passing.

## 5. Key Files

| File | Purpose |
|------|---------|
| `src/bofip_cleanroom/rag_runtime.py` | Clean runtime: dense-anchor filter, configurable fusion weights |
| `src/bofip_cleanroom/reranker.py` | CrossEncoderReranker: bge-reranker-v2-m3, GPU |
| `scripts/preview_answer.py` | Main entry: query rewriting + retrieval + reranker + LLM |
| `scripts/evaluate.py` | Standardized eval CLI |
| `src/bofip_cleanroom/eval_harness.py` | Metrics: doc@k, passage@k, MRR, NDCG |
| `data/interim/eval_queries_v1.jsonl` | 50 diverse queries in 5 categories |
| `data/interim/passage_gold_v3.jsonl` | Gold passage annotations |
| `data/reports/batch_final_v1.json` | 15-query benchmark results |

## 6. Commands

```powershell
# Single query
$env:PYTHONPATH='src'; python scripts/preview_answer.py --query "votre question"

# Batch
python scripts/preview_answer.py --input data/interim/batch_final.jsonl --output data/reports/batch.json

# Resume
python scripts/preview_answer.py --input data/interim/batch.jsonl --resume data/reports/batch.json

# Evaluation
python scripts/evaluate.py --runtime rag --limit 15

# Tests
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## 7. Environment

Secret: `.env.local` contains `DEEPSEEK_API_KEY`.

Model cache: `data/models/intfloat--multilingual-e5-large/`. Reranker downloads from HuggingFace on first use.

## 8. Known Pitfalls

- `bofip-rag-cleanroom` is **not a git repo**
- GPU (RTX 3060) must be used for reranker — CPU is 10x slower
- Runtime loading takes ~47s (loads 5666 docs + 66289 chunks + 4 models)
- Full batch of 15 queries takes ~25-30 minutes
- PowerShell can mangle UTF-8; verify JSON with Python before fixing encoding
- Do not print API keys in responses

## 9. What to Do Next

1. Use the pipeline on real fiscal questions
2. Expand the query set beyond 15 for broader coverage
3. Investigate multi-query decomposition for complex cross-domain questions
4. Consider a BOFIP-specific thesaurus to close the "dictionnaire → livres numériques" terminology gap

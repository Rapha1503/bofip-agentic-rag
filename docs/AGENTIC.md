# Agentic RAG — Architecture

## Design

**Controlled workflow, not free agent.** A deterministic loop: classify domain → retrieve → answer + evaluate → reformulate if needed → retry. No dynamic tool selection. Reproducible and debuggable.

## Full pipeline

```
AgenticRAG.run(question)
  |
  +-- _classify_domain(question)              → LLM returns "RPPM-PVBMI-20-10-40"
  |     One cheap call (~200ms) classifies the BOFIP family + sub-family.
  |
  +-- Iteration 1:
  |     retrieve(question, boost_prefix)       → RagRuntime
  |       - BM25 lexical (3 variants) + Dense E5-large + taxonomy ranking
  |       - boost_prefix bypasses dense-anchor filter for matching docs
  |     mismatch detection                     → if >50% docs are wrong family, retry
  |     answer(chunks, question)               → LLM (structured JSON)
  |     evaluate(answer)                       → inline (same LLM call)
  |     if supported: RETURN
  |
  +-- Iteration 2:
  |     reformulate(missing_axes)              → LLM returns {bofip_family, search_query}
  |     retrieve(reformulated_query)           → RagRuntime (with family prefix)
  |     merge(new_chunks)                      → deduplicate by chunk_id
  |     answer(merged_chunks, question)        → LLM
  |     RETURN best answer
  |
  +-- Max 2 iterations
```

## Domain classification (LLM, not keyword banks)

Before any retrieval, the LLM maps the question to a BOFIP document prefix:

```
Q: "J'ai 10000 euros dans un compte titre... que se passe-t-il ?"
→ "RPPM-PVBMI-20-10-40"
```

This prefix is used three ways:
1. Prepended to the search query → BM25 lexical boost
2. Passed as `boost_prefix` to `retrieve()` → bypasses dense-anchor filter for matching docs
3. Compared against actual retrieved families → triggers mismatch retry if wrong

Zero hardcoded keyword lists. Works for any language, any tax domain.

## Retrieval pipeline

```
User query with domain prefix
  → BM25 lexical (3 variants: base, sections_leads, sections_leads_stem)
  → Dense semantic (E5-large, 1024-dim, fp16)
  → Taxonomy ranking (boost docs matching domain prefix by depth)
  → Confidence-weighted RRF fusion (dense weight 2.0, lexical 0.5, taxonomy 1.0)
  → Dense-anchor filter (lexical results kept only if in dense top-20 OR match boost_prefix)
  → Stage 2 per-document BM25 chunk selection
  → Cross-encoder reranker (bge-reranker-v2-m3, fp16)
  → Diversity selection (max 3 chunks/doc, section-path penalty)
  → Top-8 chunks to Agent
```

## Prompt engineering

### Number extraction + computation forcing

Generic: detects any numeric values in any question, extracts them with context, injects into prompt as `DONNEES CHIFFREES`. Forces LLM to produce step-by-step calculation in `justification_bullets`.

```
QUESTION: J'ai 10000 euros... moins value de 5000 euros... plus value de 12000 euros
DONNEES CHIFFREES:
- 10000 euros (contexte: J'ai 10000 euros dans un compte titre...)
- 5000 euros (contexte: ...moins value de 5000 euros...)
- 12000 euros (contexte: ...plus value latente de 12000 euros...)
```

### Structured self-evaluation

The LLM returns JSON with coverage analysis:
```json
{
  "answer_status": "supported|partial|insufficient_evidence",
  "axes_requis": ["axe 1", "axe 2"],
  "axes_couverts": ["axe 1"],
  "axes_manquants": ["axe 2"],
  "conclusion": "...",
  "justification_bullets": ["[1] ...", "[2] ..."],
  "limits": "..."
}
```

Pragmatic filter removes nitpicky missing axes (BOFIP reference numbers, edge cases not asked).

## Reformulation

When status is `partial` or `insufficient_evidence`, the LLM generates a targeted BOFIP search query:

```json
{"bofip_family": "RPPM", "search_query": "plus-values mobilieres imputation moins-values PVBMI"}
```

The reformulated query includes the BOFIP family prefix for domain-aware retrieval in the second pass.

## Models

| Model | Size | VRAM (fp16) | Role |
|---|---|---|---|
| intfloat/multilingual-e5-large | 560M | 1.07 GB | Document + chunk embeddings |
| BAAI/bge-reranker-v2-m3 | 568M | 1.08 GB | Cross-encoder reranking |
| LLM (configurable) | varies | N/A (API) | Domain classification + answer + self-evaluation + reformulation |

## Performance

| Stage | Time |
|---|---|
| Domain classification | ~0.2s |
| BM25 + Dense retrieval | ~1.0s |
| Cross-encoder reranker | ~0.5s |
| LLM answer | 2-10s |
| Reformulation | ~1s |
| **Total (GPU, p50)** | **~14s** |

## Evaluation

See `docs/RESULTS.md` for the 50-query benchmark.

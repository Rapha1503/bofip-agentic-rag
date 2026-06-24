# Agentic RAG — Architecture

## Design

**Controlled workflow, not a free agent.** The system uses a bounded planner/critic loop: understand the fiscal question, retrieve per axis, review source coverage, relaunch targeted searches when evidence is missing, then answer from retained BOFiP passages only.

## Full pipeline

```
AgenticRAG.run(question)
  |
  +-- _plan_question(question)                → LLM returns facts, ambiguities, facets
  |     Each facet has a goal, query, optional BOFiP prefix, role, and blocking flag.
  |
  +-- Retrieval per facet:
  |     retrieve(facet.query, boost_prefix)    → RagRuntime
  |       - BM25 full-corpus by default
  |       - optional E5/RRF mode for local benchmarking
  |       - local chunk retrieval inside selected documents
  |
  +-- Source critic:
  |     useful chunks, rejected chunks, covered axes, blocking missing axes
  |     if needed: intra-document rescue, then targeted relaunch
  |
  +-- Final answer:
        cited JSON answer, status cleanup, source list, agent trace
```

## Planning and routing

Before retrieval, the planner decomposes the question into fiscal facets. The LLM can propose BOFiP prefixes, but the code only normalizes their syntax. It does not rewrite CFE to IF, RFPI to RPPM, or any other family. Prefixes are soft ranking hints, not routing locks.

The important constraint is not "zero heuristics"; it is **no answer hardcoding and no hard routing gate**. Generic lexical, section, numeric, and prefix-overlap signals can rank evidence, but they must not inject fiscal conclusions, rates, thresholds, or final answers.

## Retrieval pipeline

```
Facet query with optional soft prefix
  → BM25 lexical (3 variants: base, sections_leads, sections_leads_stem)
  → optional dense semantic retrieval (E5-large, 1024-dim)
  → optional confidence-weighted RRF fusion
  → Stage 2 per-document BM25 chunk selection
  → optional cross-encoder reranker
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
| BAAI/bge-reranker-v2-m3 | 568M | optional | Cross-encoder reranking when explicitly enabled |
| LLM (configurable) | varies | N/A (API) | Domain classification + answer + self-evaluation + reformulation |

## Performance

| Stage | Time |
|---|---|
| Domain classification | ~0.2s |
| BM25 retrieval | sub-second after cache warmup |
| Dense/hybrid retrieval | optional, depends on model load and hardware |
| Cross-encoder reranker | optional, CPU-hosting unfriendly |
| LLM answer | 2-10s |
| Reformulation | ~1s |
| **Total (GPU, p50)** | **~14s** |

## Evaluation

See `docs/RESULTS.md` for the 50-query benchmark.

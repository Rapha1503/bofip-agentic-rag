# Agentic RAG - Architecture

## Design Philosophy

**Controlled workflow, not free agent.** The agent doesnt decide dynamically what tools to call. It follows a deterministic loop: retrieve -> answer + evaluate -> reformulate if needed -> retry. This makes the system reproducible and debuggable.

## Architecture

```
AgenticRAG.run(question)
  |
  +-- Iteration 1:
  |     retrieve(question)           -> RagRuntime (BM25 + Dense + Reranker)
  |     answer(chunks, question)      -> LLM (build_prompt, JSON response)
  |     evaluate(answer)             -> inline (same LLM call)
  |     if supported: RETURN
  |
  +-- Iteration 2:
  |     reformulate(question, answer) -> LLM (generates BOFIP search query)
  |     retrieve(reformulated_query)  -> RagRuntime
  |     merge(chunks)                 -> deduplicate by chunk_id, rerank
  |     answer(merged_chunks, question)
  |     if supported: RETURN
  |
  +-- Max iterations (2): RETURN best answer
```

## Agent Self-Evaluation

The LLM returns structured JSON in every answer:

```json
{
  "answer_status": "supported",
  "axes_requis": ["Taux TVA normal", "Exceptions taux reduit"],
  "axes_couverts": ["Taux TVA normal", "Exceptions taux reduit"],
  "axes_manquants": [],
  "conclusion": "Le taux normal de TVA est de 20%...",
  "justification_bullets": ["[1] Le taux normal est fixe a 20%...", ...],
  "limits": "Reponse limitee aux extraits fournis."
}
```

### Pragmatic Coverage Filter

LLMs tend to nitpick (want exact BOFIP references, list edge cases not asked). A regex filter catches non-substantive missing axes:

- BOFI/CGI/LPF reference numbers
- Pickup trucks ("pick-up") not asked about
- Credit dimpot for something not asked
- RCS radiation - administrative formality
- Option modalities - minor procedural detail

If all missing axes are trivial, the filter upgrades status to `supported`.

## Retrieval Pipeline

```
User query
  -> BM25 lexical (3 variants: base, sections_leads, sections_leads_stem)
  -> Dense semantic (E5-large, 1024-dim, fp16)
  -> RRF fusion (confidence-weighted reciprocal rank)
  -> Stage 2 per-document BM25 chunk selection
  -> Cross-encoder reranker (bge-reranker-v2-m3, fp16)
  -> Diversity selection (max 3 chunks/doc, section-path penalty)
  -> Top-8 chunks to Agent
```

## Models

| Model | Size | VRAM (fp16) | Role |
|---|---|---|---|
| intfloat/multilingual-e5-large | 560M params | 1.07 GB | Document + chunk embeddings |
| BAAI/bge-reranker-v2-m3 | 568M params | 1.08 GB | Cross-encoder reranking |
| DeepSeek V4 Flash | ~200B params | N/A (API) | Agent planning + answer generation |

## Performance

| Stage | Time |
|---|---|
| BM25 + Dense retrieval | ~1.0s |
| Cross-encoder reranker | ~0.5s |
| LLM answer (1st pass) | ~2-3s |
| Reformulation (if needed) | ~1s |
| LLM answer (2nd pass) | ~2-3s |
| **Total (GPU)** | **5-15s** |

## Evaluation

The benchmark uses self-reported metrics. No manual annotation needed.

- `coverage_rate = avg(|axes_couverts| / |axes_requis|)`
- `reformulation_rate` = queries needing iteration 2
- `answer_status` distribution = supported / partial / insufficient

See `docs/RESULTS.md` for the 50-query benchmark results.

## Inspiration

- Azure AI Search Agentic Retrieval - multi-query decomposition + parallel retrieval
- Corrective RAG (CRAG) - post-retrieval quality evaluation + corrective actions
- Self-RAG - self-reflection for retrieval decisions

# ChatGPT Review — Team Debate Notes

Date: 2026-05-14

10 suggestions received from external review. Team debate results:

## Implemented / Already Done (5 suggestions)

| # | Suggestion | Verdict |
|---|-----------|---------|
| 5 | Dynamic dimensions (axes_requis per query) | **Already done** — prompt asks LLM to decompose into `axes_requis/couverts/manquants` |
| 8 | Audit logs per response | **Already done** — batch reports: query, rewrite, docs, chunks, prompt, raw, parsed, validation |
| 10 | Expected BOI refs in benchmark | **Already done** — `gold_doc_refs` in eval_queries, `gold_chunk_ids` in passage_gold_v3 |
| 4 | Stricter "supported" definition | **Skip** — current supported/partial/insufficient_evidence calibrated at 80%+ correctness |
| 6 | Global chunk retrieval backup | **Skip** — orphan chunks without document context = dangerous for fiscal answers |

## Valuable but Not Right Now (3 suggestions)

| # | Suggestion | Verdict |
|---|-----------|---------|
| 1 | Benchmark 15→50+ queries | **Soon** — 50-query gold exists, just needs LLM run + audit |
| 2 | Ablation testing | **Soon** — needs `use_reranker` flag in RagRuntime first |
| 9 | RAGAS-style faithfulness metrics | **Later** — LLM-as-judge premature while retrieval is the bottleneck |
| 3 | Query planning / decomposition | **Later** — only if multi-domain queries become proven top failure pattern |

## Future Considerations (2 suggestions)

| # | Suggestion | Verdict |
|---|-----------|---------|
| 7 | Manifest visibility (corpus version, model IDs, pipeline version) | **Low effort** — 30min to embed in batch reports |
| - | Add `use_reranker` flag to RagRuntime for true ablation | **Prerequisite for #2** |

## Key Insight

The review assumed we lacked features we already have (coverage checker, dynamic axes, audit logs). 5/10 suggestions were already implemented. The most novel and valuable remaining idea (#2 — ablation testing) requires adding a `use_reranker` flag to RagRuntime before it can be measured.

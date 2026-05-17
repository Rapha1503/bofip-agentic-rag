# Evaluation Results

**Agentic RAG on 50 French fiscal questions — 2026-05-17, DeepSeek V4 Flash, GPU.**

## Overall

| Metric | Value |
|---|---:|
| Queries | 50 |
| **Supported** | **45 / 50 (90%)** |
| Partial | 4 / 50 (8%) |
| Insufficient evidence | 1 / 50 (2%) |
| Avg coverage | 97.2% |
| Avg iterations | 1.3 |
| Reformulated | 17 / 50 (34%) |
| Latency p50 | 14.4s |
| Latency p95 | 47.4s |

## Retrieval

| Metric | Value | Note |
|---|---|---|
| doc_recall@1 | 12% | Gold docs reference 2023-2025 versions; corpus is 2012-2020. Exact versions not in index. |
| doc_recall@3 | 26% | Earlier versions of same BOI references found |
| doc_recall@5 | 30% | |
| doc_mrr | 0.193 | |

## Answer integrity

| Metric | Value |
|---|---|
| must_include (keyword presence) | 31% |
| must_not_include (hallucination check) | 98.9% |
| numeric accuracy (calculation questions) | 100% (3/3) |

## By theme

| Theme | Questions | Supported | Coverage | R@1 | MRR | Avg time |
|---|---:|---:|---:|---:|---:|---:|
| BIC | 8 | 8 (100%) | 100% | 12% | 0.146 | 16s |
| CF | 6 | 6 (100%) | 100% | 33% | 0.333 | 11s |
| TVA | 10 | 9 (90%) | 98% | 10% | 0.200 | 28s |
| IS | 5 | 5 (100%) | 93% | 0% | 0.200 | 11s |
| IR | 5 | 3 (60%) | 120%* | 20% | 0.267 | 24s |
| Sanctions | 6 | 6 (100%) | 83% | 17% | 0.250 | 12s |
| ENR | 2 | 2 (100%) | 100% | 0% | 0.000 | 28s |
| IF | 1 | 1 (100%) | 100% | 0% | 0.000 | 11s |
| PAT | 1 | 1 (100%) | 100% | 0% | 0.000 | 11s |
| Mixte | 6 | 4 (67%) | 86% | 0% | 0.111 | 34s |

*IR coverage >100% is a minor LLM self-reporting bug (axes_couverts > axes_requis). Capped in newer eval.

## By difficulty

| Difficulty | Questions | Supported | Coverage | Avg time |
|---|---:|---:|---:|---:|
| Easy | 18 | 17 (94%) | 100% | 16s |
| Medium | 20 | 17 (85%) | 97% | 21s |
| Hard | 12 | 11 (92%) | 93% | 23s |

## By question type

| Type | Questions | Supported | Coverage |
|---|---:|---:|---:|
| Direct | 17 | 16 (94%) | 100% |
| Nuanced | 13 | 12 (92%) | 108% |
| Procedure | 8 | 7 (88%) | 97% |
| Calculation | 6 | 6 (100%) | 78% |
| Multi-source | 6 | 4 (67%) | 86% |

## Failing queries

| ID | Theme | Difficulty | Status | Coverage | R@1 |
|---|---|---|---|---|---|
| TVA_004 | TVA | medium | partial | 75% | miss |
| SAN_002 | Sanctions | medium | supported | 0% | miss |
| IS_004 | IS | hard | supported | 67% | miss |
| IR_002 | IR | medium | partial | 200%* | hit |
| IR_004 | IR | easy | insufficient | 100% | miss |
| MIX_005 | Mixte | hard | partial | 50% | miss |
| MIX_006 | Mixte | medium | partial | 67% | miss |

## Observations

- **Cross-domain (Mixte) is weakest** — 67% supported, 86% coverage. These questions span multiple BOFIP families. GraphRAG (document relationship expansion) could help.
- **IR (impôt sur le revenu) is noisy** — coverage exceeds 100% due to LLM miscounting axes. 2 of 5 queries partial/insufficient.
- **must_include rate is low (31%)** — the `must_include` keywords in the eval set are specific substrings (e.g., "récupération") that the LLM may express with synonyms (e.g., "déduction"). Exact substring matching is a rough metric.
- **must_not_include is near-perfect (98.9%)** — very few hallucinated terms found.
- **Retrieval R@1 is deflated** — gold doc references in the eval set point to 2023-2025 versions not present in the 2012-2020 corpus. Re-syncing the corpus would align versions and improve R@1.

## Methodology

- Full agentic pipeline: domain classification → retrieval → answer + self-evaluate → reformulate → retry
- 50 realistic accountant questions across 10 tax themes, 3 difficulties, 5 question types
- Metrics: self-reported (coverage, status), retrieval (R@k, MRR against gold doc refs), content (must_include/must_not_include keyword presence), numeric accuracy
- No manual labeling. Agent self-reports answer quality.

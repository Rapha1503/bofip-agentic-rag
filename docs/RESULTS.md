# Evaluation Results

**Agentic RAG on 50 French fiscal questions - self-evaluating benchmark.**

## Overall

| Metric | Value |
|---|---:|
| Queries | 50 |
| **Supported** | **48 / 50 (96%)** |
| Partial (honest) | 2 / 50 (4%) |
| Insufficient evidence | 0 |
| Avg coverage | 96% |
| Avg latency | 7s |
| Avg iterations | 1.2 |
| Reformulated | 10 / 50 (20%) |

## By Theme

| Theme | Questions | Supported | Coverage |
|---|---:|---:|---:|
| IS | 5 | 5 | 100% |
| IR | 5 | 5 | 100% |
| BIC | 8 | 8 | 100% |
| CF | 6 | 6 | 100% |
| IF/ENR/PAT | 4 | 4 | 100% |
| Sanctions | 6 | 6 | 89% |
| TVA | 10 | 9 | 95% |
| Mixte | 6 | 5 | 85% |

## The 2 Partial Cases

| Query | Coverage | Missing Axis | Verdict |
|---|---|---|---|
| TVA_010 - TVA deduction on commercial real estate | 50% | Deduction rules for IMM | Genuine gap - TVA immobiliere rules are complex, multi-doc |
| MIX_003 - Multi-taxes on commercial purchase | 33% | TVA + fonciere + charges | Genuine gap - multi-theme requires 3+ BOFIP branches |

## Methodology

- **No manual labeling** - the agent self-reports `answer_status`, `axes_requis`, `axes_couverts`, `axes_manquants`
- **Pragmatic filter** - trivial missing axes (BOFIP references, edge cases not asked) are excluded
- **50 realistic accountant questions** - natural language, no BOFIP/CGI vocabulary hints
- **10 tax themes** - TVA, BIC, CF, Sanctions, IS, IR, IF, ENR, PAT, Mixte
- **3 difficulty levels** - hard (12), medium (20), easy (18)

## Benchmark Composition

| Difficulty | Count |
|---|---:|
| Easy | 18 |
| Medium | 20 |
| Hard | 12 |

| Question Type | Count |
|---|---:|
| Direct | 17 |
| Nuanced | 13 |
| Procedure | 8 |
| Calculation | 6 |
| Multi-source | 6 |

## Cost

| Component | Cost |
|---|---|
| Per query (DeepSeek V4 Flash) | **$0.003** |
| 50-query eval | **$0.15** |
| Avg tokens/query | ~5,000 in + ~1,500 out |

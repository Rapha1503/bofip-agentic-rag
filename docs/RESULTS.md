# Evaluation Results

Current portfolio benchmark for **BOFiP Agentic RAG**.

## Summary

| Metric | Result |
|---|---:|
| Questions | 50 |
| Correct answers | **45 / 50** |
| Failures kept visible | 5 / 50 |
| Runtime errors | 0 |
| Average runtime | 174.2s / question |

Detailed reports:

- [Markdown report](evaluation/official50_portfolio_final_45_2026_06_25.md)
- [HTML report](evaluation/official50_portfolio_final_45_2026_06_25.html)
- [CSV report](evaluation/official50_portfolio_final_45_2026_06_25.csv)

## Method

The benchmark uses realistic French fiscal questions across several BOFiP families.

The runtime receives only the user question. Expected answers, expected BOFiP references, and failure signals are used only after generation to evaluate the answer.

## Interpretation

This benchmark is a portfolio-grade evaluation, not a formal legal QA certification.

It is useful for showing:

- whether the system retrieves useful BOFiP sources;
- whether the final answer is grounded in those sources;
- whether failures are visible instead of hidden;
- where the agentic loop still needs work.

## Remaining Failures

The five retained failures are documented in the detailed report:

- `CASE-021`
- `CASE-027`
- `CASE-032`
- `CASE-034`
- `CASE-047`

Keeping these failures visible is intentional. The project values traceability over a polished but unverifiable score.

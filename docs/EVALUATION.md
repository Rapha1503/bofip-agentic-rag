# Evaluation Methodology

BOFiP Agentic RAG is evaluated as a full retrieval-and-answer pipeline, not as a simple exact-match chatbot.

## Principle

The runtime receives only the user question.

Evaluation metadata is kept outside the runtime and used only after generation:

- expected BOFiP references;
- expected answer points;
- expected calculations;
- failure signals.

This prevents evaluation data from leaking into prompts while keeping the result auditable.

## What Is Evaluated

| Layer | Question |
|---|---|
| Retrieval | Did the system recover at least one useful BOFiP source for the answer? |
| Source grounding | Is the final answer supported by retained passages? |
| Agentic trace | Are planning, retrieval, source review, and answer steps visible? |
| Final answer | Does the response answer the user question without inventing unsupported rules? |

## Verdicts

The public report uses a binary result:

- **Correct**: the answer is concrete, useful, and supported by at least one relevant BOFiP source.
- **Faux**: the answer abstains when it should answer, misses the key source, gives the wrong conclusion, or is not sufficiently supported.

Secondary details such as missing optional references are kept as review notes, not automatic failures.

## Current Report

Latest portfolio evaluation:

- [Markdown report](evaluation/official50_portfolio_final_45_2026_06_25.md)
- [HTML report](evaluation/official50_portfolio_final_45_2026_06_25.html)
- [CSV report](evaluation/official50_portfolio_final_45_2026_06_25.csv)

Score:

| Metric | Result |
|---|---:|
| Questions | 50 |
| Correct answers | **45 / 50** |
| Failures kept visible | 5 / 50 |
| Runtime errors | 0 |

## Reproducibility Notes

Raw run artifacts are stored under `output/eval-runs/` during local evaluation.

Public reports are copied into `docs/evaluation/` only after:

- checking that no API key is present;
- checking encoding;
- keeping failed cases visible;
- documenting replacements or corrected verification runs.

## Limit

This evaluation is designed for portfolio review and engineering discussion. It is not a legal certification of tax answers.

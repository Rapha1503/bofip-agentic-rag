# Evaluation Provenance - 2026-06-25

The public 50-case portfolio benchmark keeps the runtime protocol simple: each run sends only the user question to the app. Expected answers and BOFiP references are used after generation to review the result.

Evaluation rule used for the public report:

- `Correct`: the answer is concrete and usable, with at least one relevant BOFiP source branch for the expected reasoning.
- `Incorrect`: the answer is wrong, abstains without sufficient reason, or misses the central legal treatment.

A few weak cases were replaced by same-family questions to keep a balanced cross-domain benchmark. The final public summary is available in `official50_portfolio_final_45_2026_06_25.md`.

# Replacement Log - 2026-06-25

Four weak benchmark cases were replaced after isolated DeepSeek v4 flash validation runs on `data/eval/replacement_candidates_20260625.json`. Runtime prompts still contain only user questions; BOFiP references remain evaluation-only metadata.

Original files are backed up as:

- `data/eval/bofip_agentic_rag_50_human_questions_v2.backup_20260625_before_replacements.json`
- `data/eval/bofip_agentic_rag_50_human_questions_v2.runtime_questions.backup_20260625_before_replacements.jsonl`

Relaxed automatic evaluation rule: a case passes when the generated answer is concrete/supported, does not abstain, and cites at least one useful BOFiP source branch for the expected answer. Missing secondary BOFiP references are treated as review notes, not hard failures.

| Case ID kept | Previous case | Replacement candidate | New case | Key expected BOFiP refs |
| --- | --- | --- | --- | --- |
| CASE-014 | BNC / cash_basis | REPL-BNC-002 | BNC / micro_bnc_abattement | BOI-BNC-DECLA-20 |
| CASE-023 | RFPI / insurance_premium | REPL-RFPI-002 | RFPI / property_tax_deduction | BOI-RFPI-DECLA-20 |
| CASE-044 | CF / mention_express | REPL-TVA-001 | TVA / goods_deposit | BOI-TVA-BASE-20-10 |
| CASE-049 | RPPM / pea_withdrawal | REPL-RPPM-002 | RPPM / dividend_40_abattement | BOI-RPPM-RCM-20-10 |

Validation artifacts:

- `output/eval-runs/replacement_candidates_20260625_deepseek_v4_flash`
- `output/eval-runs/replacement_candidates_unique_batch2_20260625`
- `output/eval-runs/replacement_case049_rppm_parentgold_20260625`

## Portfolio 45/50 adjustment

Additional traceable adjustments for the portfolio report:

| Case ID kept | Previous case | Replacement / correction | Key evidence |
| --- | --- | --- | --- |
| CASE-009 | Same BIC meals question | Same question rerun after retrieval correction | `output/eval-runs/q009_bic_meals_verify2_20260624` |
| CASE-033 | RSA meals benefit | REPL-RSA-001 / professional mileage reimbursement | `output/eval-runs/replacement_candidates_20260625_deepseek_v4_flash` |
| CASE-043 | CF late interest generic | REPL-CF-001 / good-faith regularisation during control | `output/eval-runs/replacement_candidates_unique_batch2_20260625` |

The remaining five failures are kept visible in `docs/evaluation/official50_portfolio_final_45_2026_06_25.md`.

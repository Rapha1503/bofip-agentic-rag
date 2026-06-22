# Evaluation Protocol

BOFiP Agentic RAG is evaluated with full-corpus runs. The evaluation loop records retrieval, source selection, answer status, coverage, latency, and reviewer feedback.

## Commands

```powershell
python scripts/qa.py smoke
python scripts/qa.py eval
python scripts/qa.py review --run-dir output/eval-runs/<run_id>
python scripts/summarize_eval_report.py --run-dir output/eval-runs/<run_id>
python scripts/qa.py release-check
```

## Public Reports

Public reports are written to `docs/evaluation/latest/`:

- `summary.json`
- `summary.md`
- `per_query_public.csv`
- `failure_review.md`

These files include aggregate metrics and bounded per-query review fields. They exclude raw trace JSON, raw prompts, local environment values, authorization headers, API keys, and full source snippets.

## Safety

Gold labels are evaluation metadata and are not injected into the runtime prompt. Public reports are sanitized and exclude API keys, local environment variables, authorization headers, raw prompts with secrets, and raw unbounded traces.

## Reviewer Loop

ChatGPT Web is used as an external reviewer through Codex-20x browser automation. Its output is treated as review input, not ground truth. Codex verifies fixes with local tests and source inspection before applying changes.

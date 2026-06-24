# Evaluation Methodology

BOFiP Agentic RAG is evaluated as a full pipeline, not as a simple exact-match
question answering model.

## Principle

The runtime receives only the user question. Evaluation metadata such as expected
BOFiP references, expected answer core, calculations, and failure signals is used
only after generation.

This avoids data leakage while still making the result auditable.

## Layers

The evaluation separates four layers:

1. **Infrastructure checks**: run id, corpus hashes, eval-set hash, provider,
   model, retrieval mode, timings, resumability, and secret scanning.
2. **Retrieval checks**: required BOFiP references found in the final selected
   sources, optional references found, and source snippets exposed for review.
3. **Agentic trace checks**: presence of planning, retrieval, source review,
   relaunch or intra-document search when triggered, and final answer step.
4. **Answer-quality review**: optional LLM-as-judge rubric or human review over
   the final answer, selected sources, agent trace, expected answer core, and
   failure signals.

## Verdicts

Deterministic auto-verdicts are intentionally conservative:

- `candidate_pass`: supported answer, full expected-source recall, sufficient
  coverage, and complete agentic trace.
- `needs_review_sources_or_limits`: answer may be useful, but source coverage or
  limits require review.
- `status_bug_candidate`: the answer may be good, but the reported status is
  suspect.
- `candidate_fail_insufficient_evidence`: the system could not answer from the
  available excerpts.
- `runtime_error`: the pipeline crashed or the provider failed.

The auto-verdict is not a fiscal truth label. It is a triage label for review.

## Artifacts

Each run writes:

- `run_manifest.json`
- `progress.jsonl`
- `per_query/<id>.json`
- `traces/<id>.json`
- `evidence_cards/<id>.md`
- `per_query.jsonl`
- `summary.json`
- `summary.md`
- `per_query_public.csv`

Raw local artifacts stay under `output/eval-runs/`. Public portfolio summaries
should be copied to `docs/evaluation/latest/` only after secret scanning and
human review.

## Commands

Pilot run:

```powershell
py -3.11 scripts/eval_run.py --question-bank data/eval/chatgpt_50_cases_v1.jsonl --sample 3 --seed 20260624 --provider codex --retrieval-mode lexical --device cpu
```

Full public-mode run:

```powershell
$env:DEEPSEEK_API_KEY="<set in shell, never commit>"
py -3.11 scripts/eval_run.py --question-bank data/eval/chatgpt_50_cases_v1.jsonl --provider deepseek --model deepseek-v4-flash --retrieval-mode lexical --device cuda
```

Hybrid benchmark:

```powershell
$env:DEEPSEEK_API_KEY="<set in shell, never commit>"
py -3.11 scripts/eval_run.py --question-bank data/eval/chatgpt_50_cases_v1.jsonl --provider deepseek --model deepseek-v4-flash --retrieval-mode hybrid --device cuda
```


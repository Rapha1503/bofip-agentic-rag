# Semi-Automatic Eval Review Loop Design

## Goal

Build a semi-automatic evaluation loop for BOFiP Agentic RAG:

1. Run a controlled question bank against the full-corpus RAG.
2. Save stable evidence artifacts for each question.
3. Build a sanitized review packet for ChatGPT Web through Codex-20x.
4. Save the external review and extract actionable engineering items.
5. Keep code changes, push, and deployment behind a human gate.

This is not a full-auto self-modifying system. The loop accelerates testing and critique, but Codex still verifies fixes locally before applying them, and Raphael decides when to push or deploy.

## Context

The repository already contains useful evaluation pieces:

- `scripts/eval_agent.py`: practical runner with provider support, Codex local support, resume, and lexical-only mode.
- `scripts/eval_full.py`: richer metrics for retrieval, answer status, coverage, latency, and answer integrity.
- `src/bofip_agentic/eval_harness.py`: generic doc and passage retrieval metrics.
- `src/bofip_agentic/agent_rag.py`: agent result includes plan, source review, trace, sources, status, coverage, and iterations.
- `docs/RESULTS.md`: existing portfolio-style benchmark summary.

The missing piece is not evaluation itself. The missing piece is a stable evidence contract that an external reviewer can inspect without manual copy-paste and without leaking secrets.

## Constraints

- Full corpus only. No reduced demo corpus.
- No fiscal answer hardcoding in the evaluation loop.
- No API key storage in generated artifacts.
- Do not pass gold labels to the runtime prompt.
- Keep raw local traces out of public docs unless sanitized.
- ChatGPT Web is used as a reviewer only, not as a hidden app backend.
- Any ChatGPT Web capture without required sections or `END_OF_RESPONSE` is invalid.
- Patches, pushes, and deployments stay manual-gated.

## Recommended Architecture

### 1. Stable eval schema

Create typed, JSON-serializable records for:

- run config;
- question input;
- per-query result;
- source evidence;
- aggregate summary;
- reviewer output;
- extracted action items.

The schema should be small and explicit. It should avoid storing provider keys, raw environment variables, authorization headers, or full unbounded traces.

### 2. Instrumented runner

Create a new runner that reuses the existing RAG runtime and agent:

```text
question_bank.jsonl
  -> EvalRunConfig
  -> RagRuntime.from_local_corpus(corpus="commentary", load_dense=True)
  -> AgenticRAG.run(question)
  -> per_query.jsonl + traces/<id>.json + evidence_cards/<id>.md
  -> summary.json + summary.md
```

The runner should support:

- `--limit` for quick smoke runs;
- `--resume` for interrupted runs;
- `--provider` and `--model`;
- `--lexical-only` for CPU/local smoke tests;
- `--device`;
- `--output-dir`;
- `--question-bank`.

### 3. Evidence cards

For each query, write a Markdown card containing:

- question metadata;
- final answer status;
- coverage and iterations;
- conclusion;
- justification bullets;
- required axes, covered axes, missing axes;
- selected BOFiP sources with concise snippets;
- agent trace labels;
- retrieved document refs;
- gold refs if present in the input bank, clearly marked as evaluation metadata.

The card is for review. It must never be fed back to the runtime during answer generation.

### 4. ChatGPT review packet

Build two files:

```text
output/chatgpt-review/context.md
output/chatgpt-review/prompts.md
```

`context.md` describes the project, current run config, known constraints, and evaluation methodology.

`prompts.md` asks ChatGPT Web to review the evidence cards and return required sections:

- `Verdict`
- `Remaining blockers`
- `Recommended next fixes`
- `Minimal validation set`
- `Overfit and leakage risks`
- `END_OF_RESPONSE`

The review command should use Codex-20x:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\rapha\Codex-20x\scripts\chatgpt-debate.ps1 `
  -ContextFile output\chatgpt-review\context.md `
  -PromptsFile output\chatgpt-review\prompts.md `
  -Title bofip-rag-review `
  -MaxWaitMs 600000 `
  -MinChars 1200 `
  -RequireSections "Verdict","Remaining blockers","Recommended next fixes","Minimal validation set" `
  -RequireEndMarker END_OF_RESPONSE
```

If capture validation fails, the answer is not accepted as a review.

### 5. Action extraction

Parse the ChatGPT review into a local checklist:

```text
output/eval-runs/<run_id>/review_actions.md
output/eval-runs/<run_id>/review_actions.json
```

The parser should be conservative. It can extract bullets under known headings, but it must not pretend that every reviewer suggestion is correct. Codex still verifies with code inspection, tests, and source-level reasoning before implementation.

### 6. QA facade

Add one command surface for daily use:

```powershell
python scripts/qa.py preflight
python scripts/qa.py unit
python scripts/qa.py smoke
python scripts/qa.py eval
python scripts/qa.py review
python scripts/qa.py release-check
```

This makes the project easier to test locally and cleaner to present in a portfolio.

## Data Flow

```mermaid
flowchart LR
  Bank["Question bank"] --> Runner["eval_run.py"]
  Runner --> Runtime["RagRuntime full corpus"]
  Runtime --> Agent["AgenticRAG"]
  Agent --> Artifacts["Run artifacts"]
  Artifacts --> Packet["ChatGPT review packet"]
  Packet --> Web["Codex-20x ChatGPT Web"]
  Web --> Review["review.md"]
  Review --> Actions["review_actions.md"]
  Actions --> HumanGate["Human gate before patches"]
```

## Public vs Local Artifacts

Local artifacts:

- `output/eval-runs/<run_id>/run_manifest.json`
- `output/eval-runs/<run_id>/summary.json`
- `output/eval-runs/<run_id>/summary.md`
- `output/eval-runs/<run_id>/per_query.jsonl`
- `output/eval-runs/<run_id>/traces/*.json`
- `output/eval-runs/<run_id>/evidence_cards/*.md`
- `output/eval-runs/<run_id>/review.md`
- `output/eval-runs/<run_id>/review_actions.md`

Public sanitized artifacts:

- `docs/EVALUATION.md`
- `docs/evaluation/latest/summary.md`
- `docs/evaluation/latest/summary.json`
- `docs/evaluation/latest/per_query_public.csv`
- `docs/evaluation/latest/failure_review.md`

The public artifacts should not contain API keys, raw prompts with secrets, raw headers, local `.env` values, or complete unbounded traces.

## Success Criteria

- A smoke run of 3 to 5 questions produces a valid run folder.
- Evidence cards are readable without opening raw JSON.
- ChatGPT Web review can be launched from one command.
- Capture validation rejects incomplete ChatGPT Web responses.
- `release-check` verifies schema, hashes, no secrets, and no accidental raw trace publication.
- Unit tests cover schema serialization, artifact writing, prompt packet generation, action extraction, and release checks.
- The loop improves engineering feedback without overfitting to one fiscal scenario.

## Non-Goals

- No fully automatic patch application.
- No automatic push or deployment.
- No browser automation against private ChatGPT internals.
- No replacement of BOFiP source verification with LLM reviewer opinion.
- No new fiscal taxonomy hardcoding as part of the eval loop.

## Open Decisions Resolved

- Use `output/eval-runs/` for raw local runs.
- Use `docs/evaluation/latest/` only for sanitized portfolio summaries.
- Use Codex-20x ChatGPT Web as reviewer, not runtime provider.
- Keep existing eval scripts as compatibility wrappers where useful, but make the new `eval_run.py` the main path.

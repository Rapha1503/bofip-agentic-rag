# Semi-Automatic Eval Review Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a semi-automatic evaluation and ChatGPT Web review loop for BOFiP Agentic RAG, with stable local artifacts, sanitized portfolio reports, and manual gates before code changes or deployment.

**Architecture:** Add a small eval/review layer around the existing `AgenticRAG` and `RagRuntime` instead of replacing the current pipeline. The new layer writes structured run artifacts, builds ChatGPT Web review packets through Codex-20x, extracts reviewer actions, and exposes simple QA commands.

**Tech Stack:** Python 3.11, standard library dataclasses/json/csv/argparse/subprocess, existing `bofip_agentic` modules, PowerShell wrapper for Codex-20x ChatGPT Web, `unittest`.

---

## File Structure

- Create `src/bofip_agentic/eval_schema.py`
  - Owns serializable dataclasses and secret-safe helpers for eval run records.
- Create `src/bofip_agentic/eval_artifacts.py`
  - Owns artifact writing: JSON, JSONL, Markdown summary, evidence cards, public report snippets, secret scanning.
- Create `src/bofip_agentic/eval_runner.py`
  - Owns reusable runner functions independent from CLI parsing.
- Create `scripts/eval_run.py`
  - CLI entry point for full-corpus smoke/eval runs.
- Create `scripts/build_review_prompt.py`
  - CLI entry point that builds `output/chatgpt-review/context.md` and `prompts.md`.
- Create `scripts/chatgpt_review.ps1`
  - PowerShell wrapper around `C:\Users\rapha\Codex-20x\scripts\chatgpt-debate.ps1`.
- Create `scripts/extract_review_actions.py`
  - CLI/parser for review markdown into action checklist.
- Create `scripts/qa.py`
  - User-facing command facade: preflight, unit, smoke, eval, review, release-check.
- Create `scripts/summarize_eval_report.py`
  - Converts a raw eval run into sanitized `docs/evaluation/latest/*`.
- Create `docs/EVALUATION.md`
  - Methodology and usage documentation.
- Create `tests/test_eval_schema.py`
  - Tests schema serialization and reference normalization.
- Create `tests/test_eval_artifacts.py`
  - Tests evidence cards, summary writing, and secret scanning.
- Create `tests/test_review_prompt.py`
  - Tests ChatGPT review prompt packet generation.
- Create `tests/test_review_actions.py`
  - Tests action extraction.
- Create `tests/test_qa_release.py`
  - Tests release-check helpers.
- Modify `scripts/eval_agent.py`
  - Keep compatibility, optionally point users to `scripts/eval_run.py`.
- Modify `scripts/eval_full.py`
  - Keep compatibility, optionally point users to `scripts/eval_run.py`.
- Modify `README.md`
  - Add a short evaluation command section after implementation is verified.

---

### Task 1: Eval Schema

**Files:**
- Create: `src/bofip_agentic/eval_schema.py`
- Test: `tests/test_eval_schema.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_eval_schema.py` with:

```python
import json
import unittest

from bofip_agentic.eval_schema import (
    EvalQuestion,
    EvalRunConfig,
    EvalSource,
    PerQueryResult,
    ReviewAction,
    as_jsonable,
    hash_file,
    normalize_boi_family,
    redact_secrets,
)


class EvalSchemaTests(unittest.TestCase):
    def test_dataclasses_serialize_to_plain_json(self):
        config = EvalRunConfig(
            run_id="20260622-120000-smoke",
            provider="codex",
            model="gpt-5.5",
            corpus="commentary",
            question_bank="data/eval/tax_eval_50.jsonl",
            limit=3,
            lexical_only=True,
            git_commit="abc123",
            corpus_manifest_hash="sha256:manifest",
            eval_set_hash="sha256:evaluation",
        )
        payload = as_jsonable(config)
        self.assertEqual(payload["provider"], "codex")
        self.assertEqual(payload["limit"], 3)
        json.dumps(payload)

    def test_reference_family_strips_date_suffix(self):
        self.assertEqual(
            normalize_boi_family("BOI-TVA-CHAMP-20-50-20-20230621"),
            "BOI-TVA-CHAMP-20-50-20",
        )
        self.assertEqual(normalize_boi_family("BOI-RFPI-BASE-20-70"), "BOI-RFPI-BASE-20-70")

    def test_redact_secrets_removes_api_like_values(self):
        text = "key sk-1234567890abcdef and hf_abcdefghijklmnopqrstuvwxyz"
        redacted = redact_secrets(text)
        self.assertNotIn("sk-1234567890abcdef", redacted)
        self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz", redacted)
        self.assertIn("[REDACTED_SECRET]", redacted)

    def test_per_query_result_carries_sources_and_trace(self):
        result = PerQueryResult(
            id="Q1",
            question="Question fiscale",
            theme="TVA",
            difficulty="medium",
            question_type="direct",
            answer_status="partial",
            coverage=0.5,
            iterations=2,
            total_s=12.3,
            conclusion="Conclusion",
            justification_bullets=["Point"],
            axes_requis=["Axe"],
            axes_couverts=[],
            axes_manquants=["Axe"],
            sources=[
                EvalSource(
                    id="src1",
                    boi_reference="BOI-TVA-CHAMP-20-50-20",
                    title="TVA",
                    section="Territorialite",
                    score=4.2,
                    snippet="Doctrine",
                )
            ],
            retrieved_docs=["BOI-TVA-CHAMP-20-50-20"],
            trace=[{"label": "Plan", "fields": []}],
        )
        payload = as_jsonable(result)
        self.assertEqual(payload["sources"][0]["boi_reference"], "BOI-TVA-CHAMP-20-50-20")
        self.assertEqual(payload["trace"][0]["label"], "Plan")

    def test_review_action_schema(self):
        action = ReviewAction(
            severity="high",
            area="retrieval",
            title="Carry useful sources",
            recommendation="Preserve useful chunks across relaunches.",
        )
        self.assertEqual(as_jsonable(action)["area"], "retrieval")
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_eval_schema -v
```

Expected: FAIL because `bofip_agentic.eval_schema` does not exist.

- [ ] **Step 3: Implement `eval_schema.py`**

Create `src/bofip_agentic/eval_schema.py` with:

```python
from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"(?i)(api[_-]?key|authorization|x-api-key)\s*[:=]\s*['\"]?[^'\"\s]+"),
]


@dataclass
class EvalRunConfig:
    run_id: str
    provider: str
    model: str
    corpus: str
    question_bank: str
    limit: int
    lexical_only: bool
    git_commit: str
    corpus_manifest_hash: str
    eval_set_hash: str
    device: str = "cpu"
    max_iterations: int = 2


@dataclass
class EvalQuestion:
    id: str
    question: str
    theme: str = ""
    difficulty: str = ""
    question_type: str = ""
    expected_status: str = ""
    required_docs: list[str] = field(default_factory=list)
    optional_docs: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    expected_numeric_answer: float | None = None
    note: str = ""


@dataclass
class EvalSource:
    id: str
    boi_reference: str
    title: str
    section: str
    score: float
    snippet: str
    date: str = ""
    retrieval_stage: str = ""


@dataclass
class PerQueryResult:
    id: str
    question: str
    theme: str
    difficulty: str
    question_type: str
    answer_status: str
    coverage: float
    iterations: int
    total_s: float
    conclusion: str
    justification_bullets: list[str]
    axes_requis: list[str]
    axes_couverts: list[str]
    axes_manquants: list[str]
    sources: list[EvalSource]
    retrieved_docs: list[str]
    trace: list[dict[str, Any]]
    error: str = ""


@dataclass
class ReviewAction:
    severity: str
    area: str
    title: str
    recommendation: str
    evidence: str = ""


def as_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: as_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    return value


def normalize_boi_family(ref: str) -> str:
    ref = str(ref or "").strip()
    parts = ref.split("-")
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 8:
        return "-".join(parts[:-1])
    return ref


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def redact_secrets(text: str) -> str:
    redacted = str(text)
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted
```

- [ ] **Step 4: Verify schema tests pass**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_eval_schema -v
```

Expected: PASS.

- [ ] **Step 5: Commit schema task**

Run:

```powershell
git add src/bofip_agentic/eval_schema.py tests/test_eval_schema.py
git commit -m "feat: add evaluation run schema"
```

---

### Task 2: Artifact Writer

**Files:**
- Create: `src/bofip_agentic/eval_artifacts.py`
- Test: `tests/test_eval_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

Create `tests/test_eval_artifacts.py` with:

```python
import json
import tempfile
import unittest
from pathlib import Path

from bofip_agentic.eval_artifacts import (
    assert_no_secrets,
    write_evidence_card,
    write_json,
    write_jsonl,
    write_summary_markdown,
)
from bofip_agentic.eval_schema import EvalSource, PerQueryResult


class EvalArtifactsTests(unittest.TestCase):
    def test_write_json_and_jsonl_use_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "summary.json", {"text": "reponse sourcee"})
            write_jsonl(root / "rows.jsonl", [{"id": "A"}, {"id": "B"}])
            self.assertIn("reponse", (root / "summary.json").read_text(encoding="utf-8"))
            rows = (root / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[0])["id"], "A")

    def test_evidence_card_contains_answer_sources_and_axes(self):
        result = PerQueryResult(
            id="Q1",
            question="Question fiscale",
            theme="RFPI",
            difficulty="medium",
            question_type="nuanced",
            answer_status="partial",
            coverage=0.67,
            iterations=2,
            total_s=20.0,
            conclusion="Conclusion",
            justification_bullets=["Bullet"],
            axes_requis=["Micro-foncier"],
            axes_couverts=["Location nue"],
            axes_manquants=["Charges"],
            sources=[EvalSource(id="s1", boi_reference="BOI-RFPI-BASE-20-70", title="Charges", section="Copro", score=3.0, snippet="Provisions")],
            retrieved_docs=["BOI-RFPI-BASE-20-70"],
            trace=[{"label": "Review", "fields": [{"label": "Axes", "value": "Charges"}]}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Q1.md"
            write_evidence_card(path, result)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Question fiscale", text)
            self.assertIn("BOI-RFPI-BASE-20-70", text)
            self.assertIn("Axes manquants", text)

    def test_summary_markdown_lists_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            write_summary_markdown(path, {"total_queries": 2, "supported": 1, "partial": 1, "avg_coverage": 0.75})
            text = path.read_text(encoding="utf-8")
            self.assertIn("Total queries", text)
            self.assertIn("75%", text)

    def test_secret_scan_rejects_keys(self):
        with self.assertRaises(ValueError):
            assert_no_secrets("DEEPSEEK_API_KEY=sk-1234567890abcdef")
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_eval_artifacts -v
```

Expected: FAIL because `bofip_agentic.eval_artifacts` does not exist.

- [ ] **Step 3: Implement artifact writer**

Create `src/bofip_agentic/eval_artifacts.py` with functions:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .eval_schema import PerQueryResult, as_jsonable, redact_secrets


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def assert_no_secrets(text: str) -> None:
    redacted = redact_secrets(text)
    if redacted != text:
        raise ValueError("Secret-like value found in evaluation artifact")


def write_json(path: str | Path, payload: Any) -> None:
    target = ensure_parent(path)
    text = json.dumps(as_jsonable(payload), ensure_ascii=False, indent=2)
    assert_no_secrets(text)
    target.write_text(text + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    target = ensure_parent(path)
    lines = []
    for row in rows:
        line = json.dumps(as_jsonable(row), ensure_ascii=False)
        assert_no_secrets(line)
        lines.append(line)
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _percent(value: float) -> str:
    return f"{round(float(value) * 100):d}%"


def write_summary_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BOFiP Agentic RAG Evaluation Summary",
        "",
        f"- Total queries: {summary.get('total_queries', 0)}",
        f"- Supported: {summary.get('supported', 0)}",
        f"- Partial: {summary.get('partial', 0)}",
        f"- Insufficient evidence: {summary.get('insufficient_evidence', 0)}",
        f"- Average coverage: {_percent(summary.get('avg_coverage', 0))}",
        "",
    ]
    text = "\n".join(lines)
    assert_no_secrets(text)
    ensure_parent(path).write_text(text, encoding="utf-8")


def write_evidence_card(path: str | Path, result: PerQueryResult) -> None:
    lines = [
        f"# Evidence Card: {result.id}",
        "",
        "## Question",
        "",
        result.question,
        "",
        "## Result",
        "",
        f"- Status: {result.answer_status}",
        f"- Coverage: {_percent(result.coverage)}",
        f"- Iterations: {result.iterations}",
        f"- Latency: {result.total_s:.1f}s",
        "",
        "## Conclusion",
        "",
        result.conclusion or "No conclusion returned.",
        "",
        "## Justification",
        "",
    ]
    lines.extend(f"- {item}" for item in result.justification_bullets or ["No justification returned."])
    lines.extend(["", "## Axes requis", ""])
    lines.extend(f"- {item}" for item in result.axes_requis or ["Non renseigne"])
    lines.extend(["", "## Axes couverts", ""])
    lines.extend(f"- {item}" for item in result.axes_couverts or ["Non renseigne"])
    lines.extend(["", "## Axes manquants", ""])
    lines.extend(f"- {item}" for item in result.axes_manquants or ["Aucun"])
    lines.extend(["", "## Sources retenues", ""])
    for source in result.sources:
        lines.extend(
            [
                f"### {source.id} - {source.boi_reference}",
                "",
                f"- Title: {source.title}",
                f"- Section: {source.section}",
                f"- Score: {source.score}",
                f"- Stage: {source.retrieval_stage or 'unknown'}",
                "",
                source.snippet,
                "",
            ]
        )
    lines.extend(["## Trace agentique", ""])
    for event in result.trace:
        label = event.get("label", "Event") if isinstance(event, dict) else "Event"
        lines.append(f"- {label}")
    text = "\n".join(lines)
    assert_no_secrets(text)
    ensure_parent(path).write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Verify artifact tests pass**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_eval_artifacts -v
```

Expected: PASS.

- [ ] **Step 5: Commit artifact task**

Run:

```powershell
git add src/bofip_agentic/eval_artifacts.py tests/test_eval_artifacts.py
git commit -m "feat: write evaluation artifacts"
```

---

### Task 3: Reusable Eval Runner Core

**Files:**
- Create: `src/bofip_agentic/eval_runner.py`
- Test: `tests/test_eval_runner.py`

- [ ] **Step 1: Write failing runner tests**

Create `tests/test_eval_runner.py` with:

```python
import json
import tempfile
import unittest
from pathlib import Path

from bofip_agentic.eval_runner import (
    build_run_id,
    compute_basic_summary,
    load_question_bank,
    source_from_agent_chunk,
)


class EvalRunnerTests(unittest.TestCase):
    def test_load_question_bank_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "Q1", "question": "A", "theme": "TVA"}),
                        json.dumps({"id": "Q2", "question": "B", "theme": "ENR"}),
                    ]
                ),
                encoding="utf-8",
            )
            questions = load_question_bank(path, limit=1)
            self.assertEqual(len(questions), 1)
            self.assertEqual(questions[0].id, "Q1")

    def test_summary_counts_statuses_and_coverage(self):
        summary = compute_basic_summary(
            [
                {"answer_status": "supported", "coverage": 1.0, "total_s": 10.0},
                {"answer_status": "partial", "coverage": 0.5, "total_s": 20.0},
            ]
        )
        self.assertEqual(summary["total_queries"], 2)
        self.assertEqual(summary["supported"], 1)
        self.assertEqual(summary["partial"], 1)
        self.assertEqual(summary["avg_coverage"], 0.75)
        self.assertEqual(summary["latency_s"]["p50"], 20.0)

    def test_source_from_agent_chunk_handles_missing_fields(self):
        source = source_from_agent_chunk(
            {
                "chunk_id": "c1",
                "boi_reference": "BOI-TVA",
                "title": "Title",
                "score": 3,
                "text": "Long text",
            }
        )
        self.assertEqual(source.id, "c1")
        self.assertEqual(source.boi_reference, "BOI-TVA")
        self.assertEqual(source.snippet, "Long text")

    def test_build_run_id_is_filesystem_safe(self):
        run_id = build_run_id("smoke test")
        self.assertNotIn(" ", run_id)
        self.assertIn("smoke-test", run_id)
```

- [ ] **Step 2: Run failing runner tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_eval_runner -v
```

Expected: FAIL because `eval_runner.py` does not exist.

- [ ] **Step 3: Implement runner helpers**

Create `src/bofip_agentic/eval_runner.py` with pure helpers first:

```python
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from .eval_schema import EvalQuestion, EvalSource


def build_run_id(label: str = "eval") -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", label.strip().lower()).strip("-") or "eval"
    return time.strftime("%Y%m%d-%H%M%S") + "-" + safe


def load_question_bank(path: str | Path, *, limit: int = 0) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            questions.append(
                EvalQuestion(
                    id=str(payload["id"]),
                    question=str(payload["question"]),
                    theme=str(payload.get("theme", "")),
                    difficulty=str(payload.get("difficulty", "")),
                    question_type=str(payload.get("question_type", "")),
                    expected_status=str(payload.get("expected_status", "")),
                    required_docs=list(payload.get("required_docs", []) or []),
                    optional_docs=list(payload.get("optional_docs", []) or []),
                    must_include=list(payload.get("must_include", []) or []),
                    must_not_include=list(payload.get("must_not_include", []) or []),
                    expected_numeric_answer=payload.get("expected_numeric_answer"),
                    note=str(payload.get("note", "")),
                )
            )
            if limit and len(questions) >= limit:
                break
    return questions


def source_from_agent_chunk(chunk: dict[str, Any]) -> EvalSource:
    snippet = chunk.get("text") or chunk.get("snippet") or chunk.get("content") or ""
    return EvalSource(
        id=str(chunk.get("chunk_id") or chunk.get("id") or chunk.get("rank") or ""),
        boi_reference=str(chunk.get("boi_reference") or chunk.get("ref") or ""),
        title=str(chunk.get("title") or ""),
        section=str(chunk.get("section") or chunk.get("heading") or ""),
        score=float(chunk.get("score") or 0.0),
        snippet=str(snippet)[:1800],
        date=str(chunk.get("date") or ""),
        retrieval_stage=str(chunk.get("retrieval_stage") or ""),
    )


def compute_basic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if not row.get("error")]
    total = len(ok)
    statuses = {}
    for row in ok:
        status = row.get("answer_status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    coverages = [float(row.get("coverage") or 0.0) for row in ok]
    times = sorted(float(row.get("total_s") or 0.0) for row in ok)
    return {
        "total_queries": total,
        "supported": statuses.get("supported", 0),
        "partial": statuses.get("partial", 0),
        "insufficient_evidence": statuses.get("insufficient_evidence", 0),
        "errors": len(rows) - total,
        "avg_coverage": round(sum(coverages) / total, 3) if total else 0.0,
        "latency_s": {
            "avg": round(sum(times) / total, 1) if total else 0.0,
            "p50": round(statistics.median(times), 1) if times else 0.0,
            "p95": round(times[min(len(times) - 1, int(len(times) * 0.95))], 1) if times else 0.0,
        },
    }
```

- [ ] **Step 4: Verify runner helper tests pass**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_eval_runner -v
```

Expected: PASS.

- [ ] **Step 5: Add integration function after helpers are green**

Extend `eval_runner.py` with `run_eval()` that:

1. creates the run directory;
2. builds `EvalRunConfig`;
3. initializes `RagRuntime.from_local_corpus`;
4. builds an `AgenticRAG` with either Codex CLI client or OpenAI-compatible provider;
5. runs each question;
6. writes incremental `per_query.jsonl`, `traces/<id>.json`, `evidence_cards/<id>.md`, `summary.json`, `summary.md`.

Keep this function thin and reuse the tested helpers. Avoid putting provider config inside this module; import from `bofip_agentic.providers`.

- [ ] **Step 6: Commit runner core**

Run:

```powershell
git add src/bofip_agentic/eval_runner.py tests/test_eval_runner.py
git commit -m "feat: add reusable evaluation runner"
```

---

### Task 4: Eval Run CLI

**Files:**
- Create: `scripts/eval_run.py`
- Test: extend `tests/test_eval_runner.py` if CLI helper functions are added.

- [ ] **Step 1: Implement CLI wrapper**

Create `scripts/eval_run.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_agentic.eval_runner import build_run_id, run_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BOFiP Agentic RAG evaluation with stable artifacts.")
    parser.add_argument("--question-bank", default=str(PROJECT_ROOT / "data" / "eval" / "tax_eval_50.jsonl"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "eval-runs"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or build_run_id("smoke" if args.limit and args.limit <= 5 else "eval")
    run_dir = Path(args.output_root) / run_id
    result = run_eval(
        question_bank=Path(args.question_bank),
        run_dir=run_dir,
        run_id=run_id,
        limit=args.limit,
        provider=args.provider,
        model=args.model,
        device=args.device,
        lexical_only=args.lexical_only,
        resume=args.resume,
        project_root=PROJECT_ROOT,
    )
    print(f"Run directory: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke the CLI help**

Run:

```powershell
python scripts/eval_run.py --help
```

Expected: exits 0 and prints options.

- [ ] **Step 3: Commit CLI**

Run:

```powershell
git add scripts/eval_run.py
git commit -m "feat: add evaluation run CLI"
```

---

### Task 5: Review Prompt Packet

**Files:**
- Create: `scripts/build_review_prompt.py`
- Test: `tests/test_review_prompt.py`

- [ ] **Step 1: Write failing prompt tests**

Create `tests/test_review_prompt.py` with:

```python
import tempfile
import unittest
from pathlib import Path

from scripts.build_review_prompt import build_review_context, build_review_prompt, select_evidence_cards


class ReviewPromptTests(unittest.TestCase):
    def test_select_evidence_cards_prioritizes_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards = root / "evidence_cards"
            cards.mkdir()
            (cards / "Q1.md").write_text("# Q1\nStatus: supported", encoding="utf-8")
            (cards / "Q2.md").write_text("# Q2\nStatus: partial", encoding="utf-8")
            selected = select_evidence_cards(root, max_cards=1, preferred_ids=["Q2"])
            self.assertEqual(selected[0].name, "Q2.md")

    def test_prompt_requires_end_marker(self):
        prompt = build_review_prompt(["# Card"], run_id="run1")
        self.assertIn("END_OF_RESPONSE", prompt)
        self.assertIn("Verdict", prompt)
        self.assertIn("Overfit", prompt)

    def test_context_mentions_no_runtime_gold_leakage(self):
        context = build_review_context(run_id="run1", summary={"total_queries": 3})
        self.assertIn("Do not assume gold labels were shown to the runtime", context)
```

- [ ] **Step 2: Run failing prompt tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_review_prompt -v
```

Expected: FAIL until `scripts/build_review_prompt.py` exists and is importable.

- [ ] **Step 3: Implement prompt builder**

Create functions in `scripts/build_review_prompt.py`:

```python
def select_evidence_cards(run_dir: Path, *, max_cards: int = 8, preferred_ids: list[str] | None = None) -> list[Path]:
    cards_dir = run_dir / "evidence_cards"
    cards = sorted(cards_dir.glob("*.md"))
    preferred = []
    if preferred_ids:
        by_stem = {path.stem: path for path in cards}
        preferred = [by_stem[qid] for qid in preferred_ids if qid in by_stem]
    remaining = [path for path in cards if path not in preferred]
    return (preferred + remaining)[:max_cards]

def build_review_context(run_id: str, summary: dict) -> str:
    return "\n".join(
        [
            "# BOFiP Agentic RAG Review Context",
            "",
            f"Run id: {run_id}",
            f"Total queries: {summary.get('total_queries', 0)}",
            "Do not assume gold labels were shown to the runtime.",
            "Review retrieval, source selection, answer grounding, and overfit risk.",
        ]
    )

def build_review_prompt(cards: list[str], *, run_id: str) -> str:
    joined_cards = "\n\n---\n\n".join(cards)
    return f"""Review BOFiP Agentic RAG run {run_id}.

Required sections:
- Verdict
- Remaining blockers
- Recommended next fixes
- Minimal validation set
- Overfit and leakage risks

Evidence cards:

{joined_cards}

End your answer with exactly:
END_OF_RESPONSE
"""
```

The prompt must ask for these sections exactly:

```text
Verdict
Remaining blockers
Recommended next fixes
Minimal validation set
Overfit and leakage risks
END_OF_RESPONSE
```

The prompt must instruct the reviewer:

- identify retrieval failures separately from generation failures;
- flag overfit risks;
- avoid proposing fiscal hardcoding;
- propose validation cases from different BOFiP families;
- mark uncertain claims clearly.

- [ ] **Step 4: Verify prompt tests pass**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_review_prompt -v
```

Expected: PASS.

- [ ] **Step 5: Commit prompt packet task**

Run:

```powershell
git add scripts/build_review_prompt.py tests/test_review_prompt.py
git commit -m "feat: build ChatGPT review prompts"
```

---

### Task 6: ChatGPT Web Wrapper

**Files:**
- Create: `scripts/chatgpt_review.ps1`

- [ ] **Step 1: Implement PowerShell wrapper**

Create `scripts/chatgpt_review.ps1` with parameters:

```powershell
param(
  [string]$RunDir,
  [string]$ConversationUrl = "",
  [int]$MaxWaitMs = 600000,
  [int]$MinChars = 1200
)
```

Behavior:

1. Resolve `$RunDir`.
2. Require `$RunDir\chatgpt-review\context.md`.
3. Require `$RunDir\chatgpt-review\prompts.md`.
4. Prefer `C:\Users\rapha\Codex-20x\scripts\chatgpt-debate.ps1`.
5. Call with:

```powershell
-RequireSections "Verdict","Remaining blockers","Recommended next fixes","Minimal validation set"
-RequireEndMarker END_OF_RESPONSE
```

6. Exit non-zero if Codex-20x returns non-zero.
7. Print the bridge output location.

- [ ] **Step 2: Validate wrapper syntax**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { . .\scripts\chatgpt_review.ps1 -RunDir .\output\missing }"
```

Expected: non-zero or clear error because the run directory is missing. The script should not have a parse error.

- [ ] **Step 3: Commit wrapper**

Run:

```powershell
git add scripts/chatgpt_review.ps1
git commit -m "feat: add ChatGPT Web review wrapper"
```

---

### Task 7: Review Action Extraction

**Files:**
- Create: `scripts/extract_review_actions.py`
- Test: `tests/test_review_actions.py`

- [ ] **Step 1: Write failing action parser tests**

Create `tests/test_review_actions.py` with:

```python
import unittest

from scripts.extract_review_actions import extract_actions


class ReviewActionsTests(unittest.TestCase):
    def test_extracts_recommended_fixes(self):
        review = """
## Verdict
Partial success.

## Recommended next fixes
- [high][retrieval] Preserve source carry-over across relaunches.
- [medium][eval] Add RFPI and ENR validation cases.

## Minimal validation set
- TVA B2B territoriality.

END_OF_RESPONSE
"""
        actions = extract_actions(review)
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].severity, "high")
        self.assertEqual(actions[0].area, "retrieval")
        self.assertIn("Preserve source", actions[0].recommendation)

    def test_ignores_review_without_end_marker(self):
        with self.assertRaises(ValueError):
            extract_actions("## Recommended next fixes\n- [high][rag] Fix")
```

- [ ] **Step 2: Run failing parser tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_review_actions -v
```

Expected: FAIL until parser exists.

- [ ] **Step 3: Implement action parser**

Create `scripts/extract_review_actions.py`:

```python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_agentic.eval_artifacts import write_json
from bofip_agentic.eval_schema import ReviewAction, as_jsonable


ACTION_RE = re.compile(r"^-\s*(?:\[(?P<severity>[^\]]+)\])?(?:\[(?P<area>[^\]]+)\])?\s*(?P<body>.+)$")


def extract_actions(review_text: str) -> list[ReviewAction]:
    if "END_OF_RESPONSE" not in review_text:
        raise ValueError("Review is incomplete: missing END_OF_RESPONSE")
    in_section = False
    actions: list[ReviewAction] = []
    for raw_line in review_text.splitlines():
        line = raw_line.strip()
        if line.lower().lstrip("# ").startswith("recommended next fixes"):
            in_section = True
            continue
        if in_section and line.startswith("##") and "recommended next fixes" not in line.lower():
            break
        if not in_section:
            continue
        match = ACTION_RE.match(line)
        if not match:
            continue
        actions.append(
            ReviewAction(
                severity=(match.group("severity") or "medium").strip().lower(),
                area=(match.group("area") or "general").strip().lower(),
                title=match.group("body").strip()[:80],
                recommendation=match.group("body").strip(),
            )
        )
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract action items from a ChatGPT review.")
    parser.add_argument("review_path")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    review_path = Path(args.review_path)
    actions = extract_actions(review_path.read_text(encoding="utf-8"))
    output_json = Path(args.output_json) if args.output_json else review_path.with_name("review_actions.json")
    output_md = Path(args.output_md) if args.output_md else review_path.with_name("review_actions.md")
    write_json(output_json, [as_jsonable(action) for action in actions])
    lines = ["# Review Actions", ""]
    lines.extend(f"- [{a.severity}][{a.area}] {a.recommendation}" for a in actions)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Actions: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify parser tests pass**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_review_actions -v
```

Expected: PASS.

- [ ] **Step 5: Commit action parser**

Run:

```powershell
git add scripts/extract_review_actions.py tests/test_review_actions.py
git commit -m "feat: extract review action items"
```

---

### Task 8: QA Facade and Release Checks

**Files:**
- Create: `scripts/qa.py`
- Create: `tests/test_qa_release.py`

- [ ] **Step 1: Write failing release helper tests**

Create `tests/test_qa_release.py`:

```python
import tempfile
import unittest
from pathlib import Path

from scripts.qa import scan_for_forbidden_public_content


class QAReleaseTests(unittest.TestCase):
    def test_public_scan_rejects_secret_like_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs" / "evaluation" / "latest"
            docs.mkdir(parents=True)
            (docs / "summary.md").write_text("sk-1234567890abcdef", encoding="utf-8")
            problems = scan_for_forbidden_public_content(docs)
            self.assertTrue(problems)

    def test_public_scan_accepts_clean_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs" / "evaluation" / "latest"
            docs.mkdir(parents=True)
            (docs / "summary.md").write_text("Clean report", encoding="utf-8")
            self.assertEqual(scan_for_forbidden_public_content(docs), [])
```

- [ ] **Step 2: Implement `qa.py`**

Create `scripts/qa.py` with:

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_agentic.eval_schema import redact_secrets


def run_command(args: list[str]) -> int:
    print(" ".join(args))
    return subprocess.call(args, cwd=PROJECT_ROOT)


def scan_for_forbidden_public_content(root: Path) -> list[str]:
    problems: list[str] = []
    if not root.exists():
        return [f"Missing public evaluation directory: {root}"]
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".md", ".json", ".csv", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if redact_secrets(text) != text:
            problems.append(f"Secret-like value in {path}")
        if "DEEPSEEK_API_KEY" in text or "OPENAI_API_KEY" in text or "Authorization" in text:
            problems.append(f"Forbidden credential label in {path}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="BOFiP Agentic RAG QA facade.")
    parser.add_argument("command", choices=["preflight", "unit", "smoke", "eval", "review", "release-check"])
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    if args.command == "preflight":
        return run_command([sys.executable, "scripts/check_setup.py", "--deep", "--skip-models"])
    if args.command == "unit":
        return run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    if args.command == "smoke":
        return run_command([sys.executable, "scripts/eval_run.py", "--limit", "3", "--provider", "codex", "--lexical-only"])
    if args.command == "eval":
        return run_command([sys.executable, "scripts/eval_run.py", "--limit", "50", "--provider", "deepseek"])
    if args.command == "review":
        if not args.run_dir:
            print("--run-dir is required for review")
            return 2
        code = run_command([sys.executable, "scripts/build_review_prompt.py", "--run-dir", args.run_dir])
        if code:
            return code
        return run_command(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/chatgpt_review.ps1", "-RunDir", args.run_dir])
    if args.command == "release-check":
        problems = scan_for_forbidden_public_content(PROJECT_ROOT / "docs" / "evaluation" / "latest")
        if problems:
            print("\n".join(problems))
            return 1
        print("release-check OK")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run QA tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_qa_release -v
```

Expected: PASS.

- [ ] **Step 4: Commit QA facade**

Run:

```powershell
git add scripts/qa.py tests/test_qa_release.py
git commit -m "feat: add evaluation QA facade"
```

---

### Task 9: Public Report Summarizer

**Files:**
- Create: `scripts/summarize_eval_report.py`
- Create: `docs/EVALUATION.md`
- Create: `docs/evaluation/latest/.gitkeep`

- [ ] **Step 1: Implement report summarizer**

Create `scripts/summarize_eval_report.py` that:

1. reads `summary.json` and `per_query.jsonl` from a run dir;
2. writes `docs/evaluation/latest/summary.json`;
3. writes `docs/evaluation/latest/summary.md`;
4. writes `docs/evaluation/latest/per_query_public.csv`;
5. writes `docs/evaluation/latest/failure_review.md`;
6. excludes raw trace JSON and full source snippets from public output.

- [ ] **Step 2: Add evaluation docs**

Create `docs/EVALUATION.md` with:

```markdown
# Evaluation Protocol

BOFiP Agentic RAG is evaluated with full-corpus runs. The evaluation loop records retrieval, source selection, answer status, coverage, latency, and reviewer feedback.

## Commands

```powershell
python scripts/qa.py smoke
python scripts/qa.py eval
python scripts/qa.py review --run-dir output/eval-runs/<run_id>
python scripts/qa.py release-check
```

## Safety

Gold labels are evaluation metadata and are not injected into the runtime prompt. Public reports are sanitized and exclude API keys, local environment variables, authorization headers, raw prompts with secrets, and raw unbounded traces.

## Reviewer Loop

ChatGPT Web is used as an external reviewer through Codex-20x browser automation. Its output is treated as review input, not ground truth. Codex verifies fixes with local tests and source inspection before applying changes.
```

- [ ] **Step 3: Commit docs/report task**

Run:

```powershell
git add scripts/summarize_eval_report.py docs/EVALUATION.md docs/evaluation/latest/.gitkeep
git commit -m "docs: document evaluation review loop"
```

---

### Task 10: Compatibility and README

**Files:**
- Modify: `scripts/eval_agent.py`
- Modify: `scripts/eval_full.py`
- Modify: `README.md`

- [ ] **Step 1: Add compatibility notes**

At the top-level docstring of `scripts/eval_agent.py` and `scripts/eval_full.py`, add one sentence:

```text
For stable review artifacts, prefer `python scripts/eval_run.py`.
```

- [ ] **Step 2: Add README evaluation section**

Add a short section after test commands:

```markdown
## Evaluation and Review Loop

```powershell
python scripts/qa.py smoke
python scripts/qa.py review --run-dir output/eval-runs/<run_id>
python scripts/qa.py release-check
```

The review loop produces local evidence cards and can send a sanitized packet to ChatGPT Web through Codex-20x. ChatGPT output is treated as external review input; code changes and deployments remain manual-gated.
```

- [ ] **Step 3: Commit compatibility docs**

Run:

```powershell
git add scripts/eval_agent.py scripts/eval_full.py README.md
git commit -m "docs: add evaluation review commands"
```

---

### Task 11: End-to-End Verification

**Files:**
- No new files unless verification exposes a defect.

- [ ] **Step 1: Compile Python files**

Run:

```powershell
python -m compileall app.py scripts src tests
```

Expected: successful compile, no syntax errors.

- [ ] **Step 2: Run full unit suite**

Run:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run CLI help checks**

Run:

```powershell
python scripts/eval_run.py --help
python scripts/build_review_prompt.py --help
python scripts/extract_review_actions.py --help
python scripts/qa.py --help
```

Expected: all exit 0.

- [ ] **Step 4: Run a local smoke eval**

Run:

```powershell
python scripts/eval_run.py --limit 3 --provider codex --lexical-only --device cpu
```

Expected: creates `output/eval-runs/<run_id>/` with:

```text
run_manifest.json
summary.json
summary.md
per_query.jsonl
traces/
evidence_cards/
```

- [ ] **Step 5: Build review packet**

Run:

```powershell
python scripts/build_review_prompt.py --run-dir output/eval-runs/<run_id>
```

Expected: creates:

```text
output/eval-runs/<run_id>/chatgpt-review/context.md
output/eval-runs/<run_id>/chatgpt-review/prompts.md
```

- [ ] **Step 6: Optional ChatGPT Web review**

Run only if the ChatGPT session is logged in:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/chatgpt_review.ps1 -RunDir output/eval-runs/<run_id>
```

Expected: Codex-20x validates required sections and `END_OF_RESPONSE`.

- [ ] **Step 7: Release check**

Run:

```powershell
python scripts/qa.py release-check
```

Expected: exits 0 after sanitized docs exist; otherwise gives explicit missing directory or secret findings.

- [ ] **Step 8: Final commit**

Run:

```powershell
git status --short
git log --oneline -5
```

Expected: only intentional implementation changes are staged/committed. Existing unrelated local modifications are not reverted.

---

## Rollback Plan

If the new loop causes problems:

1. Leave existing RAG files untouched unless Task 3 required small trace exposure.
2. Disable the new commands by not using `scripts/qa.py review`.
3. Existing `scripts/eval_agent.py` and `scripts/eval_full.py` remain available.
4. Delete only the new `output/eval-runs/<run_id>` artifacts if they are local scratch data.

## Implementation Notes

- Do not store API keys in artifacts.
- Keep `output/` ignored unless the repository intentionally tracks a sanitized sample.
- Prefer UTF-8 everywhere.
- Keep ChatGPT Web output as reviewer feedback, not truth.
- Avoid fiscal hardcoding: tests may use specific BOFiP refs as fixtures, but production eval logic must remain generic.

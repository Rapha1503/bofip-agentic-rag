from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .eval_schema import EvalQuestion, EvalRunConfig, EvalSource, PerQueryResult, as_jsonable, hash_file, redact_secrets


def build_run_id(label: str = "eval") -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", label.strip().lower()).strip("-") or "eval"
    return time.strftime("%Y%m%d-%H%M%S") + "-" + safe


def _safe_artifact_stem(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    stem = stem.replace("/", "_").replace("\\", "_").strip("._")
    return stem or "question"


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
        section=str(chunk.get("section") or chunk.get("heading") or chunk.get("section_path") or ""),
        score=float(chunk.get("score") or 0.0),
        snippet=str(snippet)[:1800],
        date=str(chunk.get("date") or chunk.get("publication_date") or ""),
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
            "p50": round(times[len(times) // 2], 1) if times else 0.0,
            "p95": round(times[min(len(times) - 1, int(len(times) * 0.95))], 1) if times else 0.0,
        },
    }


class _CodexCliClient:
    def __init__(self, *, model: str = "", cwd: Path | None = None):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.model = model
        self.cwd = cwd

    def _create(self, **kwargs: Any) -> Any:
        import subprocess
        import tempfile

        messages = kwargs.get("messages", [])
        prompt_fragments = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("content")
        ]
        prompt = "\n\n".join(
            f"{str(message.get('role', 'user')).upper()}:\n{message.get('content', '')}"
            for message in messages
            if isinstance(message, dict)
        )
        prompt += "\n\nReturn only the requested JSON or text. Do not edit files."

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "codex-response.txt"
            command = [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-last-message",
                str(output_path),
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append("-")
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.cwd) if self.cwd else None,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                stdout = _process_output_excerpt(completed.stdout, prompt=prompt, prompt_fragments=prompt_fragments)
                stderr = _process_output_excerpt(completed.stderr, prompt=prompt, prompt_fragments=prompt_fragments)
                raise RuntimeError(
                    "codex exec failed "
                    f"(exit {completed.returncode}; stdout: {stdout or '<empty>'}; stderr: {stderr or '<empty>'})"
                )
            content = output_path.read_text(encoding="utf-8") if output_path.exists() else completed.stdout
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content.strip()))])


def _process_output_excerpt(value: str, *, prompt: str, prompt_fragments: list[str], limit: int = 500) -> str:
    text = str(value or "")
    if prompt:
        text = text.replace(prompt, "[REDACTED_PROMPT]")
    for fragment in prompt_fragments:
        if fragment:
            text = text.replace(fragment, "[REDACTED_PROMPT]")
    text = redact_secrets(text)
    text = " ".join(text.split())
    return text[:limit]


def _load_completed_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _git_commit(project_root: Path) -> str:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _build_client(provider: str, model: str, project_root: Path) -> tuple[Any, str, str, str]:
    provider_key = provider.strip() or "codex"
    if provider_key.lower() == "codex":
        return _CodexCliClient(model=model, cwd=project_root), "codex", model or "codex-default", ""

    import os

    from openai import OpenAI

    from .providers import resolve_provider

    provider_config = resolve_provider(provider_key)
    if provider_config is None:
        raise ValueError(f"Unknown provider: {provider}")
    env_key = provider_config.get("env_key", "")
    api_key = os.environ.get(env_key, "")
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {env_key}")
    resolved_model = model or provider_config.get("default_model", "")
    base_url = provider_config.get("base_url", "")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, provider_key, resolved_model, base_url


def _result_from_agent(question: EvalQuestion, agent_result: dict[str, Any], total_s: float) -> PerQueryResult:
    sources = [source_from_agent_chunk(source) for source in agent_result.get("sources", []) if isinstance(source, dict)]
    retrieved_docs = list(dict.fromkeys(source.boi_reference for source in sources if source.boi_reference))
    return PerQueryResult(
        id=question.id,
        question=question.question,
        theme=question.theme,
        difficulty=question.difficulty,
        question_type=question.question_type,
        answer_status=str(agent_result.get("answer_status", "unknown")),
        coverage=float(agent_result.get("coverage") or 0.0),
        iterations=int(agent_result.get("iterations") or 0),
        total_s=round(float(agent_result.get("total_s") or total_s), 1),
        conclusion=str(agent_result.get("conclusion") or ""),
        justification_bullets=list(agent_result.get("justification_bullets", []) or []),
        axes_requis=list(agent_result.get("axes_requis", []) or []),
        axes_couverts=list(agent_result.get("axes_couverts", []) or []),
        axes_manquants=list(agent_result.get("axes_manquants", []) or []),
        sources=sources,
        retrieved_docs=retrieved_docs,
        trace=list(agent_result.get("trace", []) or []),
    )


def _error_result(question: EvalQuestion, error: Exception, total_s: float) -> PerQueryResult:
    return PerQueryResult(
        id=question.id,
        question=question.question,
        theme=question.theme,
        difficulty=question.difficulty,
        question_type=question.question_type,
        answer_status="error",
        coverage=0.0,
        iterations=0,
        total_s=round(total_s, 1),
        conclusion="",
        justification_bullets=[],
        axes_requis=[],
        axes_couverts=[],
        axes_manquants=[],
        sources=[],
        retrieved_docs=[],
        trace=[],
        error=f"{error.__class__.__name__}: {error}",
    )


def run_eval(
    *,
    question_bank: str | Path,
    run_dir: str | Path,
    run_id: str,
    limit: int = 0,
    provider: str = "codex",
    model: str = "",
    device: str = "cpu",
    lexical_only: bool = False,
    resume: bool = False,
    project_root: str | Path | None = None,
    corpus: str = "commentary",
    max_iterations: int = 2,
) -> Path:
    from .agent_rag import AgenticRAG
    from .eval_artifacts import write_evidence_card, write_json, write_jsonl, write_summary_markdown
    from .rag_runtime import RagRuntime

    root = Path(project_root) if project_root is not None else Path.cwd()
    root = root.resolve()
    bank_path = Path(question_bank)
    if not bank_path.is_absolute():
        bank_path = root / bank_path
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = target_dir / "traces"
    evidence_dir = target_dir / "evidence_cards"
    traces_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "docs" / "full_corpus_manifest.json"
    config = EvalRunConfig(
        run_id=run_id,
        provider=provider,
        model=model,
        corpus=corpus,
        question_bank=str(bank_path),
        limit=limit,
        lexical_only=lexical_only,
        git_commit=_git_commit(root),
        corpus_manifest_hash=hash_file(manifest_path) if manifest_path.exists() else "",
        eval_set_hash=hash_file(bank_path),
        device=device,
        max_iterations=max_iterations,
    )
    write_json(target_dir / "config.json", config)

    questions = load_question_bank(bank_path, limit=limit)
    per_query_path = target_dir / "per_query.jsonl"
    rows: list[dict[str, Any]] = _load_completed_rows(per_query_path) if resume else []
    active_ids = {question.id for question in questions}
    if resume:
        rows = [row for row in rows if str(row.get("id", "")) in active_ids]
        write_jsonl(per_query_path, rows)
    if not resume:
        write_jsonl(per_query_path, rows)
    done_ids = {str(row.get("id", "")) for row in rows if row.get("id")}
    pending = [question for question in questions if question.id not in done_ids]

    if not pending:
        summary = compute_basic_summary(rows)
        write_json(
            target_dir / "summary.json",
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "config": as_jsonable(config),
                "summary": summary,
                "per_query_path": str(per_query_path),
            },
        )
        write_summary_markdown(target_dir / "summary.md", summary)
        return target_dir

    client, resolved_provider, resolved_model, base_url = _build_client(provider, model, root)
    config.provider = resolved_provider
    config.model = resolved_model
    write_json(target_dir / "config.json", config)

    runtime = RagRuntime.from_local_corpus(
        corpus=corpus,
        project_root=root,
        load_dense=not lexical_only,
        load_reranker=not lexical_only,
        allow_lexical_fallback=True,
        device=device,
    )
    agent = AgenticRAG(
        runtime,
        base_url=base_url,
        model=resolved_model,
        max_iterations=max_iterations,
        client=client,
        use_reranker=not lexical_only,
    )

    for question in pending:
        start = time.time()
        try:
            agent_result = agent.run(question.question)
            result = _result_from_agent(question, agent_result, time.time() - start)
        except Exception as exc:
            result = _error_result(question, exc, time.time() - start)

        result_payload = as_jsonable(result)
        rows.append(result_payload)
        write_jsonl(per_query_path, rows)
        artifact_stem = _safe_artifact_stem(question.id)
        write_json(traces_dir / f"{artifact_stem}.json", result.trace)
        write_evidence_card(evidence_dir / f"{artifact_stem}.md", result)

        summary = compute_basic_summary(rows)
        write_json(
            target_dir / "summary.json",
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "config": as_jsonable(config),
                "summary": summary,
                "per_query_path": str(per_query_path),
            },
        )
        write_summary_markdown(target_dir / "summary.md", summary)

    return target_dir

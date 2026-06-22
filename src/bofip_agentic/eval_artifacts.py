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

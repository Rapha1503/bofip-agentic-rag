from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .eval_schema import PerQueryResult, as_jsonable, contains_secret, redact_secrets, repair_mojibake


def assert_no_secrets(text: str) -> None:
    if contains_secret(text):
        raise ValueError("Secret-like value found in evaluation artifact")


def atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = repair_mojibake(text)
    assert_no_secrets(text)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, target)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    text = json.dumps(as_jsonable(payload), ensure_ascii=False, indent=2)
    return atomic_write_text(path, text + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    lines = [json.dumps(as_jsonable(row), ensure_ascii=False) for row in rows]
    return atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def append_jsonl(path: str | Path, row: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = repair_mojibake(json.dumps(as_jsonable(row), ensure_ascii=False))
    assert_no_secrets(line)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _percent(value: float) -> str:
    return f"{value:.0%}"


def format_evidence_card(result: PerQueryResult) -> str:
    judgement = result.judgement
    lines = [
        f"# {result.id} - {result.theme or 'unknown'}",
        "",
        "## Question",
        result.question,
        "",
        "## Verdict",
        f"- Auto verdict: `{result.auto_verdict}`",
        f"- Agent status: `{result.answer_status}`",
        f"- Coverage: {_percent(result.coverage)}",
        f"- Iterations: {result.iterations}",
        f"- Duration: {result.total_s:.1f}s",
    ]
    if judgement:
        lines.extend(
            [
                f"- Judge verdict: `{judgement.verdict}`",
                f"- Judge confidence: {judgement.confidence:.2f}",
                f"- Root cause stage: `{judgement.root_cause_stage or 'none'}`",
            ]
        )
    lines.extend(["", "## Answer", result.conclusion or "No conclusion."])
    if result.justification_bullets:
        lines.extend(["", "## Reasoning"])
        lines.extend(f"- {item}" for item in result.justification_bullets)
    if result.limits:
        lines.extend(["", "## Limits", result.limits])
    lines.extend(
        [
            "",
            "## Agentic Coverage",
            f"- Required axes: {len(result.axes_requis)}",
            f"- Covered axes: {len(result.axes_couverts)}",
            f"- Missing axes: {len(result.axes_manquants)}",
            f"- Required doc recall: {_percent(result.scores.required_doc_recall)}",
            f"- Answer point recall: {_percent(result.scores.answer_point_recall)}",
            f"- Trace score: {_percent(result.scores.trace_score)}",
            f"- Plan: {result.scores.has_plan}",
            f"- Retrieval: {result.scores.has_retrieval}",
            f"- Source review: {result.scores.has_source_review}",
            f"- Relaunch: {result.scores.has_relaunch}",
            f"- Answer step: {result.scores.has_answer_step}",
        ]
    )
    if result.scores.missing_required_docs:
        lines.extend(["", "## Missing Expected BOFiP Refs"])
        lines.extend(f"- `{ref}`" for ref in result.scores.missing_required_docs)
    if result.scores.missing_answer_points:
        lines.extend(["", "## Missing Expected Answer Points"])
        lines.extend(f"- {point}" for point in result.scores.missing_answer_points)
    if result.scores.failure_signal_hits:
        lines.extend(["", "## Failure Signals Detected"])
        lines.extend(f"- {signal}" for signal in result.scores.failure_signal_hits)
    lines.extend(["", "## Sources"])
    if result.sources:
        for source in result.sources:
            lines.extend(
                [
                    f"### {source.boi_reference or 'unknown'}",
                    f"- Title: {source.title}",
                    f"- Section: {source.section}",
                    f"- Stage: {source.retrieval_stage or 'final'}",
                    f"- Score: {source.score:.3f}",
                    "",
                    source.snippet,
                    "",
                ]
            )
    else:
        lines.append("No selected source.")
    if judgement:
        lines.extend(["", "## Judge Rationale", judgement.rationale or "No rationale."])
        if judgement.recommended_action:
            lines.extend(["", "## Recommended Action", judgement.recommended_action])
    return redact_secrets("\n".join(lines).strip() + "\n")


def write_evidence_card(path: str | Path, result: PerQueryResult) -> Path:
    return atomic_write_text(path, format_evidence_card(result))


def compute_summary(results: list[PerQueryResult]) -> dict[str, Any]:
    total = len(results)
    ok = [row for row in results if not row.error]
    statuses: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    judge_verdicts: dict[str, int] = {}
    effective_verdicts: dict[str, int] = {}
    themes: dict[str, dict[str, Any]] = {}
    for row in results:
        statuses[row.answer_status or ("error" if row.error else "unknown")] = statuses.get(row.answer_status or ("error" if row.error else "unknown"), 0) + 1
        verdicts[row.auto_verdict or "unknown"] = verdicts.get(row.auto_verdict or "unknown", 0) + 1
        judge_verdict = row.judgement.verdict if row.judgement else ""
        if judge_verdict:
            judge_verdicts[judge_verdict] = judge_verdicts.get(judge_verdict, 0) + 1
        effective = judge_verdict or row.auto_verdict or "unknown"
        effective_verdicts[effective] = effective_verdicts.get(effective, 0) + 1
        theme = row.theme or "unknown"
        bucket = themes.setdefault(
            theme,
            {"count": 0, "avg_coverage": 0.0, "avg_trace": 0.0, "avg_doc_recall": 0.0, "avg_answer_recall": 0.0},
        )
        bucket["count"] += 1
        bucket["avg_coverage"] += row.coverage
        bucket["avg_trace"] += row.scores.trace_score
        bucket["avg_doc_recall"] += row.scores.required_doc_recall
        bucket["avg_answer_recall"] += row.scores.answer_point_recall
    for bucket in themes.values():
        count = bucket["count"] or 1
        bucket["avg_coverage"] = round(bucket["avg_coverage"] / count, 3)
        bucket["avg_trace"] = round(bucket["avg_trace"] / count, 3)
        bucket["avg_doc_recall"] = round(bucket["avg_doc_recall"] / count, 3)
        bucket["avg_answer_recall"] = round(bucket["avg_answer_recall"] / count, 3)
    return {
        "total_queries": total,
        "completed_queries": len(ok),
        "errors": total - len(ok),
        "answer_status": dict(sorted(statuses.items())),
        "auto_verdict": dict(sorted(verdicts.items())),
        "judge_verdict": dict(sorted(judge_verdicts.items())),
        "effective_verdict": dict(sorted(effective_verdicts.items())),
        "avg_coverage": round(sum(row.coverage for row in ok) / len(ok), 3) if ok else 0.0,
        "avg_trace_score": round(sum(row.scores.trace_score for row in ok) / len(ok), 3) if ok else 0.0,
        "avg_required_doc_recall": round(sum(row.scores.required_doc_recall for row in ok) / len(ok), 3) if ok else 0.0,
        "avg_answer_point_recall": round(sum(row.scores.answer_point_recall for row in ok) / len(ok), 3) if ok else 0.0,
        "avg_time_s": round(sum(row.total_s for row in ok) / len(ok), 1) if ok else 0.0,
        "by_theme": dict(sorted(themes.items())),
    }


def format_summary_markdown(summary: dict[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "This report evaluates the full Agentic RAG pipeline. Gold metadata is used only after the answer is generated; the runtime receives only the user question.",
        "",
        "## Aggregate",
        f"- Total queries: {summary.get('total_queries', 0)}",
        f"- Completed: {summary.get('completed_queries', 0)}",
        f"- Errors: {summary.get('errors', 0)}",
        f"- Avg coverage: {_percent(float(summary.get('avg_coverage', 0.0)))}",
        f"- Avg agentic trace score: {_percent(float(summary.get('avg_trace_score', 0.0)))}",
        f"- Avg required-doc recall: {_percent(float(summary.get('avg_required_doc_recall', 0.0)))}",
        f"- Avg answer-point recall: {_percent(float(summary.get('avg_answer_point_recall', 0.0)))}",
        f"- Avg time/query: {summary.get('avg_time_s', 0.0)}s",
        "",
        "## Status Counts",
    ]
    for status, count in (summary.get("answer_status") or {}).items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "## Auto Verdict Counts"])
    for verdict, count in (summary.get("auto_verdict") or {}).items():
        lines.append(f"- `{verdict}`: {count}")
    if summary.get("judge_verdict"):
        lines.extend(["", "## Judge Verdict Counts"])
        for verdict, count in (summary.get("judge_verdict") or {}).items():
            lines.append(f"- `{verdict}`: {count}")
    lines.extend(["", "## Effective Verdict Counts"])
    for verdict, count in (summary.get("effective_verdict") or {}).items():
        lines.append(f"- `{verdict}`: {count}")
    lines.extend(
        [
            "",
            "## By Theme",
            "| Theme | Count | Coverage | Trace | Doc Recall | Answer Recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for theme, row in (summary.get("by_theme") or {}).items():
        lines.append(
            f"| {theme} | {row['count']} | {_percent(row['avg_coverage'])} | {_percent(row['avg_trace'])} | {_percent(row['avg_doc_recall'])} | {_percent(row['avg_answer_recall'])} |"
        )
    return "\n".join(lines) + "\n"


def write_summary_markdown(path: str | Path, summary: dict[str, Any], *, title: str) -> Path:
    return atomic_write_text(path, format_summary_markdown(summary, title=title))


def write_public_csv(path: str | Path, results: list[PerQueryResult]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "theme",
                "difficulty",
                "answer_status",
                "auto_verdict",
                "coverage",
                "trace_score",
                "required_doc_recall",
                "answer_point_recall",
                "iterations",
                "total_s",
                "judge_verdict",
                "effective_verdict",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "id": row.id,
                    "theme": row.theme,
                    "difficulty": row.difficulty,
                    "answer_status": row.answer_status,
                    "auto_verdict": row.auto_verdict,
                    "coverage": row.coverage,
                    "trace_score": row.scores.trace_score,
                    "required_doc_recall": row.scores.required_doc_recall,
                    "answer_point_recall": row.scores.answer_point_recall,
                    "iterations": row.iterations,
                    "total_s": row.total_s,
                    "judge_verdict": row.judgement.verdict if row.judgement else "",
                    "effective_verdict": row.judgement.verdict if row.judgement else row.auto_verdict,
                }
            )
    text = tmp.read_text(encoding="utf-8")
    text = repair_mojibake(text)
    assert_no_secrets(text)
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, target)
    return target

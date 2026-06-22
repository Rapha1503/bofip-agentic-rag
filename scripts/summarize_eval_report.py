from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_agentic.eval_artifacts import assert_no_secrets, ensure_parent
from bofip_agentic.eval_schema import redact_secrets

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "evaluation" / "latest"
PUBLIC_CONFIG_KEYS = ("run_id", "provider", "model", "corpus", "limit", "lexical_only", "git_commit")
FORBIDDEN_PUBLIC_LABELS = re.compile(r"(?i)(?:[a-z0-9_]*api[_-]?key|authorization|x-api-key)\b")
HEADER_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?<![a-z0-9_])(?:authorization|x-api-key|[a-z0-9_]*api[_-]?key)\b"
    r"\s*[:=]?\s*(?:bearer\s+)?[^\s,;]+"
)
PUBLIC_SUMMARY_KEYS = (
    "total_queries",
    "supported",
    "partial",
    "insufficient_evidence",
    "errors",
    "avg_coverage",
)
PUBLIC_LATENCY_KEYS = ("avg", "p50", "p95")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on {path}:{line_number}")
            rows.append(payload)
    return rows


def _sanitize_text(value: Any) -> str:
    text = HEADER_SECRET_ASSIGNMENT.sub("[REDACTED_SECRET]", str(value or ""))
    text = redact_secrets(text)
    text = HEADER_SECRET_ASSIGNMENT.sub("[REDACTED_SECRET]", text)
    return FORBIDDEN_PUBLIC_LABELS.sub("[REDACTED_SECRET]", text)


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_json(item)
            for key, item in value.items()
            if not FORBIDDEN_PUBLIC_LABELS.search(str(key))
        }
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _assert_public_text(text: str) -> None:
    assert_no_secrets(text)
    if FORBIDDEN_PUBLIC_LABELS.search(text):
        raise ValueError("Credential-like label found in public evaluation artifact")


def _write_public_text(path: Path, text: str) -> None:
    sanitized = _sanitize_text(text)
    _assert_public_text(sanitized)
    ensure_parent(path).write_text(sanitized, encoding="utf-8")


def _write_public_json(path: Path, payload: Any) -> None:
    sanitized = _sanitize_json(payload)
    text = json.dumps(sanitized, ensure_ascii=False, indent=2)
    _assert_public_text(text)
    ensure_parent(path).write_text(text + "\n", encoding="utf-8")


def _public_summary(summary_payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    config = summary_payload.get("config", {})
    if not isinstance(config, dict):
        config = {}
    public_config = {key: config[key] for key in PUBLIC_CONFIG_KEYS if key in config}
    metrics = summary_payload.get("summary", {})
    if not isinstance(metrics, dict):
        metrics = {}
    public_metrics = {key: metrics[key] for key in PUBLIC_SUMMARY_KEYS if key in metrics}
    latency = metrics.get("latency_s", {})
    if isinstance(latency, dict):
        public_metrics["latency_s"] = {key: latency[key] for key in PUBLIC_LATENCY_KEYS if key in latency}
    public_metrics["public_query_rows"] = len(rows)
    public: dict[str, Any] = {
        "generated_at": summary_payload.get("generated_at", ""),
        "config": public_config,
        "summary": public_metrics,
        "public_artifacts": {
            "summary": "summary.md",
            "per_query": "per_query_public.csv",
            "failure_review": "failure_review.md",
        },
    }
    return public


def _percent(value: Any) -> str:
    try:
        return f"{round(float(value) * 100):d}%"
    except (TypeError, ValueError):
        return "0%"


def _source_refs(row: dict[str, Any]) -> str:
    refs = row.get("retrieved_docs") or []
    if not refs and isinstance(row.get("sources"), list):
        refs = [source.get("boi_reference", "") for source in row["sources"] if isinstance(source, dict)]
    clean_refs = [_sanitize_text(ref) for ref in refs if str(ref or "").strip()]
    return "; ".join(dict.fromkeys(clean_refs))


def _public_row(row: dict[str, Any]) -> dict[str, str]:
    missing_axes = row.get("axes_manquants") or []
    return {
        "id": _sanitize_text(row.get("id", "")),
        "theme": _sanitize_text(row.get("theme", "")),
        "difficulty": _sanitize_text(row.get("difficulty", "")),
        "question_type": _sanitize_text(row.get("question_type", "")),
        "answer_status": _sanitize_text(row.get("answer_status", "")),
        "coverage": str(row.get("coverage", "")),
        "iterations": str(row.get("iterations", "")),
        "total_s": str(row.get("total_s", "")),
        "retrieved_docs": _source_refs(row),
        "missing_axes": "; ".join(_sanitize_text(item) for item in missing_axes),
        "conclusion": _sanitize_text(row.get("conclusion", "")),
    }


def _write_per_query_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "theme",
        "difficulty",
        "question_type",
        "answer_status",
        "coverage",
        "iterations",
        "total_s",
        "retrieved_docs",
        "missing_axes",
        "conclusion",
    ]
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_public_row(row))
    _assert_public_text(target.read_text(encoding="utf-8"))


def _write_summary_markdown(path: Path, public_summary: dict[str, Any]) -> None:
    metrics = public_summary.get("summary", {})
    latency = metrics.get("latency_s", {}) if isinstance(metrics.get("latency_s"), dict) else {}
    lines = [
        "# BOFiP Agentic RAG Evaluation Summary",
        "",
        f"- Total queries: {metrics.get('total_queries', 0)}",
        f"- Supported: {metrics.get('supported', 0)}",
        f"- Partial: {metrics.get('partial', 0)}",
        f"- Insufficient evidence: {metrics.get('insufficient_evidence', 0)}",
        f"- Errors: {metrics.get('errors', 0)}",
        f"- Average coverage: {_percent(metrics.get('avg_coverage', 0))}",
        f"- Average latency: {latency.get('avg', 0)}s",
        "",
        "Public artifacts omit raw agent logs, prompts, environment values, and source snippets.",
        "",
    ]
    _write_public_text(path, "\n".join(lines))


def _write_failure_review(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [
        row
        for row in rows
        if row.get("error") or str(row.get("answer_status", "")).lower() not in {"supported", "ok"}
    ]
    lines = ["# Failure Review", ""]
    if not failures:
        lines.extend(["No public failures to review.", ""])
    for row in failures:
        public = _public_row(row)
        lines.extend(
            [
                f"## {public['id']}",
                "",
                f"- Status: {public['answer_status'] or 'unknown'}",
                f"- Coverage: {_percent(row.get('coverage', 0))}",
                f"- Missing axes: {public['missing_axes'] or 'None reported'}",
                f"- Retrieved docs: {public['retrieved_docs'] or 'None reported'}",
                "",
                public["conclusion"] or "No public conclusion returned.",
                "",
            ]
        )
    _write_public_text(path, "\n".join(lines))


def summarize_eval_report(run_dir: str | Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    run_path = Path(run_dir)
    out_path = Path(output_dir)
    summary_path = run_path / "summary.json"
    per_query_path = run_path / "per_query.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not per_query_path.exists():
        raise FileNotFoundError(f"Missing {per_query_path}")

    rows = _load_jsonl(per_query_path)
    public_summary = _public_summary(_load_json(summary_path), rows)
    out_path.mkdir(parents=True, exist_ok=True)
    _write_public_json(out_path / "summary.json", public_summary)
    _write_summary_markdown(out_path / "summary.md", public_summary)
    _write_per_query_csv(out_path / "per_query_public.csv", rows)
    _write_failure_review(out_path / "failure_review.md", rows)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write sanitized public evaluation reports.")
    parser.add_argument("--run-dir", required=True, help="Evaluation run directory containing summary.json and per_query.jsonl.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Public report output directory. Defaults to docs/evaluation/latest.",
    )
    args = parser.parse_args(argv)
    output_dir = summarize_eval_report(args.run_dir, args.output_dir)
    print(f"Wrote public evaluation report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

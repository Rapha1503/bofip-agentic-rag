from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.llm_preview import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_PREVIEW_PROVIDER,
    PREVIEW_ANSWER_CONTRACT_VERSION,
    build_citation_prompt,
    generate_preview_answer_with_retry,
    preview_row_is_valid,
    review_batch_preview_payload,
)
from bofip_cleanroom.preview_runtime import DEFAULT_PREVIEW_CORPUS, Phase8bPreviewRuntime
from bofip_cleanroom.settings import INTERIM_DIR, REPORTS_DIR, ensure_data_dirs


def _default_review_path(report_path: Path) -> Path:
    filename = report_path.name
    if "preview_eval" in filename:
        return report_path.with_name(filename.replace("preview_eval", "review"))
    return report_path.with_name(report_path.stem + "__review.json")


def _error_row(
    *,
    case: dict,
    provider: str,
    model: str,
    error: BaseException,
    prompt_text: str = "",
    retrieval: dict | None = None,
) -> dict:
    message = f"{type(error).__name__}: {error}"
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "category": case["category"],
        "note": case.get("note", ""),
        "provider": provider,
        "model": model,
        "api_called": False,
        "answer_text": message,
        "raw_answer_text": message,
        "structured_answer": None,
        "answer_validation": {
            "valid": False,
            "answer_status": None,
            "has_conclusion": False,
            "has_justification": False,
            "has_limits": False,
            "bullet_count": 0,
            "citation_ids": [],
            "citation_count": 0,
            "with_any_citation": False,
            "errors": [message],
            "warnings": [],
            "parsed_from": None,
        },
        "response_metadata": {},
        "attempt_count": 0,
        "prompt_text": prompt_text,
        "retrieval": retrieval or {},
        "runtime_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small phase9 batch shadow eval.")
    parser.add_argument("--input", type=str, default=str(INTERIM_DIR / "phase9_shadow_cases_v1.jsonl"))
    parser.add_argument("--corpus", type=str, default=DEFAULT_PREVIEW_CORPUS, choices=["commentary"])
    parser.add_argument("--provider", type=str, default=DEFAULT_PREVIEW_PROVIDER, choices=["gemini", "openai"])
    parser.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-delay-seconds", type=float, default=5.0)
    parser.add_argument("--case-ids", type=str, default="", help="Comma-separated case ids to run.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of cases after filtering.")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--review-output", type=str, default="")
    parser.add_argument(
        "--resume-input",
        type=str,
        default="",
        help="Optional existing preview report. Valid rows from that report are reused instead of calling the API again.",
    )
    args = parser.parse_args()

    ensure_data_dirs()
    cases = read_jsonl(Path(args.input))
    if args.case_ids:
        selected_ids = {value.strip() for value in args.case_ids.split(",") if value.strip()}
        cases = [case for case in cases if case["case_id"] in selected_ids]
    if args.limit > 0:
        cases = cases[: args.limit]

    reused_case_count = 0
    executed_case_count = 0
    resumed_rows_by_case_id: dict[str, dict] = {}
    resume_input_path: Path | None = None
    if args.resume_input:
        resume_input_path = Path(args.resume_input).resolve()
        resume_payload = _load_json(resume_input_path)
        if resume_payload.get("corpus") == args.corpus:
            resumed_rows_by_case_id = {
                row["case_id"]: row
                for row in resume_payload.get("rows", [])
                if preview_row_is_valid(row, provider=args.provider, model=args.model)
            }

    runtime: Phase8bPreviewRuntime | None = None
    rows = []
    for case in cases:
        resumed_row = resumed_rows_by_case_id.get(case["case_id"])
        if resumed_row is not None:
            rows.append(resumed_row)
            reused_case_count += 1
            continue

        if runtime is None:
            runtime = Phase8bPreviewRuntime.from_local_corpus(corpus=args.corpus, device=args.device)
        try:
            retrieval = runtime.retrieve(
                case["query"],
                top_docs=args.top_docs,
                chunks_per_doc=args.chunks_per_doc,
                max_chunks=args.max_chunks,
            )
        except Exception as exc:
            rows.append(
                _error_row(
                    case=case,
                    provider=args.provider,
                    model=args.model,
                    error=exc,
                )
            )
            executed_case_count += 1
            continue

        try:
            preview = generate_preview_answer_with_retry(
                retrieval,
                provider=args.provider,
                model=args.model,
                max_attempts=args.max_attempts,
                base_delay_seconds=args.base_delay_seconds,
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "category": case["category"],
                    "note": case.get("note", ""),
                    "provider": args.provider,
                    "model": args.model,
                    "api_called": preview.api_called,
                    "answer_text": preview.answer_text,
                    "raw_answer_text": preview.raw_answer_text,
                    "structured_answer": preview.structured_answer,
                    "answer_validation": preview.answer_validation,
                    "response_metadata": preview.response_metadata,
                    "attempt_count": preview.attempt_count,
                    "prompt_text": preview.prompt_text,
                    "retrieval": preview.retrieval_payload,
                }
            )
        except Exception as exc:
            rows.append(
                _error_row(
                    case=case,
                    provider=args.provider,
                    model=args.model,
                    error=exc,
                    prompt_text=build_citation_prompt(retrieval),
                    retrieval=Phase8bPreviewRuntime.as_dict(retrieval),
                )
            )
        executed_case_count += 1

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "corpus": args.corpus,
        "answer_contract_version": PREVIEW_ANSWER_CONTRACT_VERSION,
        "resume_input": None if resume_input_path is None else str(resume_input_path),
        "reused_case_count": reused_case_count,
        "executed_case_count": executed_case_count,
        "case_count": len(rows),
        "rows": rows,
    }
    report_path = Path(args.output).resolve() if args.output else REPORTS_DIR / "phase9_batch_preview_eval.json"
    write_json(report_path, payload)
    review_payload = review_batch_preview_payload(
        {
            **payload,
            "source_report": str(report_path),
        }
    )
    review_path = Path(args.review_output).resolve() if args.review_output else _default_review_path(report_path)
    write_json(review_path, review_payload)
    print(f"Phase9 batch preview written to: {report_path}")
    print(f"Phase9 batch review written to: {review_path}")
    print(f"case_count = {len(rows)}")
    print(f"reused_case_count = {reused_case_count}")
    print(f"executed_case_count = {executed_case_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

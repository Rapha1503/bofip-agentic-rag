from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.jsonio import write_json
from bofip_cleanroom.llm_preview import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_PREVIEW_PROVIDER,
    PREVIEW_ANSWER_CONTRACT_VERSION,
    generate_preview_answer,
)
from bofip_cleanroom.preview_runtime import DEFAULT_PREVIEW_CORPUS, Phase8bPreviewRuntime
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated phase9 LLM preview on top of phase8b retrieval.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--corpus", type=str, default=DEFAULT_PREVIEW_CORPUS, choices=["commentary"])
    parser.add_argument("--provider", type=str, default=DEFAULT_PREVIEW_PROVIDER, choices=["gemini", "openai"])
    parser.add_argument("--model", type=str, default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-docs", type=int, default=5)
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ensure_data_dirs()
    runtime = Phase8bPreviewRuntime.from_local_corpus(corpus=args.corpus, device=args.device)
    retrieval = runtime.retrieve(
        args.query,
        top_docs=args.top_docs,
        chunks_per_doc=args.chunks_per_doc,
        max_chunks=args.max_chunks,
    )
    preview = generate_preview_answer(retrieval, provider=args.provider, model=args.model)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "query": args.query,
        "corpus": args.corpus,
        "provider": args.provider,
        "llm_model": args.model,
        "answer_contract_version": PREVIEW_ANSWER_CONTRACT_VERSION,
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

    report_path = (
        Path(args.output).resolve()
        if args.output
        else REPORTS_DIR / "phase9_preview_answer.json"
    )
    write_json(report_path, payload)
    print(f"Phase9 preview written to: {report_path}")
    print(f"api_called = {preview.api_called}")
    print(preview.answer_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

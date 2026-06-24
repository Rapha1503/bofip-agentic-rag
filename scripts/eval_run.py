from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_agentic.eval_runner import build_run_id, run_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BOFiP Agentic RAG evaluation with stable artifacts.")
    parser.add_argument("--question-bank", default=str(PROJECT_ROOT / "data" / "eval" / "chatgpt_50_cases_v1.jsonl"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "eval-runs"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--provider", default="codex", help="codex, deepseek, openai, mistral, google")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--judge-provider", default="none", help="none, codex, deepseek, openai, mistral, google")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0, help="Random sample size before limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-mode", choices=["lexical", "hybrid"], default="lexical")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    label = "chatgpt50" if not args.limit and not args.sample else f"pilot{args.sample or args.limit}"
    run_id = args.run_id or build_run_id(f"{label}_{args.provider}_{args.retrieval_mode}")
    output_dir = Path(args.output_root) / run_id

    result = run_eval(
        question_bank=args.question_bank,
        output_dir=output_dir,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_api_key=args.judge_api_key,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
        retrieval_mode=args.retrieval_mode,
        reranker=args.reranker,
        device=args.device,
        max_iterations=args.max_iterations,
        resume=args.resume,
    )
    print(f"\nRun dir: {result['run_dir']}")
    print(f"Summary: {Path(result['run_dir']) / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


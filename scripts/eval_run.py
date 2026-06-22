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

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = PROJECT_ROOT / "data" / "eval" / "bofip_agentic_rag_50_human_questions_v2.json"
DEFAULT_PILOT_IDS: tuple[str, ...] = (
    "CASE-006",
    "CASE-010",
    "CASE-017",
    "CASE-023",
    "CASE-026",
    "CASE-027",
    "CASE-029",
    "CASE-032",
    "CASE-037",
    "CASE-041",
    "CASE-042",
    "CASE-043",
    "CASE-045",
    "CASE-048",
    "CASE-050",
)


@dataclass(frozen=True)
class ReviewVariant:
    name: str
    source_review_mode: str = "full"
    chunk_limit: int = 16
    text_limit: int = 900
    post_relaunch_review: bool = True
    max_missing_axes: int = 3
    max_iterations: int | None = None


DEFAULT_VARIANTS: tuple[ReviewVariant, ...] = (
    ReviewVariant("full16"),
    ReviewVariant("full12", chunk_limit=12),
    ReviewVariant("full8", chunk_limit=8),
    ReviewVariant("full8_short500", chunk_limit=8, text_limit=500),
    ReviewVariant("full6_short400", chunk_limit=6, text_limit=400),
    ReviewVariant("review16_onepass", max_iterations=1),
    ReviewVariant("review8_onepass", chunk_limit=8, max_iterations=1),
    ReviewVariant("initial16", source_review_mode="initial_only", post_relaunch_review=False),
    ReviewVariant("initial8", source_review_mode="initial_only", chunk_limit=8, post_relaunch_review=False),
    ReviewVariant("one_axis16", max_missing_axes=1),
    ReviewVariant("one_axis8_initial", source_review_mode="initial_only", chunk_limit=8, post_relaunch_review=False, max_missing_axes=1),
    ReviewVariant("no_review", source_review_mode="none", post_relaunch_review=False, max_missing_axes=0, max_iterations=1),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-review ablation variants on the same BOFiP eval sample.")
    parser.add_argument("--question-bank", default=str(DEFAULT_BANK))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "output" / "eval-runs"))
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--retrieval-mode", choices=["lexical", "hybrid"], default="lexical")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument(
        "--case-ids",
        default=",".join(DEFAULT_PILOT_IDS),
        help="Comma-separated case IDs. Use 'random' to use --sample/--limit instead. Defaults to a frozen stratified pilot.",
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names, or 'all'. Defaults to all built-in variants.",
    )
    return parser.parse_args()


def selected_variants(value: str) -> list[ReviewVariant]:
    by_name = {variant.name: variant for variant in DEFAULT_VARIANTS}
    if value.strip().lower() == "all":
        return list(DEFAULT_VARIANTS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}. Known: {', '.join(by_name)}")
    return [by_name[name] for name in names]


def command_for(args: argparse.Namespace, variant: ReviewVariant, run_id: str) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_run.py"),
        "--question-bank",
        str(args.question_bank),
        "--output-root",
        str(args.output_root),
        "--run-id",
        run_id,
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--retrieval-mode",
        args.retrieval_mode,
        "--device",
        args.device,
        "--max-iterations",
        str(variant.max_iterations or args.max_iterations),
        "--source-review-mode",
        variant.source_review_mode,
        "--source-review-chunk-limit",
        str(variant.chunk_limit),
        "--source-review-text-limit",
        str(variant.text_limit),
        "--max-missing-axes",
        str(variant.max_missing_axes),
    ]
    case_ids = str(args.case_ids or "").strip()
    if case_ids and case_ids.lower() not in {"random", "none"}:
        command.extend(["--case-ids", case_ids])
    elif args.sample:
        command.extend(["--sample", str(args.sample), "--seed", str(args.seed)])
    elif args.limit:
        command.extend(["--limit", str(args.limit)])
    if not variant.post_relaunch_review:
        command.append("--no-post-relaunch-review")
    if args.resume:
        command.append("--resume")
    return command


def main() -> int:
    args = parse_args()
    variants = selected_variants(args.variants)
    using_fixed_case_ids = bool(str(args.case_ids or "").strip()) and str(args.case_ids).strip().lower() not in {
        "random",
        "none",
    }
    if using_fixed_case_ids and (args.sample or args.limit):
        raise SystemExit("--sample/--limit require --case-ids random; default mode uses a frozen stratified pilot.")
    if args.provider.lower() == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY must be set in the environment; it is not accepted as a process argument here.")

    stamp = args.run_prefix or datetime.now().strftime("source_review_ablation_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["BOFIP_EVAL_ABLATION_ID"] = stamp

    pending = list(variants)
    running: dict[subprocess.Popen, tuple[ReviewVariant, Path]] = {}
    completed: list[tuple[str, int, Path]] = []
    parallel = max(1, int(args.parallel))
    if using_fixed_case_ids:
        print(f"case_ids={args.case_ids}", flush=True)

    while pending or running:
        while pending and len(running) < parallel:
            variant = pending.pop(0)
            run_id = f"{stamp}_{variant.name}"
            run_dir = output_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_dir / "stdout.log"
            stderr_path = run_dir / "stderr.log"
            command = command_for(args, variant, run_id)
            stdout = stdout_path.open("a", encoding="utf-8")
            stderr = stderr_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            running[process] = (variant, run_dir)
            print(f"started {variant.name} pid={process.pid} run_dir={run_dir}", flush=True)

        time.sleep(5)
        for process in list(running):
            status = process.poll()
            if status is None:
                continue
            variant, run_dir = running.pop(process)
            completed.append((variant.name, int(status), run_dir))
            print(f"finished {variant.name} exit={status} summary={run_dir / 'summary.md'}", flush=True)

    failed = [name for name, status, _run_dir in completed if status != 0]
    comparison_paths = write_comparison(output_root, stamp, completed)
    print("\nCompleted variants:")
    for name, status, _run_dir in completed:
        print(f"- {name}: exit={status}")
    print(f"Comparison JSON: {comparison_paths[0]}")
    print(f"Comparison Markdown: {comparison_paths[1]}")
    return 1 if failed else 0


def write_comparison(output_root: Path, stamp: str, completed: list[tuple[str, int, Path]]) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    for name, status, run_dir in sorted(completed, key=lambda item: item[0]):
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "run_manifest.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        rows.append(
            {
                "variant": name,
                "exit": status,
                "run_dir": str(run_dir),
                "source_review_mode": manifest.get("source_review_mode"),
                "source_review_chunk_limit": manifest.get("source_review_chunk_limit"),
                "source_review_text_limit": manifest.get("source_review_text_limit"),
                "post_relaunch_review": manifest.get("post_relaunch_review"),
                "max_missing_axes": manifest.get("max_missing_axes"),
                "max_iterations": manifest.get("max_iterations"),
                "expected_queries": summary.get("expected_queries"),
                "total_queries": summary.get("total_queries"),
                "is_complete": summary.get("is_complete"),
                "errors": summary.get("errors"),
                "avg_time_s": summary.get("avg_time_s"),
                "avg_coverage": summary.get("avg_coverage"),
                "avg_trace_score": summary.get("avg_trace_score"),
                "avg_required_doc_recall": summary.get("avg_required_doc_recall"),
                "avg_answer_point_recall": summary.get("avg_answer_point_recall"),
                "auto_verdict": summary.get("auto_verdict") or {},
                "answer_status": summary.get("answer_status") or {},
            }
        )
    json_path = output_root / f"{stamp}_comparison.json"
    md_path = output_root / f"{stamp}_comparison.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Source Review Ablation Comparison",
        "",
        "Same question set per variant. Gold metadata is used only after generation for scoring; runtime receives only the user question.",
        "",
        "| Variant | Complete | Errors | Avg s | Coverage | Trace | Doc recall | Answer recall | Verdicts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        verdicts = ", ".join(f"{key}:{value}" for key, value in sorted(dict(row["auto_verdict"]).items()))
        lines.append(
            "| {variant} | {complete} | {errors} | {avg_s} | {cov} | {trace} | {doc} | {answer} | {verdicts} |".format(
                variant=row["variant"],
                complete=row["is_complete"],
                errors=row["errors"],
                avg_s=row["avg_time_s"],
                cov=row["avg_coverage"],
                trace=row["avg_trace_score"],
                doc=row["avg_required_doc_recall"],
                answer=row["avg_answer_point_recall"],
                verdicts=verdicts,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


if __name__ == "__main__":
    raise SystemExit(main())

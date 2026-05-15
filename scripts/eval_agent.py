"""
Agentic RAG evaluation — self-reporting, no manual labeling, incremental save.

Usage:
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py --resume
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py --limit 10
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-1422bc6ce16e41fb8123f4a1723cfa49")

from bofip_agentic.agent_rag import AgenticRAG
from bofip_agentic.rag_runtime import RagRuntime
from bofip_agentic.settings import REPORTS_DIR, ensure_data_dirs

INPUT_PATH = PROJECT_ROOT / "data" / "eval" / "tax_eval_50.jsonl"
OUTPUT_PATH = REPORTS_DIR / "eval_agent_50.json"


def load_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_summary(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"total": 0}

    coverage = sum(r.get("coverage", 0) for r in ok) / len(ok)
    iters = sum(r.get("iterations", 1) for r in ok) / len(ok)
    total_s = sum(r.get("total_s", 0) for r in ok) / len(ok)

    statuses = dict(Counter(r.get("answer_status", "?") for r in ok))
    supported = statuses.get("supported", 0)
    partial = statuses.get("partial", 0)
    insufficient = statuses.get("insufficient_evidence", 0)
    reformed = sum(1 for r in ok if r.get("iterations", 1) > 1)

    themes = {}
    for r in ok:
        t = r.get("theme", "unknown")
        themes.setdefault(t, {"count": 0, "coverage_sum": 0, "supported": 0, "partial": 0})
        themes[t]["count"] += 1
        themes[t]["coverage_sum"] += r.get("coverage", 0)
        if r.get("answer_status") == "supported":
            themes[t]["supported"] += 1
        elif r.get("answer_status") == "partial":
            themes[t]["partial"] += 1

    by_theme = {}
    for t, v in sorted(themes.items()):
        by_theme[t] = {
            "count": v["count"],
            "avg_coverage": round(v["coverage_sum"] / v["count"], 3) if v["count"] else 0,
            "supported": v["supported"],
            "partial": v["partial"],
        }

    return {
        "total_queries": len(ok),
        "avg_coverage": round(coverage, 3),
        "avg_iterations": round(iters, 1),
        "avg_time_s": round(total_s, 1),
        "supported": supported,
        "partial": partial,
        "insufficient_evidence": insufficient,
        "reformulated": "{} / {}".format(reformed, len(ok)),
        "by_theme": by_theme,
    }


def main():
    p = argparse.ArgumentParser(description="Agentic RAG self-evaluating benchmark")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: set DEEPSEEK_API_KEY")
        return 1

    ensure_data_dirs()
    queries = load_queries(INPUT_PATH)
    if args.limit > 0:
        queries = queries[:args.limit]

    done_ids = set()
    results = []
    out_path = Path(args.output) if args.output else OUTPUT_PATH

    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
            for r in prev.get("per_query", []):
                done_ids.add(r["id"])
                results.append(r)
        print("Resumed: {} done, {} remaining".format(len(done_ids), len(queries) - len(done_ids)))
    else:
        print("Loaded {} queries from {}".format(len(queries), INPUT_PATH))

    pending = [q for q in queries if q["id"] not in done_ids]
    if not pending:
        summary = compute_summary(results)
        print_summary(summary)
        return 0

    print("Init RagRuntime (GPU)...")
    rt = RagRuntime.from_local_corpus(corpus="commentary", device=args.device)
    agent = AgenticRAG(rt, api_key=api_key, max_iterations=2)
    print("Agent ready.\n")

    for idx, q in enumerate(pending, 1):
        qid = q["id"]
        question = q["question"]
        total_done = len(done_ids) + idx
        print("[{}/{}] {} ({} / {})...".format(total_done, len(queries), qid, q.get("theme", "?"), q.get("difficulty", "?")), flush=True)

        t0 = time.time()
        try:
            r = agent.run(question)
            r["id"] = qid
            r["question"] = question[:120]
            r["theme"] = q.get("theme", "")
            r["difficulty"] = q.get("difficulty", "")
            r["question_type"] = q.get("question_type", "")
            results.append(r)

            s = r["answer_status"]
            cov = r["coverage"]
            it = r["iterations"]
            print("  -> {} | coverage={:.0%} | {} iter(s) | {:.1f}s".format(s, cov, it, time.time() - t0), flush=True)
        except Exception as e:
            results.append({"id": qid, "error": str(e), "question": question[:120]})
            print("  -> ERROR: {}".format(e), flush=True)

        # Incremental save
        summary = compute_summary(results)
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "config": {"device": args.device, "max_iterations": 2, "model": "deepseek-chat"},
            "summary": summary,
            "per_query": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    summary = compute_summary(results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {"device": args.device, "max_iterations": 2, "model": "deepseek-chat"},
        "summary": summary,
        "per_query": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print("\nReport: {}".format(out_path))
    return 0


def print_summary(s: dict):
    if not s.get("total_queries"):
        return
    print("\n" + "=" * 60)
    print("AGENTIC RAG EVALUATION — {} queries".format(s["total_queries"]))
    print("=" * 60)
    print("Avg coverage:        {:.0%}".format(s["avg_coverage"]))
    print("Avg iterations:      {}".format(s["avg_iterations"]))
    print("Avg time/query:      {:.0f}s".format(s["avg_time_s"]))
    print("Supported:           {}".format(s["supported"]))
    print("Partial:             {}".format(s["partial"]))
    print("Insufficient:        {}".format(s["insufficient_evidence"]))
    print("Reformulated:        {}".format(s["reformulated"]))
    print()
    print("By theme:")
    print("{:<20} {:>6} {:>10} {:>10}".format("Theme", "Count", "Coverage", "Supported"))
    for t, v in s.get("by_theme", {}).items():
        print("{:<20} {:>6} {:>9.0%} {:>9}".format(t, v["count"], v["avg_coverage"], "{}/{}".format(v["supported"], v["count"])))


if __name__ == "__main__":
    raise SystemExit(main())

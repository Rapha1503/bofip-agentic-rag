"""
Self-evaluating agent benchmark — no manual labeling needed.

Usage:
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py --limit 3
"""
from __future__ import annotations

import argparse, json, os, sys, time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_agentic.agent_rag import AgenticRAG
from bofip_agentic.rag_runtime import RagRuntime
from bofip_agentic.settings import ensure_data_dirs, REPORTS_DIR

INPUT_PATH = PROJECT_ROOT / "data" / "eval" / "pilot_5.jsonl"


def load_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: set DEEPSEEK_API_KEY environment variable")
        return 1

    ensure_data_dirs()
    queries = load_queries(INPUT_PATH)
    if args.limit > 0:
        queries = queries[:args.limit]
    print(f"Loaded {len(queries)} queries from {INPUT_PATH}")

    print("Init RagRuntime...", flush=True)
    rt = RagRuntime.from_local_corpus(corpus="commentary", device=args.device)

    agent = AgenticRAG(rt, api_key=api_key, max_iterations=2)
    print("Agent ready.\n", flush=True)

    results = []
    for idx, q in enumerate(queries, 1):
        qid = q["id"]
        question = q["question"]
        print(f"[{idx}/{len(queries)}] {qid}: {question[:80]}...", flush=True)
        t0 = time.time()
        try:
            r = agent.run(question)
            r["id"] = qid
            r["theme"] = q.get("theme", "")
            r["difficulty"] = q.get("difficulty", "")
            r["expected_status"] = q.get("expected_status", "")
            results.append(r)
            print(f"  → {r['answer_status']} | coverage={r['coverage']:.0%} | "
                  f"iters={r['iterations']} | {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            results.append({"id": qid, "error": str(e), "question": question})
            print(f"  → ERROR: {e}", flush=True)

    # ---- Summary ----
    ok = [r for r in results if "error" not in r]
    if not ok:
        return 1

    coverage_avg = sum(r["coverage"] for r in ok) / len(ok)
    supported = sum(1 for r in ok if r["answer_status"] == "supported")
    partial = sum(1 for r in ok if r["answer_status"] == "partial")
    insuff = sum(1 for r in ok if r["answer_status"] == "insufficient_evidence")
    reformed = sum(1 for r in ok if r["iterations"] > 1)
    avg_iters = sum(r["iterations"] for r in ok) / len(ok)
    avg_s = sum(r["total_s"] for r in ok) / len(ok)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {"device": args.device, "max_iterations": 2},
        "summary": {
            "total_queries": len(ok),
            "coverage_avg": round(coverage_avg, 3),
            "supported": supported,
            "partial": partial,
            "insufficient_evidence": insuff,
            "reformulation_rate": f"{reformed}/{len(ok)}",
            "avg_iterations": round(avg_iters, 2),
            "avg_time_s": round(avg_s, 1),
        },
        "per_query": results,
    }

    out = Path(args.output) if args.output else REPORTS_DIR / "eval_agent_{}.json".format(
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n=== RESULTS ===")
    print(f"Coverage:     {coverage_avg:.0%}")
    print(f"Supported:    {supported}/{len(ok)}")
    print(f"Partial:      {partial}/{len(ok)}")
    print(f"Reformulated: {reformed}/{len(ok)}")
    print(f"Avg iters:    {avg_iters:.1f}")
    print(f"Avg time:     {avg_s:.0f}s")
    print(f"Report:       {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

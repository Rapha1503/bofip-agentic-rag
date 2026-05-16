"""
Compare old agent (v1.0 tag) vs new agent (domain routing) on 10 random queries.
Tracks: answer_status, coverage, iterations, time, LLM calls.

Usage:
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/compare_agents.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-1422bc6ce16e41fb8123f4a1723cfa49")

from bofip_agentic.agent_rag import AgenticRAG
from bofip_agentic.rag_runtime import RagRuntime

INPUT_PATH = PROJECT_ROOT / "data" / "eval" / "tax_eval_50.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reports" / "compare_agents.json"


def load_queries(n: int = 10) -> list[dict]:
    with open(INPUT_PATH, encoding="utf-8") as f:
        all_q = [json.loads(line) for line in f if line.strip()]
    random.seed(42)
    return random.sample(all_q, min(n, len(all_q)))


def run_baseline(question: str, rt: RagRuntime, api_key: str) -> dict:
    """Single-pass: retrieve + answer (no reformulation loop)."""
    t0 = time.time()
    r = rt.retrieve(question, top_docs=8, max_chunks=8)
    chunks = [
        {"rank": i + 1, "boi_reference": c.boi_reference, "title": c.title,
         "publication_date": c.publication_date, "section_path": c.section_path,
         "text": c.text, "chunk_id": c.chunk_id}
        for i, c in enumerate(r.stage2_chunks)
    ]

    from bofip_agentic.agent_rag import _parse_json
    from bofip_agentic.prompt_utils import build_prompt

    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai not installed"}

    prompt = build_prompt(question, chunks)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "Assistant fiscal. JSON strict."},
                  {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=2800,
        response_format={"type": "json_object"},
    )
    answer = _parse_json(resp.choices[0].message.content or "") or {}
    coverage = (
        len(answer.get("axes_couverts", [])) / max(len(answer.get("axes_requis", [])), 1)
    )
    return {
        "answer_status": answer.get("answer_status", "?"),
        "coverage": round(coverage, 3),
        "total_s": round(time.time() - t0, 1),
        "llm_calls": 1,
        "iterations": 1,
        "sources_count": len(chunks),
    }


def run_agent(question: str, agent: AgenticRAG) -> dict:
    """Run full agent loop."""
    t0 = time.time()
    result = agent.run(question)
    elapsed = time.time() - t0
    return {
        "answer_status": result.get("answer_status", "?"),
        "coverage": result.get("coverage", 0),
        "total_s": round(elapsed, 1),
        "llm_calls": result.get("iterations", 1) + 1,  # answer calls + reformulation calls
        "iterations": result.get("iterations", 1),
        "sources_count": result.get("chunks_used", 0),
        "trace": result.get("trace", []),
    }


def main():
    print("Loading 10 random queries...")
    queries = load_queries(10)
    print("Queries selected:")
    for q in queries:
        print("  {} [{}] {}: {}".format(q["id"], q["theme"], q["difficulty"], q["question"][:80]))

    print("\nInit RagRuntime (GPU)...")
    rt = RagRuntime.from_local_corpus(corpus="commentary", device="cuda")
    api_key = os.environ["DEEPSEEK_API_KEY"]

    agent = AgenticRAG(rt, api_key=api_key, max_iterations=3)

    results = []
    for idx, q in enumerate(queries, 1):
        qid = q["id"]
        question = q["question"]
        theme = q.get("theme", "")
        diff = q.get("difficulty", "")
        print("\n[{}/{}] {} ({} / {})".format(idx, len(queries), qid, theme, diff))
        print("  Q: {}".format(question[:100]))

        # Baseline (single pass, no loop)
        print("  Baseline...", end=" ", flush=True)
        b = run_baseline(question, rt, api_key)
        print("{} | cov={:.0%} | {:.1f}s".format(b["answer_status"], b["coverage"], b["total_s"]))

        # Agent (full loop with domain routing)
        print("  Agent...", end=" ", flush=True)
        a = run_agent(question, agent)
        trace_info = ""
        for t in a.get("trace", []):
            hint = t.get("branch_hint", "")
            conf = t.get("branch_confidence", 0)
            ref_q = t.get("reformulated_query", "")[:60]
            trace_info += " | iter{}:{}".format(t.get("iteration"), t.get("answer_status", "?"))
            if hint:
                trace_info += "+branch={}({:.0f}%)".format(hint, conf * 100)
        print("{} | cov={:.0%} | {:.1f}s | {} calls | {} iter{}".format(
            a["answer_status"], a["coverage"], a["total_s"],
            a["llm_calls"], a["iterations"], trace_info))

        results.append({
            "id": qid, "question": question[:120], "theme": theme, "difficulty": diff,
            "baseline": b, "agent": a,
        })

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    n = len(results)
    b_cov = sum(r["baseline"]["coverage"] for r in results) / n
    a_cov = sum(r["agent"]["coverage"] for r in results) / n
    b_time = sum(r["baseline"]["total_s"] for r in results) / n
    a_time = sum(r["agent"]["total_s"] for r in results) / n
    b_sup = sum(1 for r in results if r["baseline"]["answer_status"] == "supported")
    a_sup = sum(1 for r in results if r["agent"]["answer_status"] == "supported")
    a_calls = sum(r["agent"]["llm_calls"] for r in results) / n
    a_iters = sum(r["agent"]["iterations"] for r in results) / n
    a_branch = sum(1 for r in results if any(
        t.get("branch_hint") for t in r["agent"].get("trace", [])
    ))

    print("| Metric | Baseline | Agent (w/ routing) |")
    print("|---|---|---|")
    print("| Avg coverage | {:.0%} | {:.0%} |".format(b_cov, a_cov))
    print("| Avg time | {:.1f}s | {:.1f}s |".format(b_time, a_time))
    print("| Supported | {}/{} | {}/{} |".format(b_sup, n, a_sup, n))
    print("| Avg LLM calls | 1.0 | {:.1f} |".format(a_calls))
    print("| Avg iterations | 1.0 | {:.1f} |".format(a_iters))
    print("| Branch routing used | N/A | {}/{} queries |".format(a_branch, n))

    detail = []
    for r in results:
        b_ok = r["baseline"]["answer_status"] == "supported"
        a_ok = r["agent"]["answer_status"] == "supported"
        if not b_ok or not a_ok:
            detail.append(r)

    if detail:
        print("\nQueries where agents differ:")
        for r in detail:
            print("  {} | base={} agent={} | cov: {:.0%} -> {:.0%}".format(
                r["id"], r["baseline"]["answer_status"], r["agent"]["answer_status"],
                r["baseline"]["coverage"], r["agent"]["coverage"]))

    # Save
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "queries": n,
            "baseline_avg_coverage": round(b_cov, 3),
            "agent_avg_coverage": round(a_cov, 3),
            "baseline_avg_time": round(b_time, 1),
            "agent_avg_time": round(a_time, 1),
            "baseline_supported": b_sup,
            "agent_supported": a_sup,
            "agent_avg_llm_calls": round(a_calls, 1),
            "agent_avg_iterations": round(a_iters, 1),
            "branch_routing_used": a_branch,
        },
        "per_query": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nReport: {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    raise SystemExit(main())

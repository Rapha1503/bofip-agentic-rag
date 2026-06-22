"""Comprehensive evaluation of the full Agentic RAG pipeline.

Measures: retrieval recall, answer integrity, coverage, latency, reformulation
effectiveness — across themes, difficulties, and question types.

For stable review artifacts, prefer `python scripts/eval_run.py`.

Usage:
    $env:PYTHONPATH="src"
    $env:DEEPSEEK_API_KEY="sk-..."
    python scripts/eval_full.py --limit 3          # pilot
    python scripts/eval_full.py                    # full 50
    python scripts/eval_full.py --resume           # continue interrupted
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load .env files (same as app.py)
for _env in (PROJECT_ROOT / ".env.local", PROJECT_ROOT / ".env"):
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8-sig").splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _k and _k not in os.environ:
                os.environ[_k] = _v

PROVIDERS = {
    "deepseek":  {"base_url": "https://api.deepseek.com/v1",    "default_model": "deepseek-v4-flash",  "env_key": "DEEPSEEK_API_KEY"},
    "openai":    {"base_url": "https://api.openai.com/v1",       "default_model": "gpt-5.4-mini",      "env_key": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1",    "default_model": "claude-haiku-4-5",  "env_key": "ANTHROPIC_API_KEY"},
    "mistral":   {"base_url": "https://api.mistral.ai/v1",       "default_model": "mistral-small-4",   "env_key": "MISTRAL_API_KEY"},
    "google":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "default_model": "gemini-3.1-flash", "env_key": "GEMINI_API_KEY"},
    "groq":      {"base_url": "https://api.groq.com/openai/v1",  "default_model": "llama-4-maverick",  "env_key": "GROQ_API_KEY"},
    "together":  {"base_url": "https://api.together.xyz/v1",     "default_model": "meta-llama/Llama-4-Maverick", "env_key": "TOGETHER_API_KEY"},
}

INPUT_PATH = PROJECT_ROOT / "data" / "eval" / "tax_eval_50.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reports" / "eval_full.json"


def load_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_date(boi_ref: str) -> str:
    """Remove date suffix from BOI reference: 'BOI-RPPM-PVBMI-20-10-40-20250410' -> 'BOI-RPPM-PVBMI-20-10-40'"""
    parts = boi_ref.split("-")
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 8:
        return "-".join(parts[:-1])
    return boi_ref


def hit_at_k(retrieved_refs: list[str], gold_refs: list[str], k: int) -> bool:
    if not gold_refs:
        return False
    return any(ref in retrieved_refs[:k] for ref in gold_refs)


def mrr(retrieved_refs: list[str], gold_refs: list[str]) -> float:
    if not gold_refs:
        return 0.0
    for i, ref in enumerate(retrieved_refs, start=1):
        if ref in gold_refs:
            return 1.0 / i
    return 0.0


def check_keywords(text: str, keywords: list[str]) -> tuple[int, int]:
    lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in lower)
    return found, len(keywords)


def compute_summary(results: list[dict], queries: list[dict]) -> dict:
    qmap = {q["id"]: q for q in queries}
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"total": 0}

    total = len(ok)

    # --- Agent self-report ---
    supported = sum(1 for r in ok if r.get("answer_status") == "supported")
    partial = sum(1 for r in ok if r.get("answer_status") == "partial")
    insufficient = sum(1 for r in ok if r.get("answer_status") == "insufficient_evidence")
    avg_coverage = sum(r.get("coverage", 0) for r in ok) / total
    avg_iterations = sum(r.get("iterations", 1) for r in ok) / total
    reformed = sum(1 for r in ok if r.get("iterations", 1) > 1)

    times = sorted(r.get("total_s", 0) for r in ok)
    avg_time = sum(times) / total
    p50_time = times[len(times) // 2]
    p95_time = times[int(len(times) * 0.95)]

    # --- Retrieval ---
    doc_hits_1, doc_hits_3, doc_hits_5 = 0, 0, 0
    family_hits_1, family_hits_3, family_hits_5 = 0, 0, 0
    mrr_sum = 0.0
    family_mrr_sum = 0.0
    for r in ok:
        qid = r.get("id", "")
        gold = qmap.get(qid, {}).get("required_docs", [])
        retrieved = r.get("retrieved_docs", [])
        if hit_at_k(retrieved, gold, 1):
            doc_hits_1 += 1
        if hit_at_k(retrieved, gold, 3):
            doc_hits_3 += 1
        if hit_at_k(retrieved, gold, 5):
            doc_hits_5 += 1
        mrr_sum += mrr(retrieved, gold)
        # Family-level (date-stripped)
        gold_family = [strip_date(g) for g in gold]
        retrieved_family = [strip_date(r) for r in retrieved]
        if hit_at_k(retrieved_family, gold_family, 1):
            family_hits_1 += 1
        if hit_at_k(retrieved_family, gold_family, 3):
            family_hits_3 += 1
        if hit_at_k(retrieved_family, gold_family, 5):
            family_hits_5 += 1
        family_mrr_sum += mrr(retrieved_family, gold_family)
    doc_recall_1 = doc_hits_1 / total if total else 0
    doc_recall_3 = doc_hits_3 / total if total else 0
    doc_recall_5 = doc_hits_5 / total if total else 0
    avg_mrr = mrr_sum / total if total else 0
    family_recall_1 = family_hits_1 / total if total else 0
    family_recall_3 = family_hits_3 / total if total else 0
    family_recall_5 = family_hits_5 / total if total else 0
    avg_family_mrr = family_mrr_sum / total if total else 0

    # --- Answer integrity ---
    must_include_total, must_include_found = 0, 0
    must_not_include_total, must_not_include_violations = 0, 0
    numeric_correct, numeric_total = 0, 0
    for r in ok:
        qid = r.get("id", "")
        q = qmap.get(qid, {})
        concl = r.get("conclusion", "") + " " + " ".join(r.get("justification_bullets", []))

        mi = q.get("must_include", [])
        if mi:
            f, t = check_keywords(concl, mi)
            must_include_found += f
            must_include_total += t

        mni = q.get("must_not_include", [])
        if mni:
            f, t = check_keywords(concl, mni)
            must_not_include_violations += f
            must_not_include_total += t

        expected = q.get("expected_numeric_answer")
        if expected is not None:
            numeric_total += 1
            import re as _re
            nums_in_answer = []
            for nm in _re.findall(r"[\d\s]+[\d,.]+\d", concl):
                nm_clean = nm.replace(" ", "").replace(",", ".")
                try:
                    nums_in_answer.append(float(nm_clean))
                except ValueError:
                    pass
            for num in nums_in_answer:
                if abs(num - float(expected)) < 0.01:
                    numeric_correct += 1
                    break

    must_include_rate = must_include_found / must_include_total if must_include_total else 1.0
    must_not_include_rate = 1.0 - (must_not_include_violations / must_not_include_total) if must_not_include_total else 1.0
    numeric_accuracy = numeric_correct / numeric_total if numeric_total else 1.0

    # --- Per-theme ---
    themes = defaultdict(lambda: {"count": 0, "coverage_sum": 0, "supported": 0, "partial": 0,
                                   "doc_recall_1": 0, "doc_mrr_sum": 0, "time_sum": 0})
    for r in ok:
        qid = r.get("id", "")
        q = qmap.get(qid, {})
        t = q.get("theme", "unknown")
        themes[t]["count"] += 1
        themes[t]["coverage_sum"] += r.get("coverage", 0)
        themes[t]["time_sum"] += r.get("total_s", 0)
        if r.get("answer_status") == "supported":
            themes[t]["supported"] += 1
        elif r.get("answer_status") == "partial":
            themes[t]["partial"] += 1
        gold = q.get("required_docs", [])
        retrieved = r.get("retrieved_docs", [])
        if hit_at_k(retrieved, gold, 1):
            themes[t]["doc_recall_1"] += 1
        themes[t]["doc_mrr_sum"] += mrr(retrieved, gold)

    by_theme = {}
    for t, v in sorted(themes.items()):
        n = v["count"]
        by_theme[t] = {
            "count": n,
            "avg_coverage": round(v["coverage_sum"] / n, 3),
            "supported": v["supported"],
            "partial": v["partial"],
            "doc_recall_1": round(v["doc_recall_1"] / n, 3),
            "doc_mrr": round(v["doc_mrr_sum"] / n, 3),
            "avg_time_s": round(v["time_sum"] / n, 1),
        }

    # --- Per-difficulty ---
    difficulties = defaultdict(lambda: {"count": 0, "coverage_sum": 0, "supported": 0, "time_sum": 0})
    for r in ok:
        qid = r.get("id", "")
        q = qmap.get(qid, {})
        d = q.get("difficulty", "unknown")
        difficulties[d]["count"] += 1
        difficulties[d]["coverage_sum"] += r.get("coverage", 0)
        difficulties[d]["time_sum"] += r.get("total_s", 0)
        if r.get("answer_status") == "supported":
            difficulties[d]["supported"] += 1
    by_difficulty = {}
    for d, v in sorted(difficulties.items()):
        n = v["count"]
        by_difficulty[d] = {
            "count": n,
            "avg_coverage": round(v["coverage_sum"] / n, 3) if n else 0,
            "supported_rate": round(v["supported"] / n, 3) if n else 0,
            "avg_time_s": round(v["time_sum"] / n, 1) if n else 0,
        }

    # --- Per question-type ---
    qtypes = defaultdict(lambda: {"count": 0, "coverage_sum": 0, "supported": 0})
    for r in ok:
        qid = r.get("id", "")
        q = qmap.get(qid, {})
        qt = q.get("question_type", "unknown")
        qtypes[qt]["count"] += 1
        qtypes[qt]["coverage_sum"] += r.get("coverage", 0)
        if r.get("answer_status") == "supported":
            qtypes[qt]["supported"] += 1
    by_qtype = {}
    for qt, v in sorted(qtypes.items()):
        n = v["count"]
        by_qtype[qt] = {
            "count": n,
            "avg_coverage": round(v["coverage_sum"] / n, 3) if n else 0,
            "supported_rate": round(v["supported"] / n, 3) if n else 0,
        }

    return {
        "total_queries": total,
        "answer_status": {"supported": supported, "partial": partial, "insufficient_evidence": insufficient},
        "avg_coverage": round(avg_coverage, 3),
        "avg_iterations": round(avg_iterations, 1),
        "reformulated": f"{reformed}/{total}",
        "latency_s": {"avg": round(avg_time, 1), "p50": round(p50_time, 1), "p95": round(p95_time, 1)},
        "retrieval": {
            "doc_recall@1": round(doc_recall_1, 3),
            "doc_recall@3": round(doc_recall_3, 3),
            "doc_recall@5": round(doc_recall_5, 3),
            "doc_mrr": round(avg_mrr, 3),
            "family_recall@1": round(family_recall_1, 3),
            "family_recall@3": round(family_recall_3, 3),
            "family_recall@5": round(family_recall_5, 3),
            "family_mrr": round(avg_family_mrr, 3),
        },
        "answer_integrity": {
            "must_include_rate": round(must_include_rate, 3),
            "must_not_include_rate": round(must_not_include_rate, 3),
            "numeric_accuracy": round(numeric_accuracy, 3) if numeric_total else None,
        },
        "by_theme": by_theme,
        "by_difficulty": by_difficulty,
        "by_question_type": by_qtype,
    }


def print_summary(s: dict):
    if not s.get("total_queries"):
        return
    n = s["total_queries"]
    print("\n" + "=" * 65)
    print(f"  AGENTIC RAG EVALUATION — {n} queries")
    print("=" * 65)
    print(f"  Answer:  supported={s['answer_status']['supported']}  "
          f"partial={s['answer_status']['partial']}  "
          f"insufficient={s['answer_status']['insufficient_evidence']}")
    print(f"  Coverage: {s['avg_coverage']:.1%}  |  Iterations: {s['avg_iterations']}  "
          f"|  Reformulated: {s['reformulated']}")
    print(f"  Latency:  avg={s['latency_s']['avg']}s  p50={s['latency_s']['p50']}s  "
          f"p95={s['latency_s']['p95']}s")
    r = s["retrieval"]
    print(f"  Exact version:  R@1={r['doc_recall@1']:.1%}  R@3={r['doc_recall@3']:.1%}  "
          f"R@5={r['doc_recall@5']:.1%}  MRR={r['doc_mrr']:.3f}")
    if "family_recall@1" in r:
        print(f"  Family (date-stripped): R@1={r['family_recall@1']:.1%}  "
              f"R@3={r['family_recall@3']:.1%}  R@5={r['family_recall@5']:.1%}  "
              f"MRR={r['family_mrr']:.3f}")
    a = s["answer_integrity"]
    print(f"  Integrity: must_include={a['must_include_rate']:.1%}  "
          f"must_not_include={a['must_not_include_rate']:.1%}"
          + (f"  numeric={a['numeric_accuracy']:.1%}" if a.get("numeric_accuracy") is not None else ""))

    print(f"\n  {'Theme':<15} {'Count':>5} {'Coverage':>9} {'Sup':>4} {'Part':>4} {'R@1':>6} {'MRR':>6} {'Time':>6}")
    print(f"  {'-'*15} {'-'*5} {'-'*9} {'-'*4} {'-'*4} {'-'*6} {'-'*6} {'-'*6}")
    for t, v in s.get("by_theme", {}).items():
        print(f"  {t:<15} {v['count']:>5} {v['avg_coverage']:>8.0%} "
              f"{v['supported']:>4} {v['partial']:>4} "
              f"{v['doc_recall_1']:>6.0%} {v['doc_mrr']:>6.3f} {v['avg_time_s']:>5.0f}s")

    print(f"\n  {'Difficulty':<12} {'Count':>5} {'Coverage':>9} {'Supported':>10} {'Time':>6}")
    for d, v in s.get("by_difficulty", {}).items():
        print(f"  {d:<12} {v['count']:>5} {v['avg_coverage']:>8.0%} "
              f"{v['supported_rate']:>9.0%} {v['avg_time_s']:>5.0f}s")

    print(f"\n  {'Q-Type':<15} {'Count':>5} {'Coverage':>9} {'Supported':>10}")
    for qt, v in s.get("by_question_type", {}).items():
        print(f"  {qt:<15} {v['count']:>5} {v['avg_coverage']:>8.0%} {v['supported_rate']:>9.0%}")

    # Failing queries
    fails = s.get("_fail_detail", [])
    if fails:
        print(f"\n  FAILING QUERIES ({len(fails)}):")
        for f in fails:
            print(f"    [{f['id']}] {f['theme']}/{f['difficulty']} — "
                  f"status={f['status']} cov={f['cov']:.0%} "
                  f"recall={f['recall']}")


def main():
    p = argparse.ArgumentParser(description="Full agentic RAG evaluation")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--provider", type=str, default="deepseek")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    provider_cfg = PROVIDERS.get(args.provider.lower(), PROVIDERS["deepseek"])
    api_key = args.api_key or os.environ.get(provider_cfg["env_key"], "")
    if not api_key:
        print(f"ERROR: set {provider_cfg['env_key']} or use --api-key")
        return 1
    model = args.model or provider_cfg["default_model"]
    base_url = provider_cfg["base_url"]

    queries = load_queries(INPUT_PATH)
    if args.limit > 0:
        queries = queries[:args.limit]

    out_path = Path(args.output) if args.output else OUTPUT_PATH
    done_ids = set()
    results = []

    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
            for r in prev.get("per_query", []):
                done_ids.add(r["id"])
                results.append(r)
        print(f"Resumed: {len(done_ids)} done, {len(queries) - len(done_ids)} remaining")

    pending = [q for q in queries if q["id"] not in done_ids]
    if not pending:
        summary = compute_summary(results, queries)
        print_summary(summary)
        return 0

    # --- Init runtime ---
    print("Init RagRuntime (GPU)...")
    from bofip_agentic.rag_runtime import RagRuntime
    from bofip_agentic.agent_rag import AgenticRAG
    from openai import OpenAI

    rt = RagRuntime.from_local_corpus(corpus="commentary", device=args.device)
    client = OpenAI(api_key=api_key, base_url=base_url, **(  # type: ignore[arg-type]
        {"default_headers": {"x-api-key": api_key}} if args.provider.lower() == "anthropic" else {}))
    agent = AgenticRAG(rt, api_key=api_key, base_url=base_url, model=model, max_iterations=2, client=client)
    print(f"Agent ready ({args.provider}/{model}).\n")

    for idx, q in enumerate(pending, 1):
        qid = q["id"]
        question = q["question"]
        total_done = len(done_ids) + idx
        theme = q.get("theme", "?")
        diff = q.get("difficulty", "?")
        print(f"[{total_done}/{len(queries)}] {qid} ({theme}/{diff})...", end=" ", flush=True)

        t0 = time.time()
        try:
            agent_result = agent.run(question)

            # Extract retrieved document refs from sources
            sources = agent_result.get("sources", [])
            retrieved_docs = list(dict.fromkeys(s.get("boi_reference", "") for s in sources))

            r = {
                "id": qid,
                "theme": theme,
                "difficulty": diff,
                "question_type": q.get("question_type", ""),
                "question": question[:200],
                "answer_status": agent_result.get("answer_status", "?"),
                "coverage": agent_result.get("coverage", 0),
                "iterations": agent_result.get("iterations", 1),
                "total_s": round(time.time() - t0, 1),
                "conclusion": agent_result.get("conclusion", ""),
                "justification_bullets": agent_result.get("justification_bullets", []),
                "retrieved_docs": retrieved_docs,
                "trace": agent_result.get("trace", []),
            }
            results.append(r)

            cov = r["coverage"]
            it = r["iterations"]
            s = r["answer_status"]
            gold = q.get("required_docs", [])
            recall = "hit" if hit_at_k(retrieved_docs, gold, 1) else "miss"
            print(f"{s} | cov={cov:.0%} | {it}it | {r['total_s']}s | R@1={recall}", flush=True)

        except Exception as e:
            results.append({"id": qid, "error": str(e), "question": question[:200]})
            print(f"ERROR: {e}", flush=True)

        # Incremental save
        try:
            summary = compute_summary(results, queries)
        except Exception as e:
            summary = {"total_queries": len(results), "error": str(e)}
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {"provider": args.provider, "model": model, "device": args.device, "max_iterations": 2},
            "summary": summary,
            "per_query": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # Final report
    try:
        summary = compute_summary(results, queries)
    except Exception as e:
        summary = {"total_queries": len(results), "error": str(e)}
    # Collect failing queries
    fail_detail = []
    for r in results:
        if r.get("answer_status") in ("partial", "insufficient_evidence") or r.get("coverage", 1) < 0.8:
            qid = r.get("id", "")
            qm = {q["id"]: q for q in queries}
            gold = qm.get(qid, {}).get("required_docs", [])
            fail_detail.append({
                "id": qid,
                "theme": r.get("theme", ""),
                "difficulty": r.get("difficulty", ""),
                "status": r.get("answer_status", ""),
                "cov": r.get("coverage", 0),
                "recall": "hit" if hit_at_k(r.get("retrieved_docs", []), gold, 1) else "miss",
            })
    summary["_fail_detail"] = fail_detail

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"provider": args.provider, "model": model, "device": args.device, "max_iterations": 2},
        "summary": summary,
        "per_query": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\nReport: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

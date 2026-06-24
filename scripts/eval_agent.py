"""
Agentic RAG evaluation — self-reporting, no manual labeling, incremental save.

Usage:
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py --resume
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py --limit 10
    $env:PYTHONPATH="src"; $env:DEEPSEEK_API_KEY="sk-..."; python scripts/eval_agent.py --input data/eval/chatgpt_50_cases_v1.jsonl
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
    "codex":     {"base_url": "codex-cli://local",              "default_model": "gpt-5.5",            "env_key": "", "type": "codex_cli", "requires_api_key": False},
    "deepseek":  {"base_url": "https://api.deepseek.com/v1",    "default_model": "deepseek-v4-flash",  "env_key": "DEEPSEEK_API_KEY"},
    "openai":    {"base_url": "https://api.openai.com/v1",       "default_model": "gpt-5.4-mini",      "env_key": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1",    "default_model": "claude-haiku-4-5",  "env_key": "ANTHROPIC_API_KEY"},
    "mistral":   {"base_url": "https://api.mistral.ai/v1",       "default_model": "mistral-small-4",   "env_key": "MISTRAL_API_KEY"},
    "google":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "default_model": "gemini-3.1-flash", "env_key": "GEMINI_API_KEY"},
    "groq":      {"base_url": "https://api.groq.com/openai/v1",  "default_model": "llama-4-maverick",  "env_key": "GROQ_API_KEY"},
    "together":  {"base_url": "https://api.together.xyz/v1",     "default_model": "meta-llama/Llama-4-Maverick", "env_key": "TOGETHER_API_KEY"},
}

from bofip_agentic.agent_rag import AgenticRAG
from bofip_agentic.rag_runtime import RagRuntime
from bofip_agentic.settings import REPORTS_DIR, ensure_data_dirs

INPUT_PATH = PROJECT_ROOT / "data" / "eval" / "tax_eval_50.jsonl"
OUTPUT_PATH = REPORTS_DIR / "eval_agent_50.json"


def _normalize_query(row: dict, index: int) -> dict:
    question = row.get("question") or row.get("user_question")
    if not question:
        raise ValueError(f"missing question/user_question at line {index}")
    normalized = dict(row)
    normalized["id"] = str(row.get("id") or row.get("query_id") or f"q{index:03d}")
    normalized["question"] = str(question)
    normalized["theme"] = row.get("theme") or row.get("domain") or ""
    normalized["question_type"] = row.get("question_type") or row.get("type") or ""
    normalized["required_docs"] = row.get("required_docs") or row.get("must_include_sources") or []
    normalized["optional_docs"] = row.get("optional_docs") or row.get("should_include_sources") or []
    return normalized


def load_queries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [_normalize_query(json.loads(line), index) for index, line in enumerate(f, start=1) if line.strip()]


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
    p.add_argument("--input", type=str, default="", help="JSONL eval input. Supports question or user_question fields.")
    p.add_argument("--provider", type=str, default="deepseek", help="LLM provider key in PROVIDERS dict")
    p.add_argument("--model", type=str, default="", help="Model name (defaults to provider default)")
    p.add_argument("--api-key", type=str, default="", help="API key (or set env var)")
    p.add_argument("--base-url", type=str, default="", help="Base URL override")
    p.add_argument(
        "--lexical-only",
        action="store_true",
        help="Use the full corpus with BM25 lexical retrieval only. Useful for local CPU/Codex smoke evals.",
    )
    args = p.parse_args()

    provider_key = args.provider.lower()
    provider_config = PROVIDERS.get(provider_key, PROVIDERS.get("deepseek", {}))
    if not provider_config:
        print(f"ERROR: unknown provider '{args.provider}'. Options: {', '.join(PROVIDERS)}")
        return 1

    requires_api_key = provider_config.get("requires_api_key", True)
    api_key = args.api_key or os.environ.get(provider_config.get("env_key", ""), "")
    if requires_api_key and not api_key:
        print(f"ERROR: set {provider_config.get('env_key', 'API key')} or use --api-key")
        return 1

    model = args.model or provider_config.get("default_model", "deepseek-chat")
    base_url = args.base_url or provider_config.get("base_url", "https://api.deepseek.com/v1")

    ensure_data_dirs()
    input_path = Path(args.input) if args.input else INPUT_PATH
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    queries = load_queries(input_path)
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
        print("Loaded {} queries from {}".format(len(queries), input_path))

    pending = [q for q in queries if q["id"] not in done_ids]
    if not pending:
        summary = compute_summary(results)
        print_summary(summary)
        return 0

    retrieval_mode = "lexical_only" if args.lexical_only else "hybrid"
    print("Init RagRuntime ({}, {})...".format(args.device, retrieval_mode))
    rt = RagRuntime.from_local_corpus(
        corpus="commentary",
        device=args.device,
        load_dense=not args.lexical_only,
        load_reranker=not args.lexical_only,
    )
    if provider_config.get("type") == "codex_cli":
        from bofip_agentic.codex_cli_client import CodexCliClient

        agent = AgenticRAG(
            rt,
            client=CodexCliClient(model=model, project_root=PROJECT_ROOT),
            model=model,
            max_iterations=2,
        )
    else:
        agent = AgenticRAG(rt, api_key=api_key, base_url=base_url, model=model, max_iterations=2)
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
            r["gold"] = {
                "expected_status": q.get("expected_status", ""),
                "required_docs": q.get("required_docs", []),
                "optional_docs": q.get("optional_docs", []),
                "failure_signals": q.get("failure_signals", []),
            }
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
            "config": {
                "device": args.device,
                "max_iterations": 2,
                "model": model,
                "provider": provider_key,
                "retrieval_mode": retrieval_mode,
                "input": str(input_path),
            },
            "summary": summary,
            "per_query": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    summary = compute_summary(results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "device": args.device,
            "max_iterations": 2,
            "model": model,
            "provider": provider_key,
            "retrieval_mode": retrieval_mode,
            "input": str(input_path),
        },
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
    print("AGENTIC RAG EVALUATION - {} queries".format(s["total_queries"]))
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

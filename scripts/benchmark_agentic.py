"""Compare baseline pipeline vs Agentic RAG on 3 diverse queries."""
import json, os, sys, time
from pathlib import Path
from datetime import UTC, datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ["DEEPSEEK_API_KEY"] = "sk-1422bc6ce16e41fb8123f4a1723cfa49"

from bofip_agentic.rag_runtime import RagRuntime
from bofip_agentic.agent_rag import AgenticRAG
from bofip_agentic.prompt_utils import build_prompt

# ── 3 diverse queries ──
QUERIES = [
    {
        "id": "TVA_008",
        "question": "Mon auto-entreprise de services informatiques a fait 36000 euros de chiffre d'affaires cette annee. L'annee derniere j'etais a 33000. Est-ce que je depasse le seuil de franchise de TVA ? Si oui, a partir de quand dois-je facturer la TVA ?",
        "theme": "TVA",
        "difficulty": "hard",
    },
    {
        "id": "CF_001",
        "question": "J'ai recu une proposition de rectification du fisc qui me reclame 8000 euros. Quels sont mes droits ? Dans quel delai dois-je repondre ?",
        "theme": "CF",
        "difficulty": "medium",
    },
    {
        "id": "TVA_006",
        "question": "Ma societe a achete une voiture de tourisme pour les deplacements de mes commerciaux. Est-ce que je peux recuperer la TVA sur cet achat ?",
        "theme": "TVA",
        "difficulty": "medium",
    },
]

print("=" * 70)
print("AGENTIC RAG BENCHMARK — 3 queries")
print("=" * 70)

# ── Init runtime once ──
print("\nLoading RagRuntime (GPU)...")
t0 = time.time()
rt = RagRuntime.from_local_corpus(corpus="commentary", device="cuda")
print(f"Init: {time.time()-t0:.1f}s\n")

agent = AgenticRAG(rt, api_key=os.environ["DEEPSEEK_API_KEY"], max_iterations=2)

results = []
total_baseline_s = 0
total_agent_s = 0
total_agent_tokens = 0
total_agent_iters = 0
total_coverage = 0

for qi, q in enumerate(QUERIES, 1):
    qid = q["id"]
    question = q["question"]
    print(f"[{qi}/3] {qid} ({q['difficulty']})")
    print(f"  Q: {question[:90]}...")
    print()

    # ── BASELINE: single retrieval + answer ──
    entry = {
        "id": qid,
        "question": question,
        "theme": q["theme"],
        "difficulty": q["difficulty"],
    }

    # Baseline retrieval
    t_b = time.time()
    r_base = rt.retrieve(question, top_docs=8, max_chunks=8)
    base_retrieve_s = round(time.time() - t_b, 2)
    base_docs = [h.boi_reference for h in r_base.stage1_hits]
    base_chunks_count = len(r_base.stage2_chunks)
    entry["baseline"] = {
        "retrieve_s": base_retrieve_s,
        "docs": base_docs,
        "chunks_count": base_chunks_count,
    }

    # ── AGENTIC ──
    t_a = time.time()
    agent_result = agent.run(question)
    agent_total_s = round(time.time() - t_a, 2)

    entry["agent"] = {
        "total_s": agent_total_s,
        "iterations": agent_result["iterations"],
        "answer_status": agent_result["answer_status"],
        "axes_requis": agent_result["axes_requis"],
        "axes_couverts": agent_result["axes_couverts"],
        "axes_manquants": agent_result["axes_manquants"],
        "coverage": agent_result["coverage"],
        "chunks_used": agent_result["chunks_used"],
        "trace": agent_result["trace"],
        "conclusion": agent_result.get("conclusion", "")[:200],
    }

    # ── Display ──
    print(f"  Baseline: {base_retrieve_s}s | {base_chunks_count} chunks | docs: {base_docs[0][:50]}...")
    print(f"  Agent:    {agent_total_s}s | {agent_result['iterations']} iter | "
          f"status={agent_result['answer_status']} | coverage={agent_result['coverage']:.0%}")
    print(f"  Requis:   {agent_result['axes_requis']}")
    print(f"  Couverts: {agent_result['axes_couverts']}")
    if agent_result['axes_manquants']:
        print(f"  Manquants:{agent_result['axes_manquants']}")
    print(f"  Conclusion: {agent_result.get('conclusion', 'N/A')[:150]}")
    print()

    total_baseline_s += base_retrieve_s
    total_agent_s += agent_total_s
    total_agent_iters += agent_result["iterations"]
    total_coverage += agent_result["coverage"]
    results.append(entry)

# ── Summary ──
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Queries tested:           {len(QUERIES)}")
print(f"Baseline avg time:        {total_baseline_s/len(QUERIES):.1f}s (retrieval only, no LLM)")
print(f"Agent avg time:           {total_agent_s/len(QUERIES):.1f}s (full pipeline)")
print(f"Agent avg iterations:     {total_agent_iters/len(QUERIES):.1f}")
print(f"Agent avg coverage:       {total_coverage/len(QUERIES):.0%}")
print()

statuses = {}
for r in results:
    s = r["agent"]["answer_status"]
    statuses[s] = statuses.get(s, 0) + 1
print(f"Answer status: {statuses}")
print(f"Reformulated:  {sum(1 for r in results if r['agent']['iterations'] > 1)}/{len(QUERIES)}")

# ── Save report ──
report = {
    "generated_at": datetime.now(UTC).isoformat(),
    "summary": {
        "queries": len(QUERIES),
        "baseline_avg_s": round(total_baseline_s / len(QUERIES), 1),
        "agent_avg_s": round(total_agent_s / len(QUERIES), 1),
        "agent_avg_iterations": round(total_agent_iters / len(QUERIES), 1),
        "agent_avg_coverage": round(total_coverage / len(QUERIES), 3),
        "statuses": statuses,
        "reformulated": sum(1 for r in results if r["agent"]["iterations"] > 1),
    },
    "per_query": results,
}

out = PROJECT_ROOT / "data" / "reports" / "agentic_bench_3.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\nReport saved: {out}")

del rt

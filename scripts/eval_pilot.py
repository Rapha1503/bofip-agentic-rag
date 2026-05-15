"""
Honest RAG evaluation — multi-doc retrieval + passage recall + LLM judge.

Usage:
  $env:PYTHONPATH='src'; python scripts/eval_pilot.py
  $env:PYTHONPATH='src'; python scripts/eval_pilot.py --judge   # with LLM judge
  $env:PYTHONPATH='src'; python scripts/eval_pilot.py --resume   # resume interrupted run
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bofip_cleanroom.rag_runtime import RagRuntime
from bofip_cleanroom.settings import REPORTS_DIR, ensure_data_dirs
from bofip_cleanroom.jsonio import read_jsonl, write_json

INPUT_PATH = PROJECT_ROOT / "data" / "eval" / "pilot_5.jsonl"
OUTPUT_PATH = REPORTS_DIR / "eval_pilot.json"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall(retrieved, golds, k):
    ret_k = retrieved[:k]
    gold_set = set(golds)
    hits = [d for d in ret_k if d in gold_set]
    rec = len(hits) / len(golds) if golds else 0.0
    prec = len(hits) / len(ret_k) if ret_k else 0.0
    return prec, rec


def mrr(retrieved, golds):
    gold_set = set(golds)
    for i, d in enumerate(retrieved):
        if d in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(retrieved, golds, k):
    gold_set = set(golds)
    rel = [1 if d in gold_set else 0 for d in retrieved[:k]]
    ideal = sorted(rel, reverse=True)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """Tu es un expert fiscal independant qui evalue la qualite d'une reponse generee par un assistant fiscal.

Question de l'utilisateur:
{question}

Reponse de l'assistant:
{answer}

Extraits BOFiP fournis a l'assistant:
{chunks}

Evalue la reponse avec ce schema JSON strict:
{{
  "correctness": <1 a 5>,
  "completeness": <1 a 5>,
  "faithfulness": <1 a 5>,
  "hallucinated_claims": <entier>,
  "verdict": "correct|partially_correct|incorrect|insufficient_evidence",
  "explanation": "<1 phrase>"
}}

Criteres:
- correctness: les informations fiscales sont-elles exactes par rapport aux extraits ET au droit fiscal francais ?
- completeness: tous les axes requis par la question sont-ils traites ?
- faithfulness: chaque affirmation est-elle justifiee par un extrait fourni ?
- hallucinated_claims: nombre d'affirmations inventees (non presentes dans les extraits)
- verdict: correct = tout est bon, partially_correct = 1-2 erreurs mineures, incorrect = erreur majeure, insufficient_evidence = impossible de repondre avec les extraits
"""


def llm_judge(question, answer, chunks, api_key):
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai not installed"}

    chunks_text = "\n\n".join(
        "[{}] {}: {}".format(c["rank"], c["boi_reference"], c["text"][:500])
        for c in chunks[:5]
    )
    answer_text = json.dumps(answer, ensure_ascii=False, indent=2)
    prompt = JUDGE_PROMPT.format(question=question, answer=answer_text, chunks=chunks_text)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Expert fiscal. Reponds en JSON strict."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0, max_tokens=500,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            parsed = json.loads(content.strip())
            return parsed if isinstance(parsed, dict) else {"raw": content, "error": "not dict"}
        except Exception as e:
            if attempt >= 3:
                return {"error": str(e)}
            time.sleep(3 * attempt)
    return {"error": "max attempts"}


def llm_answer(query, chunks_data, api_key):
    try:
        from openai import OpenAI
    except ImportError:
        return None

    from bofip_cleanroom.prompt_utils import build_prompt

    prompt = build_prompt(query, chunks_data)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Assistant fiscal. Reponds depuis les extraits. JSON strict."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0, max_tokens=2800,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            parsed = json.loads(content.strip())
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            if attempt >= 3:
                return None
            time.sleep(3 * attempt)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", action="store_true", help="Run LLM judge on answers")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    ensure_data_dirs()
    queries = read_jsonl(INPUT_PATH)

    api_key = os.environ.get("DEEPSEEK_API_KEY") if args.judge else None
    if args.judge and not api_key:
        print("ERROR: --judge requires DEEPSEEK_API_KEY")
        return 1

    done_ids = set()
    results = []
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            prev = json.load(f)
            for r in prev.get("per_query", []):
                done_ids.add(r["id"])
                results.append(r)
        print("Resumed: {} done".format(len(done_ids)))

    pending = [q for q in queries if q["id"] not in done_ids]
    if not pending:
        return print_and_save(results, OUTPUT_PATH)

    print("Init RagRuntime ({})...".format(args.device), flush=True)
    rt = RagRuntime.from_local_corpus(corpus="commentary", device=args.device)
    print("Ready. {} pending queries.".format(len(pending)), flush=True)

    for idx, raw in enumerate(pending):
        qid = raw["id"]
        question = raw["question"]
        print("[{}/{}] {}...".format(idx + 1, len(pending), qid), end=" ", flush=True)
        t0 = time.time()

        try:
            r = rt.retrieve(question, top_docs=8, chunks_per_doc=8, max_chunks=8)
        except Exception as e:
            results.append({"id": qid, "error": str(e)})
            continue

        docs = [hit.boi_reference for hit in r.stage1_hits]
        chunk_ids = [c.chunk_id for c in r.stage2_chunks]
        chunks_full = list(r.stage2_chunks)
        gold_docs = raw["required_docs"]
        gold_chunks = raw.get("gold_chunk_ids", [])

        # Retrieval metrics
        prec_5, rec_5 = precision_recall(docs, gold_docs, 5)
        prec_8, rec_8 = precision_recall(docs, gold_docs, 8)
        m = mrr(docs, gold_docs)
        n_8 = ndcg(docs, gold_docs, 8)

        p_prec_8, p_rec_8 = precision_recall(chunk_ids, gold_chunks, 8)
        p_mrr = mrr(chunk_ids, gold_chunks)

        # Diversity
        doc_counts = defaultdict(int)
        for c in chunks_full:
            doc_counts[c.boi_reference] += 1

        entry = {
            "id": qid,
            "question": question,
            "theme": raw.get("theme", ""),
            "difficulty": raw.get("difficulty", ""),
            "doc_recall@5": rec_5,
            "doc_precision@5": prec_5,
            "doc_recall@8": rec_8,
            "doc_precision@8": prec_8,
            "mrr_doc": m,
            "ndcg_doc@8": n_8,
            "passage_recall@8": p_rec_8,
            "passage_mrr": p_mrr,
            "unique_docs": len(doc_counts),
            "max_chunks_per_doc": max(doc_counts.values()) if doc_counts else 0,
            "gold_docs": gold_docs,
            "retrieved_docs": docs,
            "gold_chunks": gold_chunks,
            "retrieved_chunks": chunk_ids,
        }

        # LLM Judge (optional)
        if args.judge and api_key:
            chunks_data = [{
                "rank": c.rank, "boi_reference": c.boi_reference,
                "title": c.title, "publication_date": c.publication_date,
                "section_path": c.section_path, "text": c.text, "chunk_id": c.chunk_id,
            } for c in chunks_full]
            answer = llm_answer(question, chunks_data, api_key)
            entry["llm_answer"] = answer
            if answer:
                entry["judge"] = llm_judge(question, answer, chunks_data, api_key)
            else:
                entry["judge"] = {"error": "answer generation failed"}

        results.append(entry)
        print("{:.1f}s | rec@5={:.0%}".format(time.time() - t0, rec_5), flush=True)

        # Incremental save
        print_and_save(results, OUTPUT_PATH, verbose=False)

    return print_and_save(results, OUTPUT_PATH, verbose=True)


def print_and_save(results, output_path, verbose=True):
    total = len(results)
    if not total:
        return 0

    avg = lambda vals: sum(vals) / len(vals) if vals else 0
    rec5 = avg([r.get("doc_recall@5", 0) for r in results])
    rec8 = avg([r.get("doc_recall@8", 0) for r in results])
    prec5 = avg([r.get("doc_precision@5", 0) for r in results])
    prec8 = avg([r.get("doc_precision@8", 0) for r in results])
    mrr_d = avg([r.get("mrr_doc", 0) for r in results])
    ndcg_d = avg([r.get("ndcg_doc@8", 0) for r in results])
    p_rec8 = avg([r.get("passage_recall@8", 0) for r in results if r.get("gold_chunks")])
    p_mrr = avg([r.get("passage_mrr", 0) for r in results if r.get("gold_chunks")])

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_queries": total,
            "avg_doc_recall@5": rec5,
            "avg_doc_precision@5": prec5,
            "avg_doc_recall@8": rec8,
            "avg_doc_precision@8": prec8,
            "avg_mrr_doc": mrr_d,
            "avg_ndcg_doc@8": ndcg_d,
            "avg_passage_recall@8": p_rec8,
            "avg_passage_mrr": p_mrr,
        },
        "per_query": results,
    }

    # Add judge summary
    judged = [r for r in results if "judge" in r and "error" not in r.get("judge", {})]
    if judged:
        jc = avg([j["judge"].get("correctness", 0) for j in judged])
        jf = avg([j["judge"].get("faithfulness", 0) for j in judged])
        jh = sum(j["judge"].get("hallucinated_claims", 0) for j in judged)
        verdicts = defaultdict(int)
        for j in judged:
            verdicts[j["judge"].get("verdict", "?")] += 1
        report["judge_summary"] = {
            "avg_correctness": jc,
            "avg_faithfulness": jf,
            "total_hallucinations": jh,
            "verdicts": dict(verdicts),
        }

    write_json(output_path, report)

    if verbose:
        print("\n=== RESULTS ({} queries) ===".format(total))
        print("doc_recall@5:  {:.0%}".format(rec5))
        print("doc_precision@5: {:.0%}".format(prec5))
        print("doc_recall@8:  {:.0%}".format(rec8))
        print("doc_precision@8: {:.0%}".format(prec8))
        print("MRR doc:       {:.3f}".format(mrr_d))
        print("NDCG doc@8:    {:.3f}".format(ndcg_d))
        if p_rec8 > 0:
            print("passage_recall@8: {:.0%}".format(p_rec8))
            print("passage MRR:      {:.3f}".format(p_mrr))
        if judged:
            print("\n=== LLM JUDGE ===")
            print("correctness avg:  {:.1f}/5".format(jc))
            print("faithfulness avg: {:.1f}/5".format(jf))
            print("hallucinations:   {}".format(jh))
            print("verdicts: {}".format(dict(verdicts)))
        print("\nReport: {}".format(output_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

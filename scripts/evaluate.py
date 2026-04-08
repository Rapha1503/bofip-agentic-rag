"""
BOFIP RAG Evaluation Script

Tests the RAG system on a corpus of questions across multiple domains.
Run: python scripts/evaluate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
from datetime import datetime

from src.retrieval.hybrid import get_hybrid_retriever
from src.generation.llm import get_llm_client

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

# Test corpus - questions across multiple BOFIP domains
TEST_QUESTIONS = [
    # TVA
    {
        "question": "Quel est le taux de TVA applicable a la restauration sur place?",
        "domain": "TVA",
        "expected_keywords": ["10%", "taux reduit", "restauration"],
    },
    {
        "question": "Quelles sont les conditions d'exoneration de TVA pour les formations professionnelles?",
        "domain": "TVA",
        "expected_keywords": ["exoneration", "formation", "organisme"],
    },
    # RFPI - Plus-values immobilieres
    {
        "question": "Quel est l'abattement pour duree de detention sur les plus-values immobilieres?",
        "domain": "RFPI",
        "expected_keywords": ["abattement", "6%", "22 ans", "detention"],
    },
    {
        "question": "Une SCI a l'IR qui detient un immeuble depuis 8 ans souhaite le vendre. Comment s'applique l'abattement pour duree de detention?",
        "domain": "RFPI",
        "expected_keywords": ["SCI", "IR", "abattement", "associe", "parts"],
    },
    # IR - Impot sur le revenu
    {
        "question": "Comment fonctionne le quotient familial pour le calcul de l'impot sur le revenu?",
        "domain": "IR",
        "expected_keywords": ["quotient", "parts", "enfant", "plafond"],
    },
    # BIC - Micro-entreprise
    {
        "question": "Quels sont les seuils du regime micro-BIC?",
        "domain": "BIC",
        "expected_keywords": ["micro", "seuil", "chiffre d'affaires", "188 700"],
    },
    # ENR - Droits de mutation
    {
        "question": "Quels sont les abattements applicables aux donations entre parents et enfants?",
        "domain": "ENR",
        "expected_keywords": ["donation", "abattement", "100 000", "enfant"],
    },
]


def evaluate_question(question_data: dict, retriever, llm) -> dict:
    """Evaluate a single question."""
    question = question_data["question"]

    # Retrieval using production method
    chunks = retriever.search_simple(question, n_results=20)

    # Generation
    result = llm.generate_with_sources(question, chunks, use_cache=False)

    # Check for expected keywords in answer
    answer_lower = result["answer"].lower()
    keywords_found = []
    keywords_missing = []
    for kw in question_data.get("expected_keywords", []):
        if kw.lower() in answer_lower:
            keywords_found.append(kw)
        else:
            keywords_missing.append(kw)

    keyword_score = len(keywords_found) / len(question_data.get("expected_keywords", [1])) if question_data.get("expected_keywords") else 1.0

    return {
        "question": question,
        "domain": question_data["domain"],
        "chunks_retrieved": len(chunks),
        "sources": [s["boi_reference"] for s in result.get("sources", [])],
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing,
        "keyword_score": keyword_score,
        "answer_preview": result["answer"][:300] + "..." if len(result["answer"]) > 300 else result["answer"],
    }


def run_evaluation():
    """Run full evaluation on test corpus."""
    print("=" * 70)
    print("BOFIP RAG EVALUATION")
    print("=" * 70)

    # Initialize components
    print("\nLoading components...")
    retriever = get_hybrid_retriever()
    llm = get_llm_client()

    print(f"Testing {len(TEST_QUESTIONS)} questions across domains")
    print("-" * 70)

    results = []
    total_score = 0

    for i, q_data in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {q_data['domain']}: {q_data['question'][:50]}...")

        try:
            result = evaluate_question(q_data, retriever, llm)
            results.append(result)
            total_score += result["keyword_score"]

            # Print summary
            status = "OK" if result["keyword_score"] >= 0.5 else "PARTIAL" if result["keyword_score"] > 0 else "MISS"
            print(f"   [{status}] Score: {result['keyword_score']:.0%} | Sources: {len(result['sources'])}")
            if result["keywords_missing"]:
                print(f"   Missing: {result['keywords_missing']}")
        except Exception as e:
            print(f"   [ERROR] {e}")
            results.append({"question": q_data["question"], "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_score = total_score / len(TEST_QUESTIONS)
    print(f"Average keyword score: {avg_score:.0%}")
    print(f"Questions evaluated: {len(results)}")

    # Domain breakdown
    print("\nBy domain:")
    domains = {}
    for r in results:
        d = r.get("domain", "unknown")
        if d not in domains:
            domains[d] = []
        domains[d].append(r.get("keyword_score", 0))

    for domain, scores in sorted(domains.items()):
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {domain}: {avg:.0%} ({len(scores)} questions)")

    # Save results
    output_file = Path(__file__).parent.parent / "data" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()

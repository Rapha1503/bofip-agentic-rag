"""
Test the full BOFIP RAG pipeline end-to-end.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.hybrid import get_hybrid_retriever
from src.generation.llm import get_llm_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_rag(question: str):
    """Test the full RAG pipeline."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    # Step 1: Retrieve relevant chunks
    print("\n[1] Retrieving relevant chunks...")
    retriever = get_hybrid_retriever()
    chunks = retriever.search_simple(question, n_results=20)

    print(f"    Found {len(chunks)} relevant chunks")
    for i, chunk in enumerate(chunks[:10]):
        ref = chunk['metadata'].get('boi_reference', 'N/A')
        score = chunk.get('combined_score', chunk.get('score', 0))
        print(f"    {i+1}. {ref} (score: {score:.4f})")

    # Step 2: Generate answer with LLM
    print("\n[2] Generating answer with LLM...")
    llm = get_llm_client()

    if not llm.client:
        print("    ERROR: No Groq API key configured!")
        return None

    result = llm.generate_with_sources(question, chunks, use_cache=False)

    # Display results
    print("\n" + "="*60)
    print("REPONSE:")
    print("="*60)
    print(result['answer'])

    print("\n" + "-"*60)
    print("SOURCES:")
    print("-"*60)
    for s in result['sources']:
        print(f"  - {s['boi_reference']}")
        if s.get('source_url'):
            print(f"    URL: {s['source_url']}")

    faith = result.get("faithfulness", {})
    if faith:
        print("\n" + "-"*60)
        print("FAITHFULNESS:")
        print("-"*60)
        print(
            f"  pass={faith.get('pass')} | mode={faith.get('mode')} | "
            f"reason={faith.get('reason', '')}"
        )

    print("\n" + "-"*60)
    print(result['disclaimer'])

    return result


if __name__ == "__main__":
    # Test questions
    questions = [
        "Quel est le taux de TVA applicable a la restauration sur place?",
    ]

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]

    for q in questions:
        test_rag(q)

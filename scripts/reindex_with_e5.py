"""
Reindex BOFIP chunks with new E5 embedding model.

This script:
1. Clears the old ChromaDB collection
2. Clears the embedding cache (old model's embeddings)
3. Reindexes all 81,101 chunks with multilingual-e5-base

Usage:
    python scripts/reindex_with_e5.py

Expected time: ~30-60 minutes depending on GPU/CPU
"""

import logging
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHROMA_DB_DIR, CACHE_DIR, PROCESSED_DATA_DIR, EMBEDDING_MODEL
from src.retrieval.vector_store import BOFIPVectorStore, index_chunks_from_file
from src.retrieval.embeddings import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_old_data():
    """Clear old embedding cache. ChromaDB is cleared via API in index_chunks_from_file."""

    # Skip ChromaDB directory deletion - let index_chunks_from_file handle it
    # via store.clear() which uses ChromaDB API (avoids file lock issues on Windows)
    logger.info("ChromaDB will be cleared via API during reindexing")

    # Clear embedding cache
    embed_cache = CACHE_DIR / 'embeddings'
    if embed_cache.exists():
        logger.info(f"Clearing old embedding cache at {embed_cache}")
        try:
            shutil.rmtree(embed_cache)
            logger.info("Embedding cache cleared")
        except PermissionError as e:
            logger.warning(f"Could not clear embedding cache (file locked): {e}")
            logger.warning("Continuing anyway - embeddings will be regenerated")


def verify_model():
    """Verify the new embedding model is loaded."""
    logger.info(f"Configured model: {EMBEDDING_MODEL}")

    # Test loading
    model = get_embedding_model()
    logger.info(f"Model loaded: {model.model_name}")
    logger.info(f"E5 model: {model.is_e5_model}")
    logger.info(f"Embedding dimension: {model.dimension}")

    # Test embedding
    test_query = "credit impot emploi domicile"
    test_passage = "Les depenses pour emploi a domicile ouvrent droit a un credit d'impot."

    query_emb = model.embed_query(test_query)
    passage_emb = model.embed_batch([test_passage], show_progress=False)

    logger.info(f"Query embedding shape: {query_emb.shape}")
    logger.info(f"Passage embedding shape: {passage_emb.shape}")

    return True


def reindex():
    """Reindex all chunks with new model."""
    chunks_file = PROCESSED_DATA_DIR / "chunks.json"

    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return False

    logger.info(f"Starting reindex from {chunks_file}")
    start_time = datetime.now()

    # Index chunks (clear_existing=True will handle the collection)
    count = index_chunks_from_file("chunks.json", clear_existing=True)

    elapsed = datetime.now() - start_time
    logger.info(f"Reindexing complete!")
    logger.info(f"  Chunks indexed: {count}")
    logger.info(f"  Time elapsed: {elapsed}")

    return True


def verify_search():
    """Verify search works with new embeddings."""
    logger.info("Verifying search functionality...")

    store = BOFIPVectorStore()

    test_queries = [
        ("credit impot emploi domicile", "IR-RICI"),
        ("plus-value residence principale", "RFPI-PVI"),
        ("TVA taux reduit restauration", "TVA"),
    ]

    for query, expected_pattern in test_queries:
        results = store.search(query, n_results=5)
        found = any(expected_pattern in r['metadata'].get('boi_reference', '') for r in results)
        status = "[OK]" if found else "[X]"
        top_ref = results[0]['metadata'].get('boi_reference', 'N/A') if results else 'No results'
        logger.info(f"  {status} '{query[:30]}...' -> {top_ref}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BOFIP Reindexing with E5 Embedding Model")
    logger.info("=" * 60)

    # Step 1: Verify model
    logger.info("\n[Step 1/4] Verifying new embedding model...")
    if not verify_model():
        sys.exit(1)

    # Step 2: Clear old data
    logger.info("\n[Step 2/4] Clearing old data...")
    clear_old_data()

    # Step 3: Reindex
    logger.info("\n[Step 3/4] Reindexing chunks (this will take a while)...")
    if not reindex():
        sys.exit(1)

    # Step 4: Verify search
    logger.info("\n[Step 4/4] Verifying search...")
    verify_search()

    logger.info("\n" + "=" * 60)
    logger.info("Reindexing complete! You can now test the app.")
    logger.info("=" * 60)

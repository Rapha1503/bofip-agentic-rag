"""
BOFIP Reranker Module

Uses a cross-encoder model to rerank retrieved chunks for better precision.
Cross-encoders are more accurate than bi-encoders for ranking but slower.
"""

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Multilingual cross-encoder model - good balance of speed and quality
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class BOFIPReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.

    Cross-encoders process (query, document) pairs together,
    giving more accurate relevance scores than bi-encoder similarity.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the reranker.

        Args:
            model_name: Cross-encoder model name (default: ms-marco-MiniLM)
        """
        self.model_name = model_name or RERANKER_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]],
               top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on relevance to query.

        Args:
            query: The search query
            chunks: List of chunk dictionaries with 'text' field
            top_k: Number of top results to return (default: all)

        Returns:
            Reranked list of chunks with 'rerank_score' added
        """
        if not chunks:
            return []

        if self.model is None:
            logger.warning("Reranker model not available, returning original order")
            return chunks

        try:
            # Prepare (query, document) pairs for cross-encoder
            pairs = [(query, chunk.get('text', '')) for chunk in chunks]

            # Get relevance scores
            scores = self.model.predict(pairs)

            # Add scores to chunks and sort
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                chunk_copy = chunk.copy()
                chunk_copy['rerank_score'] = float(score)
                scored_chunks.append(chunk_copy)

            # Sort by rerank score (descending)
            scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)

            # Return top_k if specified
            if top_k is not None:
                scored_chunks = scored_chunks[:top_k]

            logger.debug(f"Reranked {len(chunks)} chunks, returning {len(scored_chunks)}")
            return scored_chunks

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks

    def is_available(self) -> bool:
        """Check if reranker model is loaded and available."""
        return self.model is not None


# Singleton instance
_reranker_instance: Optional[BOFIPReranker] = None


def get_reranker() -> BOFIPReranker:
    """Get or create singleton reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BOFIPReranker()
    return _reranker_instance


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    reranker = get_reranker()

    # Test with sample chunks
    query = "Quel est le taux de TVA pour la restauration?"
    chunks = [
        {"text": "Le taux normal de TVA est de 20%.", "chunk_id": "1"},
        {"text": "La restauration beneficie du taux reduit de 10%.", "chunk_id": "2"},
        {"text": "Les medicaments ont un taux de 2.1%.", "chunk_id": "3"},
    ]

    print(f"Query: {query}")
    print("\nBefore reranking:")
    for c in chunks:
        print(f"  - {c['text'][:50]}...")

    reranked = reranker.rerank(query, chunks)

    print("\nAfter reranking:")
    for c in reranked:
        print(f"  - Score {c['rerank_score']:.3f}: {c['text'][:50]}...")

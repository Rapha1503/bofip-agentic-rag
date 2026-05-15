"""
BOFIP Embeddings Module

Handles embedding generation with caching using sentence-transformers.
"""

import logging
from typing import List, Optional
from pathlib import Path
import hashlib
import re
import gc

from sentence_transformers import SentenceTransformer
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import EMBEDDING_MODEL, CACHE_DIR

logger = logging.getLogger(__name__)


def _model_slug(model_name: str) -> str:
    """Filesystem-safe model identifier for cache partitioning."""
    if not model_name:
        return "default"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip().lower())
    return slug.strip("_") or "default"


class EmbeddingModel:
    """
    Wrapper for sentence-transformers with caching.
    Handles E5 model prefixes automatically.
    """

    def __init__(self, model_name: str = None, cache_dir: Path = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name or EMBEDDING_MODEL
        self.cache_dir = cache_dir or CACHE_DIR / 'embeddings'
        self.model_cache_dir = self.cache_dir / _model_slug(self.model_name)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Detect if this is an E5 model (requires prefixes)
        self.is_e5_model = 'e5' in self.model_name.lower()

        logger.info(f"Loading embedding model: {self.model_name}")
        if self.is_e5_model:
            logger.info("E5 model detected - will use query/passage prefixes")

        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key"""
        return self.model_cache_dir / f"{cache_key}.npy"

    def embed_single(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed a single text with optional caching.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            cache_path = self._get_cache_path(cache_key)

            if cache_path.exists():
                return np.load(cache_path)

        embedding = self.model.encode(text, convert_to_numpy=True)

        if use_cache:
            np.save(cache_path, embedding)

        return embedding

    def embed_batch(self, texts: List[str],
                    batch_size: int = 64,
                    show_progress: bool = True,
                    is_query: bool = False) -> np.ndarray:
        """
        Embed a batch of texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            is_query: If True, treat as queries (for E5 models)

        Returns:
            Array of embeddings (n_texts x dimension)
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

        # Add E5 prefixes if needed
        if self.is_e5_model:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
            logger.debug(f"Added E5 prefix: '{prefix}'")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query (no caching for queries).

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        # Add E5 query prefix if needed
        if self.is_e5_model:
            query = "query: " + query

        return self.model.encode(query, convert_to_numpy=True)


# Singleton instances for reuse (one per model)
_model_instances: dict[str, EmbeddingModel] = {}


def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """Get or create a singleton embedding model instance per model name."""
    resolved_name = model_name or EMBEDDING_MODEL
    if resolved_name not in _model_instances:
        _model_instances[resolved_name] = EmbeddingModel(model_name=resolved_name)
    return _model_instances[resolved_name]


def reset_embedding_models() -> None:
    """Drop cached model instances to free memory between benchmark runs."""
    _model_instances.clear()
    gc.collect()


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = get_embedding_model()

    # Test single embedding
    text = "Les frais de repas sont-ils deductibles?"
    embedding = model.embed_single(text)
    print(f"Single embedding shape: {embedding.shape}")

    # Test batch embedding
    texts = [
        "Quel est le taux de TVA applicable?",
        "Comment calculer l'impot sur le revenu?",
        "Les dividendes sont-ils imposables?"
    ]
    embeddings = model.embed_batch(texts, show_progress=False)
    print(f"Batch embeddings shape: {embeddings.shape}")

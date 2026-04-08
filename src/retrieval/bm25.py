"""
BOFIP BM25 Index

BM25 keyword-based search for hybrid retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import re

from rank_bm25 import BM25Okapi
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DATA_DIR
from src.data_pipeline.chunker import BOFIPChunk, load_chunks_from_json

logger = logging.getLogger(__name__)

# BM25 index file
BM25_INDEX_FILE = PROCESSED_DATA_DIR / "bm25_index.pkl"


def tokenize_french(text: str) -> List[str]:
    """
    French tokenizer for BM25 search.

    Key features:
    - Preserves hyphenated words (plus-value, micro-BIC, crédit-bail)
    - Preserves apostrophe contractions (d'impôt, l'article, qu'il)
    - Normalizes French numbers (18 300 → 18300)
    - Normalizes French decimal commas (5,5% → 5.5%)
    - Allows 2-char tokens for tax codes (IS, IR, etc.)
    - Does NOT remove domain-critical words like "plus", "moins"
    """
    # French stopwords - WITHOUT domain-critical terms
    # REMOVED: 'plus', 'moins' (critical for "plus-value", "moins-value")
    # REMOVED: 'd', 'l', 'n', 's', 'c', 'qu', 'j', 'm', 't' (keep French contractions intact)
    stopwords = {
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'au', 'aux',
        'ce', 'cette', 'ces', 'que', 'qui', 'quoi', 'dont', 'dans', 'sur', 'sous', 'par',
        'pour', 'avec', 'sans', 'est', 'sont', 'etre', 'être', 'avoir', 'il', 'elle', 'ils', 'elles',
        'en', 'ne', 'pas', 'si', 'se', 'son', 'sa', 'ses', 'leur', 'leurs',
        'nous', 'vous', 'tout', 'tous', 'toute', 'toutes', 'autre', 'autres',
        'même', 'meme', 'aussi', 'ainsi', 'donc', 'car', 'mais', 'où', 'ou'
    }

    # Normalize French numbers: "18 300" → "18300", "1 000 000" → "1000000"
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    # Apply multiple times for numbers like "1 000 000"
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    # Normalize French decimal commas: "5,5" → "5.5", "12,75%" → "12.75%"
    # This allows "5,5%" to be tokenized as "5.5" instead of splitting at comma
    text = re.sub(r'(\d),(\d)', r'\1.\2', text)

    text = text.lower()

    # Match words including hyphens and apostrophes (French compounds)
    # Examples:
    #   "plus-value" → ["plus-value"] (single token, preserves compound)
    #   "d'impôt" → ["d'impôt"] (single token, preserves contraction)
    #   "crédit d'impôt" → ["crédit", "d'impôt"]
    #   "micro-BIC" → ["micro-bic"]
    tokens = re.findall(
        r"[a-zA-Z0-9àâäéèêëïîôùûüçœæÀÂÄÉÈÊËÏÎÔÙÛÜÇŒÆ]+(?:[-'][a-zA-Z0-9àâäéèêëïîôùûüçœæÀÂÄÉÈÊËÏÎÔÙÛÜÇŒÆ]+)*",
        text
    )

    # Filter stopwords, allow 2-char tokens (for "IS", "IR", "TVA" becomes "tva", etc.)
    tokens = [t for t in tokens if t not in stopwords and len(t) >= 2]

    return tokens


class BOFIPBM25:
    """
    BM25 index for BOFIP chunks.
    """

    def __init__(self):
        """Initialize empty BM25 index"""
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[BOFIPChunk] = []
        self.chunk_ids: List[str] = []

    def build_index(self, chunks: List[BOFIPChunk]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of BOFIPChunk objects
        """
        logger.info(f"Building BM25 index for {len(chunks)} chunks")

        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]

        # Tokenize all documents
        tokenized_docs = [tokenize_french(c.text_with_context) for c in chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        logger.info("BM25 index built successfully")

    def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using BM25.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of search results with scores
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index first.")

        # Tokenize query
        query_tokens = tokenize_french(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:n_results]

        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive score
                chunk = self.chunks[idx]
                results.append({
                    'chunk_id': chunk.chunk_id,
                    'text': chunk.text_with_context,
                    'metadata': {
                        'doc_id': chunk.doc_id,
                        'boi_reference': chunk.boi_reference,
                        'series': chunk.series,
                        'section_title': chunk.section_title,
                        'paragraph_number': chunk.paragraph_number,
                        'publication_date': chunk.publication_date,
                        'source_url': chunk.source_url,
                    },
                    'score': float(scores[idx])
                })

        return results

    def save(self, filepath: Path = None):
        """Save BM25 index to disk"""
        filepath = filepath or BM25_INDEX_FILE
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'bm25': self.bm25,
            'chunk_ids': self.chunk_ids,
            'chunks': [c.to_dict() for c in self.chunks]
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to {filepath}")

    def load(self, filepath: Path = None):
        """Load BM25 index from disk"""
        filepath = filepath or BM25_INDEX_FILE

        if not filepath.exists():
            raise FileNotFoundError(f"BM25 index not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.bm25 = data['bm25']
        self.chunk_ids = data['chunk_ids']
        self.chunks = [BOFIPChunk.from_dict(d) for d in data['chunks']]

        logger.info(f"BM25 index loaded. {len(self.chunks)} chunks")


def build_bm25_index(chunks_file: str = "chunks.json") -> BOFIPBM25:
    """
    Build and save BM25 index from chunks file.

    Args:
        chunks_file: Path to chunks JSON file

    Returns:
        BOFIPBM25 index
    """
    chunks_path = PROCESSED_DATA_DIR / chunks_file

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    logger.info(f"Loading chunks from {chunks_path}")
    chunks = load_chunks_from_json(str(chunks_path))
    logger.info(f"Loaded {len(chunks)} chunks")

    # Build index
    bm25 = BOFIPBM25()
    bm25.build_index(chunks)

    # Save index
    bm25.save()

    return bm25


def get_bm25_index() -> BOFIPBM25:
    """
    Get BM25 index, loading from disk if available.

    Returns:
        BOFIPBM25 index
    """
    bm25 = BOFIPBM25()

    if BM25_INDEX_FILE.exists():
        bm25.load()
    else:
        logger.info("BM25 index not found, building...")
        bm25 = build_bm25_index()

    return bm25


# Test
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Build BM25 index')
    parser.add_argument('--chunks-file', default='chunks.json', help='Chunks JSON file')
    parser.add_argument('--test-query', type=str, help='Test query after building')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild index')

    args = parser.parse_args()

    # Build or load index
    if args.rebuild or not BM25_INDEX_FILE.exists():
        bm25 = build_bm25_index(args.chunks_file)
    else:
        bm25 = get_bm25_index()

    print(f"BM25 index ready with {len(bm25.chunks)} chunks")

    # Test query
    if args.test_query:
        results = bm25.search(args.test_query, n_results=5)
        print(f"\nBM25 search results for: '{args.test_query}'")
        for i, r in enumerate(results):
            print(f"\n{i+1}. {r['metadata']['boi_reference']}")
            print(f"   Score: {r['score']:.3f}")
            print(f"   Text: {r['text'][:200]}...")

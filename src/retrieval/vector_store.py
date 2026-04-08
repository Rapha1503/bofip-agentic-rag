"""
BOFIP Vector Store

ChromaDB-based vector store for BOFIP chunks.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CHROMA_DB_DIR, PROCESSED_DATA_DIR, EMBEDDING_MODEL
from src.retrieval.embeddings import get_embedding_model
from src.data_pipeline.chunker import BOFIPChunk, load_chunks_from_json

logger = logging.getLogger(__name__)

# ChromaDB collection name
COLLECTION_NAME = "bofip_chunks"


class BOFIPVectorStore:
    """
    ChromaDB vector store for BOFIP chunks.
    """

    def __init__(
        self,
        persist_dir: Path = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: Optional[str] = None,
    ):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to persist ChromaDB data
        """
        self.persist_dir = persist_dir or CHROMA_DB_DIR
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at {self.persist_dir}")
        logger.info(
            f"Using collection='{self.collection_name}' with embedding_model='{self.embedding_model_name}'"
        )

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "BOFIP fiscal documentation chunks",
                "embedding_model": self.embedding_model_name,
            },
        )
        self.embed_model = None

        logger.info(f"Collection '{self.collection_name}' ready. Count: {self.collection.count()}")

    def _get_embed_model(self):
        """Lazy-load embedding model only when needed (search/index)."""
        if self.embed_model is None:
            self.embed_model = get_embedding_model(self.embedding_model_name)
        return self.embed_model

    def add_chunks(self, chunks: List[BOFIPChunk],
                   batch_size: int = 500) -> int:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of BOFIPChunk objects
            batch_size: Batch size for adding

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Ensure unique IDs by tracking seen IDs
        seen_ids = set()

        # Process in batches
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Prepare data with unique IDs
            ids = []
            for c in batch:
                chunk_id = c.chunk_id
                # Make unique if duplicate
                counter = 0
                while chunk_id in seen_ids:
                    counter += 1
                    chunk_id = f"{c.chunk_id}_{counter}"
                seen_ids.add(chunk_id)
                ids.append(chunk_id)
            texts = [c.text_with_context for c in batch]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "boi_reference": c.boi_reference,
                    "series": ",".join(c.series) if c.series else "",
                    "section_title": c.section_title or "",
                    "paragraph_number": c.paragraph_number or "",
                    "publication_date": c.publication_date,
                    "source_url": c.source_url,
                    "content_type": c.content_type,
                    "contains_table": c.contains_table,
                    "token_count": c.token_count
                }
                for c in batch
            ]

            # Generate embeddings
            embeddings = self._get_embed_model().embed_batch(texts, show_progress=False)

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )

            added += len(batch)
            logger.info(f"Added {added}/{len(chunks)} chunks")

        return added

    def search(self, query: str,
               n_results: int = 10,
               filter_series: List[str] = None,
               where: Dict = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_series: Optional list of series to filter by
            where: Optional ChromaDB where filter (overrides filter_series)

        Returns:
            List of search results with scores and metadata
        """
        # Get query embedding
        query_embedding = self._get_embed_model().embed_query(query)

        # Build filter
        if filter_series:
            where = {"series": {"$in": filter_series}}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })

        return formatted

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks belonging to a specific document.

        Args:
            doc_id: Document ID (e.g., "1032-PGP")

        Returns:
            List of chunks sorted by paragraph number
        """
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"]
            )

            if not results['ids']:
                logger.debug(f"No chunks found for doc_id: {doc_id}")
                return []

            chunks = []
            for i, doc_text in enumerate(results['documents']):
                chunks.append({
                    'chunk_id': results['ids'][i],
                    'text': doc_text,
                    'metadata': results['metadatas'][i]
                })

            # Sort by paragraph number (handle both string and int, including ranges like "90-100")
            def get_para_num(chunk):
                para = chunk['metadata'].get('paragraph_number', '0')
                try:
                    if isinstance(para, str) and '-' in para:
                        # Handle ranges like "90-100" by taking the first number
                        para = para.split('-')[0]
                    return int(para) if para else 0
                except (ValueError, TypeError):
                    return 0

            chunks.sort(key=get_para_num)
            logger.debug(f"Found {len(chunks)} chunks for doc_id: {doc_id}")
            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks for doc_id {doc_id}: {e}")
            return []

    def get_count(self) -> int:
        """Get number of chunks in the store"""
        return self.collection.count()

    def clear(self):
        """Clear all chunks from the store"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "BOFIP fiscal documentation chunks",
                "embedding_model": self.embedding_model_name,
            },
        )
        logger.info("Vector store cleared")


def index_chunks_from_file(chunks_file: str = "chunks.json",
                           clear_existing: bool = False,
                           persist_dir: Path | None = None,
                           collection_name: str = COLLECTION_NAME,
                           embedding_model_name: Optional[str] = None) -> int:
    """
    Index all chunks from a JSON file into the vector store.

    Args:
        chunks_file: Path to chunks JSON file (relative to PROCESSED_DATA_DIR)
        clear_existing: Whether to clear existing data first

    Returns:
        Number of chunks indexed
    """
    chunks_candidate = Path(chunks_file)
    if chunks_candidate.is_absolute():
        chunks_path = chunks_candidate
    elif chunks_candidate.exists():
        chunks_path = chunks_candidate.resolve()
    else:
        chunks_path = PROCESSED_DATA_DIR / chunks_candidate

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    logger.info(f"Loading chunks from {chunks_path}")
    chunks = load_chunks_from_json(str(chunks_path))
    logger.info(f"Loaded {len(chunks)} chunks")

    # Initialize store
    store = BOFIPVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
    )

    if clear_existing:
        store.clear()

    # Check if already indexed
    current_count = store.get_count()
    if current_count > 0 and not clear_existing:
        logger.warning(f"Store already contains {current_count} chunks. Use --clear to reindex.")
        return current_count

    # Add chunks
    added = store.add_chunks(chunks)

    logger.info(f"Indexing complete. Total chunks: {store.get_count()}")
    return added


# Test
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Index BOFIP chunks')
    parser.add_argument('--chunks-file', default='chunks.json', help='Chunks JSON file')
    parser.add_argument('--clear', action='store_true', help='Clear existing index')
    parser.add_argument('--test-query', type=str, help='Test query after indexing')
    parser.add_argument('--persist-dir', type=Path, help='Custom Chroma persistence directory')
    parser.add_argument('--collection-name', default=COLLECTION_NAME, help='Chroma collection name')
    parser.add_argument('--embedding-model', type=str, help='SentenceTransformer model name')

    args = parser.parse_args()

    # Index chunks
    count = index_chunks_from_file(
        args.chunks_file,
        clear_existing=args.clear,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
    )
    print(f"Indexed {count} chunks")

    # Test query
    if args.test_query:
        store = BOFIPVectorStore(
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
        )
        results = store.search(args.test_query, n_results=5)
        print(f"\nSearch results for: '{args.test_query}'")
        for i, r in enumerate(results):
            print(f"\n{i+1}. {r['metadata']['boi_reference']}")
            print(f"   Score: {r['score']:.3f}")
            print(f"   Text: {r['text'][:200]}...")

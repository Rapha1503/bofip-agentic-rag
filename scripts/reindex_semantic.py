"""
BOFIP Semantic Reindexing Script

Reprocesses all BOFIP documents using semantic chunking (1 rule = 1 chunk)
and rebuilds the vector store and BM25 index.

Usage:
    python scripts/reindex_semantic.py                    # Full reindex
    python scripts/reindex_semantic.py --sample 100       # Test with 100 docs
    python scripts/reindex_semantic.py --chunks-only      # Only create chunks, don't index
"""

import logging
import json
import sys
import shutil
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR
from src.data_pipeline.parser import parse_metadata
from src.data_pipeline.semantic_chunker import (
    SemanticBOFIPChunker, SemanticChunk, save_semantic_chunks
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Content types to process
CONTENT_TYPES = ['Commentaire', 'Barème', 'Autres annexes']
BOFIP_ROOT = RAW_DATA_DIR / 'bofip_extracted' / 'BOFiP' / 'documents' / 'Contenu'


def find_all_documents(content_types: List[str] = None):
    """Find all BOFIP document directories"""
    content_types = content_types or CONTENT_TYPES

    for content_type in content_types:
        type_dir = BOFIP_ROOT / content_type
        if not type_dir.exists():
            continue

        for series_dir in type_dir.iterdir():
            if not series_dir.is_dir():
                continue

            for doc_dir in series_dir.iterdir():
                if not doc_dir.is_dir():
                    continue

                for date_dir in doc_dir.iterdir():
                    if not date_dir.is_dir():
                        continue

                    if (date_dir / 'document.xml').exists():
                        yield date_dir, content_type


def process_document_semantic(doc_dir: Path, content_type: str, chunker: SemanticBOFIPChunker) -> List[SemanticChunk]:
    """Process a single document with semantic chunking"""
    try:
        # Parse metadata
        metadata = parse_metadata(doc_dir / 'document.xml')

        # Check for HTML content
        html_path = doc_dir / 'data.html'
        if not html_path.exists():
            return []

        # Prepare metadata dict for chunker
        meta_dict = {
            'boi_reference': metadata.boi_reference,
            'doc_id': metadata.doc_id,
            'series': metadata.series,
            'publication_date': metadata.publication_date,
            'source_url': metadata.source_url,
            'content_type': content_type
        }

        # Create semantic chunks
        chunks = chunker.parse_and_chunk(html_path, meta_dict)
        return chunks

    except Exception as e:
        logger.error(f"Error processing {doc_dir}: {e}")
        return []


def create_semantic_chunks(content_types: List[str] = None,
                           sample_size: Optional[int] = None,
                           n_workers: int = 4) -> List[SemanticChunk]:
    """
    Process all documents with semantic chunking.

    Args:
        content_types: Content types to process
        sample_size: Limit number of documents (for testing)
        n_workers: Parallel workers

    Returns:
        List of all semantic chunks
    """
    logger.info("Finding all documents...")
    doc_list = list(find_all_documents(content_types))

    if sample_size:
        doc_list = doc_list[:sample_size]

    logger.info(f"Processing {len(doc_list)} documents with SEMANTIC chunking...")

    chunker = SemanticBOFIPChunker(min_tokens=50, max_tokens=1500)
    all_chunks = []
    errors = 0

    # Process with progress bar
    with tqdm(total=len(doc_list), desc="Semantic chunking") as pbar:
        # Sequential processing (semantic parsing is CPU-bound)
        for doc_dir, content_type in doc_list:
            try:
                chunks = process_document_semantic(doc_dir, content_type, chunker)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed: {doc_dir}: {e}")
                errors += 1
            pbar.update(1)

    logger.info(f"Processed {len(doc_list)} documents, {errors} errors")
    logger.info(f"Generated {len(all_chunks)} SEMANTIC chunks")

    # Statistics
    if all_chunks:
        token_counts = [c.token_count for c in all_chunks]
        logger.info(f"Token stats: avg={sum(token_counts)//len(token_counts)}, "
                   f"min={min(token_counts)}, max={max(token_counts)}")

    return all_chunks


def save_chunks_for_indexing(chunks: List[SemanticChunk], output_file: str = 'semantic_chunks.json'):
    """Save chunks in format compatible with existing indexer"""
    output_path = PROCESSED_DATA_DIR / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict format expected by indexer
    chunk_dicts = []
    for chunk in chunks:
        chunk_dicts.append({
            'chunk_id': chunk.chunk_id,
            'text': chunk.text,
            'text_with_context': chunk.text_with_context,
            'boi_reference': chunk.boi_reference,
            'doc_id': chunk.doc_id,
            'series': chunk.series,
            'section_title': chunk.section_title,
            'paragraph_number': chunk.paragraph_number,
            'publication_date': chunk.publication_date,
            'source_url': chunk.source_url,
            'content_type': chunk.content_type,
            'contains_table': chunk.contains_table,
            'is_header': False,
            'token_count': chunk.token_count,
            # Additional semantic fields
            'section_path': chunk.section_path,
            'contains_list': chunk.contains_list,
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(chunk_dicts)} chunks to {output_path}")
    return output_path


def rebuild_indexes(chunks_file: Path):
    """Rebuild ChromaDB and BM25 indexes from chunks"""
    logger.info("Rebuilding indexes...")

    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    logger.info(f"Loaded {len(chunks_data)} chunks")

    # Import indexing modules
    from src.retrieval.vector_store import BOFIPVectorStore
    from src.retrieval.bm25 import BOFIPBM25

    # Try to backup old ChromaDB (may fail if locked)
    backup_succeeded = False
    if CHROMA_DB_DIR.exists():
        backup_dir = CHROMA_DB_DIR.parent / 'chroma_db_backup'
        try:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(str(CHROMA_DB_DIR), str(backup_dir))
            logger.info(f"Backed up old ChromaDB to {backup_dir}")
            backup_succeeded = True
        except PermissionError:
            logger.warning("Could not backup ChromaDB (locked). Will reset collection instead.")

    # Rebuild vector store
    logger.info("Building ChromaDB vector store...")
    vector_store = BOFIPVectorStore()

    # If backup failed (DB was locked), clear the collection
    if not backup_succeeded and vector_store.get_count() > 0:
        logger.info("Clearing existing collection...")
        vector_store.clear()

    # Convert all chunks to BOFIPChunk format
    from src.data_pipeline.chunker import BOFIPChunk
    logger.info("Converting chunks to BOFIPChunk format...")

    bofip_chunks = []
    for c in chunks_data:
        chunk = BOFIPChunk(
            chunk_id=c['chunk_id'],
            text=c['text'],
            text_with_context=c['text_with_context'],
            boi_reference=c['boi_reference'],
            doc_id=c['doc_id'],
            series=c['series'] if isinstance(c['series'], list) else c['series'].split(','),
            section_title=c.get('section_title'),
            paragraph_number=c.get('paragraph_number'),
            publication_date=c['publication_date'],
            source_url=c['source_url'],
            content_type=c['content_type'],
            contains_table=c.get('contains_table', False),
            is_header=False,
            token_count=c['token_count']
        )
        bofip_chunks.append(chunk)

    # Index into vector store
    logger.info(f"Indexing {len(bofip_chunks)} chunks into vector store...")
    vector_store.add_chunks(bofip_chunks)

    logger.info(f"Vector store: {vector_store.get_count()} chunks indexed")

    # Rebuild BM25 index
    logger.info("Building BM25 index...")
    bm25 = BOFIPBM25()
    bm25.build_index(bofip_chunks)
    bm25.save()
    logger.info(f"BM25 index: {len(bm25.chunks)} chunks indexed")

    logger.info("Indexing complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Reindex BOFIP with semantic chunking')
    parser.add_argument('--sample', type=int, help='Process only N documents (for testing)')
    parser.add_argument('--types', nargs='+', default=['Commentaire'],
                       help='Content types to process')
    parser.add_argument('--chunks-only', action='store_true',
                       help='Only create chunks, skip indexing')
    parser.add_argument('--output', default='semantic_chunks.json',
                       help='Output chunks filename')

    args = parser.parse_args()

    # Check data exists
    if not BOFIP_ROOT.exists():
        logger.error(f"BOFIP data not found at {BOFIP_ROOT}")
        return

    # Step 1: Create semantic chunks
    chunks = create_semantic_chunks(
        content_types=args.types,
        sample_size=args.sample
    )

    if not chunks:
        logger.error("No chunks created!")
        return

    # Step 2: Save chunks
    chunks_file = save_chunks_for_indexing(chunks, args.output)

    # Step 3: Rebuild indexes (unless --chunks-only)
    if not args.chunks_only:
        rebuild_indexes(chunks_file)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SEMANTIC REINDEXING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Chunks file: {chunks_file}")
    if not args.chunks_only:
        logger.info("Vector store and BM25 index rebuilt!")


if __name__ == '__main__':
    main()

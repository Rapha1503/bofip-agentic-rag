"""
BOFIP Data Processing Pipeline

Processes all BOFIP documents: parse, chunk, and prepare for embedding.
"""

import logging
from pathlib import Path
from typing import List, Optional, Generator
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_pipeline.parser import parse_document, BOFIPMetadata
from src.data_pipeline.chunker import (
    BOFIPChunk, process_chunks, save_chunks_to_json, estimate_tokens
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Content types to process (focus on Commentaire for main content)
CONTENT_TYPES = ['Commentaire', 'Barème', 'Autres annexes']

# Path to extracted BOFIP data
BOFIP_ROOT = RAW_DATA_DIR / 'bofip_extracted' / 'BOFiP' / 'documents' / 'Contenu'


def find_all_documents(content_types: List[str] = None) -> Generator[Path, None, None]:
    """
    Find all BOFIP document directories.

    Args:
        content_types: List of content types to include (default: all)

    Yields:
        Path to each document date directory containing document.xml
    """
    content_types = content_types or CONTENT_TYPES

    for content_type in content_types:
        type_dir = BOFIP_ROOT / content_type
        if not type_dir.exists():
            logger.warning(f"Content type directory not found: {type_dir}")
            continue

        # Walk through series directories
        for series_dir in type_dir.iterdir():
            if not series_dir.is_dir():
                continue

            # Walk through document ID directories
            for doc_dir in series_dir.iterdir():
                if not doc_dir.is_dir():
                    continue

                # Walk through date directories
                for date_dir in doc_dir.iterdir():
                    if not date_dir.is_dir():
                        continue

                    # Check if document.xml exists
                    if (date_dir / 'document.xml').exists():
                        yield date_dir


def process_single_document(doc_dir: Path,
                           min_tokens: int = 100,
                           max_tokens: int = 800) -> List[BOFIPChunk]:
    """
    Process a single BOFIP document.

    Args:
        doc_dir: Path to document date directory
        min_tokens: Minimum chunk size
        max_tokens: Maximum chunk size

    Returns:
        List of processed chunks
    """
    try:
        metadata, raw_chunks = parse_document(doc_dir)

        if not raw_chunks:
            return []

        # Convert parser output to BOFIPChunk format
        chunks = []
        for rc in raw_chunks:
            chunk = BOFIPChunk(
                chunk_id=rc.chunk_id,
                text=rc.text,
                text_with_context=rc.text_with_context,
                boi_reference=rc.boi_reference,
                doc_id=rc.doc_id,
                series=rc.series,
                section_title=rc.section_title,
                paragraph_number=rc.paragraph_number,
                publication_date=rc.publication_date,
                source_url=rc.source_url,
                content_type=metadata.content_type,
                contains_table=rc.contains_table,
                is_header=rc.is_header,
                token_count=rc.token_count
            )
            chunks.append(chunk)

        # Process chunks (merge small, split large)
        processed = process_chunks(chunks, min_tokens, max_tokens)

        return processed

    except Exception as e:
        logger.error(f"Error processing {doc_dir}: {e}")
        return []


def process_all_documents(content_types: List[str] = None,
                          sample_size: Optional[int] = None,
                          output_file: str = 'chunks.json',
                          n_workers: int = 4) -> List[BOFIPChunk]:
    """
    Process all BOFIP documents and save chunks.

    Args:
        content_types: List of content types to process
        sample_size: If set, only process this many documents (for testing)
        output_file: Output JSON filename
        n_workers: Number of parallel workers

    Returns:
        List of all chunks
    """
    logger.info("Finding all documents...")
    doc_dirs = list(find_all_documents(content_types))

    if sample_size:
        doc_dirs = doc_dirs[:sample_size]

    logger.info(f"Processing {len(doc_dirs)} documents...")

    all_chunks = []
    errors = 0

    # Process documents with progress bar
    with tqdm(total=len(doc_dirs), desc="Processing") as pbar:
        # Use thread pool for I/O bound parsing
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_single_document, doc_dir): doc_dir
                for doc_dir in doc_dirs
            }

            for future in as_completed(futures):
                doc_dir = futures[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {doc_dir}: {e}")
                    errors += 1
                pbar.update(1)

    logger.info(f"Processed {len(doc_dirs)} documents, {errors} errors")
    logger.info(f"Generated {len(all_chunks)} chunks")

    # Calculate statistics
    total_tokens = sum(c.token_count for c in all_chunks)
    avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0
    logger.info(f"Total tokens: ~{total_tokens:,}, Average: ~{avg_tokens:.0f} per chunk")

    # Save chunks
    output_path = PROCESSED_DATA_DIR / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving chunks to {output_path}...")
    save_chunks_to_json(all_chunks, str(output_path))

    # Save statistics
    stats = {
        'total_documents': len(doc_dirs),
        'total_chunks': len(all_chunks),
        'total_tokens': total_tokens,
        'avg_tokens_per_chunk': avg_tokens,
        'errors': errors,
        'content_types': content_types or CONTENT_TYPES
    }
    stats_path = PROCESSED_DATA_DIR / 'processing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("Processing complete!")
    return all_chunks


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Process BOFIP documents')
    parser.add_argument('--sample', type=int, help='Process only N documents (for testing)')
    parser.add_argument('--types', nargs='+', default=['Commentaire'],
                       help='Content types to process')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output', default='chunks.json', help='Output filename')

    args = parser.parse_args()

    # Check if data exists
    if not BOFIP_ROOT.exists():
        logger.error(f"BOFIP data not found at {BOFIP_ROOT}")
        logger.error("Please download and extract BOFIP data first:")
        logger.error("  curl -L -o data/raw/bofip_stock.tgz https://bofip.impots.gouv.fr/opendata/stock/1")
        logger.error("  tar -xzf data/raw/bofip_stock.tgz -C data/raw/bofip_extracted")
        return

    process_all_documents(
        content_types=args.types,
        sample_size=args.sample,
        output_file=args.output,
        n_workers=args.workers
    )


if __name__ == '__main__':
    main()

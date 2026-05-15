"""
Process CGI and LPF PDFs and append chunks to the existing chunks.json.

Usage:
    python scripts/process_pdfs.py
    python scripts/process_pdfs.py --pdf-dir "Data to include later" --dry-run
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from config import PROCESSED_DATA_DIR, PDF_DATA_DIR
from data_pipeline.pdf_parser import PDFDocumentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def pdf_chunk_to_dict(chunk) -> dict:
    """Convert PDFChunk to BOFIPChunk-compatible dict."""
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "text_with_context": chunk.text_with_context,
        "boi_reference": chunk.boi_reference or "",
        "doc_id": chunk.doc_id,
        "series": chunk.series,
        "section_title": chunk.section_title,
        "paragraph_number": chunk.paragraph_number,
        "publication_date": chunk.publication_date,
        "source_url": chunk.source_url,
        "content_type": chunk.content_type,
        "contains_table": chunk.contains_table,
        "is_header": False,
        "token_count": chunk.token_count,
        "source": chunk.source,
    }


def process_pdfs(pdf_dir: Path, dry_run: bool = False) -> list:
    """
    Process all PDFs in the given directory.

    Returns list of chunk dicts.
    """
    parser = PDFDocumentParser()
    all_chunks = []

    # Auto-detect PDFs and their source type
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        name_lower = pdf_file.name.lower()

        if "code" in name_lower and "imp" in name_lower:
            source = "CGI"
        elif "livre" in name_lower and "proc" in name_lower:
            source = "LPF"
        else:
            logger.warning(f"Unknown PDF type: {pdf_file.name}, skipping")
            continue

        logger.info(f"Processing {pdf_file.name} as {source}...")
        chunks = parser.parse_pdf(pdf_file, source=source)

        # Convert to dict format
        chunk_dicts = [pdf_chunk_to_dict(c) for c in chunks]
        all_chunks.extend(chunk_dicts)

        logger.info(f"  -> {len(chunk_dicts)} chunks from {source}")

    return all_chunks


def append_to_chunks_json(new_chunks: list, chunks_path: Path):
    """
    Append new chunks to existing chunks.json.

    First removes any existing PDF chunks (by source field),
    then appends new ones.
    """
    # Load existing chunks
    if chunks_path.exists():
        with open(chunks_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        logger.info(f"Loaded {len(existing)} existing chunks")
    else:
        existing = []
        logger.info("No existing chunks.json, creating new")

    # Remove old PDF chunks (to allow re-running)
    pdf_sources = {"CGI", "LPF"}
    bofip_chunks = [c for c in existing if c.get("source", "BOFIP") not in pdf_sources]
    removed = len(existing) - len(bofip_chunks)
    if removed > 0:
        logger.info(f"Removed {removed} old PDF chunks")

    # Append new
    combined = bofip_chunks + new_chunks
    logger.info(f"Total chunks: {len(combined)} ({len(bofip_chunks)} BOFIP + {len(new_chunks)} PDF)")

    # Save
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {chunks_path}")

    return len(combined)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process CGI/LPF PDFs")
    parser.add_argument('--pdf-dir', default=str(PDF_DATA_DIR),
                        help='Directory containing PDFs')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse PDFs but do not write to chunks.json')
    args = parser.parse_args()

    pdf_dir = project_root / args.pdf_dir

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)

    # Process PDFs
    new_chunks = process_pdfs(pdf_dir)

    if not new_chunks:
        logger.warning("No chunks created from PDFs")
        sys.exit(1)

    # Show stats
    sources = {}
    for c in new_chunks:
        src = c.get("source", "UNKNOWN")
        sources[src] = sources.get(src, 0) + 1

    logger.info(f"\nChunk summary:")
    for src, count in sorted(sources.items()):
        logger.info(f"  {src}: {count} chunks")

    if args.dry_run:
        logger.info("\n[DRY RUN] Would append to chunks.json but not writing")
        return

    # Append to chunks.json
    chunks_path = PROCESSED_DATA_DIR / "chunks.json"
    total = append_to_chunks_json(new_chunks, chunks_path)

    logger.info(f"\nDone! Total chunks in index: {total}")
    logger.info("Next steps:")
    logger.info("  1. Rebuild BM25: python -m src.retrieval.bm25 --rebuild")
    logger.info("  2. Rebuild ChromaDB: python scripts/reindex_semantic.py")


if __name__ == "__main__":
    main()
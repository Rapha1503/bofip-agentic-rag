"""
BOFIP-RAG Bootstrap

Single-command, reproducible build of the entire knowledge base from public sources.

This script:
  1. Downloads the BOFIP open-data stock from data.economie.gouv.fr
  2. Extracts the archive
  3. Parses all BOFIP documents and creates semantic chunks
  4. Builds the ChromaDB vector store + BM25 sparse index
  5. (--full only) Downloads and ingests the LEGI archive (CGI / LPF)

Usage:
    # Fast demo (~10 minutes, BOFIP only, 500 documents)
    python scripts/bootstrap.py --sample 500

    # Full build (several hours, full BOFIP + CGI + LPF)
    python scripts/bootstrap.py --full
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional

# Make the project root importable so we can reuse pipeline functions.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bootstrap")


# Public BOFIP open-data stock (full corpus, ~116 MB compressed).
BOFIP_STOCK_URL = "https://bofip.impots.gouv.fr/opendata/stock/1"
BOFIP_ARCHIVE_PATH = RAW_DATA_DIR / "bofip_stock.tgz"
BOFIP_EXTRACT_DIR = RAW_DATA_DIR / "bofip_extracted"
BOFIP_MARKER = BOFIP_EXTRACT_DIR / "BOFiP" / "documents" / "Contenu"


def _check_groq_key() -> None:
    """Warn (don't fail) if GROQ_API_KEY is missing — indexing works without it."""
    if not os.getenv("GROQ_API_KEY"):
        logger.warning(
            "GROQ_API_KEY is not set. The index will be built, but the Streamlit "
            "app will not be able to generate answers until you add a key to .env."
        )
    else:
        logger.info("GROQ_API_KEY detected.")


def _download_bofip_archive() -> None:
    """Download the BOFIP stock archive with resume support."""
    # Reuse the resume-capable downloader from the LEGI pipeline.
    from scripts.process_legi_archive import _download_with_resume, _get_remote_size

    BOFIP_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if BOFIP_MARKER.exists():
        logger.info(f"BOFIP data already extracted at {BOFIP_EXTRACT_DIR}, skipping download.")
        return

    remote_size = _get_remote_size(BOFIP_STOCK_URL)
    if BOFIP_ARCHIVE_PATH.exists() and remote_size and BOFIP_ARCHIVE_PATH.stat().st_size >= remote_size:
        logger.info(f"BOFIP archive already present at {BOFIP_ARCHIVE_PATH}, skipping download.")
        return

    logger.info(f"Downloading BOFIP stock from {BOFIP_STOCK_URL}")
    logger.info(f"  -> {BOFIP_ARCHIVE_PATH}")
    _download_with_resume(
        url=BOFIP_STOCK_URL,
        target_path=BOFIP_ARCHIVE_PATH,
        remote_size=remote_size,
    )
    size_mb = BOFIP_ARCHIVE_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Downloaded BOFIP archive ({size_mb:.1f} MB)")


def _extract_bofip_archive() -> None:
    """Extract the BOFIP archive if not already extracted."""
    if BOFIP_MARKER.exists():
        logger.info("BOFIP archive already extracted.")
        return

    if not BOFIP_ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"BOFIP archive missing: {BOFIP_ARCHIVE_PATH}. Run download step first."
        )

    BOFIP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {BOFIP_ARCHIVE_PATH} to {BOFIP_EXTRACT_DIR}")
    with tarfile.open(BOFIP_ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(BOFIP_EXTRACT_DIR)
    logger.info("Extraction complete.")


def _build_bofip_index(sample_size: Optional[int]) -> int:
    """
    Run the semantic chunker over BOFIP documents and build vector + sparse indexes.

    Returns:
        Total number of BOFIP chunks created.
    """
    # Reuse the existing pipeline — do not duplicate logic.
    from scripts.reindex_semantic import (
        create_semantic_chunks,
        save_chunks_for_indexing,
        rebuild_indexes,
    )

    label = f"sample={sample_size}" if sample_size else "full corpus"
    logger.info(f"Creating BOFIP semantic chunks ({label})...")
    chunks = create_semantic_chunks(
        content_types=["Commentaire"],
        sample_size=sample_size,
    )

    if not chunks:
        raise RuntimeError(
            "No BOFIP chunks were produced. Check that the archive was downloaded "
            "and extracted correctly under data/raw/bofip_extracted/."
        )

    chunks_file = save_chunks_for_indexing(chunks, output_file="chunks.json")
    logger.info(f"Saved {len(chunks)} BOFIP chunks to {chunks_file}")

    rebuild_indexes(chunks_file)
    return len(chunks)


def _ingest_legi() -> int:
    """
    Download and ingest the latest LEGI archive (CGI / LPF), then sync the
    legal chunks into ChromaDB.

    Returns:
        Number of legal chunks ingested.
    """
    # Reuse process_legi_archive's plumbing rather than re-implementing.
    from scripts.process_legi_archive import (
        _resolve_archive_path,
        _save_chunks,
        _append_replace_sources,
        _infer_archive_mode,
        _append_delta_upsert,
    )
    from src.data_pipeline.legi_tar_parser import LEGITarCodeParser
    import datetime as dt

    raw_legi_dir = RAW_DATA_DIR / "legi"
    logger.info("Resolving latest LEGI archive (downloads if missing)...")
    archive_path = _resolve_archive_path(
        archive_arg="latest-full",
        raw_legi_dir=raw_legi_dir,
        auto_download=True,
    )
    logger.info(f"Using LEGI archive: {archive_path}")

    target_sources = ["CGI", "LPF"]
    parser_obj = LEGITarCodeParser()
    per_source = parser_obj.parse_archive(
        archive_path=archive_path,
        target_sources=target_sources,
        as_of=dt.date.today(),
        include_future=False,
    )

    merged_chunks = []
    found_sources = []
    for source in target_sources:
        source_chunks = [c.to_dict() for c in per_source.get(source, [])]
        logger.info(f"  {source}: {len(source_chunks)} chunks")
        if source_chunks:
            found_sources.append(source)
            merged_chunks.extend(source_chunks)

    if not merged_chunks:
        logger.warning("No LEGI chunks were produced. Skipping legal ingestion.")
        return 0

    legi_file = PROCESSED_DATA_DIR / "legi_chunks.json"
    _save_chunks(merged_chunks, legi_file)

    main_chunks_path = PROCESSED_DATA_DIR / "chunks.json"
    append_mode = _infer_archive_mode(archive_path)
    if append_mode == "delta_upsert":
        _append_delta_upsert(merged_chunks, main_chunks_path, found_sources)
    else:
        _append_replace_sources(merged_chunks, main_chunks_path, found_sources)

    # Rebuild BM25 over the combined corpus.
    logger.info("Rebuilding BM25 index over combined corpus...")
    from src.retrieval.bm25 import BOFIPBM25
    from src.data_pipeline.chunker import load_chunks_from_json

    all_chunks = load_chunks_from_json(str(main_chunks_path))
    bm25 = BOFIPBM25()
    bm25.build_index(all_chunks)
    bm25.save()
    logger.info(f"BM25: {len(bm25.chunks)} chunks indexed.")

    # Sync legal vectors into Chroma.
    logger.info("Syncing legal chunks into ChromaDB...")
    from scripts import sync_legal_chunks  # noqa: F401
    # The sync script reads chunks.json and updates Chroma in place.
    # Call its main() programmatically.
    original_argv = sys.argv[:]
    try:
        sys.argv = ["sync_legal_chunks.py", "--skip-restore"]
        sync_legal_chunks.main()
    finally:
        sys.argv = original_argv

    return len(merged_chunks)


def _print_summary(
    bofip_chunks: int,
    legi_chunks: int,
    started_at: float,
) -> None:
    duration = time.time() - started_at
    minutes, seconds = divmod(int(duration), 60)
    hours, minutes = divmod(minutes, 60)

    logger.info("=" * 60)
    logger.info("BOFIP-RAG BOOTSTRAP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"BOFIP chunks indexed : {bofip_chunks:,}")
    if legi_chunks:
        logger.info(f"LEGI chunks indexed  : {legi_chunks:,}")
        logger.info(f"Total chunks         : {bofip_chunks + legi_chunks:,}")
    logger.info(f"ChromaDB directory   : {CHROMA_DB_DIR}")
    logger.info(f"Processed data dir   : {PROCESSED_DATA_DIR}")
    logger.info(f"Total duration       : {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info("")
    logger.info("Next step: launch the app")
    logger.info("    streamlit run app.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap the BOFIP-RAG knowledge base from public sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Build only N BOFIP documents (fast demo, BOFIP only, ~10 min).",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Full build: complete BOFIP corpus + LEGI (CGI/LPF). Takes hours.",
    )
    args = parser.parse_args()

    started_at = time.time()
    _check_groq_key()

    # Step 1 — BOFIP archive
    _download_bofip_archive()
    _extract_bofip_archive()

    # Step 2 — BOFIP chunking + indexing
    sample_size = args.sample if args.sample else None
    bofip_chunks = _build_bofip_index(sample_size=sample_size)

    # Step 3 — LEGI (full mode only)
    legi_chunks = 0
    if args.full:
        try:
            legi_chunks = _ingest_legi()
        except Exception as exc:
            logger.error(f"LEGI ingestion failed: {exc}")
            logger.error(
                "BOFIP indexing succeeded, but the legal corpus (CGI/LPF) is missing. "
                "You can retry with: python scripts/process_legi_archive.py --archive latest-full --append"
            )

    _print_summary(bofip_chunks, legi_chunks, started_at)


if __name__ == "__main__":
    main()

"""
Synchronize legal chunks (CGI/LPF) in ChromaDB with data/processed/chunks.json.

Goal:
- Keep BOFIP embeddings untouched (fast)
- Replace only legal chunks after parser changes (safe + quick)
- Recover automatically from interrupted index rebuilds by restoring backup

Usage:
    python scripts/sync_legal_chunks.py
    python scripts/sync_legal_chunks.py --skip-restore
    python scripts/sync_legal_chunks.py --delta-file data/processed/legi_chunks.json
"""

import sys
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHROMA_DB_DIR, PROCESSED_DATA_DIR
from src.retrieval.vector_store import BOFIPVectorStore
from src.data_pipeline.chunker import load_chunks_from_json, BOFIPChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LEGAL_CONTENT_TYPES = ("CGI", "LPF")
BACKUP_DIR = CHROMA_DB_DIR.parent / "chroma_db_backup"


def _get_chroma_count(persist_dir: Path) -> int:
    """
    Read embedding count directly from SQLite (no Chroma client lock).
    Returns -1 if DB is missing/unreadable.
    """
    db_path = persist_dir / "chroma.sqlite3"
    if not db_path.exists():
        return -1

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM embeddings")
        count = int(cur.fetchone()[0])
        conn.close()
        return count
    except Exception as exc:
        logger.warning(f"Could not read Chroma count at {db_path}: {exc}")
        return -1


def resolve_target_dir(min_expected_count: int = 50000, skip_restore: bool = False) -> Path:
    """Choose safest target Chroma directory and restore from backup when possible."""
    current_count = _get_chroma_count(CHROMA_DB_DIR)
    backup_count = _get_chroma_count(BACKUP_DIR) if BACKUP_DIR.exists() else -1

    logger.info(f"Current Chroma count ({CHROMA_DB_DIR}): {current_count}")
    if BACKUP_DIR.exists():
        logger.info(f"Backup Chroma count ({BACKUP_DIR}): {backup_count}")

    if current_count >= min_expected_count:
        return CHROMA_DB_DIR

    if not BACKUP_DIR.exists() or backup_count < 0:
        logger.warning(
            f"Chroma count is low ({current_count}) but backup is missing at {BACKUP_DIR}. "
            "Continuing without restore."
        )
        return CHROMA_DB_DIR

    logger.warning(
        f"Detected incomplete Chroma index ({current_count}). "
        f"Backup has {backup_count} embeddings."
    )

    if skip_restore:
        logger.info("Skipping restore (--skip-restore). Using backup as sync target.")
        return BACKUP_DIR

    # Try to restore backup into active directory before opening Chroma clients.
    try:
        if CHROMA_DB_DIR.exists():
            shutil.rmtree(CHROMA_DB_DIR)
        shutil.copytree(BACKUP_DIR, CHROMA_DB_DIR)
        restored_count = _get_chroma_count(CHROMA_DB_DIR)
        logger.info(f"Backup restored to active directory. Count: {restored_count}")
        return CHROMA_DB_DIR
    except PermissionError as exc:
        logger.warning(
            f"Could not restore backup due file lock ({exc}). "
            "Using backup directory directly."
        )
        return BACKUP_DIR


def _get_ids_for_content_type(store: BOFIPVectorStore, content_type: str) -> List[str]:
    try:
        res = store.collection.get(where={"content_type": content_type}, include=[])
        return res.get("ids", []) or []
    except Exception as exc:
        logger.error(f"Failed to fetch IDs for content_type={content_type}: {exc}")
        return []


def _delete_ids_in_batches(store: BOFIPVectorStore, ids: List[str], batch_size: int = 1000) -> int:
    deleted = 0
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        store.collection.delete(ids=batch)
        deleted += len(batch)
    return deleted


def _load_chunks_for_counts(chunks_path: Path) -> List[BOFIPChunk]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")
    return load_chunks_from_json(str(chunks_path))


def _load_delta_chunks(delta_file: Path) -> List[BOFIPChunk]:
    if not delta_file.exists():
        raise FileNotFoundError(f"Missing delta file: {delta_file}")
    chunks = load_chunks_from_json(str(delta_file))
    return [c for c in chunks if c.content_type in LEGAL_CONTENT_TYPES]


def _dedupe_legal_by_reference(chunks: List[BOFIPChunk]) -> List[BOFIPChunk]:
    by_ref: Dict[Tuple[str, str], BOFIPChunk] = {}
    for chunk in chunks:
        ref = str(chunk.boi_reference or "").strip()
        if not ref:
            continue
        by_ref[(chunk.content_type, ref)] = chunk
    return list(by_ref.values())


def _get_ids_for_reference(store: BOFIPVectorStore, content_type: str, boi_reference: str) -> List[str]:
    """
    Fetch Chroma IDs matching one legal reference.

    Uses boi_reference filter first, then validates content_type in metadata.
    """
    try:
        res = store.collection.get(where={"boi_reference": boi_reference}, include=["metadatas"])
    except Exception as exc:
        logger.error(f"Failed to fetch IDs for boi_reference={boi_reference}: {exc}")
        return []

    ids = res.get("ids", []) or []
    metas = res.get("metadatas", []) or []
    matched = []
    for idx, metadata in zip(ids, metas):
        if str((metadata or {}).get("content_type", "")).upper() == content_type:
            matched.append(idx)
    return matched


def sync_legal_chunks(
    batch_size: int = 500,
    persist_dir: Path = CHROMA_DB_DIR,
    delta_file: Path | None = None,
) -> None:
    chunks_path = PROCESSED_DATA_DIR / "chunks.json"
    all_chunks = _load_chunks_for_counts(chunks_path)
    legal_chunks_full: List[BOFIPChunk] = [c for c in all_chunks if c.content_type in LEGAL_CONTENT_TYPES]
    bofip_count = len(all_chunks) - len(legal_chunks_full)

    logger.info(f"Loaded {len(all_chunks)} total chunks")
    logger.info(f"  BOFIP: {bofip_count}")
    logger.info(f"  Legal (CGI/LPF): {len(legal_chunks_full)}")

    store = BOFIPVectorStore(persist_dir=persist_dir)
    before_count = store.get_count()
    logger.info(f"Current Chroma count ({persist_dir}): {before_count}")

    if delta_file:
        legal_chunks_delta = _dedupe_legal_by_reference(_load_delta_chunks(delta_file))
        logger.info(f"Delta mode enabled with {len(legal_chunks_delta)} legal chunks from {delta_file}")

        if not legal_chunks_delta:
            logger.info("Delta file has no legal chunks; nothing to sync.")
            return

        # Delete existing vectors for changed legal references only.
        ids_to_delete = []
        for chunk in legal_chunks_delta:
            ids_to_delete.extend(_get_ids_for_reference(store, chunk.content_type, chunk.boi_reference))
        ids_to_delete = list(set(ids_to_delete))
        deleted = _delete_ids_in_batches(store, ids_to_delete)
        logger.info(f"Deleted {deleted} existing legal vectors for delta references")

        added = store.add_chunks(legal_chunks_delta, batch_size=batch_size)
    else:
        # Full legal refresh mode.
        legal_ids = []
        for content_type in LEGAL_CONTENT_TYPES:
            ids = _get_ids_for_content_type(store, content_type)
            logger.info(f"Found {len(ids)} existing {content_type} chunks in Chroma")
            legal_ids.extend(ids)

        deleted = _delete_ids_in_batches(store, legal_ids)
        after_delete_count = store.get_count()
        logger.info(f"Deleted {deleted} legal chunks. Chroma count now: {after_delete_count}")

        added = store.add_chunks(legal_chunks_full, batch_size=batch_size)

    final_count = store.get_count()
    expected_total = bofip_count + len(legal_chunks_full)

    logger.info(f"Added {added} legal chunks")
    logger.info(f"Final Chroma count ({persist_dir}): {final_count}")
    logger.info(f"Expected count from chunks.json: {expected_total}")

    if final_count != expected_total:
        logger.warning(
            "Count mismatch after legal sync. "
            f"Expected {expected_total}, got {final_count}. "
            "Run full vector rebuild if this persists."
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sync CGI/LPF chunks in ChromaDB")
    parser.add_argument(
        "--skip-restore",
        action="store_true",
        help="Do not auto-restore chroma_db_backup when current index looks incomplete",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for embedding/indexing legal chunks",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        help="Explicit Chroma persist directory (default: auto-select active/backup)",
    )
    parser.add_argument(
        "--delta-file",
        type=str,
        help=(
            "Optional legal delta JSON (e.g. data/processed/legi_chunks.json). "
            "When set, only changed legal references are upserted."
        ),
    )
    args = parser.parse_args()

    target_dir = Path(args.persist_dir) if args.persist_dir else resolve_target_dir(skip_restore=args.skip_restore)
    logger.info(f"Using Chroma target directory: {target_dir}")
    delta_file = Path(args.delta_file) if args.delta_file else None
    sync_legal_chunks(batch_size=args.batch_size, persist_dir=target_dir, delta_file=delta_file)


if __name__ == "__main__":
    main()

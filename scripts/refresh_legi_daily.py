"""
Automated daily LEGI refresh orchestration.

Workflow (KISS):
1) Ensure legal baseline is healthy (run latest-full when required)
2) Apply latest daily delta (safe upsert mode)
3) Rebuild BM25
4) Sync legal vectors in Chroma
5) Persist refresh state for audit/debug

Usage:
    python scripts/refresh_legi_daily.py
    python scripts/refresh_legi_daily.py --force-full
    python scripts/refresh_legi_daily.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pickle
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHROMA_DB_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.retrieval.bm25 import BM25_INDEX_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent
PROCESS_ARCHIVE_SCRIPT = PROJECT_ROOT / "scripts" / "process_legi_archive.py"
SYNC_LEGAL_SCRIPT = PROJECT_ROOT / "scripts" / "sync_legal_chunks.py"
STATE_FILE = PROCESSED_DATA_DIR / "legi_refresh_state.json"
CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"
LEGI_DELTA_FILE = PROCESSED_DATA_DIR / "legi_chunks.json"


def _parse_as_of(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format YYYY-MM-DD."
        ) from exc


def _latest_local_archive(pattern: str) -> str:
    legi_dir = RAW_DATA_DIR / "legi"
    if not legi_dir.exists():
        return ""
    matches = sorted(p.name for p in legi_dir.glob(pattern))
    return matches[-1] if matches else ""


def _chunks_counts() -> Dict[str, int]:
    if not CHUNKS_FILE.exists():
        return {"total": 0, "CGI": 0, "LPF": 0}

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    cgi = 0
    lpf = 0
    for chunk in chunks:
        content_type = str(chunk.get("content_type", "")).upper()
        if content_type == "CGI":
            cgi += 1
        elif content_type == "LPF":
            lpf += 1
    return {"total": len(chunks), "CGI": cgi, "LPF": lpf}


def _bm25_count() -> int:
    if not BM25_INDEX_FILE.exists():
        return -1
    try:
        with open(BM25_INDEX_FILE, "rb") as f:
            data = pickle.load(f)
        chunks = data.get("chunks", [])
        return len(chunks)
    except Exception as exc:
        logger.warning(f"Could not read BM25 index count: {exc}")
        return -1


def _chroma_count() -> int:
    db_path = CHROMA_DB_DIR / "chroma.sqlite3"
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
        logger.warning(f"Could not read Chroma count: {exc}")
        return -1


def _run_command(cmd: List[str], dry_run: bool = False) -> None:
    cmd_display = " ".join(cmd)
    logger.info(f"Running: {cmd_display}")
    if dry_run:
        return

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.stdout:
        logger.info(proc.stdout.strip())
    if proc.stderr:
        logger.warning(proc.stderr.strip())

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {cmd_display}")


def _save_state(payload: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved state: {STATE_FILE}")


def _build_process_cmd(archive_mode: str, as_of: dt.date, include_future: bool) -> List[str]:
    cmd = [
        sys.executable,
        str(PROCESS_ARCHIVE_SCRIPT),
        "--archive",
        archive_mode,
        "--append",
        "--allow-empty",
        "--as-of",
        as_of.isoformat(),
    ]
    if include_future:
        cmd.append("--include-future")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate daily LEGI refresh")
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=dt.date.today(),
        help="Reference date for legal version selection (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--include-future",
        action="store_true",
        help="Include nearest future version when no active version exists",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Force full snapshot ingestion before daily delta",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip full snapshot step",
    )
    parser.add_argument(
        "--skip-delta",
        action="store_true",
        help="Skip daily delta step",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 rebuild",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip Chroma legal sync",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing",
    )
    args = parser.parse_args()

    started_at = dt.datetime.now().isoformat()
    actions: List[str] = []

    initial_counts = _chunks_counts()
    logger.info(f"Initial counts: {initial_counts}")

    latest_full_local = _latest_local_archive("Freemium_legi_global_*.tar.gz")
    has_full_local = bool(latest_full_local)
    legal_count = initial_counts["CGI"] + initial_counts["LPF"]

    need_full = False
    if not args.skip_full:
        need_full = args.force_full or (legal_count < 1000) or (not has_full_local)

    try:
        full_ingested = False
        delta_attempted = False

        if need_full:
            _run_command(
                _build_process_cmd(
                    archive_mode="latest-full",
                    as_of=args.as_of,
                    include_future=args.include_future,
                ),
                dry_run=args.dry_run,
            )
            full_ingested = True
            actions.append("full_snapshot_ingested")
        else:
            actions.append("full_snapshot_skipped")

        if not args.skip_delta:
            _run_command(
                _build_process_cmd(
                    archive_mode="latest",
                    as_of=args.as_of,
                    include_future=args.include_future,
                ),
                dry_run=args.dry_run,
            )
            delta_attempted = True
            actions.append("daily_delta_applied")
        else:
            actions.append("daily_delta_skipped")

        if not args.skip_bm25:
            _run_command(
                [sys.executable, "-m", "src.retrieval.bm25", "--rebuild"],
                dry_run=args.dry_run,
            )
            actions.append("bm25_rebuilt")
        else:
            actions.append("bm25_skipped")

        if not args.skip_sync:
            sync_cmd = [sys.executable, str(SYNC_LEGAL_SCRIPT)]
            # If we only ran daily delta (no full snapshot), do fast incremental sync.
            if delta_attempted and not full_ingested:
                sync_cmd.extend(["--delta-file", str(LEGI_DELTA_FILE)])
                actions.append("chroma_legal_synced_delta")
            else:
                actions.append("chroma_legal_synced_full")
            _run_command(
                sync_cmd,
                dry_run=args.dry_run,
            )
        else:
            actions.append("chroma_sync_skipped")

        final_counts = _chunks_counts() if not args.dry_run else initial_counts
        state = {
            "timestamp": dt.datetime.now().isoformat(),
            "started_at": started_at,
            "as_of": args.as_of.isoformat(),
            "include_future": args.include_future,
            "dry_run": args.dry_run,
            "actions": actions,
            "status": "ok",
            "initial_counts": initial_counts,
            "final_counts": final_counts,
            "bm25_count": _bm25_count() if not args.dry_run else -1,
            "chroma_count": _chroma_count() if not args.dry_run else -1,
            "latest_full_local": _latest_local_archive("Freemium_legi_global_*.tar.gz"),
            "latest_delta_local": _latest_local_archive("LEGI_*.tar.gz"),
        }
        _save_state(state)

        logger.info("LEGI daily refresh completed.")
        logger.info(f"Final counts: {state['final_counts']}")
        logger.info(f"BM25 count: {state['bm25_count']} | Chroma count: {state['chroma_count']}")
    except Exception as exc:
        failure_state = {
            "timestamp": dt.datetime.now().isoformat(),
            "started_at": started_at,
            "as_of": args.as_of.isoformat(),
            "include_future": args.include_future,
            "dry_run": args.dry_run,
            "actions": actions,
            "status": "failed",
            "error": str(exc),
            "initial_counts": initial_counts,
            "latest_full_local": _latest_local_archive("Freemium_legi_global_*.tar.gz"),
            "latest_delta_local": _latest_local_archive("LEGI_*.tar.gz"),
        }
        _save_state(failure_state)
        raise


if __name__ == "__main__":
    main()

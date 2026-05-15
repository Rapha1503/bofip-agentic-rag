"""
Process LEGI tar archives and build legal chunks (CGI/LPF).

This script is the production-oriented path for LEGI migration:
- parses official DILA LEGI archives directly (.tar.gz)
- selects "as-of" article versions
- writes BOFIPChunk-compatible JSON
- optionally replaces existing CGI/LPF chunks in chunks.json

Usage examples:
    python scripts/process_legi_archive.py
    python scripts/process_legi_archive.py --archive latest --append
    python scripts/process_legi_archive.py --archive LEGI_20260209-211306.tar.gz --codes CGI
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_pipeline.legi_tar_parser import LEGITarCodeParser, DEFAULT_CODE_IDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


LEGI_INDEX_URL = "https://echanges.dila.gouv.fr/OPENDATA/LEGI/"
DAILY_ARCHIVE_PATTERN = re.compile(r"(LEGI_\d{8}-\d{6}\.tar\.gz)")
FULL_ARCHIVE_PATTERN = re.compile(r"(Freemium_legi_global_\d{8}-\d{6}\.tar\.gz)")


def _fetch_latest_archive_name(index_url: str = LEGI_INDEX_URL, full: bool = False) -> str:
    try:
        with urlopen(index_url, timeout=30) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to fetch LEGI index {index_url}: {exc}") from exc

    pattern = FULL_ARCHIVE_PATTERN if full else DAILY_ARCHIVE_PATTERN
    names = sorted(set(pattern.findall(html)))
    if not names:
        kind = "full" if full else "daily"
        raise RuntimeError(f"No {kind} LEGI archive names found at {index_url}")
    return names[-1]


def _find_latest_local_archive(raw_legi_dir: Path, full: bool = False) -> Optional[str]:
    if not raw_legi_dir.exists():
        return None
    pattern = FULL_ARCHIVE_PATTERN if full else DAILY_ARCHIVE_PATTERN
    names = [p.name for p in raw_legi_dir.glob("*.tar.gz") if pattern.fullmatch(p.name)]
    if not names:
        return None
    return sorted(names)[-1]


def _download_archive(archive_name: str, target_path: Path, index_url: str = LEGI_INDEX_URL) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{index_url}{archive_name}"
    logger.info(f"Downloading LEGI archive: {url}")

    remote_size = _get_remote_size(url)
    _download_with_resume(url=url, target_path=target_path, remote_size=remote_size)
    logger.info(f"Downloaded to: {target_path} ({target_path.stat().st_size} bytes)")


def _get_remote_size(url: str) -> int:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            return int(content_length) if content_length else 0
    except Exception:
        return 0


def _download_with_resume(url: str, target_path: Path, remote_size: int = 0, max_attempts: int = 6) -> None:
    for attempt in range(1, max_attempts + 1):
        existing = target_path.stat().st_size if target_path.exists() else 0
        if remote_size and existing >= remote_size:
            logger.info("Archive already present locally with expected size; skipping download")
            return

        headers = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            logger.info(f"Resuming download from byte {existing}")

        req = Request(url, headers=headers)

        try:
            with urlopen(req, timeout=180) as response:
                status = getattr(response, "status", None)

                # Server ignored Range and sent full content.
                if existing > 0 and status == 200:
                    logger.warning("Server does not support resume for this request; restarting full download")
                    existing = 0

                mode = "ab" if existing > 0 and status == 206 else "wb"
                with open(target_path, mode) as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            current_size = target_path.stat().st_size if target_path.exists() else 0
            if not remote_size or current_size >= remote_size:
                return

            logger.warning(
                f"Download incomplete after attempt {attempt}/{max_attempts} "
                f"({current_size}/{remote_size} bytes). Retrying..."
            )
            if attempt < max_attempts:
                sleep_s = min(120, 10 * attempt)
                logger.info(f"Waiting {sleep_s}s before next attempt...")
                time.sleep(sleep_s)
        except (HTTPError, URLError, TimeoutError) as exc:
            logger.warning(f"Download attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                sleep_s = min(120, 15 * attempt)
                logger.info(f"Waiting {sleep_s}s before retry...")
                time.sleep(sleep_s)

    current_size = target_path.stat().st_size if target_path.exists() else 0
    if remote_size:
        raise RuntimeError(
            f"Failed downloading {url}: {current_size}/{remote_size} bytes after {max_attempts} attempts"
        )
    raise RuntimeError(f"Failed downloading {url} after {max_attempts} attempts")


def _resolve_archive_path(archive_arg: str, raw_legi_dir: Path, auto_download: bool = True) -> Path:
    """
    Resolve archive path from:
    - explicit local path
    - filename under data/raw/legi/
    - 'latest' (fetch from index, optionally download)
    """
    lower_arg = archive_arg.lower()

    if lower_arg in ("latest", "latest-full"):
        is_full = lower_arg == "latest-full"
        try:
            latest_name = _fetch_latest_archive_name(full=is_full)
        except RuntimeError as exc:
            latest_name = _find_latest_local_archive(raw_legi_dir=raw_legi_dir, full=is_full)
            if not latest_name:
                raise
            logger.warning(f"{exc} | Falling back to latest local archive: {latest_name}")
        archive_path = raw_legi_dir / latest_name
        if auto_download:
            _download_archive(latest_name, archive_path)
        elif archive_path.exists():
            logger.info(f"Using cached latest archive: {archive_path}")
        else:
            raise FileNotFoundError(
                f"Latest archive not found locally: {archive_path}. "
                "Rerun with --download."
            )
        return archive_path

    candidate = Path(archive_arg)
    if candidate.exists():
        return candidate

    # Try relative to raw LEGI dir.
    candidate = raw_legi_dir / archive_arg
    if candidate.exists():
        return candidate

    # If archive_arg looks like a LEGI filename, download it.
    if DAILY_ARCHIVE_PATTERN.fullmatch(archive_arg) or FULL_ARCHIVE_PATTERN.fullmatch(archive_arg):
        if auto_download:
            _download_archive(archive_arg, candidate)
            return candidate
        raise FileNotFoundError(
            f"Archive not found locally: {candidate}. Rerun with --download."
        )

    raise FileNotFoundError(f"Archive not found: {archive_arg}")


def _save_chunks(chunks: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def _append_replace_sources(new_chunks: List[dict], main_chunks_path: Path, replaced_sources: List[str]) -> None:
    if main_chunks_path.exists():
        with open(main_chunks_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    replaced = {s.upper() for s in replaced_sources}
    kept = [
        c for c in existing
        if str(c.get("content_type", "")).upper() not in replaced
    ]
    removed = len(existing) - len(kept)

    combined = kept + new_chunks
    with open(main_chunks_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Updated {main_chunks_path}: total={len(combined)} "
        f"(removed={removed} from {sorted(replaced)}, added={len(new_chunks)})"
    )


def _append_delta_upsert(new_chunks: List[dict], main_chunks_path: Path, target_sources: List[str]) -> None:
    """
    Safe append mode for daily LEGI deltas:
    - replace only matching legal article references
    - add new legal article references
    - keep untouched legal references
    """
    if main_chunks_path.exists():
        with open(main_chunks_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    target = {s.upper() for s in target_sources}
    deduped_new = {}
    for chunk in new_chunks:
        source = str(chunk.get("content_type", "")).upper()
        if source not in target:
            continue
        ref = str(chunk.get("boi_reference", "")).strip()
        if not ref:
            continue
        deduped_new[(source, ref)] = chunk

    kept = []
    replaced = 0
    for chunk in existing:
        source = str(chunk.get("content_type", "")).upper()
        ref = str(chunk.get("boi_reference", "")).strip()
        if source in target and (source, ref) in deduped_new:
            replaced += 1
            continue
        kept.append(chunk)

    combined = kept + list(deduped_new.values())
    with open(main_chunks_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Updated {main_chunks_path} in delta-upsert mode: total={len(combined)} "
        f"(replaced={replaced}, upserted={len(deduped_new)}, sources={sorted(target)})"
    )


def _infer_archive_mode(archive_path: Path) -> str:
    name = archive_path.name
    if FULL_ARCHIVE_PATTERN.fullmatch(name):
        return "full_replace"
    if DAILY_ARCHIVE_PATTERN.fullmatch(name):
        return "delta_upsert"
    return "full_replace"


def _parse_as_of(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format YYYY-MM-DD."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Process LEGI archive for CGI/LPF")
    parser.add_argument(
        "--archive",
        default="latest",
        help="Archive path, archive filename, 'latest' (daily), or 'latest-full'",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow download when archive is missing locally",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=["CGI", "LPF"],
        help="Target legal sources to extract (default: CGI LPF)",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=dt.date.today(),
        help="Reference date for active article versions (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--include-future",
        action="store_true",
        help="If no active version exists for an article, include nearest future version",
    )
    parser.add_argument(
        "--output-file",
        default=str(PROCESSED_DATA_DIR / "legi_chunks.json"),
        help="Output JSON file for parsed LEGI chunks",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Replace existing chunks for processed sources in data/processed/chunks.json",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not fail when no target chunks are found (useful for daily delta runs)",
    )
    args = parser.parse_args()

    target_sources = []
    for code in args.codes:
        code_upper = code.upper()
        if code_upper not in DEFAULT_CODE_IDS:
            raise ValueError(f"Unsupported code '{code}'. Supported: {sorted(DEFAULT_CODE_IDS)}")
        if code_upper not in target_sources:
            target_sources.append(code_upper)

    raw_legi_dir = RAW_DATA_DIR / "legi"
    archive_path = _resolve_archive_path(
        archive_arg=args.archive,
        raw_legi_dir=raw_legi_dir,
        auto_download=args.download or args.archive.lower() in ("latest", "latest-full"),
    )

    parser_obj = LEGITarCodeParser()
    per_source = parser_obj.parse_archive(
        archive_path=archive_path,
        target_sources=target_sources,
        as_of=args.as_of,
        include_future=args.include_future,
    )

    merged_chunks = []
    found_sources = []
    for source in target_sources:
        source_chunks = [c.to_dict() for c in per_source.get(source, [])]
        logger.info(f"{source}: {len(source_chunks)} selected chunks")
        if source_chunks:
            found_sources.append(source)
            merged_chunks.extend(source_chunks)
        else:
            logger.warning(
                f"{source}: no chunks found in {archive_path.name}. "
                "This can happen with daily deltas if the code did not change."
            )

    if not merged_chunks:
        if args.allow_empty:
            output_file = Path(args.output_file)
            _save_chunks([], output_file)
            logger.info(
                "No LEGI chunks were produced for requested sources. "
                "Nothing to update."
            )
            return
        logger.error("No LEGI chunks were produced.")
        sys.exit(1)

    output_file = Path(args.output_file)
    _save_chunks(merged_chunks, output_file)

    if args.append:
        main_chunks_path = PROCESSED_DATA_DIR / "chunks.json"
        append_mode = _infer_archive_mode(archive_path)
        if append_mode == "delta_upsert":
            _append_delta_upsert(
                new_chunks=merged_chunks,
                main_chunks_path=main_chunks_path,
                target_sources=found_sources,
            )
        else:
            _append_replace_sources(
                new_chunks=merged_chunks,
                main_chunks_path=main_chunks_path,
                replaced_sources=found_sources,
            )

        logger.info("Next steps:")
        logger.info("  1. Rebuild BM25: python -m src.retrieval.bm25 --rebuild")
        logger.info("  2. Sync legal vectors: python scripts/sync_legal_chunks.py")


if __name__ == "__main__":
    main()

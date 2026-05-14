from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def build_manifest(
    *,
    raw_root: Path,
    parser_version: str,
    chunker_version: str = "0.0.0",
    index_version: str = "0.0.0",
    benchmark_version: str = "0.0.0",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "raw_root": str(raw_root),
        "raw_root_exists": raw_root.exists(),
        "parser_version": parser_version,
        "chunker_version": chunker_version,
        "index_version": index_version,
        "benchmark_version": benchmark_version,
    }
    if extra:
        payload.update(extra)
    return payload


def fingerprint_paths(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        hasher.update(str(path).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
    return hasher.hexdigest()

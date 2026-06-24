from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "docs" / "full_corpus_manifest.json"


@dataclass(frozen=True)
class CheckResult:
    name: str
    path: str
    ok: bool
    detail: str


RUNTIME_FILES = [
    ("raw_docs", Path("data/interim/raw_docs.jsonl"), "expected full BOFiP parsed runtime rows"),
    ("chunks", Path("data/interim/chunks.jsonl"), "expected full-corpus section-window rows"),
    (
        "doc_embeddings",
        Path("data/interim/doc_dense_cache.npy"),
        "expected full-corpus document embeddings with 1024 dimensions",
    ),
    (
        "chunk_embeddings",
        Path("data/interim/chunk_dense_cache.npy"),
        "expected full-corpus chunk embeddings with 1024 dimensions",
    ),
]

TRACKED_FILES = [
    ("eval_queries", Path("data/interim/eval_queries_v1.jsonl"), "tracked evaluation queries"),
    ("passage_gold", Path("data/interim/passage_gold_v3.jsonl"), "tracked passage gold"),
]

EXPECTED_MODELS = [
    ("e5_large_model", Path("data/models/intfloat--multilingual-e5-large"), "local E5-large model directory"),
]

OPTIONAL_MODELS = [
    # No optional local model is part of the default artifact contract. The
    # reranker can still be downloaded by sentence-transformers when enabled.
]

EXPECTED_JSONL_COUNTS = {
    "eval_queries": 50,
    "passage_gold": 50,
}
MIN_JSONL_COUNTS = {
    "raw_docs": 9048,
    "chunks": 79000,
}
MIN_NPY_ROWS = {
    "doc_embeddings": 9048,
    "chunk_embeddings": 79000,
}
EXPECTED_EMBEDDING_DIM = 1024


def _manifest_artifacts() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        str(item.get("path", "")).replace("\\", "/"): item
        for item in manifest.get("artifacts", [])
        if item.get("path")
    }


def _count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("not a NumPy .npy file")
        major = handle.read(1)[0]
        handle.read(1)
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        else:
            header_len = struct.unpack("<I", handle.read(4))[0]
        header = handle.read(header_len).decode("latin1")
    payload = ast.literal_eval(header)
    return tuple(payload["shape"])


def _check_file(
    name: str,
    rel_path: Path,
    detail: str,
    *,
    deep: bool,
    manifest_artifacts: dict[str, dict],
) -> CheckResult:
    path = PROJECT_ROOT / rel_path
    display = rel_path.as_posix()
    if not path.exists():
        return CheckResult(name, display, False, f"missing; {detail}")
    if not path.is_file():
        return CheckResult(name, display, False, "exists but is not a file")
    if not deep:
        size_mb = path.stat().st_size / (1024 * 1024)
        return CheckResult(name, display, True, f"present ({size_mb:.1f} MB)")

    manifest_item = manifest_artifacts.get(display)
    if manifest_item and "rows" in manifest_item:
        count = _count_jsonl(path)
        expected = int(manifest_item["rows"])
        return CheckResult(name, display, count == expected, f"{count} rows; manifest expects {expected}")
    if manifest_item and "shape" in manifest_item:
        shape = _npy_shape(path)
        expected = tuple(int(value) for value in manifest_item["shape"])
        return CheckResult(name, display, shape == expected, f"shape {shape}; manifest expects {expected}")

    if name in EXPECTED_JSONL_COUNTS:
        count = _count_jsonl(path)
        expected = EXPECTED_JSONL_COUNTS[name]
        return CheckResult(name, display, count == expected, f"{count} rows; expected {expected}")
    if name in MIN_JSONL_COUNTS:
        count = _count_jsonl(path)
        minimum = MIN_JSONL_COUNTS[name]
        return CheckResult(name, display, count >= minimum, f"{count} rows; expected at least {minimum}")
    if name in MIN_NPY_ROWS:
        shape = _npy_shape(path)
        expected_rows = MIN_NPY_ROWS[name]
        ok = len(shape) == 2 and shape[0] >= expected_rows and shape[1] == EXPECTED_EMBEDDING_DIM
        return CheckResult(
            name,
            display,
            ok,
            f"shape {shape}; expected >= {expected_rows} rows and {EXPECTED_EMBEDDING_DIM} dims",
        )
    return CheckResult(name, display, True, "present")


def _check_model(name: str, rel_path: Path, detail: str) -> CheckResult:
    path = PROJECT_ROOT / rel_path
    display = rel_path.as_posix()
    if not path.exists():
        return CheckResult(name, display, False, f"missing; {detail}")
    if not path.is_dir():
        return CheckResult(name, display, False, "exists but is not a directory")
    return CheckResult(name, display, True, "present")


def run_checks(*, deep: bool, skip_models: bool, tracked_only: bool) -> list[CheckResult]:
    files = TRACKED_FILES if tracked_only else [*RUNTIME_FILES, *TRACKED_FILES]
    manifest_artifacts = _manifest_artifacts() if deep else {}
    results = [
        _check_file(name, rel_path, detail, deep=deep, manifest_artifacts=manifest_artifacts)
        for name, rel_path, detail in files
    ]
    if not skip_models:
        results.extend(_check_model(name, rel_path, detail) for name, rel_path, detail in EXPECTED_MODELS)
        if not tracked_only:
            results.extend(
                _check_model(name, rel_path, detail)
                for name, rel_path, detail in OPTIONAL_MODELS
                if (PROJECT_ROOT / rel_path).exists()
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local BOFiP Agentic RAG setup and required artifacts.")
    parser.add_argument("--deep", action="store_true", help="Count JSONL rows and validate .npy shapes.")
    parser.add_argument("--skip-models", action="store_true", help="Do not require local model directories.")
    parser.add_argument("--tracked-only", action="store_true", help="Only check files that are committed to Git.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    results = run_checks(deep=args.deep, skip_models=args.skip_models, tracked_only=args.tracked_only)
    ok = all(result.ok for result in results)

    if args.json:
        print(json.dumps({"ok": ok, "checks": [asdict(result) for result in results]}, indent=2))
    else:
        print("BOFiP Agentic RAG setup check")
        print(f"Project root: {PROJECT_ROOT}")
        for result in results:
            mark = "OK" if result.ok else "MISSING"
            print(f"[{mark}] {result.name}: {result.path} - {result.detail}")
        if not ok:
            print("\nPlace the missing full-corpus artifacts locally, then rerun this command.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

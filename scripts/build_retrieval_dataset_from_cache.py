"""
Build an expanded retrieval benchmark dataset from cached LLM responses.

Goal:
- Keep existing gold questions from scripts/test_questions.json
- Add "silver" questions discovered in data/cache/llm_responses
- Extract cited BOI/CGI/LPF references from model responses
- Keep only references that exist in the indexed corpus

Usage:
    python scripts/build_retrieval_dataset_from_cache.py
    python scripts/build_retrieval_dataset_from_cache.py --out scripts/test_questions_expanded.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25 import get_bm25_index

DEFAULT_BASE_DATASET = Path(__file__).parent / "test_questions.json"
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "llm_responses"
DEFAULT_OUTPUT = Path(__file__).parent / "test_questions_expanded.json"

BOI_RE = re.compile(r"\b(BOI-[A-Z0-9-]+)\b", re.IGNORECASE)
LEGAL_RE = re.compile(
    r"\b((?:CGI|LPF)\s+Art\.\s*[A-Z]?\*?\.?\s*\d[\dA-Z\-\s\.]*?(?:\s+(?:BIS|TER|QUATER|QUINQUIES|SEXIES|SEPTIES|OCTIES|NONIES|DECIES))?)\b",
    re.IGNORECASE,
)


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def _normalize_reference_for_match(reference: str) -> str:
    """Normalize references so matching is robust to spacing/case variance."""
    ref = _normalize_spaces(reference).replace("’", "'").rstrip(".,;:")
    return ref.upper()


def _canonicalize_expected_ref(reference: str) -> str:
    """
    Canonical expected reference stored in dataset.

    BOI: strip trailing date segment (YYYYMMDD) to avoid date lock-in.
    Legal: normalize prefix formatting.
    """
    ref = _normalize_spaces(reference).rstrip(".,;:")
    upper = ref.upper()

    if upper.startswith("BOI-"):
        ref = upper
        ref = re.sub(r"-\d{8}$", "", ref)
        return ref

    if upper.startswith("CGI ART.") or upper.startswith("LPF ART."):
        source = "CGI" if upper.startswith("CGI") else "LPF"
        article = ref.split("Art.", 1)[-1] if "Art." in ref else ref.split("ART.", 1)[-1]
        article = _normalize_spaces(article)
        return f"{source} Art. {article}"

    return upper


def _extract_references_from_response(response_text: str) -> List[str]:
    """Extract BOI/CGI/LPF references from free-form answer text."""
    text = response_text or ""

    refs: List[str] = []
    refs.extend(m.group(1) for m in BOI_RE.finditer(text))
    refs.extend(m.group(1) for m in LEGAL_RE.finditer(text))

    cleaned: List[str] = []
    seen: Set[str] = set()
    for ref in refs:
        # Drop paragraph marker if present in captured snippet.
        ref = ref.split("§", 1)[0].strip()
        canonical = _canonicalize_expected_ref(ref)
        key = _normalize_reference_for_match(canonical)
        if not canonical or key in seen:
            continue
        seen.add(key)
        cleaned.append(canonical)

    return cleaned


def _reference_exists_in_corpus(expected_ref: str, corpus_refs_norm: List[str]) -> bool:
    """True when at least one corpus reference starts with expected prefix."""
    target = _normalize_reference_for_match(expected_ref)
    return any(c.startswith(target) for c in corpus_refs_norm)


def _infer_domain(refs: List[str]) -> str:
    for ref in refs:
        if ref.startswith("BOI-"):
            parts = ref.split("-")
            if len(parts) >= 2:
                return parts[1]
        if ref.startswith("CGI Art.") or ref.startswith("LPF Art."):
            return "LEGAL"
    return "UNKNOWN"


def _load_base_questions(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("questions", [])
    if isinstance(data, list):
        return data
    return []


def build_dataset(
    base_dataset: Path,
    cache_dir: Path,
    output_path: Path,
    max_refs_per_question: int = 3,
) -> Dict[str, Any]:
    bm25 = get_bm25_index()
    corpus_refs_norm = [
        _normalize_reference_for_match(chunk.boi_reference)
        for chunk in bm25.chunks
        if chunk.boi_reference
    ]

    base_questions = _load_base_questions(base_dataset)
    by_query_key: Dict[str, Dict[str, Any]] = {}
    for q in base_questions:
        query = _normalize_spaces(q.get("query", ""))
        if query:
            by_query_key[query.lower()] = q

    cache_files = sorted(cache_dir.glob("*.json"))
    added = 0
    skipped_no_refs = 0
    skipped_invalid_refs = 0
    skipped_duplicate_query = 0
    silver_idx = 1

    for cache_file in cache_files:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                row = json.load(f)
        except Exception:
            continue

        query = _normalize_spaces(row.get("question", ""))
        if not query:
            continue
        query_key = query.lower()
        if query_key in by_query_key:
            skipped_duplicate_query += 1
            continue

        refs = _extract_references_from_response(row.get("response", ""))
        if not refs:
            skipped_no_refs += 1
            continue

        valid_refs = [r for r in refs if _reference_exists_in_corpus(r, corpus_refs_norm)]
        if not valid_refs:
            skipped_invalid_refs += 1
            continue

        valid_refs = valid_refs[:max_refs_per_question]

        candidate = {
            "id": f"S{silver_idx:03d}",
            "query": query,
            "expected_boi": valid_refs,
            "domain": _infer_domain(valid_refs),
            "complexity": "Unknown",
            "source": "llm_cache_silver",
            "notes": f"Auto-extracted from {cache_file.name}",
        }
        by_query_key[query_key] = candidate
        silver_idx += 1
        added += 1

    questions = list(by_query_key.values())
    payload = {
        "version": "1.1",
        "description": "Expanded retrieval benchmark (gold + silver from llm cache)",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "base_dataset": str(base_dataset),
        "cache_dir": str(cache_dir),
        "stats": {
            "base_questions": len(base_questions),
            "cache_files": len(cache_files),
            "silver_added": added,
            "total_questions": len(questions),
            "skipped_duplicate_query": skipped_duplicate_query,
            "skipped_no_refs": skipped_no_refs,
            "skipped_invalid_refs": skipped_invalid_refs,
        },
        "questions": questions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded retrieval dataset from LLM cache")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE_DATASET, help="Gold dataset path")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="LLM cache directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output dataset path")
    parser.add_argument(
        "--max-refs-per-question",
        type=int,
        default=3,
        help="Maximum expected references kept per question",
    )
    args = parser.parse_args()

    payload = build_dataset(
        base_dataset=args.base.resolve(),
        cache_dir=args.cache_dir.resolve(),
        output_path=args.out.resolve(),
        max_refs_per_question=max(1, args.max_refs_per_question),
    )

    print("=" * 72)
    print("RETRIEVAL DATASET BUILD SUMMARY")
    print("=" * 72)
    print(f"Base dataset: {payload['base_dataset']}")
    print(f"Cache dir: {payload['cache_dir']}")
    print(f"Total questions: {payload['stats']['total_questions']}")
    print(f"Silver added: {payload['stats']['silver_added']}")
    print(f"Skipped (duplicate query): {payload['stats']['skipped_duplicate_query']}")
    print(f"Skipped (no refs): {payload['stats']['skipped_no_refs']}")
    print(f"Skipped (invalid refs): {payload['stats']['skipped_invalid_refs']}")
    print(f"Saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()

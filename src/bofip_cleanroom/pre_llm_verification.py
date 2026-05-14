from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_STAGE1_HIT_FIELDS = (
    "rank",
    "score",
    "boi_reference",
    "title",
    "sources",
    "ranks",
)

REQUIRED_STAGE2_CHUNK_FIELDS = (
    "citation_id",
    "boi_reference",
    "title",
    "publication_date",
    "section_path",
    "chunk_id",
    "chunk_kind",
    "text",
)


def infer_documents_root(raw_xml_path: str | Path) -> Path:
    path = Path(raw_xml_path).resolve()
    for parent in (path, *path.parents):
        if parent.name == "documents":
            return parent
    raise ValueError(f"Could not infer BOFiP documents root from: {path}")


def infer_doc_dense_cache_prefix(
    *,
    interim_dir: Path,
    document_count: int,
    dense_mode: str,
    model_name: str,
) -> Path | None:
    model_lower = model_name.lower()
    if "e5-large" in model_lower or "e5large" in model_lower:
        suffix = "e5large"
    elif "e5-base" in model_lower or "multilingual-e5-base" in model_lower or "e5" in model_lower:
        suffix = "e5"
    else:
        return None

    prefix = interim_dir / f"doc_dense_cache_{document_count}_{dense_mode}_{suffix}"
    return prefix if prefix.with_suffix(".npy").exists() else None


def build_stage1_replay_command(
    *,
    python_executable: str,
    project_root: Path,
    interim_dir: Path,
    reference_payload: dict[str, Any],
    output_path: Path,
) -> list[str]:
    command = [
        python_executable,
        str((project_root / "scripts" / "phase3_doc_multiview_hybrid_eval.py").resolve()),
        "--raw-docs",
        str(Path(reference_payload["raw_docs_path"]).resolve()),
        "--queries",
        str(Path(reference_payload["queries_path"]).resolve()),
        "--model",
        str(reference_payload["model_name"]),
        "--lexical-modes",
        ",".join(reference_payload["lexical_modes"]),
        "--dense-mode",
        str(reference_payload["dense_mode"]),
        "--device",
        "cpu",
        "--weights",
        ",".join(f"{name}={value}" for name, value in reference_payload["weights"].items()),
        "--candidate-k",
        str(reference_payload["candidate_k"]),
        "--rank-constant",
        str(reference_payload["rank_constant"]),
        "--fusion-mode",
        str(reference_payload["fusion_mode"]),
        "--confidence-top-n",
        str(reference_payload["confidence_top_n"]),
        "--confidence-alpha",
        str(reference_payload["confidence_alpha"]),
        "--score-alpha",
        str(reference_payload["score_alpha"]),
        "--output",
        str(output_path.resolve()),
    ]
    if reference_payload.get("stem_lexical"):
        command.append("--stem-lexical")
    if reference_payload.get("query_acronym_expansion"):
        command.extend(
            [
                "--query-acronym-expansion",
                "--query-acronym-max-expansions",
                str(reference_payload.get("query_acronym_max_expansions", 3)),
            ]
        )
    if reference_payload.get("local_title_rerank_top_n", 0) > 0:
        command.extend(
            [
                "--local-title-rerank-top-n",
                str(reference_payload["local_title_rerank_top_n"]),
                "--local-title-rerank-weight",
                str(reference_payload["local_title_rerank_weight"]),
            ]
        )
        if reference_payload.get("local_title_rerank_stem"):
            command.append("--local-title-rerank-stem")
    if reference_payload.get("specificity_rerank_top_n", 0) > 0:
        command.extend(
            [
                "--specificity-rerank-top-n",
                str(reference_payload["specificity_rerank_top_n"]),
                "--specificity-rerank-weight",
                str(reference_payload["specificity_rerank_weight"]),
            ]
        )
    cache_prefix = infer_doc_dense_cache_prefix(
        interim_dir=interim_dir,
        document_count=int(reference_payload["document_count"]),
        dense_mode=str(reference_payload["dense_mode"]),
        model_name=str(reference_payload["model_name"]),
    )
    if cache_prefix is not None:
        command.extend(["--cache-prefix", str(cache_prefix.resolve())])
    if reference_payload.get("chunk_dense_enabled"):
        command.extend(
            [
                "--chunk-dense-cache",
                str(Path(reference_payload["chunk_dense_cache"]).resolve()),
                "--chunks",
                str(Path(reference_payload["chunks_path"]).resolve()),
                "--chunk-dense-device",
                "cpu",
            ]
        )
    return command


def build_direct_passage_replay_command(
    *,
    python_executable: str,
    project_root: Path,
    reference_payload: dict[str, Any],
    replay_stage1_report_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        python_executable,
        str((project_root / "scripts" / "phase5_direct_chunk_eval.py").resolve()),
        "--chunks",
        str(Path(reference_payload["chunks_path"]).resolve()),
        "--stage1-report",
        str(replay_stage1_report_path.resolve()),
        "--queries",
        str(Path(reference_payload["queries_path"]).resolve()),
        "--top-docs",
        str(reference_payload["top_docs"]),
        "--chunks-per-doc",
        str(reference_payload["chunks_per_doc"]),
        "--max-chunks",
        str(reference_payload["max_chunks"]),
        "--chunk-mode",
        str(reference_payload["chunk_mode"]),
        "--output",
        str(output_path.resolve()),
    ]


def compare_numeric_metrics(
    *,
    expected: dict[str, Any],
    observed: dict[str, Any],
    metric_keys: list[str],
    tolerance: float,
) -> dict[str, Any]:
    drifts: dict[str, dict[str, float]] = {}
    passed = True
    for key in metric_keys:
        expected_value = float(expected[key])
        observed_value = float(observed[key])
        delta = observed_value - expected_value
        within_tolerance = abs(delta) <= tolerance
        drifts[key] = {
            "expected": expected_value,
            "observed": observed_value,
            "delta": delta,
            "within_tolerance": within_tolerance,
        }
        if not within_tolerance:
            passed = False
    return {
        "passed": passed,
        "tolerance": tolerance,
        "metrics": drifts,
    }


def summarize_chunk_document_coverage(
    raw_docs_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_document_ids = {
        str(row["document_id"])
        for row in raw_docs_rows
        if row.get("document_id")
    }
    chunk_document_ids = {
        str(row["document_id"])
        for row in chunk_rows
        if row.get("document_id")
    }
    raw_unique_boi_references = {
        str(row["boi_reference"])
        for row in raw_docs_rows
        if row.get("boi_reference")
    }
    chunk_unique_boi_references = {
        str(row["boi_reference"])
        for row in chunk_rows
        if row.get("boi_reference")
    }
    missing_document_ids = sorted(raw_document_ids - chunk_document_ids)
    extra_document_ids = sorted(chunk_document_ids - raw_document_ids)

    return {
        "raw_document_count": len(raw_docs_rows),
        "raw_document_id_count": len(raw_document_ids),
        "chunk_document_id_count": len(chunk_document_ids),
        "raw_unique_boi_reference_count": len(raw_unique_boi_references),
        "chunk_unique_boi_reference_count": len(chunk_unique_boi_references),
        "raw_duplicate_boi_reference_count": len(raw_docs_rows) - len(raw_unique_boi_references),
        "missing_document_id_count": len(missing_document_ids),
        "extra_document_id_count": len(extra_document_ids),
        "missing_document_ids_sample": missing_document_ids[:10],
        "extra_document_ids_sample": extra_document_ids[:10],
    }


def validate_retrieval_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top_level = (
        "query",
        "lexical_query",
        "acronym_expansions",
        "source_confidences",
        "stage1_hits",
        "family_selection",
        "stage2_chunks",
    )
    for field in required_top_level:
        if field not in payload:
            errors.append(f"missing top-level field: {field}")

    stage1_hits = payload.get("stage1_hits", [])
    if not isinstance(stage1_hits, list) or not stage1_hits:
        errors.append("stage1_hits must be a non-empty list")
    else:
        for index, hit in enumerate(stage1_hits, start=1):
            for field in REQUIRED_STAGE1_HIT_FIELDS:
                if field not in hit:
                    errors.append(f"stage1_hits[{index}] missing field: {field}")

    stage2_chunks = payload.get("stage2_chunks", [])
    if not isinstance(stage2_chunks, list) or not stage2_chunks:
        errors.append("stage2_chunks must be a non-empty list")
    else:
        citation_ids: list[int] = []
        chunk_ids: set[str] = set()
        for index, chunk in enumerate(stage2_chunks, start=1):
            for field in REQUIRED_STAGE2_CHUNK_FIELDS:
                if field not in chunk:
                    errors.append(f"stage2_chunks[{index}] missing field: {field}")
            citation_id = chunk.get("citation_id")
            if not isinstance(citation_id, int):
                errors.append(f"stage2_chunks[{index}] citation_id must be an integer")
            else:
                citation_ids.append(citation_id)
            chunk_id = chunk.get("chunk_id")
            if isinstance(chunk_id, str):
                if chunk_id in chunk_ids:
                    errors.append(f"duplicate chunk_id in stage2_chunks: {chunk_id}")
                chunk_ids.add(chunk_id)
            text = chunk.get("text", "")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"stage2_chunks[{index}] text must be non-empty")
        if citation_ids:
            expected_ids = list(range(1, len(citation_ids) + 1))
            if citation_ids != expected_ids:
                errors.append(
                    "stage2_chunks citation_id sequence must be contiguous starting at 1: "
                    f"observed={citation_ids}"
                )

    return errors


def summarize_order_match(expected: list[str], observed: list[str]) -> dict[str, Any]:
    return {
        "matches": expected == observed,
        "expected": expected,
        "observed": observed,
    }

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bofip_cleanroom.discovery import discover_content_documents
from bofip_cleanroom.jsonio import read_jsonl, write_json
from bofip_cleanroom.pre_llm_verification import (
    build_direct_passage_replay_command,
    build_stage1_replay_command,
    compare_numeric_metrics,
    infer_documents_root,
    summarize_chunk_document_coverage,
    summarize_order_match,
    validate_retrieval_payload,
)
from bofip_cleanroom.preview_runtime import Phase8bPreviewRuntime
from bofip_cleanroom.settings import INTERIM_DIR, REPORTS_DIR, ensure_data_dirs


REFERENCE_STAGE1_REPORTS = {
    "retrieval_queries_sample_1000_v3": REPORTS_DIR / "phase6_stage1_v3.json",
    "retrieval_queries_full_v4": REPORTS_DIR / "phase6_stage1_v4.json",
    "retrieval_queries_full_v5": REPORTS_DIR / "phase6_stage1_v5.json",
    "retrieval_queries_full_v6": REPORTS_DIR / "phase6_stage1_v6.json",
    "passage_gold_v2": REPORTS_DIR / "phase6_stage1_passage_gold_v2.json",
}

REFERENCE_STAGE1_PASSAGE_V1 = REPORTS_DIR / "phase6_stage1_passage_gold_v1.json"
REFERENCE_DIRECT_PASSAGE_V1 = REPORTS_DIR / "phase6_direct_passage_v1.json"
REFERENCE_DIRECT_PASSAGE_V2 = REPORTS_DIR / "phase6_direct_passage_v2.json"
REFERENCE_CROSS_SUMMARY = REPORTS_DIR / "phase6_cross_benchmark_summary.json"
REFERENCE_PHASE9_BATCH = REPORTS_DIR / "phase9_batch_preview_eval_gemini_v1.json"

REFERENCE_FAMILY_REPORTS = [
    REPORTS_DIR / "phase6_family_v3.json",
    REPORTS_DIR / "phase6_family_v4.json",
    REPORTS_DIR / "phase6_family_v5.json",
    REPORTS_DIR / "phase6_family_v6.json",
    REPORTS_DIR / "phase6_family_passage_v1.json",
    REPORTS_DIR / "phase6_family_passage_v2.json",
]

STAGE1_METRIC_KEYS = ["hit@1", "hit@3", "hit@5"]
DIRECT_PASSAGE_METRIC_KEYS = [
    "stage1_doc_hit@1",
    "stage1_doc_hit@3",
    "stage1_doc_hit@5",
    "stage2_doc_hit@1",
    "stage2_doc_hit@3",
    "stage2_doc_hit@5",
    "passage_hit@1",
    "passage_hit@3",
    "passage_hit@5",
]
CROSS_SUMMARY_NUMERIC_KEYS = [
    "stage1_doc_hit@1",
    "stage1_doc_hit@3",
    "stage1_doc_hit@5",
    "family_hit_rate",
    "family_doc_hit@1",
    "family_doc_hit@3",
    "family_doc_hit@5",
    "chunk_doc_hit@1",
    "chunk_doc_hit@3",
    "chunk_doc_hit@5",
    "passage_hit@1",
    "passage_hit@3",
    "passage_hit@5",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail_text(text: str, *, max_lines: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text.strip()
    return "\n".join(lines[-max_lines:]).strip()


def _status_from_messages(*, failures: list[str], warnings: list[str]) -> str:
    if failures:
        return "fail"
    if warnings:
        return "pass_with_warnings"
    return "pass"


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int = 7200,
    env: dict[str, str] | None = None,
) -> dict:
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        env=env,
        check=False,
    )
    duration = time.time() - start
    return {
        "command": command,
        "returncode": completed.returncode,
        "duration_seconds": round(duration, 2),
        "stdout_tail": _tail_text(completed.stdout),
        "stderr_tail": _tail_text(completed.stderr),
        "succeeded": completed.returncode == 0,
    }


def _build_reused_command_result(*, output_path: Path, reason: str) -> dict:
    return {
        "command": None,
        "returncode": 0,
        "duration_seconds": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
        "succeeded": True,
        "reused_existing_output": True,
        "reused_output_path": str(output_path.resolve()),
        "reason": reason,
    }


def _exception_payload(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _normalize_dataset_rows(summary: dict) -> dict[str, dict]:
    return {row["dataset"]: row for row in summary.get("datasets", [])}


def _compare_dataset_rows(expected_row: dict, observed_row: dict, *, tolerance: float) -> dict:
    failures: list[str] = []
    warnings: list[str] = []

    for field in ("query_count", "supported_query_count", "unsupported_query_count", "has_passage_gold"):
        if expected_row.get(field) != observed_row.get(field):
            failures.append(
                f"{field} mismatch: expected={expected_row.get(field)!r}, observed={observed_row.get(field)!r}"
            )

    numeric_expected = {
        key: expected_row[key]
        for key in CROSS_SUMMARY_NUMERIC_KEYS
        if key in expected_row and expected_row[key] is not None
    }
    numeric_observed = {
        key: observed_row[key]
        for key in numeric_expected
    }
    drift = compare_numeric_metrics(
        expected=numeric_expected,
        observed=numeric_observed,
        metric_keys=sorted(numeric_expected),
        tolerance=tolerance,
    )
    if not drift["passed"]:
        failures.append("numeric metric drift exceeds tolerance")

    return {
        "status": _status_from_messages(failures=failures, warnings=warnings),
        "failures": failures,
        "warnings": warnings,
        "drift": drift,
    }


def _compare_stage1_payload(reference_payload: dict, observed_payload: dict, *, tolerance: float) -> dict:
    comparison_failures: list[str] = []
    for field in ("document_count", "query_count", "supported_query_count", "unsupported_query_count"):
        if observed_payload.get(field) != reference_payload.get(field):
            comparison_failures.append(
                f"{field} mismatch: expected={reference_payload.get(field)!r}, observed={observed_payload.get(field)!r}"
            )
    metric_comparison = compare_numeric_metrics(
        expected=reference_payload["metrics"],
        observed=observed_payload["metrics"],
        metric_keys=STAGE1_METRIC_KEYS,
        tolerance=tolerance,
    )
    if not metric_comparison["passed"]:
        comparison_failures.append("stage1 metrics drift exceeds tolerance")
    return {
        "status": "pass" if not comparison_failures else "fail",
        "failures": comparison_failures,
        "metric_comparison": metric_comparison,
    }


def _compare_direct_passage_payload(reference_payload: dict, observed_payload: dict, *, tolerance: float) -> dict:
    comparison_failures: list[str] = []
    if observed_payload.get("query_count") != reference_payload.get("query_count"):
        comparison_failures.append(
            f"query_count mismatch: expected={reference_payload.get('query_count')!r}, observed={observed_payload.get('query_count')!r}"
        )
    metric_comparison = compare_numeric_metrics(
        expected=reference_payload["metrics"],
        observed=observed_payload["metrics"],
        metric_keys=DIRECT_PASSAGE_METRIC_KEYS,
        tolerance=tolerance,
    )
    if not metric_comparison["passed"]:
        comparison_failures.append("direct passage metrics drift exceeds tolerance")
    return {
        "status": "pass" if not comparison_failures else "fail",
        "failures": comparison_failures,
        "metric_comparison": metric_comparison,
    }


def _compare_cross_summary_payload(expected_summary: dict, observed_summary: dict, *, tolerance: float) -> dict:
    expected_rows = _normalize_dataset_rows(expected_summary)
    observed_rows = _normalize_dataset_rows(observed_summary)

    dataset_comparisons: dict[str, dict] = {}
    comparison_failures: list[str] = []
    for dataset, expected_row in expected_rows.items():
        observed_row = observed_rows.get(dataset)
        if observed_row is None:
            comparison_failures.append(f"dataset missing from replayed cross summary: {dataset}")
            continue
        row_comparison = _compare_dataset_rows(expected_row, observed_row, tolerance=tolerance)
        dataset_comparisons[dataset] = row_comparison
        if row_comparison["status"] == "fail":
            comparison_failures.append(f"cross summary dataset drift: {dataset}")

    unexpected_datasets = sorted(set(observed_rows) - set(expected_rows))
    for dataset in unexpected_datasets:
        comparison_failures.append(f"unexpected dataset present in replayed cross summary: {dataset}")

    if observed_summary.get("gate_a") != expected_summary.get("gate_a"):
        comparison_failures.append("gate_a payload differs from reference")

    return {
        "status": "pass" if not comparison_failures else "fail",
        "failures": comparison_failures,
        "dataset_comparisons": dataset_comparisons,
        "unexpected_datasets": unexpected_datasets,
        "gate_a_expected": expected_summary.get("gate_a"),
        "gate_a_observed": observed_summary.get("gate_a"),
    }


def check_data_integrity() -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    details: dict = {}

    reference_stage1 = _load_json(REFERENCE_STAGE1_REPORTS["retrieval_queries_sample_1000_v3"])
    raw_docs_path = Path(reference_stage1["raw_docs_path"]).resolve()
    chunks_path = Path(reference_stage1["chunks_path"]).resolve()

    raw_docs_rows = read_jsonl(raw_docs_path)
    chunk_rows = read_jsonl(chunks_path)

    raw_doc_count = len(raw_docs_rows)
    chunk_count = len(chunk_rows)
    details["raw_docs_path"] = str(raw_docs_path)
    details["chunks_path"] = str(chunks_path)
    details["raw_doc_count"] = raw_doc_count
    details["chunk_count"] = chunk_count

    if raw_doc_count != 5666:
        failures.append(f"commentary raw document count mismatch: expected 5666, observed {raw_doc_count}")

    required_doc_fields = ("document_id", "boi_reference", "title", "publication_date", "raw_xml_path", "raw_html_path")
    missing_field_counts = {field: 0 for field in required_doc_fields}
    missing_file_count = 0
    raw_doc_keys: set[tuple[str, str]] = set()
    for row in raw_docs_rows:
        raw_doc_keys.add((row["document_id"], row["publication_date"]))
        for field in required_doc_fields:
            value = row.get(field)
            if not value:
                missing_field_counts[field] += 1
        if not Path(row["raw_xml_path"]).exists() or not Path(row["raw_html_path"]).exists():
            missing_file_count += 1
    details["missing_doc_field_counts"] = missing_field_counts
    details["missing_source_file_count"] = missing_file_count
    for field, count in missing_field_counts.items():
        if count:
            failures.append(f"raw document field {field} missing on {count} rows")
    if missing_file_count:
        failures.append(f"{missing_file_count} raw source file pairs referenced by raw docs are missing")

    documents_root = infer_documents_root(raw_docs_rows[0]["raw_xml_path"])
    discovered = discover_content_documents(documents_root)
    commentary_discovered = [item for item in discovered if item.category_path and item.category_path[0] == "Commentaire"]
    discovered_keys = {(item.document_id, item.publication_date) for item in commentary_discovered}
    details["documents_root"] = str(documents_root)
    details["discovered_content_count"] = len(discovered)
    details["discovered_commentary_count"] = len(commentary_discovered)
    if len(commentary_discovered) != 5666:
        failures.append(
            f"commentary discovery count mismatch: expected 5666, observed {len(commentary_discovered)}"
        )
    if discovered_keys != raw_doc_keys:
        missing_from_raw = len(discovered_keys - raw_doc_keys)
        missing_from_discovery = len(raw_doc_keys - discovered_keys)
        failures.append(
            "discovery/raw-doc coverage mismatch: "
            f"missing_from_raw={missing_from_raw}, missing_from_discovery={missing_from_discovery}"
        )
        details["discovery_diff"] = {
            "missing_from_raw_count": missing_from_raw,
            "missing_from_discovery_count": missing_from_discovery,
        }

    empty_chunk_count = sum(1 for row in chunk_rows if not str(row.get("text", "")).strip())
    chunk_ids = [row.get("chunk_id") for row in chunk_rows]
    duplicate_chunk_count = len(chunk_ids) - len(set(chunk_ids))
    chunk_coverage = summarize_chunk_document_coverage(raw_docs_rows, chunk_rows)
    details["empty_chunk_count"] = empty_chunk_count
    details["duplicate_chunk_count"] = duplicate_chunk_count
    details["chunk_document_coverage"] = chunk_coverage
    if empty_chunk_count:
        failures.append(f"empty chunk count must be 0, observed {empty_chunk_count}")
    if duplicate_chunk_count:
        failures.append(f"duplicate chunk_id count must be 0, observed {duplicate_chunk_count}")
    if chunk_coverage["raw_document_id_count"] != raw_doc_count:
        failures.append(
            "raw document_id coverage is internally inconsistent: "
            f"raw_document_id_count={chunk_coverage['raw_document_id_count']}, raw_doc_count={raw_doc_count}"
        )
    if chunk_coverage["chunk_document_id_count"] != raw_doc_count:
        failures.append(
            "chunk coverage differs from raw docs: "
            f"chunk_document_id_count={chunk_coverage['chunk_document_id_count']}, raw_doc_count={raw_doc_count}"
        )
    if chunk_coverage["missing_document_id_count"] or chunk_coverage["extra_document_id_count"]:
        failures.append(
            "chunk/raw document_id set mismatch: "
            f"missing={chunk_coverage['missing_document_id_count']}, extra={chunk_coverage['extra_document_id_count']}"
        )

    cache_checks = []
    cache_paths = sorted(INTERIM_DIR.glob("doc_dense_cache_5666_*.npy")) + sorted(INTERIM_DIR.glob("chunk_dense_cache_5666_*.npy"))
    for cache_path in cache_paths:
        cache = np.load(cache_path, mmap_mode="r")
        expected_rows = raw_doc_count if cache_path.name.startswith("doc_dense_cache") else chunk_count
        passed = cache.ndim == 2 and cache.shape[0] == expected_rows
        cache_checks.append(
            {
                "path": str(cache_path.resolve()),
                "shape": list(cache.shape),
                "expected_rows": expected_rows,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(
                f"cache shape mismatch for {cache_path.name}: shape={tuple(cache.shape)}, expected_rows={expected_rows}"
            )
    details["cache_checks"] = cache_checks

    return {
        "status": _status_from_messages(failures=failures, warnings=warnings),
        "failures": failures,
        "warnings": warnings,
        "details": details,
    }


def run_unit_tests() -> dict:
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(SRC_ROOT) if not current_pythonpath else str(SRC_ROOT) + os.pathsep + current_pythonpath
    result = _run_command(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=PROJECT_ROOT,
        env=env,
    )
    failures = [] if result["succeeded"] else ["unit test suite failed"]
    warnings: list[str] = []
    return {
        "status": _status_from_messages(failures=failures, warnings=warnings),
        "failures": failures,
        "warnings": warnings,
        "command": result,
    }


def replay_stage1_reports(*, work_dir: Path, tolerance: float, force: bool) -> dict:
    replay_dir = work_dir / "stage1_replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    warnings: list[str] = []
    comparisons: dict[str, dict] = {}
    outputs: dict[str, str] = {}

    for dataset, reference_path in REFERENCE_STAGE1_REPORTS.items():
        reference_payload = _load_json(reference_path)
        output_path = replay_dir / f"{reference_path.stem}__replay.json"
        command = build_stage1_replay_command(
            python_executable=sys.executable,
            project_root=PROJECT_ROOT,
            interim_dir=INTERIM_DIR,
            reference_payload=reference_payload,
            output_path=output_path,
        )
        dataset_result = {
            "reference_path": str(reference_path.resolve()),
            "output_path": str(output_path.resolve()),
        }
        if not force and output_path.exists():
            observed_payload = _load_json(output_path)
            comparison = _compare_stage1_payload(reference_payload, observed_payload, tolerance=tolerance)
            if comparison["status"] == "pass":
                dataset_result["command"] = _build_reused_command_result(
                    output_path=output_path,
                    reason="existing replay matches phase6 baseline",
                )
                dataset_result["comparison"] = comparison
                outputs[dataset] = str(output_path.resolve())
                comparisons[dataset] = dataset_result
                continue

        command_result = _run_command(command, cwd=PROJECT_ROOT)
        dataset_result["command"] = command_result
        if not command_result["succeeded"]:
            failures.append(f"stage1 replay failed for {dataset}")
            comparisons[dataset] = dataset_result
            continue

        observed_payload = _load_json(output_path)
        dataset_result["comparison"] = _compare_stage1_payload(reference_payload, observed_payload, tolerance=tolerance)
        if dataset_result["comparison"]["status"] == "fail":
            failures.append(f"stage1 replay drift detected for {dataset}")
        outputs[dataset] = str(output_path.resolve())
        comparisons[dataset] = dataset_result

    return {
        "status": _status_from_messages(failures=failures, warnings=warnings),
        "failures": failures,
        "warnings": warnings,
        "comparisons": comparisons,
        "outputs": outputs,
    }


def replay_direct_passage(*, work_dir: Path, stage1_replay_outputs: dict[str, str], tolerance: float, force: bool) -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    reference_payload = _load_json(REFERENCE_DIRECT_PASSAGE_V2)
    output_path = work_dir / "direct_passage_replays" / f"{REFERENCE_DIRECT_PASSAGE_V2.stem}__replay.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    replay_stage1_path = stage1_replay_outputs.get("passage_gold_v2")
    if not replay_stage1_path:
        failures.append("missing replayed stage1 passage_gold_v2 report; cannot replay direct passage v2")
        return {
            "status": _status_from_messages(failures=failures, warnings=warnings),
            "failures": failures,
            "warnings": warnings,
        }

    command = build_direct_passage_replay_command(
        python_executable=sys.executable,
        project_root=PROJECT_ROOT,
        reference_payload=reference_payload,
        replay_stage1_report_path=Path(replay_stage1_path),
        output_path=output_path,
    )
    result: dict = {
        "reference_path": str(REFERENCE_DIRECT_PASSAGE_V2.resolve()),
        "output_path": str(output_path.resolve()),
    }
    if not force and output_path.exists():
        observed_payload = _load_json(output_path)
        comparison = _compare_direct_passage_payload(reference_payload, observed_payload, tolerance=tolerance)
        if comparison["status"] == "pass":
            result["command"] = _build_reused_command_result(
                output_path=output_path,
                reason="existing direct-passage replay matches phase6 baseline",
            )
            result["comparison"] = comparison
            return {
                "status": _status_from_messages(failures=failures, warnings=warnings),
                "failures": failures,
                "warnings": warnings,
                "comparison": result,
                "output": str(output_path.resolve()),
            }

    command_result = _run_command(command, cwd=PROJECT_ROOT)
    result["command"] = command_result
    if not command_result["succeeded"]:
        failures.append("direct passage replay failed for passage_gold_v2")
        result["status"] = "fail"
        return {
            "status": _status_from_messages(failures=failures, warnings=warnings),
            "failures": failures,
            "warnings": warnings,
            "comparison": result,
        }

    observed_payload = _load_json(output_path)
    result["comparison"] = _compare_direct_passage_payload(reference_payload, observed_payload, tolerance=tolerance)
    if result["comparison"]["status"] == "fail":
        failures.append("direct passage replay drift detected for passage_gold_v2")

    return {
        "status": _status_from_messages(failures=failures, warnings=warnings),
        "failures": failures,
        "warnings": warnings,
        "comparison": result,
        "output": str(output_path.resolve()),
    }


def replay_cross_summary(
    *,
    work_dir: Path,
    stage1_replay_outputs: dict[str, str],
    direct_passage_output: str | None,
    tolerance: float,
    force: bool,
) -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    output_path = work_dir / "cross_summary" / "phase6_cross_benchmark_summary__replay.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not direct_passage_output:
        failures.append("missing replayed direct passage report; cannot recompute cross summary")
        return {
            "status": _status_from_messages(failures=failures, warnings=warnings),
            "failures": failures,
            "warnings": warnings,
        }

    required_stage1_datasets = [
        "passage_gold_v2",
        "retrieval_queries_sample_1000_v3",
        "retrieval_queries_full_v4",
        "retrieval_queries_full_v5",
        "retrieval_queries_full_v6",
    ]
    missing_stage1_outputs = [dataset for dataset in required_stage1_datasets if dataset not in stage1_replay_outputs]
    if missing_stage1_outputs:
        failures.append(
            "missing replayed stage1 outputs required for cross summary: "
            + ", ".join(missing_stage1_outputs)
        )
        return {
            "status": _status_from_messages(failures=failures, warnings=warnings),
            "failures": failures,
            "warnings": warnings,
        }

    stage1_reports = [
        str(REFERENCE_STAGE1_PASSAGE_V1.resolve()),
        str(Path(stage1_replay_outputs["passage_gold_v2"]).resolve()),
        str(Path(stage1_replay_outputs["retrieval_queries_sample_1000_v3"]).resolve()),
        str(Path(stage1_replay_outputs["retrieval_queries_full_v4"]).resolve()),
        str(Path(stage1_replay_outputs["retrieval_queries_full_v5"]).resolve()),
        str(Path(stage1_replay_outputs["retrieval_queries_full_v6"]).resolve()),
    ]
    family_reports = [str(path.resolve()) for path in REFERENCE_FAMILY_REPORTS]
    direct_reports = [
        str(REFERENCE_DIRECT_PASSAGE_V1.resolve()),
        str(Path(direct_passage_output).resolve()),
    ]
    command = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "phase6_cross_benchmark_summary.py").resolve()),
        "--stage1-reports",
        *stage1_reports,
        "--family-reports",
        *family_reports,
        "--direct-passage-reports",
        *direct_reports,
        "--output",
        str(output_path.resolve()),
    ]
    result: dict = {
        "reference_path": str(REFERENCE_CROSS_SUMMARY.resolve()),
        "output_path": str(output_path.resolve()),
    }
    expected_summary = _load_json(REFERENCE_CROSS_SUMMARY)
    if not force and output_path.exists():
        observed_summary = _load_json(output_path)
        comparison = _compare_cross_summary_payload(expected_summary, observed_summary, tolerance=tolerance)
        if comparison["status"] == "pass":
            result["command"] = _build_reused_command_result(
                output_path=output_path,
                reason="existing cross-summary replay matches phase6 baseline",
            )
            result["comparison"] = comparison
            return {
                "status": _status_from_messages(failures=failures, warnings=warnings),
                "failures": failures,
                "warnings": warnings,
                "comparison": result,
                "output": str(output_path.resolve()),
            }

    command_result = _run_command(command, cwd=PROJECT_ROOT)
    result["command"] = command_result
    if not command_result["succeeded"]:
        failures.append("cross benchmark summary replay failed")
        result["status"] = "fail"
        return {
            "status": _status_from_messages(failures=failures, warnings=warnings),
            "failures": failures,
            "warnings": warnings,
            "comparison": result,
        }

    observed_summary = _load_json(output_path)
    result["comparison"] = _compare_cross_summary_payload(expected_summary, observed_summary, tolerance=tolerance)
    if result["comparison"]["status"] == "fail":
        failures.append("cross benchmark summary drift detected")

    return {
        "status": _status_from_messages(failures=failures, warnings=warnings),
        "failures": failures,
        "warnings": warnings,
        "comparison": result,
        "output": str(output_path.resolve()),
    }


def run_phase9_retrieval_smoke() -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    details = {
        "reference_path": str(REFERENCE_PHASE9_BATCH.resolve()),
        "cases": [],
    }

    reference_report = _load_json(REFERENCE_PHASE9_BATCH)
    runtime = Phase8bPreviewRuntime.from_local_corpus(corpus="commentary", device="cpu")
    for row in reference_report["rows"]:
        observed_payload = Phase8bPreviewRuntime.as_dict(runtime.retrieve(row["query"]))
        errors = validate_retrieval_payload(observed_payload)
        case_warnings: list[str] = []

        reference_payload = row["retrieval"]
        stage1_order = summarize_order_match(
            [hit["boi_reference"] for hit in reference_payload["stage1_hits"]],
            [hit["boi_reference"] for hit in observed_payload["stage1_hits"]],
        )
        stage2_order = summarize_order_match(
            [chunk["chunk_id"] for chunk in reference_payload["stage2_chunks"]],
            [chunk["chunk_id"] for chunk in observed_payload["stage2_chunks"]],
        )
        if observed_payload["lexical_query"] != reference_payload["lexical_query"]:
            case_warnings.append("lexical_query changed relative to phase9 reference")
        if observed_payload["acronym_expansions"] != reference_payload["acronym_expansions"]:
            case_warnings.append("acronym_expansions changed relative to phase9 reference")
        if not stage1_order["matches"]:
            case_warnings.append("stage1 hit order changed relative to phase9 reference")
        if not stage2_order["matches"]:
            case_warnings.append("stage2 chunk order changed relative to phase9 reference")

        if errors:
            failures.append(f"phase9 retrieval payload invalid for case {row['case_id']}")
        elif case_warnings:
            warnings.append(f"phase9 retrieval drift warning for case {row['case_id']}")

        details["cases"].append(
            {
                "case_id": row["case_id"],
                "query": row["query"],
                "errors": errors,
                "warnings": case_warnings,
                "lexical_query": {
                    "matches": observed_payload["lexical_query"] == reference_payload["lexical_query"],
                    "expected": reference_payload["lexical_query"],
                    "observed": observed_payload["lexical_query"],
                },
                "acronym_expansions": {
                    "matches": observed_payload["acronym_expansions"] == reference_payload["acronym_expansions"],
                    "expected": reference_payload["acronym_expansions"],
                    "observed": observed_payload["acronym_expansions"],
                },
                "stage1_order": stage1_order,
                "stage2_order": stage2_order,
            }
        )

    return {
        "status": _status_from_messages(failures=failures, warnings=warnings),
        "failures": failures,
        "warnings": warnings,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the pre-LLM BOFiP cleanroom stack without relaunching the full experimentation history.")
    parser.add_argument("--output", type=str, default=str(REPORTS_DIR / "pre_llm_verification_summary.json"))
    parser.add_argument("--work-dir", type=str, default=str(REPORTS_DIR / "pre_llm_verification"))
    parser.add_argument("--metrics-tolerance", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true", help="Rerun heavy replay steps even if matching replay outputs already exist.")
    args = parser.parse_args()

    ensure_data_dirs()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "project_root": str(PROJECT_ROOT.resolve()),
        "official_baseline": "phase6",
        "diagnostic_reference": "phase7",
        "work_dir": str(work_dir),
        "metrics_tolerance": args.metrics_tolerance,
        "checks": {},
    }

    try:
        summary["checks"]["data_integrity"] = check_data_integrity()
    except Exception as exc:  # pragma: no cover - defensive reporting
        summary["checks"]["data_integrity"] = {
            "status": "fail",
            "failures": ["data integrity check crashed"],
            "warnings": [],
            "exception": _exception_payload(exc),
        }

    try:
        summary["checks"]["unit_tests"] = run_unit_tests()
    except Exception as exc:  # pragma: no cover - defensive reporting
        summary["checks"]["unit_tests"] = {
            "status": "fail",
            "failures": ["unit test step crashed"],
            "warnings": [],
            "exception": _exception_payload(exc),
        }

    stage1_outputs: dict[str, str] = {}
    try:
        stage1_check = replay_stage1_reports(work_dir=work_dir, tolerance=args.metrics_tolerance, force=args.force)
        summary["checks"]["stage1_replay"] = stage1_check
        stage1_outputs = stage1_check.get("outputs", {})
    except Exception as exc:  # pragma: no cover - defensive reporting
        summary["checks"]["stage1_replay"] = {
            "status": "fail",
            "failures": ["stage1 replay step crashed"],
            "warnings": [],
            "exception": _exception_payload(exc),
        }

    direct_output: str | None = None
    try:
        direct_check = replay_direct_passage(
            work_dir=work_dir,
            stage1_replay_outputs=stage1_outputs,
            tolerance=args.metrics_tolerance,
            force=args.force,
        )
        summary["checks"]["direct_passage_replay"] = direct_check
        direct_output = direct_check.get("output")
    except Exception as exc:  # pragma: no cover - defensive reporting
        summary["checks"]["direct_passage_replay"] = {
            "status": "fail",
            "failures": ["direct passage replay step crashed"],
            "warnings": [],
            "exception": _exception_payload(exc),
        }

    try:
        summary["checks"]["cross_benchmark_summary"] = replay_cross_summary(
            work_dir=work_dir,
            stage1_replay_outputs=stage1_outputs,
            direct_passage_output=direct_output,
            tolerance=args.metrics_tolerance,
            force=args.force,
        )
    except Exception as exc:  # pragma: no cover - defensive reporting
        summary["checks"]["cross_benchmark_summary"] = {
            "status": "fail",
            "failures": ["cross benchmark summary step crashed"],
            "warnings": [],
            "exception": _exception_payload(exc),
        }

    try:
        summary["checks"]["phase9_retrieval_smoke"] = run_phase9_retrieval_smoke()
    except Exception as exc:  # pragma: no cover - defensive reporting
        summary["checks"]["phase9_retrieval_smoke"] = {
            "status": "fail",
            "failures": ["phase9 retrieval smoke step crashed"],
            "warnings": [],
            "exception": _exception_payload(exc),
        }

    statuses = [check["status"] for check in summary["checks"].values()]
    overall_failures = [name for name, check in summary["checks"].items() if check["status"] == "fail"]
    overall_warnings = [name for name, check in summary["checks"].items() if check["status"] == "pass_with_warnings"]
    summary["overall_pass"] = not overall_failures
    summary["overall_status"] = _status_from_messages(
        failures=overall_failures,
        warnings=overall_warnings,
    )
    summary["overall_failed_checks"] = overall_failures
    summary["overall_warning_checks"] = overall_warnings
    summary["statuses"] = statuses

    output_path = Path(args.output).resolve()
    write_json(output_path, summary)
    print(f"Pre-LLM verification summary written to: {output_path}")
    print(f"overall_status = {summary['overall_status']}")
    if overall_failures:
        print(f"failed_checks = {', '.join(overall_failures)}")
    elif overall_warnings:
        print(f"warning_checks = {', '.join(overall_warnings)}")
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

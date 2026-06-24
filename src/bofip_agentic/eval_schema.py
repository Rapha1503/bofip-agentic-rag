from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|authorization|x-api-key)\s*[:=]\s*['\"]?[^'\"\s]+"),
]
MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "â€™", "â€œ", "â€�", "â€“", "â€”", "â‚¬")


@dataclass(frozen=True)
class EvalRunConfig:
    run_id: str
    provider: str
    model: str
    judge_provider: str
    judge_model: str
    corpus: str
    question_bank: str
    limit: int
    sample: int
    seed: int
    retrieval_mode: str
    reranker: bool
    device: str
    max_iterations: int
    source_review_mode: str = "full"
    source_review_chunk_limit: int = 16
    source_review_text_limit: int = 900
    post_relaunch_review: bool = True
    max_missing_axes: int = 3
    git_commit: str = ""
    git_dirty: bool = False
    git_status_hash: str = ""
    corpus_manifest_hash: str = ""
    eval_set_hash: str = ""
    question_count: int = 0
    case_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    theme: str = ""
    difficulty: str = ""
    question_type: str = ""
    expected_status: str = ""
    expected_answer_core: list[str] = field(default_factory=list)
    expected_calculation: str = ""
    required_docs: list[str] = field(default_factory=list)
    optional_docs: list[str] = field(default_factory=list)
    failure_signals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalSource:
    chunk_id: str
    boi_reference: str
    title: str = ""
    section: str = ""
    score: float = 0.0
    snippet: str = ""
    retrieval_stage: str = ""


@dataclass(frozen=True)
class AgenticScores:
    required_doc_hits: list[str] = field(default_factory=list)
    missing_required_docs: list[str] = field(default_factory=list)
    required_doc_recall: float = 0.0
    optional_doc_hits: list[str] = field(default_factory=list)
    answer_point_hits: list[str] = field(default_factory=list)
    missing_answer_points: list[str] = field(default_factory=list)
    answer_point_recall: float = 1.0
    failure_signal_hits: list[str] = field(default_factory=list)
    trace_score: float = 0.0
    has_plan: bool = False
    has_source_review: bool = False
    has_retrieval: bool = False
    has_answer_step: bool = False
    has_relaunch: bool = False


@dataclass(frozen=True)
class EvalJudgement:
    verdict: str
    confidence: float = 0.0
    answer_quality: int = 0
    grounding: int = 0
    agentic_process: int = 0
    status_quality: int = 0
    root_cause_stage: str = ""
    rationale: str = ""
    recommended_action: str = ""


@dataclass(frozen=True)
class PerQueryResult:
    id: str
    question: str
    theme: str = ""
    difficulty: str = ""
    question_type: str = ""
    answer_status: str = ""
    auto_verdict: str = ""
    coverage: float = 0.0
    iterations: int = 0
    total_s: float = 0.0
    conclusion: str = ""
    justification_bullets: list[str] = field(default_factory=list)
    limits: str = ""
    axes_requis: list[str] = field(default_factory=list)
    axes_couverts: list[str] = field(default_factory=list)
    axes_manquants: list[str] = field(default_factory=list)
    sources: list[EvalSource] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)
    scores: AgenticScores = field(default_factory=AgenticScores)
    judgement: EvalJudgement | None = None
    error: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    step_timings: list[dict[str, Any]] = field(default_factory=list)


def as_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: as_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return repair_mojibake(value)
    return value


def normalize_boi_family(ref: str) -> str:
    value = str(ref or "").strip().upper()
    if not value:
        return ""
    parts = value.split("-")
    if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
        return "-".join(parts[:-1])
    return value


def boi_matches(retrieved: str, expected: str) -> bool:
    retrieved_family = normalize_boi_family(retrieved)
    expected_family = normalize_boi_family(expected)
    if not retrieved_family or not expected_family:
        return False
    return (
        retrieved_family == expected_family
        or retrieved_family.startswith(expected_family + "-")
        or expected_family.startswith(retrieved_family + "-")
    )


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def redact_secrets(text: str) -> str:
    redacted = str(text)
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def contains_secret(text: str) -> bool:
    value = str(text)
    return redact_secrets(value) != value


def repair_mojibake(text: str) -> str:
    value = str(text)
    if not any(marker in value for marker in MOJIBAKE_MARKERS):
        return value
    try:
        candidate = value.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return value
    original_markers = sum(value.count(marker) for marker in MOJIBAKE_MARKERS)
    candidate_markers = sum(candidate.count(marker) for marker in MOJIBAKE_MARKERS)
    return candidate if candidate_markers < original_markers else value

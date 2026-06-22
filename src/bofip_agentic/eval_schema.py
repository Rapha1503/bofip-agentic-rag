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
    re.compile(r"(?i)(api[_-]?key|authorization|x-api-key)\s*[:=]\s*['\"]?[^'\"\s]+"),
]


@dataclass
class EvalRunConfig:
    run_id: str
    provider: str
    model: str
    corpus: str
    question_bank: str
    limit: int
    lexical_only: bool
    git_commit: str
    corpus_manifest_hash: str
    eval_set_hash: str
    device: str = "cpu"
    max_iterations: int = 2


@dataclass
class EvalQuestion:
    id: str
    question: str
    theme: str = ""
    difficulty: str = ""
    question_type: str = ""
    expected_status: str = ""
    required_docs: list[str] = field(default_factory=list)
    optional_docs: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    expected_numeric_answer: float | None = None
    note: str = ""


@dataclass
class EvalSource:
    id: str
    boi_reference: str
    title: str
    section: str
    score: float
    snippet: str
    date: str = ""
    retrieval_stage: str = ""


@dataclass
class PerQueryResult:
    id: str
    question: str
    theme: str
    difficulty: str
    question_type: str
    answer_status: str
    coverage: float
    iterations: int
    total_s: float
    conclusion: str
    justification_bullets: list[str]
    axes_requis: list[str]
    axes_couverts: list[str]
    axes_manquants: list[str]
    sources: list[EvalSource]
    retrieved_docs: list[str]
    trace: list[dict[str, Any]]
    error: str = ""


@dataclass
class ReviewAction:
    severity: str
    area: str
    title: str
    recommendation: str
    evidence: str = ""


def as_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: as_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    return value


def normalize_boi_family(ref: str) -> str:
    ref = str(ref or "").strip()
    parts = ref.split("-")
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 8:
        return "-".join(parts[:-1])
    return ref


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def redact_secrets(text: str) -> str:
    redacted = str(text)
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted

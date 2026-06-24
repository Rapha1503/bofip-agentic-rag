from __future__ import annotations

import json
import os
import random
import re
import hashlib
import subprocess
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .agent_rag import AgenticRAG
from .eval_artifacts import (
    append_jsonl,
    compute_summary,
    write_evidence_card,
    write_json,
    write_jsonl,
    write_public_csv,
    write_summary_markdown,
)
from .eval_schema import (
    AgenticScores,
    EvalJudgement,
    EvalQuestion,
    EvalRunConfig,
    EvalSource,
    PerQueryResult,
    as_jsonable,
    boi_matches,
    hash_file,
)
from .providers import PROVIDERS
from .rag_runtime import RagRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANSWER_POINT_STOPWORDS = {
    "afin",
    "ainsi",
    "alors",
    "apres",
    "apprecier",
    "avec",
    "avant",
    "cette",
    "confondre",
    "dans",
    "dont",
    "elle",
    "elles",
    "entre",
    "etre",
    "eventuel",
    "eventuels",
    "effet",
    "effets",
    "examiner",
    "faut",
    "indique",
    "indiquee",
    "leur",
    "leurs",
    "maniere",
    "mais",
    "meme",
    "permettre",
    "position",
    "pour",
    "quand",
    "quelle",
    "quelles",
    "sans",
    "selon",
    "situation",
    "sont",
    "sous",
    "suffisamment",
    "tous",
    "toute",
    "toutes",
    "verifier",
}


def build_run_id(label: str = "eval") -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "-", label.strip().lower()).strip("-") or "eval"
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{safe}"


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def load_question_bank(
    path: str | Path,
    *,
    limit: int = 0,
    sample: int = 0,
    seed: int = 42,
    case_ids: Sequence[str] | None = None,
) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for index, row in enumerate(_iter_question_rows(path), start=1):
        gold = row.get("gold_eval") or {}
        refs = gold.get("expected_bofip_refs") or {}
        question = _runtime_question_from_row(row, index)
        if not question:
            raise ValueError(f"missing user_question/question/runtime_question at row {index}")
        questions.append(
            EvalQuestion(
                id=str(row.get("id") or row.get("query_id") or row.get("legacy_id") or f"Q{index:03d}"),
                question=str(question),
                theme=str(row.get("theme") or row.get("domain") or ""),
                difficulty=str(row.get("difficulty") or ""),
                question_type=str(row.get("question_type") or row.get("type") or ""),
                expected_status=str(row.get("expected_status") or gold.get("expected_status") or ""),
                expected_answer_core=_coerce_list(
                    row.get("expected_answer_core") or gold.get("expected_answer_points")
                ),
                expected_calculation=str(row.get("expected_calculation") or gold.get("expected_calculation") or ""),
                required_docs=_coerce_list(
                    row.get("required_docs")
                    or row.get("must_include_sources")
                    or refs.get("must_include")
                ),
                optional_docs=_coerce_list(
                    row.get("optional_docs")
                    or row.get("should_include_sources")
                    or refs.get("should_include")
                ),
                failure_signals=_coerce_list(row.get("failure_signals") or gold.get("failure_signals")),
            )
        )
    if case_ids:
        wanted = [str(item).strip() for item in case_ids if str(item).strip()]
        by_id = {question.id: question for question in questions}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            raise ValueError(f"unknown case id(s): {', '.join(missing)}")
        questions = [by_id[item] for item in wanted]
    if sample:
        rng = random.Random(seed)
        count = min(sample, len(questions))
        questions = rng.sample(questions, count)
    if limit:
        questions = questions[:limit]
    return questions


def _runtime_question_from_row(row: dict[str, Any], index: int) -> str:
    runtime_question = str(row.get("runtime_question") or "").strip()
    candidates = {
        key: str(row.get(key) or "").strip()
        for key in ("user_question", "question")
        if row.get(key)
    }
    if runtime_question:
        for key, value in candidates.items():
            if value and value != runtime_question:
                raise ValueError(
                    f"row {index} has runtime_question and {key} with different text; "
                    "refusing to guess which prompt is safe to send"
                )
        return runtime_question
    return candidates.get("user_question") or candidates.get("question") or ""


def _iter_question_rows(path: str | Path) -> list[dict[str, Any]]:
    bank_path = Path(path)
    if bank_path.suffix.lower() == ".json":
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
        rows = payload.get("cases") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"expected JSON array or object with cases: {bank_path}")
        return [dict(row) for row in rows]

    rows: list[dict[str, Any]] = []
    with bank_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def provider_config(provider_name: str) -> tuple[str, dict[str, Any]]:
    normalized = provider_name.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "codex": "Codex local",
        "codex local": "Codex local",
        "deepseek": "DeepSeek",
        "openai": "OpenAI",
        "mistral": "Mistral",
        "google": "Google",
    }
    key = aliases.get(normalized, provider_name)
    for name, config in PROVIDERS.items():
        if name.lower() == key.lower():
            return name, config
    raise ValueError(f"unknown provider: {provider_name}")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def git_status_porcelain() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def git_status_hash() -> str:
    status = git_status_porcelain()
    if not status:
        return ""
    return hashlib.sha256(status.encode("utf-8")).hexdigest()[:16]


def source_from_agent_chunk(chunk: dict[str, Any]) -> EvalSource:
    section_path = chunk.get("section_path") or chunk.get("section") or ""
    if isinstance(section_path, list):
        section = " > ".join(str(item) for item in section_path if str(item).strip())
    else:
        section = str(section_path)
    score_value = chunk.get("score", chunk.get("local_score", 0.0))
    try:
        score = float(score_value)
    except (TypeError, ValueError):
        score = 0.0
    return EvalSource(
        chunk_id=str(chunk.get("chunk_id") or chunk.get("id") or ""),
        boi_reference=str(chunk.get("boi_reference") or chunk.get("document_id") or ""),
        title=str(chunk.get("title") or ""),
        section=section,
        score=score,
        snippet=_short(str(chunk.get("text") or chunk.get("snippet") or ""), 900),
        retrieval_stage=str(chunk.get("retrieval_stage") or ""),
    )


def compute_agentic_scores(question: EvalQuestion, result: dict[str, Any]) -> AgenticScores:
    sources = [source_from_agent_chunk(chunk) for chunk in result.get("sources", []) or []]
    retrieved_docs = [source.boi_reference for source in sources if source.boi_reference]
    required_hits: list[str] = []
    missing_required: list[str] = []
    for expected in question.required_docs:
        if any(boi_matches(retrieved, expected) for retrieved in retrieved_docs):
            required_hits.append(expected)
        else:
            missing_required.append(expected)
    optional_hits = [
        expected
        for expected in question.optional_docs
        if any(boi_matches(retrieved, expected) for retrieved in retrieved_docs)
    ]
    required_recall = len(required_hits) / len(question.required_docs) if question.required_docs else 1.0
    answer_points = list(question.expected_answer_core)
    if question.expected_calculation:
        answer_points.append(question.expected_calculation)
    answer_text = _answer_eval_text(result)
    answer_point_hits: list[str] = []
    missing_answer_points: list[str] = []
    for point in answer_points:
        if _answer_point_is_covered(point, answer_text):
            answer_point_hits.append(point)
        else:
            missing_answer_points.append(point)
    answer_point_recall = len(answer_point_hits) / len(answer_points) if answer_points else 1.0
    failure_signal_hits = [
        signal
        for signal in question.failure_signals
        if _normalized_phrase(signal) and _normalized_phrase(signal) in _normalized_phrase(answer_text)
    ]
    trace = result.get("trace", []) or []
    timings = result.get("step_timings", []) or []
    labels = " ".join(str(item.get("label", "")) for item in timings).lower()
    stages = " ".join(str(item.get("stage", "")) for item in trace).lower()
    source_review_skipped = any(
        isinstance(item, dict)
        and isinstance(item.get("source_review"), dict)
        and item["source_review"].get("coverage_status") == "skipped"
        for item in trace
    )
    has_plan = "plan" in labels or "plan" in stages
    has_source_review = (
        "critique" in labels or "source_review" in json.dumps(trace, ensure_ascii=False).lower()
    ) and not source_review_skipped
    has_retrieval = "recherche" in labels or "routes" in json.dumps(trace, ensure_ascii=False).lower()
    has_answer_step = "réponse" in labels or "reponse" in labels or bool(result.get("conclusion"))
    has_relaunch = int(result.get("iterations", 1) or 1) > 1 or "relance" in labels or "intra-document" in labels
    trace_parts = [has_plan, has_source_review, has_retrieval, has_answer_step]
    trace_score = sum(1 for item in trace_parts if item) / len(trace_parts)
    if has_relaunch:
        trace_score = min(1.0, trace_score + 0.1)
    return AgenticScores(
        required_doc_hits=required_hits,
        missing_required_docs=missing_required,
        required_doc_recall=round(required_recall, 3),
        optional_doc_hits=optional_hits,
        answer_point_hits=answer_point_hits,
        missing_answer_points=missing_answer_points,
        answer_point_recall=round(answer_point_recall, 3),
        failure_signal_hits=failure_signal_hits,
        trace_score=round(trace_score, 3),
        has_plan=has_plan,
        has_source_review=has_source_review,
        has_retrieval=has_retrieval,
        has_answer_step=has_answer_step,
        has_relaunch=has_relaunch,
    )


def auto_verdict(result: PerQueryResult) -> str:
    if result.error:
        return "runtime_error"
    if result.answer_status == "insufficient_evidence":
        return "candidate_fail_insufficient_evidence"
    if result.answer_status == "partial":
        return "status_bug_candidate"
    if result.answer_status != "supported":
        return "needs_human_review_unknown_status"
    if result.scores.failure_signal_hits:
        return "needs_review_failure_signal"
    good_runtime = result.coverage >= 0.75 and result.scores.trace_score >= 0.75
    has_answer_gold = bool(result.scores.answer_point_hits or result.scores.missing_answer_points)
    answer_gold_strong = has_answer_gold and result.scores.answer_point_recall >= 0.75
    answer_gold_acceptable = has_answer_gold and result.scores.answer_point_recall >= 0.6
    source_gold_acceptable = result.scores.required_doc_recall >= 0.5 or not result.scores.missing_required_docs
    if good_runtime and (source_gold_acceptable or answer_gold_strong):
        return "candidate_pass"
    if good_runtime and answer_gold_acceptable:
        return "candidate_pass_source_gap"
    if result.coverage >= 0.65 and result.scores.trace_score >= 0.75:
        return "needs_review_sources_or_limits"
    return "needs_human_review"


def _answer_eval_text(result: dict[str, Any]) -> str:
    parts: list[str] = [
        str(result.get("conclusion") or ""),
        str(result.get("limits") or ""),
    ]
    parts.extend(str(item) for item in result.get("justification_bullets", []) or [])
    parts.extend(str(item) for item in result.get("axes_couverts", []) or [])
    return "\n".join(part for part in parts if part)


def _normalized_phrase(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("œ", "oe").replace("æ", "ae")
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return " ".join(text.split())


def _answer_tokens(value: str) -> set[str]:
    raw_tokens = set(re.findall(r"[a-z0-9%]+", _normalized_phrase(value)))
    tokens: set[str] = set()
    for token in raw_tokens:
        if token in ANSWER_POINT_STOPWORDS or (len(token) < 3 and not any(ch.isdigit() for ch in token)):
            continue
        tokens.add(token)
        if len(token) > 4 and token.endswith("s"):
            tokens.add(token[:-1])
    return tokens


def _answer_point_is_covered(point: str, answer_text: str) -> bool:
    expected = _answer_tokens(point)
    if not expected:
        return True
    actual = _answer_tokens(answer_text)
    hits = expected & actual
    ratio = len(hits) / len(expected)
    if len(expected) <= 5:
        return ratio >= 0.6
    return ratio >= 0.45 and len(hits) >= 3


def _make_client(provider: str, model: str, api_key: str, base_url: str = ""):
    provider_name, config = provider_config(provider)
    if config.get("type") == "codex_cli":
        from .codex_cli_client import CodexCliClient

        return CodexCliClient(model=model or config.get("default_model", "gpt-5.5"), project_root=PROJECT_ROOT)
    key = api_key or os.environ.get(str(config.get("env_key") or ""), "")
    if not key:
        raise ValueError(f"missing API key for {provider_name}: set {config.get('env_key')}")
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=base_url or str(config.get("base_url")))


def run_llm_judge(
    question: EvalQuestion,
    result: PerQueryResult,
    *,
    provider: str,
    model: str,
    api_key: str = "",
) -> EvalJudgement:
    provider_name, config = provider_config(provider)
    client = _make_client(provider_name, model or str(config.get("default_model")), api_key)
    prompt = _build_judge_prompt(question, result)
    response = client.chat.completions.create(
        model=model or str(config.get("default_model")),
        messages=[
            {
                "role": "system",
                "content": "Tu es un évaluateur indépendant de systèmes Agentic RAG fiscaux. Réponds uniquement en JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = {"verdict": "needs_human_review", "rationale": str(content)[:1000]}
    return EvalJudgement(
        verdict=str(payload.get("verdict") or "needs_human_review"),
        confidence=_float(payload.get("confidence"), 0.0),
        answer_quality=int(_float(payload.get("answer_quality"), 0)),
        grounding=int(_float(payload.get("grounding"), 0)),
        agentic_process=int(_float(payload.get("agentic_process"), 0)),
        status_quality=int(_float(payload.get("status_quality"), 0)),
        root_cause_stage=str(payload.get("root_cause_stage") or ""),
        rationale=str(payload.get("rationale") or ""),
        recommended_action=str(payload.get("recommended_action") or ""),
    )


def run_eval(
    *,
    question_bank: str | Path,
    output_dir: str | Path,
    provider: str = "codex",
    model: str = "",
    api_key: str = "",
    judge_provider: str = "none",
    judge_model: str = "",
    judge_api_key: str = "",
    limit: int = 0,
    sample: int = 0,
    seed: int = 42,
    case_ids: Sequence[str] | None = None,
    retrieval_mode: str = "lexical",
    reranker: bool = False,
    device: str = "cpu",
    max_iterations: int = 2,
    source_review_mode: str = "full",
    source_review_chunk_limit: int = 16,
    source_review_text_limit: int = 900,
    post_relaunch_review: bool = True,
    max_missing_axes: int = 3,
    resume: bool = False,
    runtime_factory: Callable[..., Any] | None = None,
    agent_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    bank_path = Path(question_bank)
    if not bank_path.is_absolute():
        bank_path = PROJECT_ROOT / bank_path
    questions = load_question_bank(bank_path, limit=limit, sample=sample, seed=seed, case_ids=case_ids)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    provider_name, config = provider_config(provider)
    selected_model = model or str(config.get("default_model") or "")
    runtime_api_key = ""
    if not config.get("type") == "codex_cli":
        runtime_api_key = api_key or os.environ.get(str(config.get("env_key") or ""), "")
        if not runtime_api_key:
            raise ValueError(f"missing API key for {provider_name}: set {config.get('env_key')}")
    judge_name = "none"
    judge_selected_model = ""
    if judge_provider and judge_provider.lower() != "none":
        judge_name, judge_config = provider_config(judge_provider)
        judge_selected_model = judge_model or str(judge_config.get("default_model") or "")

    config_record = EvalRunConfig(
        run_id=run_dir.name,
        provider=provider_name,
        model=selected_model,
        judge_provider=judge_name,
        judge_model=judge_selected_model,
        corpus="commentary",
        question_bank=str(bank_path),
        limit=limit,
        sample=sample,
        seed=seed,
        retrieval_mode=retrieval_mode,
        reranker=reranker,
        device=device,
        max_iterations=max_iterations,
        source_review_mode=source_review_mode,
        source_review_chunk_limit=source_review_chunk_limit,
        source_review_text_limit=source_review_text_limit,
        post_relaunch_review=post_relaunch_review,
        max_missing_axes=max_missing_axes,
        git_commit=git_commit(),
        git_dirty=bool(git_status_porcelain()),
        git_status_hash=git_status_hash(),
        corpus_manifest_hash=_hash_if_exists(PROJECT_ROOT / "docs" / "full_corpus_manifest.json"),
        eval_set_hash=hash_file(bank_path),
        question_count=len(questions),
        case_ids=[question.id for question in questions],
    )
    manifest_path = run_dir / "run_manifest.json"
    has_existing_results = (run_dir / "per_query").exists() and bool(list((run_dir / "per_query").glob("*.json")))
    if resume and manifest_path.exists():
        _assert_resume_manifest_compatible(manifest_path, config_record)
    elif resume and has_existing_results:
        raise ValueError(f"cannot resume {run_dir}: per-query artifacts exist but run_manifest.json is missing")
    elif not resume and has_existing_results:
        raise ValueError(f"run directory already has per-query artifacts: {run_dir}; use --resume or a new --run-id")
    write_json(manifest_path, config_record)

    completed = _load_completed(run_dir / "per_query") if resume else {}
    unexpected_completed = sorted(set(completed) - {question.id for question in questions})
    if unexpected_completed:
        raise ValueError(f"resume found completed case IDs outside current question set: {', '.join(unexpected_completed)}")
    results: list[PerQueryResult] = list(completed.values())
    pending = [question for question in questions if question.id not in completed]

    if pending:
        load_dense = retrieval_mode.strip().lower() in {"hybrid", "dense"}
        if runtime_factory is None:
            runtime = RagRuntime.from_local_corpus(
                corpus="commentary",
                device=device,
                load_dense=load_dense,
                load_reranker=reranker,
            )
        else:
            runtime = runtime_factory(load_dense=load_dense, reranker=reranker, device=device)
        current_question = {"id": ""}

        def progress_callback(label: str, payload: dict[str, Any]) -> None:
            append_jsonl(
                run_dir / "progress.jsonl",
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "query_id": current_question["id"],
                    "label": label,
                    "payload": payload,
                },
            )

        if agent_factory is None:
            if config.get("type") == "codex_cli":
                from .codex_cli_client import CodexCliClient

                client = CodexCliClient(model=selected_model, project_root=PROJECT_ROOT)
                agent = AgenticRAG(
                    runtime,
                    client=client,
                    model=selected_model,
                    max_iterations=max_iterations,
                    use_reranker=reranker,
                    progress_callback=progress_callback,
                    source_review_mode=source_review_mode,
                    source_review_chunk_limit=source_review_chunk_limit,
                    source_review_text_limit=source_review_text_limit,
                    post_relaunch_review=post_relaunch_review,
                    max_missing_axes=max_missing_axes,
                )
            else:
                agent = AgenticRAG(
                    runtime,
                    api_key=runtime_api_key,
                    base_url=str(config.get("base_url")),
                    model=selected_model,
                    max_iterations=max_iterations,
                    use_reranker=reranker,
                    progress_callback=progress_callback,
                    source_review_mode=source_review_mode,
                    source_review_chunk_limit=source_review_chunk_limit,
                    source_review_text_limit=source_review_text_limit,
                    post_relaunch_review=post_relaunch_review,
                    max_missing_axes=max_missing_axes,
                )
        else:
            agent = agent_factory(
                runtime=runtime,
                progress_callback=progress_callback,
                provider=provider_name,
                model=selected_model,
                api_key=runtime_api_key,
                source_review_mode=source_review_mode,
                source_review_chunk_limit=source_review_chunk_limit,
                source_review_text_limit=source_review_text_limit,
                post_relaunch_review=post_relaunch_review,
                max_missing_axes=max_missing_axes,
            )

        for index, question in enumerate(pending, start=1):
            current_question["id"] = question.id
            started = time.time()
            print(f"[{index}/{len(pending)}] {question.id} {question.theme}...", flush=True)
            try:
                agent_result = agent.run(question.question)
                result = per_query_from_agent_result(question, agent_result)
                result = dataclasses_replace(
                    result,
                    total_s=round(time.time() - started, 2),
                )
                if judge_name != "none":
                    judgement = run_llm_judge(
                        question,
                        result,
                        provider=judge_name,
                        model=judge_selected_model,
                        api_key=judge_api_key,
                    )
                    result = dataclasses_replace(result, judgement=judgement)
            except Exception as exc:
                result = PerQueryResult(
                    id=question.id,
                    question=question.question,
                    theme=question.theme,
                    difficulty=question.difficulty,
                    question_type=question.question_type,
                    answer_status="error",
                    auto_verdict="runtime_error",
                    total_s=round(time.time() - started, 2),
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            result = dataclasses_replace(result, auto_verdict=auto_verdict(result))
            _write_query_artifacts(run_dir, result)
            results.append(result)
            _write_rollups(run_dir, results, expected_count=len(questions))
            print(
                f"  -> {result.auto_verdict} | status={result.answer_status} | "
                f"cov={result.coverage:.0%} | doc={result.scores.required_doc_recall:.0%} | "
                f"trace={result.scores.trace_score:.0%} | {result.total_s:.1f}s",
                flush=True,
            )

    _write_rollups(run_dir, results, expected_count=len(questions))
    final_summary = compute_summary(results)
    final_summary["expected_queries"] = len(questions)
    final_summary["is_complete"] = len(results) == len(questions)
    return {
        "run_dir": str(run_dir),
        "summary": final_summary,
        "results": [as_jsonable(row) for row in results],
    }


def per_query_from_agent_result(question: EvalQuestion, agent_result: dict[str, Any]) -> PerQueryResult:
    scores = compute_agentic_scores(question, agent_result)
    sources = [source_from_agent_chunk(chunk) for chunk in agent_result.get("sources", []) or []]
    retrieved_docs = list(dict.fromkeys(source.boi_reference for source in sources if source.boi_reference))
    result = PerQueryResult(
        id=question.id,
        question=question.question,
        theme=question.theme,
        difficulty=question.difficulty,
        question_type=question.question_type,
        answer_status=str(agent_result.get("answer_status") or ""),
        coverage=_float(agent_result.get("coverage"), 0.0),
        iterations=int(_float(agent_result.get("iterations"), 0)),
        total_s=_float(agent_result.get("total_s"), 0.0),
        conclusion=str(agent_result.get("conclusion") or ""),
        justification_bullets=[str(item) for item in agent_result.get("justification_bullets", []) or []],
        limits=str(agent_result.get("limits") or ""),
        axes_requis=[str(item) for item in agent_result.get("axes_requis", []) or []],
        axes_couverts=[str(item) for item in agent_result.get("axes_couverts", []) or []],
        axes_manquants=[str(item) for item in agent_result.get("axes_manquants", []) or []],
        sources=sources,
        retrieved_docs=retrieved_docs,
        scores=scores,
        trace=agent_result.get("trace", []) or [],
        step_timings=agent_result.get("step_timings", []) or [],
    )
    return dataclasses_replace(result, auto_verdict=auto_verdict(result))


def _build_judge_prompt(question: EvalQuestion, result: PerQueryResult) -> str:
    payload = {
        "task": "Evaluate an Agentic RAG answer. Gold metadata is evaluation-only and was not sent to the runtime.",
        "allowed_verdicts": [
            "pass",
            "pass_with_limits",
            "status_bug",
            "retrieval_fail",
            "generation_fail",
            "unsafe_or_hallucinated",
            "runtime_error",
            "needs_human_review",
        ],
        "rubric": {
            "answer_quality": "0-5: does the answer address the user's fiscal question and calculation?",
            "grounding": "0-5: are the cited BOFiP sources sufficient and relevant?",
            "agentic_process": "0-5: did planning, source review, relaunches and limits behave coherently?",
            "status_quality": "0-5: is supported/insufficient consistent with the actual answer?",
        },
        "question": as_jsonable(question),
        "result": {
            "answer_status": result.answer_status,
            "auto_verdict": result.auto_verdict,
            "coverage": result.coverage,
            "conclusion": result.conclusion,
            "justification_bullets": result.justification_bullets,
            "limits": result.limits,
            "sources": [as_jsonable(source) for source in result.sources],
            "scores": as_jsonable(result.scores),
            "step_timings": result.step_timings,
        },
        "output_schema": {
            "verdict": "one allowed verdict",
            "confidence": "0.0-1.0",
            "answer_quality": "0-5",
            "grounding": "0-5",
            "agentic_process": "0-5",
            "status_quality": "0-5",
            "root_cause_stage": "retrieval|source_review|planning|generation|status|none",
            "rationale": "short French explanation",
            "recommended_action": "short next action or empty",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _write_query_artifacts(run_dir: Path, result: PerQueryResult) -> None:
    write_json(run_dir / "per_query" / f"{result.id}.json", result)
    write_json(run_dir / "traces" / f"{result.id}.json", {"id": result.id, "trace": result.trace, "step_timings": result.step_timings})
    write_evidence_card(run_dir / "evidence_cards" / f"{result.id}.md", result)


def _write_rollups(run_dir: Path, results: list[PerQueryResult], *, expected_count: int | None = None) -> None:
    ordered = sorted(results, key=lambda row: row.id)
    summary = compute_summary(ordered)
    if expected_count is not None:
        summary["expected_queries"] = expected_count
        summary["is_complete"] = len(ordered) == expected_count
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir / "summary.md", summary, title="BOFiP Agentic RAG Evaluation")
    write_jsonl(run_dir / "per_query.jsonl", ordered)
    write_public_csv(run_dir / "per_query_public.csv", ordered)


def _assert_resume_manifest_compatible(path: Path, expected: EvalRunConfig) -> None:
    existing = json.loads(path.read_text(encoding="utf-8"))
    expected_payload = as_jsonable(expected)
    mismatches = [
        key
        for key, value in expected_payload.items()
        if existing.get(key) != value
    ]
    if mismatches:
        details = ", ".join(sorted(mismatches))
        raise ValueError(f"resume manifest mismatch for {path.parent}: {details}; use a new --run-id")


def _load_completed(per_query_dir: Path) -> dict[str, PerQueryResult]:
    completed: dict[str, PerQueryResult] = {}
    if not per_query_dir.exists():
        return completed
    for path in sorted(per_query_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed[str(payload["id"])] = _per_query_from_payload(payload)
    return completed


def _per_query_from_payload(payload: dict[str, Any]) -> PerQueryResult:
    scores_payload = payload.get("scores") or {}
    judgement_payload = payload.get("judgement")
    return PerQueryResult(
        id=str(payload.get("id") or ""),
        question=str(payload.get("question") or ""),
        theme=str(payload.get("theme") or ""),
        difficulty=str(payload.get("difficulty") or ""),
        question_type=str(payload.get("question_type") or ""),
        answer_status=str(payload.get("answer_status") or ""),
        auto_verdict=str(payload.get("auto_verdict") or ""),
        coverage=_float(payload.get("coverage"), 0.0),
        iterations=int(_float(payload.get("iterations"), 0)),
        total_s=_float(payload.get("total_s"), 0.0),
        conclusion=str(payload.get("conclusion") or ""),
        justification_bullets=[str(item) for item in payload.get("justification_bullets", []) or []],
        limits=str(payload.get("limits") or ""),
        axes_requis=[str(item) for item in payload.get("axes_requis", []) or []],
        axes_couverts=[str(item) for item in payload.get("axes_couverts", []) or []],
        axes_manquants=[str(item) for item in payload.get("axes_manquants", []) or []],
        sources=[
            EvalSource(
                chunk_id=str(item.get("chunk_id") or ""),
                boi_reference=str(item.get("boi_reference") or ""),
                title=str(item.get("title") or ""),
                section=str(item.get("section") or ""),
                score=_float(item.get("score"), 0.0),
                snippet=str(item.get("snippet") or ""),
                retrieval_stage=str(item.get("retrieval_stage") or ""),
            )
            for item in payload.get("sources", []) or []
        ],
        retrieved_docs=[str(item) for item in payload.get("retrieved_docs", []) or []],
        scores=AgenticScores(
            required_doc_hits=[str(item) for item in scores_payload.get("required_doc_hits", []) or []],
            missing_required_docs=[str(item) for item in scores_payload.get("missing_required_docs", []) or []],
            required_doc_recall=_float(scores_payload.get("required_doc_recall"), 0.0),
            optional_doc_hits=[str(item) for item in scores_payload.get("optional_doc_hits", []) or []],
            answer_point_hits=[str(item) for item in scores_payload.get("answer_point_hits", []) or []],
            missing_answer_points=[str(item) for item in scores_payload.get("missing_answer_points", []) or []],
            answer_point_recall=_float(scores_payload.get("answer_point_recall"), 1.0),
            failure_signal_hits=[str(item) for item in scores_payload.get("failure_signal_hits", []) or []],
            trace_score=_float(scores_payload.get("trace_score"), 0.0),
            has_plan=bool(scores_payload.get("has_plan")),
            has_source_review=bool(scores_payload.get("has_source_review")),
            has_retrieval=bool(scores_payload.get("has_retrieval")),
            has_answer_step=bool(scores_payload.get("has_answer_step")),
            has_relaunch=bool(scores_payload.get("has_relaunch")),
        ),
        judgement=EvalJudgement(**judgement_payload) if isinstance(judgement_payload, dict) else None,
        error=str(payload.get("error") or ""),
        trace=payload.get("trace", []) or [],
        step_timings=payload.get("step_timings", []) or [],
    )


def _hash_if_exists(path: Path) -> str:
    return hash_file(path) if path.exists() else ""


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _short(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def dataclasses_replace(result: PerQueryResult, **changes: Any) -> PerQueryResult:
    import dataclasses

    return dataclasses.replace(result, **changes)

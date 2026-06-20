from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from openai import APITimeoutError, InternalServerError, OpenAI, RateLimitError

from .env_utils import load_default_env_files
from .preview_runtime import PreviewRetrievalResult, Phase8bPreviewRuntime


PreviewProvider = Literal["gemini", "openai", "deepseek"]

DEFAULT_PREVIEW_PROVIDER: PreviewProvider = "deepseek"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_PREVIEW_MODEL = DEFAULT_DEEPSEEK_MODEL
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
PREVIEW_ANSWER_CONTRACT_VERSION = "v2_structured_json"

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_CITATION_BLOCK_RE = re.compile(r"\[([0-9,\s]+)\]")
_RETRY_DELAY_SECONDS_RE = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_RETRY_DELAY_DURATION_RE = re.compile(r"'retryDelay': '([0-9]+(?:\.[0-9]+)?)s'", re.IGNORECASE)
_COMPACT_RETRY_MARKERS = (
    "truncated",
    "closing brace",
    "shorter complete json object",
)


@dataclass(frozen=True)
class StructuredPreviewAnswer:
    answer_status: str
    conclusion: str
    justification_bullets: list[str]
    limits: str


@dataclass(frozen=True)
class PreviewAnswer:
    provider: str
    model: str
    answer_text: str
    raw_answer_text: str
    prompt_text: str
    retrieval_payload: dict
    api_called: bool
    contract_version: str
    structured_answer: dict[str, Any] | None
    answer_validation: dict[str, Any]
    response_metadata: dict[str, Any]
    attempt_count: int = 1


def has_api_key(provider: PreviewProvider = DEFAULT_PREVIEW_PROVIDER) -> bool:
    load_default_env_files()
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "deepseek":
        return bool(os.environ.get("DEEPSEEK_API_KEY"))
    raise ValueError(f"Unsupported provider: {provider}")


def _should_use_compact_prompt(validation_errors: list[str] | None) -> bool:
    if not validation_errors:
        return False
    combined = " ".join(validation_errors).lower()
    return any(marker in combined for marker in _COMPACT_RETRY_MARKERS)


def _compact_prompt_text(text: Any, *, max_chars: int = 420) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    clipped = compact[: max_chars + 1]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,;:") + "…"


def _extract_retry_delay_seconds(error: BaseException) -> float | None:
    text = str(error or "")
    matches = [
        float(match.group(1))
        for pattern in (_RETRY_DELAY_SECONDS_RE, _RETRY_DELAY_DURATION_RE)
        for match in pattern.finditer(text)
    ]
    if not matches:
        return None
    return max(matches)


def build_citation_prompt(
    result: PreviewRetrievalResult,
    *,
    validation_errors: list[str] | None = None,
    compact_mode: bool | None = None,
) -> str:
    if compact_mode is None:
        compact_mode = _should_use_compact_prompt(validation_errors)
    chunks = Phase8bPreviewRuntime.build_context_chunks(result)
    evidence_lines = []
    for chunk in chunks:
        chunk_text = chunk["text"] if not compact_mode else _compact_prompt_text(chunk["text"])
        evidence_lines.append(
            "\n".join(
                [
                    f"[{chunk['citation_id']}] BOI: {chunk['boi_reference']}",
                    f"Titre: {chunk['title']}",
                    f"Date: {chunk['publication_date'] or 'inconnue'}",
                    f"Section: {chunk['section_path'] or '(sans section)'}",
                    f"Texte: {chunk_text}",
                ]
            )
        )
    evidence_block = "\n\n".join(evidence_lines)
    prompt = (
        "Question utilisateur:\n"
        f"{result.query}\n\n"
        "Extraits BOFiP fournis:\n"
        f"{evidence_block}\n\n"
        "Instructions:\n"
        "- Reponds uniquement a partir des extraits fournis.\n"
        "- N'invente ni source, ni article, ni reponse.\n"
        "- Si la question contient une premisse fausse et que les extraits la contredisent, corrige-la explicitement.\n"
        "- Tu dois renvoyer un objet JSON valide et rien d'autre.\n"
        "- N'utilise ni markdown, ni bloc de code, ni texte avant ou apres le JSON.\n"
        "- Schema JSON attendu exactement:\n"
        "{\n"
        '  "answer_status": "supported" ou "insufficient_evidence",\n'
        '  "conclusion": "une phrase courte",\n'
        '  "justification_bullets": ["puce 1", "puce 2"],\n'
        '  "limits": "une phrase courte"\n'
        "}\n"
        "- Regles de validation:\n"
        "  - conclusion doit rester breve, idealement <= 25 mots.\n"
        "  - Si answer_status = 'supported', fournis 2 a 4 puces.\n"
        "  - Si answer_status = 'supported', chaque puce doit contenir au moins une citation [n] ou [n, m] et rester breve, idealement <= 35 mots.\n"
        "  - Si answer_status = 'insufficient_evidence', fournis 1 a 2 puces expliquant que les extraits sont insuffisants.\n"
        "  - Si answer_status = 'insufficient_evidence', les citations sont optionnelles et doivent rester parcimonieuses.\n"
        "  - limits est obligatoire dans tous les cas.\n"
        "  - limits doit rester bref, idealement <= 20 mots.\n"
        "  - Si les extraits suffisent, ecris exactement dans limits: 'aucune limite majeure dans les extraits fournis.'\n"
        "  - Les citations [n] doivent referencer uniquement les extraits fournis.\n"
    )
    if compact_mode:
        prompt += (
            "- Mode compact obligatoire:\n"
            "  - Renvoie le JSON sur une seule ligne.\n"
            "  - N'utilise aucun retour a la ligne dans les valeurs de chaine.\n"
            "  - Si answer_status = 'supported', produis exactement 2 puces courtes.\n"
            "  - Si answer_status = 'insufficient_evidence', produis exactement 1 puce courte.\n"
            "  - Conserve uniquement les informations strictement necessaires.\n"
        )
    if validation_errors:
        prompt += (
            "\nCorrection format obligatoire:\n"
            "- La reponse precedente a ete refusee pour les raisons suivantes:\n"
            + "".join(f"  - {error}\n" for error in validation_errors)
            + "- Repars de zero.\n"
            + "- Renvoie un JSON complet, ferme correctement et strictement conforme.\n"
            + "- N'abrege pas la reponse au milieu d'une chaine JSON.\n"
        )
    return prompt


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_bullet(value: Any) -> str:
    text = _clean_text(value)
    while text.startswith(("-", "*", "\u2022")):
        text = text[1:].strip()
    return text


def _split_justification_text(text: str) -> list[str]:
    bullets: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "\u2022")):
            if current:
                bullets.append(_clean_bullet(current))
            current = _clean_bullet(line)
            continue
        if current:
            current = f"{current} {line}".strip()
        else:
            current = line
    if current:
        bullets.append(_clean_bullet(current))
    return [bullet for bullet in bullets if bullet]


def _coerce_answer_status(payload: dict[str, Any]) -> str:
    raw_status = _clean_text(payload.get("answer_status") or payload.get("status"))
    normalized = raw_status.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"supported", "insufficient_evidence"}:
        return normalized
    if payload.get("insufficient_evidence") is True:
        return "insufficient_evidence"
    if payload.get("insufficient_evidence") is False:
        return "supported"
    return ""


def _coerce_structured_answer(payload: dict[str, Any]) -> StructuredPreviewAnswer:
    justification_raw = (
        payload.get("justification_bullets")
        if "justification_bullets" in payload
        else payload.get("justification")
    )
    if isinstance(justification_raw, list):
        bullets = [_clean_bullet(item) for item in justification_raw if _clean_bullet(item)]
    elif isinstance(justification_raw, str):
        bullets = _split_justification_text(justification_raw)
    else:
        bullets = []

    conclusion = _clean_text(payload.get("conclusion") or payload.get("Conclusion"))
    limits = _clean_text(
        payload.get("limits")
        or payload.get("limites")
        or payload.get("Limites")
        or payload.get("limits_text")
    )
    return StructuredPreviewAnswer(
        answer_status=_coerce_answer_status(payload),
        conclusion=conclusion,
        justification_bullets=bullets,
        limits=limits,
    )


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    for match in _JSON_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _parse_json_answer(text: str) -> tuple[StructuredPreviewAnswer | None, str | None]:
    for candidate in _extract_json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return _coerce_structured_answer(payload), "json"
    return None, None


def _infer_status_from_text(*parts: str) -> str:
    combined = " ".join(part.lower() for part in parts if part).strip()
    if any(marker in combined for marker in ("insuffisant", "insuffisants", "ne permettent pas", "pas de reponse fiable")):
        return "insufficient_evidence"
    return "supported"


def _parse_legacy_markdown_answer(text: str) -> tuple[StructuredPreviewAnswer | None, str | None]:
    conclusion_parts: list[str] = []
    bullet_parts: list[str] = []
    limits_parts: list[str] = []
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("conclusion:"):
            current = "conclusion"
            remainder = line.split(":", 1)[1].strip()
            if remainder:
                conclusion_parts.append(remainder)
            continue
        if lowered.startswith("justification:"):
            current = "justification"
            remainder = line.split(":", 1)[1].strip()
            if remainder:
                bullet_parts.extend(_split_justification_text(remainder))
            continue
        if lowered.startswith("limites:") or lowered.startswith("limits:"):
            current = "limits"
            remainder = line.split(":", 1)[1].strip()
            if remainder:
                limits_parts.append(remainder)
            continue

        if current == "conclusion":
            conclusion_parts.append(line)
        elif current == "justification":
            if line.startswith(("-", "*", "\u2022")):
                bullet_parts.append(_clean_bullet(line))
            elif bullet_parts:
                bullet_parts[-1] = f"{bullet_parts[-1]} {line}".strip()
            else:
                bullet_parts.append(line)
        elif current == "limits":
            limits_parts.append(line)

    if not conclusion_parts and not bullet_parts and not limits_parts:
        return None, None

    conclusion = " ".join(part for part in conclusion_parts if part).strip()
    limits = " ".join(part for part in limits_parts if part).strip()
    answer = StructuredPreviewAnswer(
        answer_status=_infer_status_from_text(conclusion, limits, " ".join(bullet_parts)),
        conclusion=conclusion,
        justification_bullets=[bullet for bullet in bullet_parts if bullet],
        limits=limits,
    )
    return answer, "legacy_markdown"


def parse_structured_preview_answer(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        return {
            "structured_answer": None,
            "parsed_from": None,
            "errors": ["model output is empty"],
        }

    structured_answer, parsed_from = _parse_json_answer(stripped)
    if structured_answer is not None:
        return {
            "structured_answer": asdict(structured_answer),
            "parsed_from": parsed_from,
            "errors": [],
        }

    structured_answer, parsed_from = _parse_legacy_markdown_answer(stripped)
    if structured_answer is not None:
        return {
            "structured_answer": asdict(structured_answer),
            "parsed_from": parsed_from,
            "errors": [],
        }

    if stripped.startswith("{") and not stripped.endswith("}"):
        return {
            "structured_answer": None,
            "parsed_from": None,
            "errors": [
                "json output appears truncated before the closing brace; return a shorter complete JSON object"
            ],
        }

    return {
        "structured_answer": None,
        "parsed_from": None,
        "errors": ["could not parse model output as JSON or legacy structured markdown"],
    }


def extract_citation_ids(text: str) -> list[int]:
    citation_ids: list[int] = []
    for block in _CITATION_BLOCK_RE.findall(text or ""):
        for raw_value in block.split(","):
            value = raw_value.strip()
            if value.isdigit():
                citation_ids.append(int(value))
    return citation_ids


def validate_structured_preview_answer(
    structured_answer: dict[str, Any],
    *,
    retrieval_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answer = _coerce_structured_answer(structured_answer)
    errors: list[str] = []
    warnings: list[str] = []

    if answer.answer_status not in {"supported", "insufficient_evidence"}:
        errors.append("answer_status must be 'supported' or 'insufficient_evidence'")
    if not answer.conclusion:
        errors.append("conclusion must be non-empty")
    if not answer.limits:
        errors.append("limits must be non-empty")
    if not answer.justification_bullets:
        errors.append("justification_bullets must be non-empty")

    all_citation_ids: list[int] = []
    if answer.answer_status == "supported":
        if not 2 <= len(answer.justification_bullets) <= 4:
            errors.append("supported answers must contain 2 to 4 justification bullets")
        for index, bullet in enumerate(answer.justification_bullets, start=1):
            citations = extract_citation_ids(bullet)
            all_citation_ids.extend(citations)
            if not citations:
                errors.append(f"supported justification bullet {index} must contain at least one citation [n]")
    elif answer.answer_status == "insufficient_evidence":
        if not 1 <= len(answer.justification_bullets) <= 2:
            errors.append("insufficient_evidence answers must contain 1 to 2 justification bullets")
        if not any(
            marker in " ".join([answer.conclusion, answer.limits, *answer.justification_bullets]).lower()
            for marker in (
                "insuffisant",
                "insuffisants",
                "ne permettent pas",
                "pas de reponse fiable",
                "aucune information",
                "ne couvrent pas",
            )
        ):
            warnings.append("insufficient_evidence answer does not explicitly mention insufficient evidence")
        for bullet in answer.justification_bullets:
            all_citation_ids.extend(extract_citation_ids(bullet))

    stage2_chunks = [] if retrieval_payload is None else retrieval_payload.get("stage2_chunks", [])
    max_citation_id = len(stage2_chunks)
    invalid_citation_ids = sorted({cid for cid in all_citation_ids if cid < 1 or cid > max_citation_id})
    if max_citation_id and invalid_citation_ids:
        errors.append(
            "citations reference unknown retrieval excerpts: "
            + ", ".join(str(cid) for cid in invalid_citation_ids)
        )

    unique_citation_ids = sorted(set(all_citation_ids))
    return {
        "valid": not errors,
        "answer_status": answer.answer_status or None,
        "has_conclusion": bool(answer.conclusion),
        "has_justification": bool(answer.justification_bullets),
        "has_limits": bool(answer.limits),
        "bullet_count": len(answer.justification_bullets),
        "citation_ids": unique_citation_ids,
        "citation_count": len(all_citation_ids),
        "with_any_citation": bool(all_citation_ids),
        "errors": errors,
        "warnings": warnings,
    }


def render_structured_preview_answer(structured_answer: dict[str, Any]) -> str:
    answer = _coerce_structured_answer(structured_answer)
    lines = [f"Conclusion: {answer.conclusion}", "", "Justification:"]
    for bullet in answer.justification_bullets:
        lines.append(f"- {bullet}")
    lines.extend(["", f"Limites: {answer.limits}"])
    return "\n".join(lines).strip()


def normalize_preview_answer(
    raw_answer_text: str,
    *,
    retrieval_payload: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    parse_result = parse_structured_preview_answer(raw_answer_text)
    structured_answer = parse_result["structured_answer"]
    if structured_answer is None:
        validation = {
            "valid": False,
            "answer_status": None,
            "has_conclusion": False,
            "has_justification": False,
            "has_limits": False,
            "bullet_count": 0,
            "citation_ids": [],
            "citation_count": 0,
            "with_any_citation": False,
            "errors": list(parse_result["errors"]),
            "warnings": [],
            "parsed_from": parse_result["parsed_from"],
        }
        return raw_answer_text.strip(), None, validation

    validation = validate_structured_preview_answer(
        structured_answer,
        retrieval_payload=retrieval_payload,
    )
    validation["parsed_from"] = parse_result["parsed_from"]
    return render_structured_preview_answer(structured_answer), structured_answer, validation


def review_batch_preview_payload(batch_payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in batch_payload.get("rows", []):
        existing_validation = row.get("answer_validation")
        if isinstance(existing_validation, dict):
            answer_text = row.get("answer_text", "")
            structured_answer = row.get("structured_answer")
            validation = dict(existing_validation)
        else:
            raw_answer_text = row.get("raw_answer_text") or row.get("answer_text", "")
            retrieval_payload = row.get("retrieval", {})
            answer_text, structured_answer, validation = normalize_preview_answer(
                raw_answer_text,
                retrieval_payload=retrieval_payload,
            )
        failure_kind = _classify_preview_failure(validation)
        rows.append(
            {
                "case_id": row.get("case_id"),
                "category": row.get("category"),
                "answer_status": validation["answer_status"],
                "format_valid": validation["valid"],
                "parsed_from": validation.get("parsed_from"),
                "has_conclusion": validation["has_conclusion"],
                "has_justification": validation["has_justification"],
                "has_limites": validation["has_limits"],
                "citation_count": validation["citation_count"],
                "citation_ids": validation["citation_ids"],
                "with_any_citation": validation["with_any_citation"],
                "errors": validation["errors"],
                "warnings": validation["warnings"],
                "failure_kind": failure_kind,
                "normalized_answer_text": answer_text,
                "structured_answer": structured_answer,
            }
        )

    return {
        "generated_from": batch_payload.get("source_report")
        or batch_payload.get("report_path")
        or batch_payload.get("generated_from")
        or "",
        "answer_contract_version": batch_payload.get("answer_contract_version", PREVIEW_ANSWER_CONTRACT_VERSION),
        "case_count": len(rows),
        "format_valid_count": sum(1 for row in rows if row["format_valid"]),
        "has_conclusion_count": sum(1 for row in rows if row["has_conclusion"]),
        "has_justification_count": sum(1 for row in rows if row["has_justification"]),
        "has_limites_count": sum(1 for row in rows if row["has_limites"]),
        "with_any_citation_count": sum(1 for row in rows if row["with_any_citation"]),
        "format_invalid_count": sum(1 for row in rows if row["failure_kind"] == "format_invalid"),
        "provider_rate_limit_count": sum(1 for row in rows if row["failure_kind"] == "provider_rate_limit"),
        "provider_timeout_count": sum(1 for row in rows if row["failure_kind"] == "provider_timeout"),
        "provider_internal_error_count": sum(1 for row in rows if row["failure_kind"] == "provider_internal"),
        "missing_api_key_count": sum(1 for row in rows if row["failure_kind"] == "missing_api_key"),
        "runtime_error_count": sum(1 for row in rows if row["failure_kind"] == "runtime_error"),
        "rows": rows,
    }


def _classify_preview_failure(validation: dict[str, Any]) -> str:
    if validation.get("valid"):
        return "valid"

    error_text = " ".join(validation.get("errors", [])).lower()
    if "api key is missing" in error_text:
        return "missing_api_key"
    if "ratelimiterror" in error_text or "resource_exhausted" in error_text or "quota exceeded" in error_text:
        return "provider_rate_limit"
    if "apitimeouterror" in error_text or "request timed out" in error_text:
        return "provider_timeout"
    if "internalservererror" in error_text or " 503 " in f" {error_text} ":
        return "provider_internal"
    if (
        "could not parse model output" in error_text
        or "json output appears truncated" in error_text
        or "must be " in error_text
        or "supported answers must contain" in error_text
        or "insufficient_evidence answers must contain" in error_text
        or "citations reference unknown retrieval excerpts" in error_text
    ):
        return "format_invalid"
    if any(
        marker in error_text
        for marker in (
            "runtimeerror:",
            "valueerror:",
            "typeerror:",
            "keyerror:",
            "attributeerror:",
        )
    ):
        return "runtime_error"
    return "format_invalid"


def preview_row_is_valid(
    row: dict[str, Any],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> bool:
    validation = row.get("answer_validation")
    if not isinstance(validation, dict) or not validation.get("valid"):
        return False
    if provider is not None and row.get("provider") != provider:
        return False
    if model is not None and row.get("model") != model:
        return False
    return True


def _missing_key_preview_answer(
    *,
    provider: str,
    model: str,
    prompt_text: str,
    result: PreviewRetrievalResult,
    env_key_name: str,
) -> PreviewAnswer:
    retrieval_payload = Phase8bPreviewRuntime.as_dict(result)
    raw_answer_text = f"{env_key_name} absente: preview LLM non executee."
    return PreviewAnswer(
        provider=provider,
        model=model,
        answer_text=raw_answer_text,
        raw_answer_text=raw_answer_text,
        prompt_text=prompt_text,
        retrieval_payload=retrieval_payload,
        api_called=False,
        contract_version=PREVIEW_ANSWER_CONTRACT_VERSION,
        structured_answer=None,
        answer_validation={
            "valid": False,
            "answer_status": None,
            "has_conclusion": False,
            "has_justification": False,
            "has_limits": False,
            "bullet_count": 0,
            "citation_ids": [],
            "citation_count": 0,
            "with_any_citation": False,
            "errors": ["LLM call skipped because API key is missing"],
            "warnings": [],
            "parsed_from": None,
        },
        response_metadata={"compact_prompt": False},
    )


def generate_preview_answer(
    result: PreviewRetrievalResult,
    *,
    provider: PreviewProvider = DEFAULT_PREVIEW_PROVIDER,
    model: str = DEFAULT_PREVIEW_MODEL,
    api_key: str | None = None,
    validation_errors: list[str] | None = None,
) -> PreviewAnswer:
    compact_mode = _should_use_compact_prompt(validation_errors)
    prompt_text = build_citation_prompt(
        result,
        validation_errors=validation_errors,
        compact_mode=compact_mode,
    )
    load_default_env_files()
    env_key_name = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }.get(provider, "")
    key = api_key or os.environ.get(env_key_name)
    if not key:
        return _missing_key_preview_answer(
            provider=provider,
            model=model,
            prompt_text=prompt_text,
            result=result,
            env_key_name=env_key_name,
        )

    retrieval_payload = Phase8bPreviewRuntime.as_dict(result)
    response_metadata: dict[str, Any] = {"compact_prompt": compact_mode}
    if provider == "gemini":
        client = OpenAI(api_key=key, base_url=GEMINI_OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant fiscal prudent. "
                        "Tu dois repondre uniquement depuis les extraits BOFiP fournis, "
                        "respecter strictement le schema JSON demande, "
                        "et ne jamais inventer de citation."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        message = response.choices[0].message
        raw_answer_text = message.content or ""
        usage = getattr(response, "usage", None)
        response_metadata.update(
            {
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "prompt_tokens": None if usage is None else getattr(usage, "prompt_tokens", None),
                "completion_tokens": None if usage is None else getattr(usage, "completion_tokens", None),
                "total_tokens": None if usage is None else getattr(usage, "total_tokens", None),
            }
        )
    elif provider == "deepseek":
        client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant fiscal prudent. "
                        "Tu dois repondre uniquement depuis les extraits BOFiP fournis, "
                        "respecter strictement le schema JSON demande, "
                        "et ne jamais inventer de citation."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        message = response.choices[0].message
        raw_answer_text = message.content or ""
        usage = getattr(response, "usage", None)
        response_metadata.update(
            {
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "prompt_tokens": None if usage is None else getattr(usage, "prompt_tokens", None),
                "completion_tokens": None if usage is None else getattr(usage, "completion_tokens", None),
                "total_tokens": None if usage is None else getattr(usage, "total_tokens", None),
            }
        )
    elif provider == "openai":
        client = OpenAI(api_key=key)
        response = client.responses.create(
            model=model,
            instructions=(
                "Tu es un assistant fiscal prudent. "
                "Tu dois repondre uniquement depuis les extraits BOFiP fournis, "
                "respecter strictement le schema JSON demande, "
                "et ne jamais inventer de citation."
            ),
            input=prompt_text,
            temperature=0.0,
            max_output_tokens=800,
        )
        raw_answer_text = response.output_text or ""
        usage = getattr(response, "usage", None)
        response_metadata.update(
            {
                "finish_reason": None,
                "input_tokens": None if usage is None else getattr(usage, "input_tokens", None),
                "output_tokens": None if usage is None else getattr(usage, "output_tokens", None),
                "total_tokens": None if usage is None else getattr(usage, "total_tokens", None),
            }
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    answer_text, structured_answer, answer_validation = normalize_preview_answer(
        raw_answer_text,
        retrieval_payload=retrieval_payload,
    )
    return PreviewAnswer(
        provider=provider,
        model=model,
        answer_text=answer_text,
        raw_answer_text=raw_answer_text,
        prompt_text=prompt_text,
        retrieval_payload=retrieval_payload,
        api_called=True,
        contract_version=PREVIEW_ANSWER_CONTRACT_VERSION,
        structured_answer=structured_answer,
        answer_validation=answer_validation,
        response_metadata=response_metadata,
    )


def generate_preview_answer_with_retry(
    result: PreviewRetrievalResult,
    *,
    provider: PreviewProvider = DEFAULT_PREVIEW_PROVIDER,
    model: str = DEFAULT_PREVIEW_MODEL,
    api_key: str | None = None,
    max_attempts: int = 5,
    base_delay_seconds: float = 5.0,
) -> PreviewAnswer:
    attempt = 1
    validation_errors: list[str] | None = None
    while True:
        try:
            preview = generate_preview_answer(
                result,
                provider=provider,
                model=model,
                api_key=api_key,
                validation_errors=validation_errors,
            )
        except (RateLimitError, InternalServerError, APITimeoutError) as exc:
            is_retryable_internal = isinstance(exc, InternalServerError) and getattr(exc, "status_code", None) == 503
            is_retryable_timeout = isinstance(exc, APITimeoutError)
            if not isinstance(exc, RateLimitError) and not is_retryable_internal and not is_retryable_timeout:
                raise
            if attempt >= max_attempts:
                raise
            retry_delay_seconds = _extract_retry_delay_seconds(exc) or 0.0
            time.sleep(max(base_delay_seconds * attempt, retry_delay_seconds))
            attempt += 1
            continue

        if not preview.api_called or preview.answer_validation["valid"] or attempt >= max_attempts:
            return replace(preview, attempt_count=attempt)

        validation_errors = list(preview.answer_validation.get("errors", [])) or [
            "the previous answer did not satisfy the local structured-output validator"
        ]
        raw_answer_text = preview.raw_answer_text.strip()
        if raw_answer_text.startswith("{") and not raw_answer_text.endswith("}"):
            validation_errors.append(
                "the previous JSON response was truncated before the closing brace; return a shorter complete JSON object"
            )
        time.sleep(base_delay_seconds * attempt)
        attempt += 1

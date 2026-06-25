from __future__ import annotations

import re
import unicodedata
from html import unescape


WHITESPACE_RE = re.compile(r"\s+")
LEGAL_REF_PATTERNS = [
    re.compile(r"\barticle\s+[A-ZL]?\s*\.?\s*\d+[A-Za-z0-9\-\s]*(?:bis|ter|quater|quinquies)?\s+du\s+(?:CGI|LPF)\b", re.IGNORECASE),
    re.compile(r"\b(?:CGI|LPF)\s+art\.?\s*[A-ZL]?\s*\.?\s*\d+[A-Za-z0-9\-\s]*(?:bis|ter|quater|quinquies)?\b", re.IGNORECASE),
]


def normalize_whitespace(text: str | None) -> str:
    value = unescape(text or "")
    value = value.replace("\xa0", " ")
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def strip_accents(text: str | None) -> str:
    value = normalize_whitespace(text)
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def estimate_token_count(text: str) -> int:
    stripped = normalize_whitespace(text)
    if not stripped:
        return 0
    return max(1, int(round(len(stripped.split()) * 1.25)))


def slugify(text: str) -> str:
    normalized = normalize_whitespace(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "node"


def extract_legal_refs(text: str) -> list[str]:
    refs: list[str] = []
    for pattern in LEGAL_REF_PATTERNS:
        refs.extend(match.group(0).strip() for match in pattern.finditer(text or ""))
    deduped = []
    seen = set()
    for ref in refs:
        key = ref.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped

from __future__ import annotations

from collections import Counter, defaultdict
import re

from .models import RawDocument
from .text_utils import normalize_whitespace, strip_accents


ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}(?:\s?[0-9])?\b")
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")
QUERY_ACRONYM_RE = re.compile(r"\b[A-Za-z]{2,6}(?:\s?[0-9])?\b")
FRENCH_FUNCTION_WORDS = {
    "a",
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "chez",
    "d",
    "dans",
    "de",
    "des",
    "du",
    "dont",
    "en",
    "et",
    "l",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "ne",
    "ou",
    "par",
    "pas",
    "plus",
    "pour",
    "que",
    "qui",
    "sa",
    "sans",
    "ses",
    "son",
    "sous",
    "sur",
    "un",
    "une",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
}


def _document_text_views(document: RawDocument, *, max_sections: int = 40) -> list[str]:
    texts = [document.title, document.html_title or ""]
    texts.extend(section.title for section in document.sections[:max_sections])
    return [text for text in texts if normalize_whitespace(text)]


def _content_words(text: str) -> list[str]:
    normalized = strip_accents(text).lower().replace("'", " ")
    words = WORD_RE.findall(normalized)
    normalized = []
    for word in words:
        cleaned = word.strip("'")
        if not cleaned or cleaned.isdigit() or cleaned in FRENCH_FUNCTION_WORDS:
            continue
        normalized.append(cleaned)
    return normalized


def _candidate_phrases(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    normalized = normalize_whitespace(text)
    for segment in re.split(r"\s+-\s+|\s*:\s*|\(|\)|;|,", normalized):
        words = _content_words(segment)
        if len(words) < 2:
            continue
        for start in range(len(words)):
            max_size = min(6, len(words) - start)
            for size in range(2, max_size + 1):
                window = words[start : start + size]
                acronym = "".join(word[0] for word in window if word and word[0].isalpha()).upper()
                if 2 <= len(acronym) <= 6:
                    candidates.append((acronym, " ".join(window)))
    return candidates


def build_acronym_expansion_map(
    documents: list[RawDocument],
    *,
    max_sections_per_doc: int = 40,
    min_document_support: int = 2,
    max_document_support: int = 500,
    min_phrase_support: int = 2,
    dominance_ratio: float = 1.5,
    max_expansions_per_acronym: int = 1,
) -> dict[str, list[str]]:
    acronym_documents: dict[str, set[str]] = defaultdict(set)
    texts_by_doc: dict[str, list[str]] = {}

    for document in documents:
        doc_ref = document.boi_reference
        views = _document_text_views(document, max_sections=max_sections_per_doc)
        texts_by_doc[doc_ref] = views
        for text in views:
            for acronym in ACRONYM_RE.findall(text):
                acronym_documents[acronym].add(doc_ref)

    expansion_map: dict[str, list[str]] = {}
    for acronym, doc_refs in acronym_documents.items():
        if acronym.lower() in FRENCH_FUNCTION_WORDS:
            continue
        if len(doc_refs) < min_document_support:
            continue
        if len(doc_refs) > max_document_support:
            continue

        phrase_counts: Counter[str] = Counter()
        for doc_ref in doc_refs:
            for text in texts_by_doc.get(doc_ref, []):
                for candidate_acronym, phrase in _candidate_phrases(text):
                    if candidate_acronym == acronym:
                        phrase_counts[phrase] += 1

        if not phrase_counts:
            continue
        top_candidates = phrase_counts.most_common(max_expansions_per_acronym + 1)
        top_phrase, top_count = top_candidates[0]
        second_count = top_candidates[1][1] if len(top_candidates) > 1 else 0
        if top_count < min_phrase_support:
            continue
        if second_count and (top_count / second_count) < dominance_ratio:
            continue

        kept = [phrase for phrase, count in top_candidates[:max_expansions_per_acronym] if count >= min_phrase_support]
        if kept:
            expansion_map[acronym] = kept
    return expansion_map


def expand_query_with_acronyms(
    query: str,
    expansion_map: dict[str, list[str]],
    *,
    max_expansions_per_query: int = 3,
) -> tuple[str, list[tuple[str, str]]]:
    normalized_query = normalize_whitespace(query)
    normalized_query_ascii = strip_accents(normalized_query).lower()
    seen_phrases: set[str] = set()
    expansions: list[tuple[str, str]] = []

    for token in QUERY_ACRONYM_RE.findall(normalized_query):
        acronym = strip_accents(token).upper()
        phrases = expansion_map.get(acronym, [])
        for phrase in phrases:
            if phrase in normalized_query_ascii:
                continue
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            expansions.append((acronym, phrase))
            if len(expansions) >= max_expansions_per_query:
                break
        if len(expansions) >= max_expansions_per_query:
            break

    if not expansions:
        return normalized_query, []
    expanded_query = normalized_query + " " + " ".join(phrase for _, phrase in expansions)
    return expanded_query, expansions

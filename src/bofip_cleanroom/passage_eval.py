from __future__ import annotations

import re

from .models import ChunkNode
from .text_utils import normalize_whitespace, strip_accents


NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_match_text(text: str | None) -> str:
    normalized = strip_accents(normalize_whitespace(text or "")).lower()
    normalized = NON_ALNUM_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def chunk_matches_passage_gold(chunk: ChunkNode, gold_row: dict) -> bool:
    if chunk.boi_reference != gold_row.get("expected_boi"):
        return False

    chunk_ids_all = gold_row.get("chunk_ids_all", [])
    if chunk_ids_all and chunk.chunk_id not in chunk_ids_all:
        return False
    chunk_ids_any = gold_row.get("chunk_ids_any", [])
    if chunk_ids_any and chunk.chunk_id not in chunk_ids_any:
        return False

    section_text = normalize_match_text(" ".join(chunk.section_path))
    body_text = normalize_match_text(chunk.text)
    combined_text = normalize_match_text(" ".join([section_text, body_text]))

    for term in gold_row.get("section_terms_all", []):
        if normalize_match_text(term) not in section_text:
            return False
    any_section_terms = gold_row.get("section_terms_any", [])
    if any_section_terms:
        if not any(normalize_match_text(term) in section_text for term in any_section_terms):
            return False

    for term in gold_row.get("text_terms_all", []):
        if normalize_match_text(term) not in body_text:
            return False
    any_text_terms = gold_row.get("text_terms_any", [])
    if any_text_terms:
        if not any(normalize_match_text(term) in body_text for term in any_text_terms):
            return False

    for term in gold_row.get("combined_terms_all", []):
        if normalize_match_text(term) not in combined_text:
            return False
    any_combined_terms = gold_row.get("combined_terms_any", [])
    if any_combined_terms:
        if not any(normalize_match_text(term) in combined_text for term in any_combined_terms):
            return False

    return True

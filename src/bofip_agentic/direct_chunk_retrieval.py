from __future__ import annotations

import re
from dataclasses import dataclass

from .lexical_retrieval import LexicalBM25Index, RetrievalHit, get_chunk_search_text_fn
from .models import ChunkNode
from .text_utils import normalize_whitespace, strip_accents


NUMERIC_INTENT_RE = re.compile(
    r"\b("
    r"seuil|seuils|montant|montants|taux|plafond|plafonds|"
    r"bareme|bar[eè]me|chiffre\s+d['’ ]affaires|combien|calcul|calcule|"
    r"pourcentage|euros?|€|ht|hors\s+taxe"
    r")\b",
    re.IGNORECASE,
)
AMOUNT_EVIDENCE_RE = re.compile(
    r"\b\d[\d\s]*(?:[,.]\d+)?\s*(?:€|euros?\b|%|pour\s*cent\b)",
    re.IGNORECASE,
)
REFERENCE_SCALE_RE = re.compile(r"\bBOI-BAREME-\d+\b|\bbar[eè]me\b", re.IGNORECASE)
NUMERIC_SECTION_RE = re.compile(
    r"\b(seuil|seuils|taux|limite|limites|plafond|plafonds|"
    r"bareme|bar[eè]me|chiffre\s+d['’ ]affaires)\b",
    re.IGNORECASE,
)
NUMERIC_TOKEN_RE = re.compile(r"\d(?:[\d\s\u00a0.,]*\d)?")
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
BOI_REFERENCE_RE = re.compile(r"\bBOI-[A-Z0-9]+(?:-[A-Z0-9]+)*\b", re.IGNORECASE)
NON_PRINCIPLE_LEAD_RE = re.compile(r"^\s*(remarque|exemple|r[ée]ponse)\s*:", re.IGNORECASE)
OVERLAP_STOPWORDS = {
    "avec",
    "dans",
    "dois",
    "dont",
    "elle",
    "egal",
    "etre",
    "fait",
    "fixe",
    "hors",
    "leur",
    "pour",
    "quel",
    "quelle",
    "sans",
    "taxe",
}


@dataclass(frozen=True)
class Stage1DocumentHit:
    rank: int
    score: float
    boi_reference: str


@dataclass
class DirectChunkHit:
    global_rank: int
    document_rank: int
    local_rank: int
    local_score: float
    boi_reference: str
    chunk: ChunkNode


@dataclass
class DirectChunkResult:
    chunk_hits: list[DirectChunkHit]


def _has_numeric_intent(query: str) -> bool:
    normalized = strip_accents(query)
    return bool(NUMERIC_INTENT_RE.search(normalized))


def _numeric_evidence_score(query: str, chunk: ChunkNode) -> float:
    if not _has_numeric_intent(query):
        return 0.0

    section_text = normalize_whitespace(" ".join(chunk.section_path))
    body_text = normalize_whitespace(chunk.text)
    full_text = f"{section_text}\n{body_text}"
    normalized_query = strip_accents(query).lower()
    normalized_full_text = strip_accents(full_text).lower()
    score = 0.0
    if AMOUNT_EVIDENCE_RE.search(body_text):
        score += 4.0
    if REFERENCE_SCALE_RE.search(full_text):
        score += 3.0
    if NUMERIC_SECTION_RE.search(section_text):
        score += 2.0
    elif NUMERIC_SECTION_RE.search(body_text[:500]):
        score += 1.0
    query_numbers = _numeric_tokens(normalized_query)
    if query_numbers:
        body_numbers = _numeric_tokens(body_text)
        section_numbers = _numeric_tokens(section_text)
        matched_body = query_numbers & body_numbers
        matched_section = query_numbers & section_numbers
        score += 3.0 * len(matched_body)
        score += 2.0 * len(matched_section - matched_body)
    score += _query_overlap_score(normalized_query, normalized_full_text)
    return score


def _numeric_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for match in NUMERIC_TOKEN_RE.finditer(strip_accents(value).lower()):
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if not digits:
            continue
        compact = re.sub(r"[\s\u00a0]", "", raw)
        if len(digits) >= 3 or re.search(r"[\s\u00a0.]", raw):
            tokens.add(digits)
        if "," in compact and "." not in compact:
            tokens.add(compact.replace(",", "."))
        elif "." in compact and not re.search(r"\.\d{3}(?:\D|$)", compact):
            tokens.add(compact)
    return tokens


def _query_overlap_score(normalized_query: str, normalized_text: str) -> float:
    query_terms = {
        token
        for token in TOKEN_RE.findall(normalized_query)
        if len(token) >= 4 and token not in OVERLAP_STOPWORDS
    }
    if not query_terms:
        return 0.0
    text_terms = set(TOKEN_RE.findall(normalized_text))
    return min(4.0, 0.75 * len(query_terms & text_terms))


def _has_general_rule_intent(query: str) -> bool:
    normalized = strip_accents(query).lower()
    return (
        "regle generale" in normalized
        or "regles generales" in normalized
        or "principes generaux" in normalized
        or "droit commun" in normalized
    )


def _has_exception_intent(query: str) -> bool:
    normalized = strip_accents(query).lower()
    return any(marker in normalized for marker in ("derogation", "derogations", "exception", "exceptions"))


def _structural_chunk_score(query: str, chunk: ChunkNode) -> float:
    if not _has_general_rule_intent(query):
        return 0.0

    section_text = strip_accents(" ".join(chunk.section_path)).lower()
    body_text = strip_accents(chunk.text).lower()
    score = 0.0
    if any(marker in section_text for marker in ("principes generaux", "regle generale", "regles generales", "droit commun")):
        score += 8.0
    if "derogation" in section_text and not _has_exception_intent(query):
        score -= 8.0
    if NON_PRINCIPLE_LEAD_RE.search(body_text):
        score -= 6.0
    return score


def _significant_tokens(value: str) -> list[str]:
    normalized = strip_accents(value).lower()
    return [
        token
        for token in TOKEN_RE.findall(normalized)
        if len(token) >= 4 and token not in OVERLAP_STOPWORDS
    ]


def _lexical_evidence_score(query: str, chunk: ChunkNode) -> float:
    query_tokens = _significant_tokens(query)
    if not query_tokens:
        return 0.0
    query_terms = set(query_tokens)
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))

    section_text = " ".join(chunk.section_path)
    full_text = f"{section_text}\n{' '.join(chunk.legal_refs)}\n{chunk.text}"
    full_tokens = _significant_tokens(full_text)
    full_terms = set(full_tokens)
    full_bigrams = set(zip(full_tokens, full_tokens[1:]))
    section_tokens = _significant_tokens(section_text)
    section_terms = set(section_tokens)
    section_bigrams = set(zip(section_tokens, section_tokens[1:]))

    score = 0.0
    score += min(5.0, 0.8 * len(query_terms & full_terms))
    score += min(12.0, 5.0 * len(query_bigrams & full_bigrams))
    score += min(12.0, 3.0 * len(query_terms & section_terms))
    score += min(28.0, 10.0 * len(query_bigrams & section_bigrams))
    return score


def _prioritize_local_hits(query: str, hits: list[RetrievalHit]) -> list[tuple[int, RetrievalHit]]:
    numeric_weight = 3.0 if _has_numeric_intent(query) else 0.0
    ranked = sorted(
        hits,
        key=lambda hit: (
            -(
                hit.score
                + 2.0 * _lexical_evidence_score(query, hit.chunk)
                + numeric_weight * _numeric_evidence_score(query, hit.chunk)
                + _structural_chunk_score(query, hit.chunk)
            ),
            hit.rank,
            hit.chunk.chunk_id,
        ),
    )
    return [(rank, hit) for rank, hit in enumerate(ranked, start=1)]


def _global_numeric_score(query: str, hit: DirectChunkHit) -> float:
    return _numeric_evidence_score(query, hit.chunk) - 0.6 * max(0, hit.document_rank - 1)


def _exact_boi_references(query: str) -> list[str]:
    references: list[str] = []
    for match in BOI_REFERENCE_RE.finditer(query):
        reference = match.group(0).upper()
        if reference not in references:
            references.append(reference)
    return references[:4]


def _normalize_boi_reference(value: str) -> str:
    return value.strip().upper().removeprefix("BOI-").strip("-")


def _is_followable_boi_reference(value: str) -> bool:
    return _normalize_boi_reference(value).count("-") >= 2


def _is_exact_or_child_reference(boi_reference: str, target_reference: str) -> bool:
    ref = _normalize_boi_reference(boi_reference)
    target = _normalize_boi_reference(target_reference)
    return bool(target and (ref == target or ref.startswith(target + "-")))


class DirectChunkRetriever:
    def __init__(self, chunks: list[ChunkNode], *, local_chunk_mode: str = "full"):
        self.chunks = list(chunks)
        self.local_chunk_mode = local_chunk_mode
        self.local_chunk_search_text_fn = get_chunk_search_text_fn(local_chunk_mode)
        self.chunks_by_reference: dict[str, list[ChunkNode]] = {}
        for chunk in self.chunks:
            self.chunks_by_reference.setdefault(chunk.boi_reference, []).append(chunk)
        self._chunk_indexes_by_reference: dict[str, LexicalBM25Index] = {}

    def _chunk_index_for_reference(self, boi_reference: str) -> LexicalBM25Index:
        if boi_reference not in self._chunk_indexes_by_reference:
            self._chunk_indexes_by_reference[boi_reference] = LexicalBM25Index(
                self.chunks_by_reference.get(boi_reference, []),
                search_text_fn=self.local_chunk_search_text_fn,
            )
        return self._chunk_indexes_by_reference[boi_reference]

    def search(
        self,
        query: str,
        *,
        lexical_query: str | None = None,
        stage1_hits: list[Stage1DocumentHit],
        top_docs: int = 5,
        chunks_per_doc: int = 2,
        max_candidates: int = 6,
    ) -> DirectChunkResult:
        effective_query = lexical_query or query
        chunk_hits: list[DirectChunkHit] = []
        seen_chunk_ids: set[str] = set()
        for doc_hit in stage1_hits[:top_docs]:
            boi_reference = doc_hit.boi_reference
            doc_chunks = self.chunks_by_reference.get(boi_reference, [])
            if not doc_chunks:
                continue
            local_hits = self._chunk_index_for_reference(boi_reference).search(
                effective_query,
                top_k=len(doc_chunks),
            )
            prioritized_hits = _prioritize_local_hits(effective_query, local_hits)[:chunks_per_doc]
            for adjusted_rank, local_hit in prioritized_hits:
                chunk_hits.append(
                    DirectChunkHit(
                        global_rank=0,
                        document_rank=doc_hit.rank,
                        local_rank=adjusted_rank,
                        local_score=local_hit.score,
                        boi_reference=boi_reference,
                        chunk=local_hit.chunk,
                    )
                )
                seen_chunk_ids.add(local_hit.chunk.chunk_id)

        query_references = _exact_boi_references(effective_query)
        followed_references = list(query_references)
        for hit in chunk_hits[:max_candidates]:
            haystack = f"{' '.join(hit.chunk.section_path)}\n{' '.join(hit.chunk.legal_refs)}\n{hit.chunk.text}"
            for reference in _exact_boi_references(haystack):
                if reference not in followed_references:
                    followed_references.append(reference)
        followed_references = followed_references[:6]

        if followed_references:
            follow_rank = 0 if query_references else len(stage1_hits[:top_docs]) + 1
            for reference in followed_references:
                if not _is_followable_boi_reference(reference):
                    continue
                for boi_reference in sorted(self.chunks_by_reference):
                    if not _is_exact_or_child_reference(boi_reference, reference):
                        continue
                    local_hits = self._chunk_index_for_reference(boi_reference).search(
                        effective_query,
                        top_k=len(self.chunks_by_reference[boi_reference]),
                    )
                    prioritized_hits = _prioritize_local_hits(effective_query, local_hits)[:chunks_per_doc]
                    for adjusted_rank, local_hit in prioritized_hits:
                        if local_hit.chunk.chunk_id in seen_chunk_ids:
                            continue
                        chunk_hits.append(
                            DirectChunkHit(
                                global_rank=0,
                                document_rank=follow_rank,
                                local_rank=adjusted_rank,
                                local_score=local_hit.score,
                                boi_reference=boi_reference,
                                chunk=local_hit.chunk,
                            )
                        )
                        seen_chunk_ids.add(local_hit.chunk.chunk_id)
                    follow_rank += 1

            reference_candidates: list[ChunkNode] = []
            for chunk in self.chunks:
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                haystack = f"{' '.join(chunk.section_path)}\n{chunk.text}".upper()
                if any(reference in haystack for reference in followed_references):
                    reference_candidates.append(chunk)
            reference_candidates.sort(
                key=lambda chunk: (
                    -_numeric_evidence_score(effective_query, chunk),
                    chunk.boi_reference,
                    chunk.chunk_id,
                )
            )
            for index, chunk in enumerate(reference_candidates[: max(2, chunks_per_doc)], start=1):
                chunk_hits.append(
                    DirectChunkHit(
                        global_rank=0,
                        document_rank=follow_rank,
                        local_rank=index,
                        local_score=0.0,
                        boi_reference=chunk.boi_reference,
                        chunk=chunk,
                    )
                )
                seen_chunk_ids.add(chunk.chunk_id)

        if _has_numeric_intent(effective_query):
            chunk_hits.sort(
                key=lambda hit: (
                    -_global_numeric_score(effective_query, hit),
                    hit.local_rank,
                    hit.document_rank,
                    -hit.local_score,
                    hit.chunk.chunk_id,
                )
            )
        else:
            chunk_hits.sort(
                key=lambda hit: (
                    hit.local_rank,
                    hit.document_rank,
                    -hit.local_score,
                    hit.chunk.chunk_id,
                )
            )
        chunk_hits = chunk_hits[:max_candidates]
        for index, hit in enumerate(chunk_hits, start=1):
            hit.global_rank = index
        return DirectChunkResult(chunk_hits=chunk_hits)

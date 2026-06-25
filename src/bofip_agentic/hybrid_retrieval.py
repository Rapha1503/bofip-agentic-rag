from __future__ import annotations

from dataclasses import dataclass
import math


EPSILON = 1e-9


@dataclass
class RankedDoc:
    boi_reference: str
    score: float
    rank: int
    source: str


@dataclass
class HybridDocHit:
    rank: int
    boi_reference: str
    score: float
    sources: list[str]
    ranks: dict[str, int]


@dataclass
class SourceRankProfile:
    source: str
    confidence: float
    uniform_probability: float
    document_strengths: dict[str, float]


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    stddev = math.sqrt(variance)
    if stddev <= EPSILON:
        return [0.0 for _ in values]
    return [(value - mean) / stddev for value in values]


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    shifted = [math.exp(value - maximum) for value in values]
    total = sum(shifted)
    if total <= EPSILON:
        return [1.0 / len(values) for _ in values]
    return [value / total for value in shifted]


def compute_source_rank_profiles(
    rankings: dict[str, list[RankedDoc]],
    *,
    top_n: int = 5,
) -> dict[str, SourceRankProfile]:
    profiles: dict[str, SourceRankProfile] = {}
    for source_name, docs in rankings.items():
        trimmed = docs[:top_n]
        if not trimmed:
            profiles[source_name] = SourceRankProfile(
                source=source_name,
                confidence=0.0,
                uniform_probability=0.0,
                document_strengths={},
            )
            continue

        if len(trimmed) == 1:
            profiles[source_name] = SourceRankProfile(
                source=source_name,
                confidence=1.0,
                uniform_probability=1.0,
                document_strengths={trimmed[0].boi_reference: 1.0},
            )
            continue

        zscores = _zscore([doc.score for doc in trimmed])
        probabilities = _softmax(zscores)
        uniform_probability = 1.0 / len(trimmed)
        top_probability = probabilities[0]
        softmax_confidence = max(0.0, (top_probability - uniform_probability) / max(EPSILON, 1.0 - uniform_probability))
        top_score = trimmed[0].score
        runner_up_score = trimmed[1].score
        relative_gap = max(0.0, (top_score - runner_up_score) / max(EPSILON, abs(top_score)))
        confidence = min(1.0, (0.25 * softmax_confidence) + (0.75 * relative_gap))

        denominator = max(EPSILON, top_probability - uniform_probability)
        document_strengths: dict[str, float] = {}
        for doc, probability in zip(trimmed, probabilities):
            strength = max(0.0, (probability - uniform_probability) / denominator)
            document_strengths[doc.boi_reference] = min(1.0, strength)

        profiles[source_name] = SourceRankProfile(
            source=source_name,
            confidence=confidence,
            uniform_probability=uniform_probability,
            document_strengths=document_strengths,
        )
    return profiles


def reciprocal_rank_fuse(
    rankings: dict[str, list[RankedDoc]],
    *,
    top_k: int = 5,
    rank_constant: int = 60,
    source_weights: dict[str, float] | None = None,
) -> list[HybridDocHit]:
    doc_scores: dict[str, float] = {}
    doc_sources: dict[str, set[str]] = {}
    doc_ranks: dict[str, dict[str, int]] = {}

    for source_name, docs in rankings.items():
        weight = 1.0 if source_weights is None else float(source_weights.get(source_name, 1.0))
        for doc in docs:
            doc_scores.setdefault(doc.boi_reference, 0.0)
            doc_scores[doc.boi_reference] += weight * (1.0 / (rank_constant + doc.rank))
            doc_sources.setdefault(doc.boi_reference, set()).add(source_name)
            doc_ranks.setdefault(doc.boi_reference, {})[source_name] = doc.rank

    ordered = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        HybridDocHit(
            rank=index + 1,
            boi_reference=boi_reference,
            score=score,
            sources=sorted(doc_sources.get(boi_reference, set())),
            ranks=doc_ranks.get(boi_reference, {}),
        )
        for index, (boi_reference, score) in enumerate(ordered)
    ]


def confidence_weighted_reciprocal_rank_fuse(
    rankings: dict[str, list[RankedDoc]],
    *,
    top_k: int = 5,
    rank_constant: int = 60,
    source_weights: dict[str, float] | None = None,
    confidence_top_n: int = 5,
    confidence_alpha: float = 1.0,
    score_alpha: float = 0.5,
) -> list[HybridDocHit]:
    profiles = compute_source_rank_profiles(rankings, top_n=confidence_top_n)
    doc_scores: dict[str, float] = {}
    doc_sources: dict[str, set[str]] = {}
    doc_ranks: dict[str, dict[str, int]] = {}

    for source_name, docs in rankings.items():
        base_weight = 1.0 if source_weights is None else float(source_weights.get(source_name, 1.0))
        profile = profiles.get(
            source_name,
            SourceRankProfile(source=source_name, confidence=0.0, uniform_probability=0.0, document_strengths={}),
        )
        source_multiplier = 1.0 + (confidence_alpha * profile.confidence)
        for doc in docs:
            rank_term = 1.0 / (rank_constant + doc.rank)
            document_strength = profile.document_strengths.get(doc.boi_reference, 0.0)
            score_multiplier = 1.0 + (score_alpha * document_strength)
            contribution = base_weight * source_multiplier * score_multiplier * rank_term
            doc_scores.setdefault(doc.boi_reference, 0.0)
            doc_scores[doc.boi_reference] += contribution
            doc_sources.setdefault(doc.boi_reference, set()).add(source_name)
            doc_ranks.setdefault(doc.boi_reference, {})[source_name] = doc.rank

    ordered = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        HybridDocHit(
            rank=index + 1,
            boi_reference=boi_reference,
            score=score,
            sources=sorted(doc_sources.get(boi_reference, set())),
            ranks=doc_ranks.get(boi_reference, {}),
        )
        for index, (boi_reference, score) in enumerate(ordered)
    ]

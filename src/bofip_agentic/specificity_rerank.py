from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from .family_routing import reference_core
from .lexical_retrieval import document_search_text_title_tail, tokenize
from .models import RawDocument


T = TypeVar("T")


@dataclass(frozen=True)
class SpecificityFeatures:
    has_family_neighbor: bool
    tail_overlap: float
    depth_strength: float
    broadness_penalty: float
    score: float


class SpecificityReranker:
    def __init__(
        self,
        documents: list[RawDocument],
        *,
        min_prefix_len: int = 4,
    ):
        self.documents_by_ref = {document.boi_reference: document for document in documents}
        self.min_prefix_len = min_prefix_len
        self.reference_cores = {
            reference: reference_core(reference)
            for reference in self.documents_by_ref
        }
        self.prefix_counts: dict[tuple[str, ...], int] = {}
        for core in self.reference_cores.values():
            for prefix_len in range(min_prefix_len, len(core) + 1):
                prefix = core[:prefix_len]
                self.prefix_counts[prefix] = self.prefix_counts.get(prefix, 0) + 1
        self.descendant_counts = {
            reference: max(0, self.prefix_counts.get(core, 1) - 1)
            for reference, core in self.reference_cores.items()
        }

    def _has_family_neighbor(self, boi_reference: str, candidates: list[str]) -> bool:
        core = self.reference_cores.get(boi_reference, ())
        if len(core) < self.min_prefix_len:
            return False
        for other in candidates:
            if other == boi_reference:
                continue
            other_core = self.reference_cores.get(other, ())
            common = 0
            for left, right in zip(core, other_core):
                if left != right:
                    break
                common += 1
            if common >= self.min_prefix_len:
                return True
        return False

    def features_for(
        self,
        query: str,
        boi_reference: str,
        *,
        candidates: list[str],
        candidate_refs_for_normalization: list[str],
        depth_weight: float = 0.2,
        broadness_weight: float = 0.35,
    ) -> SpecificityFeatures:
        document = self.documents_by_ref.get(boi_reference)
        if document is None:
            return SpecificityFeatures(False, 0.0, 0.0, 0.0, 0.0)

        if not self._has_family_neighbor(boi_reference, candidates):
            return SpecificityFeatures(False, 0.0, 0.0, 0.0, 0.0)

        query_tokens = set(tokenize(query, stem=True))
        tail_tokens = set(tokenize(document_search_text_title_tail(document), stem=True))
        if boi_reference:
            tail_tokens -= set(tokenize(boi_reference, stem=True))
        tail_overlap = (
            len(query_tokens & tail_tokens) / len(tail_tokens)
            if tail_tokens
            else 0.0
        )

        candidate_depths = [
            len(self.reference_cores.get(reference, ()))
            for reference in candidate_refs_for_normalization
        ]
        current_depth = len(self.reference_cores.get(boi_reference, ()))
        min_depth = min(candidate_depths) if candidate_depths else current_depth
        max_depth = max(candidate_depths) if candidate_depths else current_depth
        if max_depth > min_depth:
            depth_strength = (current_depth - min_depth) / (max_depth - min_depth)
        else:
            depth_strength = 0.0

        candidate_descendant_counts = [
            self.descendant_counts.get(reference, 0)
            for reference in candidate_refs_for_normalization
        ]
        current_descendants = self.descendant_counts.get(boi_reference, 0)
        max_descendants = max(candidate_descendant_counts) if candidate_descendant_counts else current_descendants
        broadness_penalty = (
            current_descendants / max_descendants
            if max_descendants > 0
            else 0.0
        )

        score = tail_overlap + (depth_weight * depth_strength) - (broadness_weight * broadness_penalty)
        return SpecificityFeatures(
            has_family_neighbor=True,
            tail_overlap=tail_overlap,
            depth_strength=depth_strength,
            broadness_penalty=broadness_penalty,
            score=score,
        )

    def rerank_hits(
        self,
        query: str,
        hits: list[T],
        *,
        get_reference: Callable[[T], str],
        get_score: Callable[[T], float],
        clone_hit: Callable[[T, int, float], T],
        top_n: int,
        weight: float,
    ) -> list[T]:
        if top_n <= 1 or weight <= 0.0 or len(hits) <= 1:
            return list(hits)

        candidate_hits = list(hits[:top_n])
        candidate_refs = [get_reference(hit) for hit in candidate_hits]
        boosted: list[tuple[float, float, T]] = []
        for hit in hits:
            reference = get_reference(hit)
            features = self.features_for(
                query,
                reference,
                candidates=candidate_refs,
                candidate_refs_for_normalization=candidate_refs,
            )
            specificity_bonus = weight * max(0.0, features.score)
            boosted.append((get_score(hit) + specificity_bonus, specificity_bonus, hit))

        ordered = sorted(
            boosted,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        return [
            clone_hit(hit, index + 1, score)
            for index, (score, _, hit) in enumerate(ordered)
        ]


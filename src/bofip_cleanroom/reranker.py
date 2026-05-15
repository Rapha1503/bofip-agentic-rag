from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class RankedItem:
    item: Any
    score: float


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model = CrossEncoder(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        self.batch_size = batch_size
        self.model_name = model_name
        # Warmup: first predict() loads model to GPU — do it now
        self.model.predict([["warmup", "warmup"]], batch_size=1, show_progress_bar=False)

    def rerank(
        self,
        query: str,
        items: list[Any],
        *,
        get_text,
        top_k: int = 8,
    ) -> list[RankedItem]:
        if not items:
            return []

        pairs = [[query, get_text(item)] for item in items]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = [
            RankedItem(item=item, score=float(score))
            for item, score in zip(items, scores)
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

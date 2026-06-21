from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def CrossEncoder(*args, **kwargs):
    from sentence_transformers import CrossEncoder as _CrossEncoder
    return _CrossEncoder(*args, **kwargs)


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
        try:
            self.model.model.half()
        except Exception:
            pass
        self.batch_size = batch_size
        self.model_name = model_name
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
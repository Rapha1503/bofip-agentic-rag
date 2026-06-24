from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RetrievalProfile:
    mode: str
    label: str
    load_dense: bool


def selected_retrieval_profile(env: Mapping[str, str] | None = None) -> RetrievalProfile:
    values = os.environ if env is None else env
    mode = values.get("BOFIP_RETRIEVAL_MODE", "").strip().lower().replace("-", "_")
    legacy_dense = values.get("BOFIP_ENABLE_DENSE", "").strip().lower() in {"1", "true", "yes"}

    if mode in {"", "lexical"} and not legacy_dense:
        return RetrievalProfile(mode="lexical", label="BM25 full-corpus", load_dense=False)
    if mode in {"hybrid", "dense"} or legacy_dense:
        return RetrievalProfile(mode="hybrid", label="BM25 + embeddings E5", load_dense=True)
    return RetrievalProfile(mode="lexical", label="BM25 full-corpus", load_dense=False)

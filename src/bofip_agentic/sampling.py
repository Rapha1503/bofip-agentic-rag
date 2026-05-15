from __future__ import annotations

from collections import defaultdict, deque
import random

from .discovery import SourceDocumentPaths
from .xml_parser import parse_document_xml


def random_sample_documents(
    documents: list[SourceDocumentPaths],
    limit: int,
    *,
    seed: int = 0,
    allowed_content_types: set[str] | None = None,
) -> list[SourceDocumentPaths]:
    if limit <= 0:
        return []

    filtered: list[SourceDocumentPaths] = []
    for document in documents:
        metadata = parse_document_xml(document.xml_path)
        content_type = metadata.get("content_type") or "UNKNOWN"
        if allowed_content_types is not None and content_type not in allowed_content_types:
            continue
        filtered.append(document)

    if limit >= len(filtered):
        return list(filtered)

    rng = random.Random(seed)
    items = list(filtered)
    rng.shuffle(items)
    return items[:limit]


def stratified_sample_documents(
    documents: list[SourceDocumentPaths],
    limit: int,
    *,
    seed: int = 0,
    allowed_content_types: set[str] | None = None,
) -> list[SourceDocumentPaths]:
    if limit <= 0:
        return []

    buckets: dict[str, deque[SourceDocumentPaths]] = defaultdict(deque)
    for document in documents:
        metadata = parse_document_xml(document.xml_path)
        content_type = metadata.get("content_type") or "UNKNOWN"
        if allowed_content_types is not None and content_type not in allowed_content_types:
            continue
        buckets[content_type].append(document)

    filtered_count = sum(len(bucket) for bucket in buckets.values())
    if limit >= filtered_count:
        return [item for bucket in buckets.values() for item in bucket]

    rng = random.Random(seed)
    shuffled_buckets: dict[str, deque[SourceDocumentPaths]] = {}
    for name, bucket in buckets.items():
        items = list(bucket)
        rng.shuffle(items)
        shuffled_buckets[name] = deque(items)

    sampled: list[SourceDocumentPaths] = []
    bucket_names = sorted(shuffled_buckets)
    while len(sampled) < limit and bucket_names:
        remaining: list[str] = []
        for name in bucket_names:
            bucket = shuffled_buckets[name]
            if bucket and len(sampled) < limit:
                sampled.append(bucket.popleft())
            if bucket:
                remaining.append(name)
        bucket_names = remaining
    return sampled

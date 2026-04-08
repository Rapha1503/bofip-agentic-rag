"""
BOFIP Content Chunker

Handles intelligent chunking of BOFIP content for RAG.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import re
import json


@dataclass
class BOFIPChunk:
    """A chunk of content ready for embedding"""
    chunk_id: str
    text: str
    text_with_context: str

    # Metadata
    boi_reference: str
    doc_id: str
    series: List[str]
    section_title: Optional[str]
    paragraph_number: Optional[str]
    publication_date: str
    source_url: str
    content_type: str

    # Flags
    contains_table: bool = False
    is_header: bool = False
    token_count: int = 0

    # Source identifier: "BOFIP", "CGI", "LPF"
    source: str = "BOFIP"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'BOFIPChunk':
        """Create from dictionary, ignoring unknown fields."""
        import inspect
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation for French text.
    ~4 characters per token is a reasonable approximation.
    """
    return len(text) // 4


def merge_small_chunks(chunks: List[BOFIPChunk], min_tokens: int = 100) -> List[BOFIPChunk]:
    """
    Merge consecutive small chunks that are below min_tokens.

    Args:
        chunks: List of chunks to potentially merge
        min_tokens: Minimum token count before merging

    Returns:
        List of chunks with small ones merged
    """
    if not chunks:
        return []

    merged = []
    buffer_chunks = []

    for chunk in chunks:
        # Skip headers - they define sections
        if chunk.is_header:
            # Flush buffer first
            if buffer_chunks:
                merged.append(_merge_chunk_list(buffer_chunks))
                buffer_chunks = []
            merged.append(chunk)
            continue

        buffer_chunks.append(chunk)
        total_tokens = sum(c.token_count for c in buffer_chunks)

        if total_tokens >= min_tokens:
            merged.append(_merge_chunk_list(buffer_chunks))
            buffer_chunks = []

    # Flush remaining buffer
    if buffer_chunks:
        merged.append(_merge_chunk_list(buffer_chunks))

    return merged


def _merge_chunk_list(chunks: List[BOFIPChunk]) -> BOFIPChunk:
    """Merge a list of chunks into one"""
    if len(chunks) == 1:
        return chunks[0]

    first = chunks[0]

    # Combine texts
    combined_text = ' '.join(c.text for c in chunks)
    combined_context = ' '.join(c.text_with_context for c in chunks)

    # Use first paragraph number, or create range
    para_nums = [c.paragraph_number for c in chunks if c.paragraph_number]
    if len(para_nums) > 1:
        para_num = f"{para_nums[0]}-{para_nums[-1]}"
    elif para_nums:
        para_num = para_nums[0]
    else:
        para_num = None

    # Create new chunk ID with merged indicator
    import uuid
    unique_suffix = uuid.uuid4().hex[:8]
    chunk_id = f"{first.boi_reference}_p{para_num}_{unique_suffix}" if para_num else f"{first.chunk_id}_{unique_suffix}"

    return BOFIPChunk(
        chunk_id=chunk_id,
        text=combined_text,
        text_with_context=combined_context,
        boi_reference=first.boi_reference,
        doc_id=first.doc_id,
        series=first.series,
        section_title=first.section_title,
        paragraph_number=para_num,
        publication_date=first.publication_date,
        source_url=first.source_url,
        content_type=first.content_type,
        contains_table=any(c.contains_table for c in chunks),
        is_header=False,
        token_count=estimate_tokens(combined_text)
    )


def split_large_chunks(chunks: List[BOFIPChunk], max_tokens: int = 800) -> List[BOFIPChunk]:
    """
    Split chunks that exceed max_tokens at sentence boundaries.

    Args:
        chunks: List of chunks to potentially split
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunks with large ones split
    """
    result = []

    for chunk in chunks:
        if chunk.token_count <= max_tokens:
            result.append(chunk)
            continue

        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)

        current_text = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = estimate_tokens(sentence)

            if current_tokens + sentence_tokens > max_tokens and current_text:
                # Create chunk from accumulated sentences
                text = ' '.join(current_text)
                result.append(BOFIPChunk(
                    chunk_id=f"{chunk.chunk_id}_part{len(result)}",
                    text=text,
                    text_with_context=f"{chunk.section_title}\n\n{text}" if chunk.section_title else text,
                    boi_reference=chunk.boi_reference,
                    doc_id=chunk.doc_id,
                    series=chunk.series,
                    section_title=chunk.section_title,
                    paragraph_number=chunk.paragraph_number,
                    publication_date=chunk.publication_date,
                    source_url=chunk.source_url,
                    content_type=chunk.content_type,
                    contains_table=chunk.contains_table,
                    is_header=False,
                    token_count=estimate_tokens(text)
                ))
                current_text = []
                current_tokens = 0

            current_text.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget last chunk
        if current_text:
            text = ' '.join(current_text)
            result.append(BOFIPChunk(
                chunk_id=f"{chunk.chunk_id}_part{len(result)}",
                text=text,
                text_with_context=f"{chunk.section_title}\n\n{text}" if chunk.section_title else text,
                boi_reference=chunk.boi_reference,
                doc_id=chunk.doc_id,
                series=chunk.series,
                section_title=chunk.section_title,
                paragraph_number=chunk.paragraph_number,
                publication_date=chunk.publication_date,
                source_url=chunk.source_url,
                content_type=chunk.content_type,
                contains_table=chunk.contains_table,
                is_header=False,
                token_count=estimate_tokens(text)
            ))

    return result


def process_chunks(chunks: List[BOFIPChunk],
                   min_tokens: int = 100,
                   max_tokens: int = 800) -> List[BOFIPChunk]:
    """
    Process chunks: merge small ones, split large ones.

    Args:
        chunks: Raw chunks from parser
        min_tokens: Minimum chunk size
        max_tokens: Maximum chunk size

    Returns:
        Processed chunks within size bounds
    """
    # First merge small chunks
    merged = merge_small_chunks(chunks, min_tokens)

    # Then split large chunks
    final = split_large_chunks(merged, max_tokens)

    # Filter out empty chunks and headers-only
    final = [c for c in final if c.text.strip() and c.token_count > 20]

    return final


def save_chunks_to_json(chunks: List[BOFIPChunk], filepath: str):
    """Save chunks to JSON file"""
    data = [c.to_dict() for c in chunks]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_chunks_from_json(filepath: str) -> List[BOFIPChunk]:
    """Load chunks from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [BOFIPChunk.from_dict(d) for d in data]

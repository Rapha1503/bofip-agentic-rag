"""
BOFIP Semantic Chunker

Chunks BOFIP documents by FISCAL RULE, not by size.
Each numbered paragraph (§10, §20, etc.) starts a new chunk.
This preserves the complete rule in one atomic unit.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup
from pathlib import Path
import re
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """A semantically meaningful chunk = one fiscal rule"""
    chunk_id: str
    text: str
    text_with_context: str  # Includes hierarchical path

    # Metadata
    boi_reference: str
    doc_id: str
    series: List[str]
    paragraph_number: Optional[str]  # §10, §20, etc.
    section_path: str  # "I > A > 1" hierarchical path
    section_title: Optional[str]  # Immediate parent section
    publication_date: str
    source_url: str
    content_type: str

    # Flags
    contains_table: bool = False
    contains_list: bool = False
    token_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'SemanticChunk':
        return cls(**data)


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for French"""
    return len(text) // 4


class SemanticBOFIPChunker:
    """
    Chunks BOFIP HTML by fiscal rule units.

    Strategy:
    - Track hierarchical context (h1 > h2 > h3 > h4)
    - Each numbered paragraph (§XX) starts a new chunk
    - Accumulate content until next §XX or section header
    - Preserve complete rules as atomic units
    """

    def __init__(self, min_tokens: int = 50, max_tokens: int = 1500):
        """
        Args:
            min_tokens: Minimum chunk size (merge if smaller)
            max_tokens: Maximum chunk size (split if larger)
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def parse_and_chunk(self, html_path: Path, metadata: dict) -> List[SemanticChunk]:
        """
        Parse HTML and create semantic chunks.

        Args:
            html_path: Path to data.html
            metadata: Dict with boi_reference, doc_id, series, publication_date, source_url, content_type

        Returns:
            List of SemanticChunk objects
        """
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml')
        body = soup.find('body')

        if not body:
            return []

        # State tracking
        chunks = []
        section_stack = []  # [(level, title), ...] for h1, h2, h3, h4
        current_paragraph_num = None
        current_content = []
        current_has_table = False
        current_has_list = False

        def get_section_path() -> str:
            """Get current hierarchical path like 'I > A > 1'"""
            if not section_stack:
                return ""
            return " > ".join(title for _, title in section_stack)

        def get_section_title() -> Optional[str]:
            """Get immediate parent section title"""
            if not section_stack:
                return None
            return section_stack[-1][1]

        def flush_chunk():
            """Create chunk from accumulated content"""
            nonlocal current_content, current_paragraph_num, current_has_table, current_has_list

            if not current_content:
                return

            text = ' '.join(current_content).strip()
            if not text:
                current_content = []
                return

            token_count = estimate_tokens(text)

            # Skip very short chunks (likely just headers)
            if token_count < 20:
                current_content = []
                return

            section_path = get_section_path()
            section_title = get_section_title()

            # Create context text with hierarchy
            if section_path:
                context_text = f"[{metadata['boi_reference']}]\n{section_path}\n\n{text}"
            else:
                context_text = f"[{metadata['boi_reference']}]\n\n{text}"

            # Create unique chunk ID
            if current_paragraph_num:
                chunk_id = f"{metadata['boi_reference']}_p{current_paragraph_num}"
            else:
                chunk_id = f"{metadata['boi_reference']}_{len(chunks)}"

            chunk = SemanticChunk(
                chunk_id=chunk_id,
                text=text,
                text_with_context=context_text,
                boi_reference=metadata['boi_reference'],
                doc_id=metadata['doc_id'],
                series=metadata.get('series', []),
                paragraph_number=current_paragraph_num,
                section_path=section_path,
                section_title=section_title,
                publication_date=metadata.get('publication_date', ''),
                source_url=metadata.get('source_url', ''),
                content_type=metadata.get('content_type', 'Commentaire'),
                contains_table=current_has_table,
                contains_list=current_has_list,
                token_count=token_count
            )

            chunks.append(chunk)

            # Reset state
            current_content = []
            current_has_table = False
            current_has_list = False

        def update_section_stack(level: int, title: str):
            """Update hierarchical section stack"""
            nonlocal section_stack
            # Remove all sections at same level or deeper
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))

        # Process all elements
        for element in body.children:
            if element.name is None:
                continue

            # Headers define sections
            if element.name == 'h1':
                flush_chunk()
                title = element.get_text(strip=True)
                update_section_stack(1, title)
                current_paragraph_num = None

            elif element.name == 'h2':
                flush_chunk()
                title = element.get_text(strip=True)
                update_section_stack(2, title)
                current_paragraph_num = None

            elif element.name == 'h3':
                flush_chunk()
                title = element.get_text(strip=True)
                update_section_stack(3, title)
                current_paragraph_num = None

            elif element.name == 'h4':
                flush_chunk()
                title = element.get_text(strip=True)
                update_section_stack(4, title)
                current_paragraph_num = None

            # Paragraphs
            elif element.name == 'p':
                text = element.get_text(strip=True)

                # Check if this is a paragraph number (e.g., "10", "20", "30")
                # Can be plain "10" or bold "<strong>10</strong>"
                if re.match(r'^\d+$', text):
                    # New numbered paragraph = new chunk
                    flush_chunk()
                    current_paragraph_num = text
                # Skip deleted paragraph markers like "(70)"
                elif re.match(r'^\(\d+\)$', text):
                    continue
                else:
                    # Regular content - accumulate
                    if text:
                        current_content.append(text)

            # Tables
            elif element.name == 'table':
                table_text = self._extract_table_text(element)
                if table_text:
                    current_content.append(table_text)
                    current_has_table = True

            # Lists
            elif element.name in ['ul', 'ol']:
                list_text = self._extract_list_text(element)
                if list_text:
                    current_content.append(list_text)
                    current_has_list = True

        # Flush remaining content
        flush_chunk()

        # Post-process: merge small chunks, split large ones
        chunks = self._post_process_chunks(chunks, metadata)

        return chunks

    def _extract_table_text(self, table_element) -> str:
        """Extract table as readable text"""
        lines = []

        caption = table_element.find('caption')
        if caption:
            lines.append(f"Tableau: {caption.get_text(strip=True)}")

        # Headers
        thead = table_element.find('thead')
        if thead:
            headers = [th.get_text(strip=True) for th in thead.find_all('th')]
            if headers:
                lines.append(" | ".join(headers))

        # Rows
        tbody = table_element.find('tbody') or table_element
        for row in tbody.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
            if cells:
                lines.append(" | ".join(cells))

        return "\n".join(lines)

    def _extract_list_text(self, list_element) -> str:
        """Extract list as readable text"""
        items = []
        for li in list_element.find_all('li', recursive=False):
            text = li.get_text(strip=True)
            if text:
                items.append(f"- {text}")
        return "\n".join(items)

    def _post_process_chunks(self, chunks: List[SemanticChunk], metadata: dict) -> List[SemanticChunk]:
        """Merge small chunks and split large ones"""
        if not chunks:
            return []

        result = []
        buffer = None

        for chunk in chunks:
            # If chunk is very small, try to merge with buffer
            if chunk.token_count < self.min_tokens:
                if buffer is None:
                    buffer = chunk
                else:
                    # Merge with buffer
                    buffer = self._merge_chunks(buffer, chunk, metadata)
            else:
                # Flush buffer first
                if buffer is not None:
                    if buffer.token_count >= self.min_tokens:
                        result.append(buffer)
                    elif result:
                        # Merge tiny buffer with previous
                        result[-1] = self._merge_chunks(result[-1], buffer, metadata)
                    else:
                        result.append(buffer)
                    buffer = None

                # Handle current chunk
                if chunk.token_count > self.max_tokens:
                    # Split large chunk
                    split_chunks = self._split_chunk(chunk, metadata)
                    result.extend(split_chunks)
                else:
                    result.append(chunk)

        # Flush remaining buffer
        if buffer is not None:
            if result and buffer.token_count < self.min_tokens:
                result[-1] = self._merge_chunks(result[-1], buffer, metadata)
            else:
                result.append(buffer)

        return result

    def _merge_chunks(self, chunk1: SemanticChunk, chunk2: SemanticChunk, metadata: dict) -> SemanticChunk:
        """Merge two chunks"""
        combined_text = f"{chunk1.text}\n\n{chunk2.text}"

        # Use first chunk's paragraph number, or create range
        if chunk1.paragraph_number and chunk2.paragraph_number:
            para_num = f"{chunk1.paragraph_number}-{chunk2.paragraph_number}"
        else:
            para_num = chunk1.paragraph_number or chunk2.paragraph_number

        return SemanticChunk(
            chunk_id=f"{metadata['boi_reference']}_p{para_num}" if para_num else chunk1.chunk_id,
            text=combined_text,
            text_with_context=f"[{metadata['boi_reference']}]\n{chunk1.section_path}\n\n{combined_text}",
            boi_reference=metadata['boi_reference'],
            doc_id=metadata['doc_id'],
            series=metadata.get('series', []),
            paragraph_number=para_num,
            section_path=chunk1.section_path,
            section_title=chunk1.section_title,
            publication_date=metadata.get('publication_date', ''),
            source_url=metadata.get('source_url', ''),
            content_type=metadata.get('content_type', 'Commentaire'),
            contains_table=chunk1.contains_table or chunk2.contains_table,
            contains_list=chunk1.contains_list or chunk2.contains_list,
            token_count=estimate_tokens(combined_text)
        )

    def _split_chunk(self, chunk: SemanticChunk, metadata: dict) -> List[SemanticChunk]:
        """Split a large chunk at natural boundaries"""
        text = chunk.text
        result = []

        # Try to split at double newlines first, then sentences
        if '\n\n' in text:
            parts = text.split('\n\n')
        else:
            # Split at sentence boundaries
            parts = re.split(r'(?<=[.!?])\s+', text)

        current_text = []
        current_tokens = 0

        for part in parts:
            part_tokens = estimate_tokens(part)

            if current_tokens + part_tokens > self.max_tokens and current_text:
                # Create chunk from accumulated text
                chunk_text = '\n\n'.join(current_text) if '\n\n' in text else ' '.join(current_text)
                result.append(SemanticChunk(
                    chunk_id=f"{chunk.chunk_id}_part{len(result)}",
                    text=chunk_text,
                    text_with_context=f"[{metadata['boi_reference']}]\n{chunk.section_path}\n\n{chunk_text}",
                    boi_reference=chunk.boi_reference,
                    doc_id=chunk.doc_id,
                    series=chunk.series,
                    paragraph_number=chunk.paragraph_number,
                    section_path=chunk.section_path,
                    section_title=chunk.section_title,
                    publication_date=chunk.publication_date,
                    source_url=chunk.source_url,
                    content_type=chunk.content_type,
                    contains_table=chunk.contains_table,
                    contains_list=chunk.contains_list,
                    token_count=estimate_tokens(chunk_text)
                ))
                current_text = []
                current_tokens = 0

            current_text.append(part)
            current_tokens += part_tokens

        # Don't forget last part
        if current_text:
            chunk_text = '\n\n'.join(current_text) if '\n\n' in text else ' '.join(current_text)
            result.append(SemanticChunk(
                chunk_id=f"{chunk.chunk_id}_part{len(result)}",
                text=chunk_text,
                text_with_context=f"[{metadata['boi_reference']}]\n{chunk.section_path}\n\n{chunk_text}",
                boi_reference=chunk.boi_reference,
                doc_id=chunk.doc_id,
                series=chunk.series,
                paragraph_number=chunk.paragraph_number,
                section_path=chunk.section_path,
                section_title=chunk.section_title,
                publication_date=chunk.publication_date,
                source_url=chunk.source_url,
                content_type=chunk.content_type,
                contains_table=chunk.contains_table,
                contains_list=chunk.contains_list,
                token_count=estimate_tokens(chunk_text)
            ))

        return result if result else [chunk]


def save_semantic_chunks(chunks: List[SemanticChunk], filepath: str):
    """Save chunks to JSON"""
    data = [c.to_dict() for c in chunks]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_semantic_chunks(filepath: str) -> List[SemanticChunk]:
    """Load chunks from JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [SemanticChunk.from_dict(d) for d in data]


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample document (resolved relative to project root)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    test_html = (
        PROJECT_ROOT
        / "data" / "raw" / "bofip_extracted" / "BOFiP" / "documents"
        / "Contenu" / "Commentaire" / "RFPI" / "1002-PGP" / "2012-09-12" / "data.html"
    )

    if test_html.exists():
        metadata = {
            'boi_reference': 'BOI-RFPI-SPEC-20-40-20-20-20120912',
            'doc_id': '1002-PGP',
            'series': ['RFPI'],
            'publication_date': '2012-09-12',
            'source_url': 'https://bofip.impots.gouv.fr/bofip/1002-PGP',
            'content_type': 'Commentaire'
        }

        chunker = SemanticBOFIPChunker()
        chunks = chunker.parse_and_chunk(test_html, metadata)

        print(f"\n{'='*60}")
        print(f"SEMANTIC CHUNKING TEST")
        print(f"{'='*60}")
        print(f"Document: {metadata['boi_reference']}")
        print(f"Chunks created: {len(chunks)}")
        print()

        for i, chunk in enumerate(chunks[:5]):
            print(f"--- Chunk {i+1} ---")
            print(f"ID: {chunk.chunk_id}")
            print(f"§: {chunk.paragraph_number}")
            print(f"Path: {chunk.section_path}")
            print(f"Tokens: {chunk.token_count}")
            print(f"Text: {chunk.text[:200]}...")
            print()

        # Stats
        token_counts = [c.token_count for c in chunks]
        print(f"\nStats:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg tokens: {sum(token_counts)//len(token_counts)}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
    else:
        print(f"Test file not found: {test_html}")

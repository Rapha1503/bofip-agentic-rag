"""
PDF Parser for CGI and LPF documents.

KISS approach: Extract text, chunk by page with article detection.
Compatible with existing SemanticChunk structure.
"""

import pdfplumber
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """A chunk from a PDF document - compatible with SemanticChunk structure."""
    chunk_id: str
    text: str
    text_with_context: str

    # Metadata - compatible with BOFIP chunks
    doc_reference: str  # Article number or section
    doc_id: str  # Unique document identifier
    source: str  # "CGI" or "LPF"
    series: List[str]  # Empty for PDF sources
    paragraph_number: Optional[str]
    section_path: str
    section_title: Optional[str]
    publication_date: str
    source_url: str
    content_type: str
    page_number: int

    # Flags
    contains_table: bool = False
    contains_list: bool = False
    token_count: int = 0

    # For compatibility with BOFIP chunks
    boi_reference: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for French."""
    return len(text) // 4


class PDFDocumentParser:
    """
    Parse PDF documents (CGI, LPF) into chunks.

    KISS Strategy:
    - Extract text page by page
    - Detect article boundaries (pattern: number + optional letters)
    - Create one chunk per article or per page if no articles found
    - Preserve French characters
    """

    def __init__(self, min_tokens: int = 50, max_tokens: int = 1500):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        # Article line pattern for CGI/LPF headings.
        # Supports variants like: "39 A.", "102 ter", "279-0 bis A", "L. 10", "L. 10-0 A. D."
        suffixes = (
            "bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|"
            "undecies|duodecies|terdecies|quaterdecies|quindecies|sexdecies"
        )
        self.article_pattern = re.compile(
            rf'^\s*(?P<ref>(?!\d{{4}}-\d{{2}}-\d{{2}}\b)(?:[LRA]\*?\.?\s*)?'
            rf'\d+[A-Z]?(?:-\d+[A-Z]?)*(?:\s+(?:[A-Z]|{suffixes}))*)'
            rf'(?:\.\s*[A-Z]\.)?\.?\s*$',
            re.MULTILINE | re.IGNORECASE
        )

    def parse_pdf(self, pdf_path: Path, source: str) -> List[PDFChunk]:
        """
        Parse a PDF and return chunks.

        Args:
            pdf_path: Path to PDF file
            source: "CGI" or "LPF"

        Returns:
            List of PDFChunk objects
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return []

        logger.info(f"Parsing PDF: {pdf_path.name} (source: {source})")

        chunks = []
        doc_id = f"{source}-{pdf_path.stem}"

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Total pages: {total_pages}")

                # Extract publication date from first page
                first_page_text = pdf.pages[0].extract_text() or ""
                pub_date = self._extract_date(first_page_text)
                last_article_ref: Optional[str] = None

                # Process each page
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""

                    if not text.strip():
                        continue

                    # Skip TOC pages (first 10 pages typically)
                    if page_num < 10 and self._is_toc_page(text):
                        continue

                    # Extract section title from header
                    section_title = self._extract_section_title(text)

                    # Chunk the page content
                    page_chunks = self._chunk_page(
                        text=text,
                        page_num=page_num + 1,
                        doc_id=doc_id,
                        source=source,
                        section_title=section_title,
                        publication_date=pub_date
                    )

                    # Map fallback page chunks to the previous article when likely a continuation.
                    # This avoids generic "p.X" references in legal code body pages.
                    page_number = page_num + 1
                    if (
                        page_number > 5
                        and len(page_chunks) == 1
                        and page_chunks[0].doc_reference.startswith("Page ")
                        and last_article_ref
                    ):
                        self._promote_page_chunk_to_article(page_chunks[0], last_article_ref)

                    # Track last seen article for continuation handling
                    for chunk in page_chunks:
                        if chunk.doc_reference.startswith("Article "):
                            last_article_ref = chunk.doc_reference.replace("Article ", "", 1).strip()

                    chunks.extend(page_chunks)

                logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise

        return chunks

    def _extract_date(self, text: str) -> str:
        """Extract publication date from text."""
        # Priority: explicit "Derniere modification" style labels
        labeled = re.search(
            r'(?:Derni[eè]re\s+modification|Mise\s+[aà]\s+jour)\s*:?\s*(\d{4}-\d{2}-\d{2})',
            text,
            re.IGNORECASE
        )
        if labeled:
            return labeled.group(1)

        # ISO format (YYYY-MM-DD)
        iso = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        if iso:
            return iso.group(1)

        # French numeric format (DD/MM/YYYY) -> normalize to YYYY-MM-DD
        fr_num = re.search(r'\b(\d{2})/(\d{2})/(\d{4})\b', text)
        if fr_num:
            day, month, year = fr_num.group(1), fr_num.group(2), fr_num.group(3)
            return f"{year}-{month}-{day}"

        # French textual format (e.g., "18 janvier 2026")
        fr_text = re.search(
            r'\b(\d{1,2})\s+'
            r'(janvier|fevrier|f[ée]vrier|mars|avril|mai|juin|juillet|'
            r'aout|ao[ûu]t|septembre|octobre|novembre|decembre|d[ée]cembre)\s+'
            r'(\d{4})\b',
            text,
            re.IGNORECASE
        )
        if fr_text:
            month_map = {
                'janvier': '01',
                'fevrier': '02', 'février': '02',
                'mars': '03',
                'avril': '04',
                'mai': '05',
                'juin': '06',
                'juillet': '07',
                'aout': '08', 'août': '08',
                'septembre': '09',
                'octobre': '10',
                'novembre': '11',
                'decembre': '12', 'décembre': '12',
            }
            day = fr_text.group(1).zfill(2)
            month = month_map.get(fr_text.group(2).lower(), '')
            year = fr_text.group(3)
            if month:
                return f"{year}-{month}-{day}"

        return ""

    def _is_toc_page(self, text: str) -> bool:
        """Check if page is a table of contents page."""
        # TOC pages have many "..." sequences and chapter references
        dots_count = text.count('...')
        lines = text.split('\n')

        # If more than 30% of lines have dots, likely TOC
        if len(lines) > 5 and dots_count > len(lines) * 0.3:
            return True

        # Check for "Plan" heading
        if text.strip().startswith('Plan'):
            return True

        return False

    def _extract_section_title(self, text: str) -> Optional[str]:
        """Extract section title from page header."""
        # First line often contains hierarchical path
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if it's a section header (contains " - ")
            if ' - ' in first_line and len(first_line) < 300:
                return first_line
        return None

    def _chunk_page(
        self,
        text: str,
        page_num: int,
        doc_id: str,
        source: str,
        section_title: Optional[str],
        publication_date: str
    ) -> List[PDFChunk]:
        """
        Chunk a single page's text.

        Strategy:
        - Try to split by article boundaries
        - If no articles found, use the whole page as one chunk
        - Respect max_tokens limit
        """
        chunks = []

        # Remove header line if it's a section path
        lines = text.split('\n')
        if section_title and lines and lines[0].strip() == section_title:
            text = '\n'.join(lines[1:])

        # Find article boundaries
        articles = self._split_by_articles(text)

        if not articles:
            # No articles found - use whole page as chunk
            articles = [("page", text)]

        for article_ref, article_text in articles:
            if not article_text.strip():
                continue

            token_count = estimate_tokens(article_text)

            # Skip very small chunks
            if token_count < self.min_tokens:
                continue

            # Split large chunks
            if token_count > self.max_tokens:
                sub_chunks = self._split_large_text(article_text, article_ref)
                for i, sub_text in enumerate(sub_chunks):
                    chunk = self._create_chunk(
                        text=sub_text,
                        article_ref=f"{article_ref}_{i+1}" if len(sub_chunks) > 1 else article_ref,
                        page_num=page_num,
                        doc_id=doc_id,
                        source=source,
                        section_title=section_title,
                        publication_date=publication_date
                    )
                    chunks.append(chunk)
            else:
                chunk = self._create_chunk(
                    text=article_text,
                    article_ref=article_ref,
                    page_num=page_num,
                    doc_id=doc_id,
                    source=source,
                    section_title=section_title,
                    publication_date=publication_date
                )
                chunks.append(chunk)

        return chunks

    def _split_by_articles(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text by article numbers.

        Returns list of (article_ref, content) tuples.
        """
        matches = list(self.article_pattern.finditer(text))

        if not matches:
            return []

        articles = []
        for i, match in enumerate(matches):
            article_ref = self._normalize_article_ref(match.group('ref'))
            start = match.end()

            # End is either next match or end of text
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            content = text[start:end].strip()
            articles.append((article_ref, content))

        return articles

    def _normalize_article_ref(self, article_ref: str) -> str:
        """Normalize an extracted article reference for stable lookups."""
        ref = article_ref.replace('\xa0', ' ')
        ref = re.sub(r'\s+', ' ', ref).strip().rstrip('.')

        # Normalize LPF prefixes like "L 64" or "L64" -> "L. 64"
        prefix_match = re.match(r'^([LRA])(\*)?\.?\s*(.+)$', ref, re.IGNORECASE)
        if prefix_match:
            letter = prefix_match.group(1).upper()
            star = prefix_match.group(2) or ''
            remainder = prefix_match.group(3).strip()
            ref = f"{letter}{star}. {remainder}"

        # Canonical lowercase for legal Latin suffixes
        ref = re.sub(
            r'\b(BIS|TER|QUATER|QUINQUIES|SEXIES|SEPTIES|OCTIES|NONIES|DECIES|'
            r'UNDECIES|DUODECIES|TERDECIES|QUATERDECIES|QUINDECIES|SEXDECIES)\b',
            lambda m: m.group(1).lower(),
            ref,
            flags=re.IGNORECASE
        )

        return ref

    def _promote_page_chunk_to_article(self, chunk: PDFChunk, article_ref: str) -> None:
        """Convert a generic page chunk into an article continuation chunk."""
        chunk.doc_reference = f"Article {article_ref}"
        chunk.boi_reference = f"{chunk.source} Art. {article_ref}"

        # Keep context coherent for embeddings and reranking.
        if not chunk.text_with_context.startswith("Article "):
            chunk.text_with_context = f"Article {article_ref}\n{chunk.text_with_context}"

    def _split_large_text(self, text: str, base_ref: str) -> List[str]:
        """Split large text into smaller chunks by paragraphs."""
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = estimate_tokens(para)

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _create_chunk(
        self,
        text: str,
        article_ref: str,
        page_num: int,
        doc_id: str,
        source: str,
        section_title: Optional[str],
        publication_date: str
    ) -> PDFChunk:
        """Create a PDFChunk object."""
        # Generate unique chunk ID
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        chunk_id = f"{doc_id}_p{page_num}_{article_ref}_{text_hash}"

        # Clean chunk ID (remove spaces and special chars)
        chunk_id = re.sub(r'[^a-zA-Z0-9_-]', '_', chunk_id)

        # Create context text with section info
        context_parts = []
        if section_title:
            context_parts.append(f"[{section_title}]")
        if article_ref != "page":
            context_parts.append(f"Article {article_ref}")
        context_parts.append(text)
        text_with_context = '\n'.join(context_parts)

        # Detect tables (simple heuristic: presence of | or multiple aligned numbers)
        contains_table = '|' in text or bool(re.search(r'\d+\s+\d+\s+\d+', text))

        # Detect lists
        contains_list = bool(re.search(r'^\s*(?:[-*]|\d+[.)]|[a-zA-Z][.)])\s', text, re.MULTILINE))

        return PDFChunk(
            chunk_id=chunk_id,
            text=text,
            text_with_context=text_with_context,
            doc_reference=f"Article {article_ref}" if article_ref != "page" else f"Page {page_num}",
            doc_id=doc_id,
            source=source,
            series=[],  # PDFs don't have BOFIP series
            paragraph_number=None,
            section_path="",
            section_title=section_title,
            publication_date=publication_date,
            source_url=f"file://{doc_id}#page={page_num}",
            content_type=source,
            page_number=page_num,
            contains_table=contains_table,
            contains_list=contains_list,
            token_count=estimate_tokens(text),
            boi_reference=f"{source} Art. {article_ref}" if article_ref != "page" else f"{source} p.{page_num}",
        )


def parse_cgi_pdf(pdf_path: Path) -> List[PDFChunk]:
    """Parse Code GÃ©nÃ©ral des ImpÃ´ts PDF."""
    parser = PDFDocumentParser()
    return parser.parse_pdf(pdf_path, source="CGI")


def parse_lpf_pdf(pdf_path: Path) -> List[PDFChunk]:
    """Parse Livre des ProcÃ©dures Fiscales PDF."""
    parser = PDFDocumentParser()
    return parser.parse_pdf(pdf_path, source="LPF")


if __name__ == "__main__":
    # Test with sample
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        source = sys.argv[2] if len(sys.argv) > 2 else "CGI"

        parser = PDFDocumentParser()
        chunks = parser.parse_pdf(pdf_path, source)

        print(f"\nTotal chunks: {len(chunks)}")
        for chunk in chunks[:5]:
            print(f"\n--- {chunk.chunk_id} ---")
            print(f"Source: {chunk.source}")
            print(f"Page: {chunk.page_number}")
            print(f"Ref: {chunk.doc_reference}")
            print(f"Tokens: {chunk.token_count}")
            print(f"Text preview: {chunk.text[:200]}...")


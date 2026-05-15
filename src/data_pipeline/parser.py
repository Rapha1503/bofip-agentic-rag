"""
BOFIP Document Parser

Parses document.xml (metadata) and data.html (content) from BOFIP Open Data.
"""

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
import re


@dataclass
class BOFIPMetadata:
    """Metadata extracted from document.xml"""
    doc_id: str                          # e.g., "1032-PGP"
    boi_reference: str                   # e.g., "BOI-IR-LIQ-10-10-10-30-20240618"
    title: str                           # Full title
    publication_date: str                # e.g., "2024-06-18"
    series: List[str]                    # e.g., ["IR", "LIQ"]
    content_type: str                    # e.g., "Commentaire"
    content_level: str                   # e.g., "Enfant" or "Parent"
    source_url: str                      # Full URL to bofip.impots.gouv.fr
    related_refs: List[str] = field(default_factory=list)  # Referenced documents
    referenced_by: List[str] = field(default_factory=list) # Documents referencing this


@dataclass
class BOFIPChunk:
    """A chunk of content ready for embedding"""
    chunk_id: str                        # Unique ID: "{boi_ref}_§{para_num}"
    text: str                            # Clean text content
    text_with_context: str               # Text with section title for retrieval

    # Metadata
    boi_reference: str
    doc_id: str
    series: List[str]
    section_title: Optional[str]
    paragraph_number: Optional[str]
    publication_date: str
    source_url: str

    # Flags
    contains_table: bool = False
    is_header: bool = False

    # Token count (approximate)
    token_count: int = 0


def parse_metadata(xml_path: Path) -> BOFIPMetadata:
    """
    Parse document.xml to extract metadata.

    Args:
        xml_path: Path to document.xml

    Returns:
        BOFIPMetadata object
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Define namespaces
    namespaces = {
        'dc': 'http://purl.org/dc/elements/1.1',
        'bofip': 'https://bofip.impots.gouv.fr'
    }

    # Find Dublin Core section
    dc = root.find('.//dc:dublincore', namespaces)
    if dc is None:
        # Try without namespace (some files may differ)
        dc = root.find('.//{http://purl.org/dc/elements/1.1}dublincore')

    # Find BOFIP section
    bofip = root.find('.//bofip:bodgfip', namespaces)
    if bofip is None:
        bofip = root.find('.//{https://bofip.impots.gouv.fr}bodgfip')

    # Extract Dublin Core fields
    title = ""
    doc_id = ""
    source_url = ""
    publication_date = ""
    series = []
    related_refs = []
    referenced_by = []

    if dc is not None:
        # Title (may be in CDATA)
        title_elem = dc.find('{http://purl.org/dc/elements/1.1}title')
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()

        # Date
        date_elem = dc.find('{http://purl.org/dc/elements/1.1}date')
        if date_elem is not None and date_elem.text:
            publication_date = date_elem.text.strip()

        # Identifiers (first is doc_id, second is URL)
        identifiers = dc.findall('{http://purl.org/dc/elements/1.1}identifier')
        for i, ident in enumerate(identifiers):
            if ident.text:
                if i == 0:
                    doc_id = ident.text.strip()
                elif ident.text.startswith('http'):
                    source_url = ident.text.strip()

        # Subjects (series)
        subjects = dc.findall('{http://purl.org/dc/elements/1.1}subject')
        for subj in subjects:
            if subj.text:
                series.append(subj.text.strip())

        # Relations
        relations = dc.findall('{http://purl.org/dc/elements/1.1}relation')
        for rel in relations:
            if rel.text:
                rel_type = rel.get('type', '')
                # Extract doc ID from format like "Contenu:2494-PGP" or "Contenu.Commentaire:2494-PGP"
                match = re.search(r':(\d+-PGP)', rel.text)
                if match:
                    ref_id = match.group(1)
                    if rel_type == 'references':
                        related_refs.append(ref_id)
                    elif rel_type == 'isReferencedBy':
                        referenced_by.append(ref_id)

    # Extract BOFIP-specific fields
    boi_reference = ""
    content_type = ""
    content_level = ""

    if bofip is not None:
        # BOI reference
        boi_elem = bofip.find('{https://bofip.impots.gouv.fr}contenu_id')
        if boi_elem is not None and boi_elem.text:
            boi_reference = boi_elem.text.strip()

        # Content type
        type_elem = bofip.find('{https://bofip.impots.gouv.fr}contenu_type')
        if type_elem is not None and type_elem.text:
            content_type = type_elem.text.strip()

        # Content level
        level_elem = bofip.find('{https://bofip.impots.gouv.fr}contenu_niveau')
        if level_elem is not None and level_elem.text:
            content_level = level_elem.text.strip()

    return BOFIPMetadata(
        doc_id=doc_id,
        boi_reference=boi_reference,
        title=title,
        publication_date=publication_date,
        series=series,
        content_type=content_type,
        content_level=content_level,
        source_url=source_url,
        related_refs=related_refs,
        referenced_by=referenced_by
    )


def parse_content(html_path: Path, metadata: BOFIPMetadata) -> List[BOFIPChunk]:
    """
    Parse data.html and create chunks for RAG.

    Args:
        html_path: Path to data.html
        metadata: BOFIPMetadata from document.xml

    Returns:
        List of BOFIPChunk objects
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'lxml')
    body = soup.find('body')

    if not body:
        return []

    chunks = []
    current_section = None
    current_paragraph_num = None
    current_text_parts = []

    def estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 chars per token for French"""
        return len(text) // 4

    def create_chunk(text: str, section: str, para_num: str,
                     contains_table: bool = False, is_header: bool = False) -> BOFIPChunk:
        """Create a BOFIPChunk from accumulated text"""
        text = text.strip()
        if not text:
            return None

        # Create text with context (section title prepended)
        context_text = f"{section}\n\n{text}" if section else text

        # Create unique chunk ID (use _p instead of § for ASCII safety, add counter for uniqueness)
        if para_num:
            chunk_id = f"{metadata.boi_reference}_p{para_num}_{len(chunks)}"
        else:
            chunk_id = f"{metadata.boi_reference}_{len(chunks)}"

        return BOFIPChunk(
            chunk_id=chunk_id,
            text=text,
            text_with_context=context_text,
            boi_reference=metadata.boi_reference,
            doc_id=metadata.doc_id,
            series=metadata.series,
            section_title=section,
            paragraph_number=para_num,
            publication_date=metadata.publication_date,
            source_url=metadata.source_url,
            contains_table=contains_table,
            is_header=is_header,
            token_count=estimate_tokens(text)
        )

    def flush_current():
        """Flush accumulated text into a chunk"""
        nonlocal current_text_parts, current_paragraph_num
        if current_text_parts:
            text = ' '.join(current_text_parts)
            chunk = create_chunk(text, current_section, current_paragraph_num)
            if chunk:
                chunks.append(chunk)
            current_text_parts = []

    # Process all elements in order
    for element in body.children:
        if element.name is None:  # Skip text nodes
            continue

        # Headers (h1, h2, h3) - start new section
        if element.name in ['h1', 'h2', 'h3']:
            flush_current()
            current_section = element.get_text(strip=True)
            # Create a header chunk
            header_chunk = create_chunk(current_section, None, None, is_header=True)
            if header_chunk:
                chunks.append(header_chunk)

        # Paragraphs
        elif element.name == 'p':
            text = element.get_text(strip=True)

            # Check if this is a paragraph number (just a number like "1", "10", "20")
            if re.match(r'^\d+$', text):
                flush_current()
                current_paragraph_num = text
            # Check if it's an empty/deleted paragraph marker like "(70)"
            elif re.match(r'^\(\d+\)$', text):
                continue  # Skip deleted paragraphs
            else:
                current_text_parts.append(text)

        # Tables
        elif element.name == 'table':
            flush_current()
            # Extract table as text
            table_text = extract_table_text(element)
            chunk = create_chunk(table_text, current_section, current_paragraph_num, contains_table=True)
            if chunk:
                chunks.append(chunk)

        # Lists
        elif element.name in ['ul', 'ol']:
            list_text = element.get_text(separator=' ', strip=True)
            current_text_parts.append(list_text)

    # Flush any remaining text
    flush_current()

    return chunks


def extract_table_text(table_element) -> str:
    """
    Extract table content as readable text.

    Args:
        table_element: BeautifulSoup table element

    Returns:
        Table content as formatted text
    """
    lines = []

    # Caption
    caption = table_element.find('caption')
    if caption:
        lines.append(f"Tableau: {caption.get_text(strip=True)}")
        lines.append("")

    # Headers
    thead = table_element.find('thead')
    if thead:
        headers = []
        for th in thead.find_all('th'):
            headers.append(th.get_text(strip=True))
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * 40)

    # Body rows
    tbody = table_element.find('tbody')
    if tbody:
        for row in tbody.find_all('tr'):
            cells = []
            for cell in row.find_all(['th', 'td']):
                cells.append(cell.get_text(strip=True))
            if cells:
                lines.append(" | ".join(cells))

    return "\n".join(lines)


def parse_document(doc_dir: Path) -> tuple[BOFIPMetadata, List[BOFIPChunk]]:
    """
    Parse a complete BOFIP document (metadata + content).

    Args:
        doc_dir: Path to the date folder containing document.xml and data.html

    Returns:
        Tuple of (BOFIPMetadata, List[BOFIPChunk])
    """
    xml_path = doc_dir / 'document.xml'
    html_path = doc_dir / 'data.html'

    if not xml_path.exists():
        raise FileNotFoundError(f"document.xml not found in {doc_dir}")

    metadata = parse_metadata(xml_path)

    chunks = []
    if html_path.exists():
        chunks = parse_content(html_path, metadata)

    return metadata, chunks


# Test function
if __name__ == "__main__":
    import sys

    # Test with a sample document (resolved relative to project root)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    test_dir = (
        PROJECT_ROOT
        / "data" / "raw" / "bofip_extracted" / "BOFiP" / "documents"
        / "Contenu" / "Commentaire" / "IR" / "1032-PGP" / "2024-06-18"
    )

    if test_dir.exists():
        print("=" * 60)
        print("Testing BOFIP Parser")
        print("=" * 60)

        metadata, chunks = parse_document(test_dir)

        print("\n--- METADATA ---")
        print(f"Doc ID: {metadata.doc_id}")
        print(f"BOI Reference: {metadata.boi_reference}")
        print(f"Title: {metadata.title[:80]}...")
        print(f"Date: {metadata.publication_date}")
        print(f"Series: {metadata.series}")
        print(f"Type: {metadata.content_type}")
        print(f"Level: {metadata.content_level}")
        print(f"Related refs: {metadata.related_refs}")

        print(f"\n--- CHUNKS ({len(chunks)} total) ---")
        for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Section: {chunk.section_title}")
            print(f"  Paragraph: {chunk.paragraph_number}")
            print(f"  Tokens: ~{chunk.token_count}")
            print(f"  Table: {chunk.contains_table}")
            print(f"  Text: {chunk.text[:100]}...")

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    else:
        print(f"Test directory not found: {test_dir}")

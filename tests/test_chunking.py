from __future__ import annotations

import unittest

from bofip_cleanroom.chunking import build_chunks
from bofip_cleanroom.models import RawDocument, RawParagraph, RawSectionNode


class ChunkingTests(unittest.TestCase):
    def test_parent_child_creates_parent_and_children(self) -> None:
        document = RawDocument(
            document_id="TESTDOC",
            boi_reference="BOI-TEST-0001",
            title="Titre",
            document_type="Contenu",
            content_type="Commentaire",
            publication_date="2024-01-01",
            source_url=None,
            language="fr",
            sections=[
                RawSectionNode(
                    section_id="sec-1",
                    parent_section_id=None,
                    level=1,
                    order_index=0,
                    title="Section 1",
                    anchor="sec-1",
                    path=["Section 1"],
                )
            ],
            paragraphs=[
                RawParagraph(
                    paragraph_id="p-1",
                    section_id="sec-1",
                    order_index=0,
                    html_tag="p",
                    anchor="p-1",
                    paragraph_number=None,
                    text="Premier paragraphe.",
                    legal_refs=[],
                    links=[],
                ),
                RawParagraph(
                    paragraph_id="p-2",
                    section_id="sec-1",
                    order_index=1,
                    html_tag="p",
                    anchor="p-2",
                    paragraph_number=None,
                    text="Second paragraphe.",
                    legal_refs=[],
                    links=[],
                ),
            ],
        )

        chunks = build_chunks(document, strategy="parent_child")

        parent_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "parent_section"]
        child_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "child_paragraph"]

        self.assertTrue(parent_chunks)
        self.assertEqual(len(child_chunks), 2)
        parent_ids = {chunk.chunk_id for chunk in parent_chunks}
        self.assertTrue(all(chunk.parent_chunk_id in parent_ids for chunk in child_chunks))

    def test_paragraph_preserving_merges_trivial_fragments(self) -> None:
        document = RawDocument(
            document_id="TESTDOC2",
            boi_reference="BOI-TEST-0002",
            title="Titre",
            document_type="Contenu",
            content_type="Commentaire",
            publication_date="2024-01-01",
            source_url=None,
            language="fr",
            sections=[
                RawSectionNode(
                    section_id="sec-1",
                    parent_section_id=None,
                    level=1,
                    order_index=0,
                    title="Section 1",
                    anchor="sec-1",
                    path=["Section 1"],
                )
            ],
            paragraphs=[
                RawParagraph(
                    paragraph_id="p-1",
                    section_id="sec-1",
                    order_index=0,
                    html_tag="p",
                    anchor="p-1",
                    paragraph_number="70",
                    text="(70)",
                    legal_refs=[],
                    links=[],
                ),
                RawParagraph(
                    paragraph_id="p-2",
                    section_id="sec-1",
                    order_index=1,
                    html_tag="p",
                    anchor="p-2",
                    paragraph_number=None,
                    text="Le texte substantiel suit immédiatement et doit absorber le renvoi.",
                    legal_refs=[],
                    links=[],
                ),
            ],
        )

        chunks = build_chunks(document, strategy="paragraph_preserving", max_tokens=100)

        self.assertEqual(len(chunks), 1)
        self.assertIn("(70)", chunks[0].text)
        self.assertIn("texte substantiel", chunks[0].text)

    def test_long_paragraph_is_split_under_limit(self) -> None:
        long_text = " ".join(f"mot{i}" for i in range(500))
        document = RawDocument(
            document_id="TESTDOC3",
            boi_reference="BOI-TEST-0003",
            title="Titre",
            document_type="Contenu",
            content_type="Commentaire",
            publication_date="2024-01-01",
            source_url=None,
            language="fr",
            sections=[
                RawSectionNode(
                    section_id="sec-1",
                    parent_section_id=None,
                    level=1,
                    order_index=0,
                    title="Section 1",
                    anchor="sec-1",
                    path=["Section 1"],
                )
            ],
            paragraphs=[
                RawParagraph(
                    paragraph_id="p-1",
                    section_id="sec-1",
                    order_index=0,
                    html_tag="p",
                    anchor="p-1",
                    paragraph_number=None,
                    text=long_text,
                    legal_refs=[],
                    links=[],
                )
            ],
        )

        chunks = build_chunks(document, strategy="section_window", max_tokens=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 80 for chunk in chunks))

    def test_section_window_merges_trivial_boundary_chunk_into_next_section(self) -> None:
        document = RawDocument(
            document_id="TESTDOC4",
            boi_reference="BOI-TEST-0004",
            title="Titre",
            document_type="Contenu",
            content_type="Commentaire",
            publication_date="2024-01-01",
            source_url=None,
            language="fr",
            sections=[
                RawSectionNode(
                    section_id="sec-a",
                    parent_section_id=None,
                    level=1,
                    order_index=0,
                    title="Section A",
                    anchor="sec-a",
                    path=["Section A"],
                ),
                RawSectionNode(
                    section_id="sec-b",
                    parent_section_id=None,
                    level=1,
                    order_index=1,
                    title="Section B",
                    anchor="sec-b",
                    path=["Section B"],
                ),
            ],
            paragraphs=[
                RawParagraph(
                    paragraph_id="p-a",
                    section_id="sec-a",
                    order_index=0,
                    html_tag="p",
                    anchor="p-a",
                    paragraph_number=None,
                    text="(60)",
                    legal_refs=[],
                    links=[],
                ),
                RawParagraph(
                    paragraph_id="p-b",
                    section_id="sec-b",
                    order_index=1,
                    html_tag="p",
                    anchor="p-b",
                    paragraph_number="70",
                    text="Le paragraphe substantiel de la section suivante doit absorber le marqueur isolé.",
                    legal_refs=[],
                    links=[],
                ),
            ],
        )

        chunks = build_chunks(document, strategy="section_window", max_tokens=100, min_tokens=40)

        self.assertEqual(len(chunks), 1)
        self.assertIn("(60)", chunks[0].text)
        self.assertIn("section suivante", chunks[0].text)


if __name__ == "__main__":
    unittest.main()

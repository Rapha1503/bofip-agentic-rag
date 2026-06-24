from __future__ import annotations

import unittest

from bofip_agentic.chunking import build_chunks
from bofip_agentic.models import RawDocument, RawParagraph, RawSectionNode, RawTable


def _document(*, paragraphs: list[RawParagraph], tables: list[RawTable] | None = None) -> RawDocument:
    return RawDocument(
        document_id="doc",
        boi_reference="BOI-TEST-1",
        title="Document test",
        document_type="BOI",
        content_type="commentary",
        publication_date="2026-01-01",
        source_url=None,
        language="fr",
        sections=[
            RawSectionNode("s1", None, 1, 0, "Section A", None, ["Section A"]),
            RawSectionNode("s2", None, 1, 1, "Section B", None, ["Section B"]),
        ],
        paragraphs=paragraphs,
        tables=tables or [],
    )


class ChunkingTests(unittest.TestCase):
    def test_small_chunks_are_not_merged_across_sections(self) -> None:
        document = _document(
            paragraphs=[
                RawParagraph("p1", "s1", 0, "p", None, None, "Court fragment A.", [], []),
                RawParagraph("p2", "s2", 1, "p", None, None, "Court fragment B.", [], []),
            ],
        )

        chunks = build_chunks(document, strategy="section_window", min_tokens=40)

        self.assertEqual(2, len(chunks))
        self.assertEqual(["Section A"], chunks[0].section_path)
        self.assertEqual(["Section B"], chunks[1].section_path)
        self.assertNotIn("Court fragment B", chunks[0].text)

    def test_tables_are_emitted_as_retrievable_chunks(self) -> None:
        document = _document(
            paragraphs=[],
            tables=[
                RawTable(
                    table_id="t1",
                    section_id="s1",
                    order_index=0,
                    headers=["Tranche", "Taux"],
                    rows=[["0 a 10 000 euros", "0 %"], ["10 001 a 25 000 euros", "11 %"]],
                    linearized_text="Tranche | Taux\n0 a 10 000 euros | 0 %\n10 001 a 25 000 euros | 11 %",
                )
            ],
        )

        chunks = build_chunks(document, strategy="section_window")

        self.assertEqual(1, len(chunks))
        self.assertEqual("table", chunks[0].chunk_kind)
        self.assertEqual(["Section A"], chunks[0].section_path)
        self.assertEqual(["t1"], chunks[0].paragraph_range)
        self.assertIn("10 001 a 25 000 euros", chunks[0].text)


if __name__ == "__main__":
    unittest.main()

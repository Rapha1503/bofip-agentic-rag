from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from bofip_cleanroom.discovery import SourceDocumentPaths
from bofip_cleanroom.sampling import stratified_sample_documents


class SamplingTests(unittest.TestCase):
    def test_stratified_sample_respects_seed_and_limit(self) -> None:
        docs = [
            SourceDocumentPaths("a1", "2024-01-01", [], Path("a1.xml"), Path("a1.html")),
            SourceDocumentPaths("a2", "2024-01-01", [], Path("a2.xml"), Path("a2.html")),
            SourceDocumentPaths("b1", "2024-01-01", [], Path("b1.xml"), Path("b1.html")),
            SourceDocumentPaths("b2", "2024-01-01", [], Path("b2.xml"), Path("b2.html")),
        ]

        def fake_parse(path: Path) -> dict:
            return {"content_type": "Commentaire" if path.stem.startswith("a") else "Barème"}

        with patch("bofip_cleanroom.sampling.parse_document_xml", side_effect=fake_parse):
            result_a = stratified_sample_documents(docs, 2, seed=7)
            result_b = stratified_sample_documents(docs, 2, seed=7)

        self.assertEqual(len(result_a), 2)
        self.assertEqual([doc.document_id for doc in result_a], [doc.document_id for doc in result_b])


if __name__ == "__main__":
    unittest.main()
